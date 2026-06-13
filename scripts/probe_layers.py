"""Pick the feature / warm-start layer for a backbone by measuring per-layer localization.

For each candidate decoder layer L we read the FROZEN backbone's attention from the last
text token to the image tokens, reshape it to the patch grid, upsample to the image, and
score how well it localizes the ground-truth mask -- with NO training. The layer that
localizes best is the natural choice for `model.backbone.selected_layer_id` and
`grounding_init_start_layer` (the original LLaVA-1.5-7B value of 14 was found the same way).

Two threshold-free scores, averaged over the probe set:
  * pointing-game : fraction of samples whose attention peak falls inside the GT mask
  * mass-in-mask  : fraction of the (normalized) attention mass that lands inside the mask

This is a cheap PROXY for the warm-started fusion head's initial attention; confirm the top
1-2 layers with a short training run (compare val gIoU). Run single-GPU:

    python scripts/probe_layers.py --config configs/qwen2vl_2b.yaml \
        --probe-set data/layer_probe.jsonl --image-root /imgs --limit 200

probe-set JSONL: {"image": "...", "question": "...", "mask": "path/to/binary_mask.png"}
(mask: white/non-zero = object). Build ~200 from your ReasonSeg / RefCOCO val split.
"""

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lens.config import load_config
from lens.models.backbones import build_mllm_backbone, force_eager_attention


def image_token_id(backbone):
    tid = getattr(backbone, "_image_token_id", None)
    if tid is not None:
        return tid
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(backbone.model.config, attr, None)
        if v is not None:
            return v
    return backbone.processor.tokenizer.convert_tokens_to_ids("<image>")


def build_inputs(backbone, image, question):
    if getattr(backbone, "image_hw", None):
        image = image.resize((backbone.image_hw, backbone.image_hw))
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = backbone.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = backbone.processor(text=[text], images=[image], return_tensors="pt", padding=True)
    return {
        k: (v.to(backbone.model.device).to(backbone.model.dtype)
            if v.is_floating_point() else v.to(backbone.model.device))
        for k, v in inputs.items() if isinstance(v, torch.Tensor)
    }


@torch.no_grad()
def per_layer_maps(backbone, inputs, img_tok_id):
    """Per decoder layer, return (attention_map, feature_sim_map) on the patch grid for the
    last text token:
      attention  = last-token -> image-token attention (mean over heads), at layer L;
      feature-sim= cosine(last-token feature, each image-token feature) at layer L's output.
    Raw attention is sink-dominated / diffuse, so the feature-similarity map is usually the
    cleaner localizer -- we report both.
    """
    out = backbone.model(**inputs, output_attentions=True,
                         output_hidden_states=True, use_cache=False)
    attns = out.attentions       # tuple[L] of [1, heads, seq, seq]
    hs = out.hidden_states       # tuple[L+1] of [1, seq, hidden]
    input_ids = inputs["input_ids"][0]
    image_cols = (input_ids == img_tok_id)
    n_img = int(image_cols.sum())
    grid = int(round(n_img ** 0.5))
    if grid * grid != n_img or n_img == 0:
        return None
    non_img = torch.nonzero(~image_cols, as_tuple=False).flatten()
    q = int(non_img[-1])  # last text token (the query token, as in LENS)
    attn_maps, feat_maps = [], []
    for li in range(len(attns)):
        a = attns[li][0].float().mean(0)[q]                       # [seq]
        attn_maps.append(a[image_cols].reshape(grid, grid))
        h = hs[li + 1][0].float()                                 # [seq, hidden] (layer li out)
        sim = F.cosine_similarity(h[image_cols], h[q][None, :], dim=-1)  # [n_img]
        feat_maps.append(sim.reshape(grid, grid))
    del out, attns, hs
    return attn_maps, feat_maps


def score_localization(heatmap_grid, mask_np):
    """Returns (pointing_hit in {0,1}, mass_in_mask, mask_area_fraction).

    concentration = mass_in_mask / mask_area_fraction is computed by the caller: it is 1.0
    for a diffuse map (chance) and > 1 when the map concentrates on the object -- area- and
    size-normalised, so it is far more stable than raw pointing-game at small N.
    """
    hm = heatmap_grid[None, None].float()
    H, W = mask_np.shape
    up = F.interpolate(hm, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    up = up.clamp(min=0)
    mask_t = torch.from_numpy(mask_np).to(up.device)
    area = float(mask_t.float().mean())
    total = up.sum()
    if total <= 0 or area <= 0:
        return 0.0, 0.0, area
    up = up / total
    idx = int(torch.argmax(up))
    py, px = idx // W, idx % W
    hit = float(bool(mask_t[py, px]))
    mass = float(up[mask_t].sum())
    return hit, mass, area


def load_probe_set(path, image_root):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("samples") or data.get("data") or [data]
    except json.JSONDecodeError:
        data = [json.loads(l) for l in text.splitlines() if l.strip()]
    items = []
    for d in data:
        img = d["image"]
        msk = d["mask"]
        if image_root:
            if not os.path.isabs(img):
                img = os.path.join(image_root, img)
            if not os.path.isabs(msk):
                msk = os.path.join(image_root, msk)
        items.append({"image": img, "question": d["question"], "mask": msk})
    return items

from tqdm import tqdm
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--probe-set", required=True, help="JSONL {image, question, mask}")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = build_mllm_backbone(cfg.model.backbone)
    backbone.to(torch.bfloat16).to(device)
    backbone.model.eval()
    force_eager_attention(backbone.model)  # output_attentions needs eager
    img_tok_id = image_token_id(backbone)

    items = load_probe_set(args.probe_set, args.image_root)
    if args.limit:
        items = items[: args.limit]

    acc = None
    n, skipped = 0, 0
    for it in tqdm(items):
        try:
            image = Image.open(it["image"]).convert("RGB")
            mask = np.array(Image.open(it["mask"]).convert("L")) > 127
            if mask.sum() == 0:
                skipped += 1
                continue
            inputs = build_inputs(backbone, image, it["question"])
            res = per_layer_maps(backbone, inputs, img_tok_id)
            if res is None:
                skipped += 1
                continue
            attn_maps, feat_maps = res
            if acc is None:
                L = len(attn_maps)
                acc = {k: [0.0] * L for k in ("a_hit", "a_conc", "f_hit", "f_conc")}
            for li in range(len(attn_maps)):
                ah, am, area = score_localization(attn_maps[li].cpu(), mask)
                fh_, fm, _ = score_localization(feat_maps[li].cpu(), mask)
                acc["a_hit"][li] += ah
                acc["a_conc"][li] += (am / area) if area > 0 else 0.0
                acc["f_hit"][li] += fh_
                acc["f_conc"][li] += (fm / area) if area > 0 else 0.0
            n += 1
        except Exception as e:
            skipped += 1
            print(f"[skip] {it['image']}: {type(e).__name__}: {e}")

    if not n:
        print("no usable samples")
        return
    if n < 50:
        print(f"WARNING: only {n} samples — pointing% is very noisy; use >=100-200 for a "
              f"trustworthy ranking.")

    L = len(acc["a_hit"])
    a_pt = [acc["a_hit"][i] / n for i in range(L)]
    a_cc = [acc["a_conc"][i] / n for i in range(L)]
    f_pt = [acc["f_hit"][i] / n for i in range(L)]
    f_cc = [acc["f_conc"][i] / n for i in range(L)]
    # Rank by attention concentration: LENS's heatmap IS attention, so attn_conc is the
    # faithful signal. feat_conc is a cross-check (clean for LLaVA, often flat for models
    # whose text/image features don't share a cosine space, e.g. Qwen2-VL).
    order = sorted(range(L), key=lambda i: a_cc[i], reverse=True)
    best = order[0]
    feat_order = sorted(range(L), key=lambda i: f_cc[i], reverse=True)

    print("=" * 78)
    print(f"Layer probe — {cfg.model.backbone.name}  ({n} samples, {skipped} skipped)")
    print("conc = (mass in mask)/(mask area);  1.0 = chance (diffuse),  >1 = localizes")
    print(f"{'layer':>5}{'attn_pt%':>10}{'attn_conc':>11}{'feat_pt%':>10}{'feat_conc':>11}")
    for i in range(L):
        star = "  <-- best (attn_conc)" if i == best else ""
        print(f"{i:>5}{100*a_pt[i]:>10.1f}{a_cc[i]:>11.2f}{100*f_pt[i]:>10.1f}{f_cc[i]:>11.2f}{star}")
    print("-" * 78)
    print(f"Top-3 by attn_conc: {order[:3]}   Top-3 by feat_conc: {feat_order[:3]}")
    if (max(a_cc) - min(a_cc)) < 0.15:
        print("NOTE: attn_conc is nearly flat -> probe can't discriminate; use the middle "
              "layer + a short training sweep.")
    if best >= L - 2:
        print("NOTE: peak is at the very last layer(s) -> often attention collapse, not "
              "localization; prefer the cleanest mid/late peak instead.")
    print(f"Suggested: selected_layer_id = grounding_init_start_layer = {best}  "
          f"(then sweep {best}±1 with a short training run, decide by val gIoU)")
    print("=" * 78)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"backbone": cfg.model.backbone.name, "n": n,
                       "attn_pointing": a_pt, "attn_conc": a_cc,
                       "feat_pointing": f_pt, "feat_conc": f_cc,
                       "ranking_by_attn_conc": order, "ranking_by_feat_conc": feat_order,
                       "suggested_layer": best}, fh, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
