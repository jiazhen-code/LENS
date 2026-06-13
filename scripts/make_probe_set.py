"""Export a layer-probe set {image, question, mask} from the same val datasets eval.py uses.

Renders each GT mask to a binary PNG and writes a JSONL the layer probe consumes, so you
don't have to hand-build masks:

    python scripts/make_probe_set.py --config configs/default.yaml --out data/layer_probe.jsonl --limit 200
    # then:
    python scripts/probe_layers.py --config configs/qwen2vl_2b.yaml --probe-set data/layer_probe.jsonl --limit 200

The val split is taken from the config's eval section (cfg.eval.val_dataset / val_split /
dataset_dir): ReasonSeg, or refcoco / refcoco+ / refcocog. Image + mask paths written are
absolute, so the probe needs no --image-root.

Geometry note: the probe upsamples the patch-grid heatmap to the mask's original H×W, which
is the exact inverse of a plain square resize (Qwen2-VL's image_hw resize). For LLaVA's
CLIP resize+crop it is approximate, but the per-layer *ranking* is unaffected (the same
geometric warp applies to every layer).
"""

import argparse
import json
import os
import sys
from functools import partial

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "LISA"))

from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor

from lens.config import load_config
from lens.data.test_dataset import TestReasoningDataset, TestReferDataset
from lens.data.trainval_dataset import collate_fn_val
from lens.models.backbones import extract_user_instruction

REASON_SETS = ["ReasonSeg"]
REFER_SETS = ["refcoco", "refcoco+", "refcocog"]


def build_val_dataset(ev):
    proc = CLIPImageProcessor.from_pretrained(ev.clip_vision_model)
    if ev.val_dataset in REASON_SETS:
        return TestReasoningDataset(
            ev.dataset_dir, proc, ev.image_size,
            datasetname=ev.val_dataset, train_test_split=ev.val_split,
            eval_only=ev.eval_only, conversation_records={},
        )
    if ev.val_dataset in REFER_SETS:
        return TestReferDataset(
            ev.dataset_dir, proc, ev.image_size,
            datasetname=ev.val_dataset, train_test_split=ev.val_split,
            conversation_records={},
        )
    raise ValueError(f"Unsupported val_dataset for probe export: {ev.val_dataset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml",
                    help="uses its eval.* section to pick the val split.")
    ap.add_argument("--out", default="data/layer_probe.jsonl")
    ap.add_argument("--mask-dir", default=None,
                    help="where rendered masks go (default: <out_dir>/probe_masks).")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = build_val_dataset(cfg.eval)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=partial(collate_fn_val, tokenizer=None,
                           use_mm_start_end=cfg.eval.use_mm_start_end),
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    mask_dir = args.mask_dir or os.path.join(out_dir, "probe_masks")
    os.makedirs(mask_dir, exist_ok=True)

    n, skipped = 0, 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for batch in loader:
            if n >= args.limit:
                break
            img_path = batch["image_paths"][0]
            convs = batch["conversation_list"]
            masks = batch["masks_list"][0]
            if masks.dim() == 2:
                masks = masks.unsqueeze(0)
            k = min(len(convs), masks.shape[0])
            for j in range(k):
                if n >= args.limit:
                    break
                m = (masks[j] > 0)
                if int(m.sum()) == 0:
                    skipped += 1
                    continue
                q = extract_user_instruction(convs[j])
                if not q:
                    skipped += 1
                    continue
                mask_path = os.path.join(mask_dir, f"{n:05d}.png")
                Image.fromarray((m.cpu().numpy().astype("uint8") * 255)).save(mask_path)
                fh.write(json.dumps({"image": os.path.abspath(img_path),
                                     "question": q, "mask": mask_path},
                                    ensure_ascii=False) + "\n")
                n += 1

    print(f"wrote {n} probe samples -> {args.out}  (masks in {mask_dir}, skipped {skipped})")


if __name__ == "__main__":
    main()
