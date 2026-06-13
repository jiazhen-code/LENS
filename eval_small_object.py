"""Small-object mitigation eval (Table_R 5) — training-free NxN image splitting, frozen backbone.

Rebuttal W5: LENS reads keypoints off a *frozen* backbone's coarse attention grid (e.g. 24x24
for LLaVA-1.5-7B), so a small target occupies only a fraction of one grid cell and its attention
peak is too blurry for reliable keypoint extraction. This script measures the training-free fix
from the rebuttal: partition the image into an NxN grid (default 2x2) of non-overlapping
quadrants, push EACH quadrant **independently** through the unchanged, frozen LENS pipeline
(frozen MLLM -> attention keypoints -> SAM mask decode), then stitch the quadrant masks back to
their original locations. Because each quadrant is upsampled to the backbone's native input by
the SAME preprocessing the baseline uses (data_deal: ResizeLongestSide + pad -> the MLLM
processor's 336/384 square), a small object becomes a larger fraction of the quadrant and spans
more visual tokens, giving a sharper, better-localized peak. No parameters and no fine-tuning;
the ONLY thing that differs from the baseline is the input (full image vs. four quadrants).

It evaluates BOTH settings on a small-object subset of ReasonSeg -- every instance whose target
area (mask pixels == 1) is below ``--area-thresh`` of the image area:

  * ``full image (single pass)``  -- byte-identical to eval.py's per-sample path; and
  * ``+ NxN split (ours)``        -- the training-free split + stitch.

gIoU/cIoU are computed EXACTLY as in eval.validate (same intersectionAndUnionGPU, same 0.6 mask
threshold, cIoU = sum(I)/sum(U), gIoU = mean per-mask IoU). One run = one backbone (chosen by
--config) = the two table rows:

    # LLaVA-1.5-7B (fixed-resolution backbone)
    accelerate launch eval_small_object.py --config configs/default.yaml            --checkpoint /path/to/full_N.pth
    # LLaVA-OneVision-7B (high-resolution backbone)
    accelerate launch eval_small_object.py --config configs/llava_onevision_7b.yaml --checkpoint /path/to/full_N.pth

(A plain ``python eval_small_object.py ...`` also works -- single GPU.) This is a standalone
script: it does NOT modify eval.py / eval_routing.py or any other evaluation path; it only
imports them read-only so the metric and preprocessing are identical to the main eval.
"""

import argparse
import json
import os
import sys
from itertools import product

# Make the third-party LISA tree importable (provides `model_lisa`, used by lens.data) and keep
# the repo root on sys.path (for `lens`) -- mirrors eval.py / eval_routing.py.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "LISA"))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import gather_object

from lens.config import load_config
from lens.models import load_full_state_dict
from lens.models.conditioner import MLLM_Conditioner, FullModel
from lens.models.decoder import Decoder
from lens.data.test_dataset import TestReasoningDataset, collate_fn_test
from lens.data.data_processing import get_mask_from_json
from lens.utils.inference import data_deal
from lens.models.backbones.base import extract_user_instruction  # pull the bare query from a prompt
from eval import intersectionAndUnionGPU  # reuse the EXACT metric the main eval uses

# Same binarization threshold as eval.validate (`(pred.sigmoid() > 0.6)`); kept here so the
# baseline row of this table matches eval.py's numbers on the subset bit-for-bit.
MASK_THRESH = 0.6


def _cast(t, precision):
    """Match eval.prepare_input's precision handling for the SAM input tensor."""
    if precision == "fp16":
        return t.half()
    if precision == "bf16":
        return t.bfloat16()
    return t.float()


def _run_pipeline(full, sam_img, pil_img, prompt, resized_size, ori_size, device, precision):
    """One forward of the frozen LENS pipeline on a single (image, prompt).

    Returns (binarized mask (1, H, W) at ``ori_size``; head_map (Gh, Gw) = the peak-normalized
    language->image attention the keypoints are read off). This is the same call eval.validate
    makes per sample; passing ``mask_raw=None`` only skips the (ignored, in eval) segmentation
    loss -- the predicted masks are identical either way (see decoder.forward).
    """
    imgs = _cast(sam_img.unsqueeze(0).to(device), precision)
    head_maps, _, pred_masks, _ = full(imgs, [pil_img], [prompt], None, [resized_size], [ori_size])
    pred = (pred_masks[0].sigmoid() > MASK_THRESH).int()  # (1, H, W)
    return pred, head_maps[0, 0].float()                  # head_maps is (B, 1, Gh, Gw)


def _inputs_from_rgb(rgb, img_size):
    """Build the (SAM tensor, MLLM PIL, resized_size) for an RGB uint8 array, EXACTLY as the eval
    data path does:
      * data_deal(is_mask=False)            == ResizeLongestSide + normalize + pad  -> SAM input
      * data_deal(is_mask=True) + ToPILImage == ResizeLongestSide + pad (0-255)     -> MLLM input
    resized_size is [W_r, H_r] to match the swap eval.validate applies to sam_input_shape.
    """
    sam_t, new_size, _ = data_deal(rgb, img_size, is_mask=False)
    pil_t, _, _ = data_deal(rgb, img_size, is_mask=True)
    return sam_t, ToPILImage()(pil_t), [new_size[1], new_size[0]]


def _grid_bounds(size, n):
    """Non-overlapping, gap-free [lo, hi) splits of [0, size) into n parts (rounded edges)."""
    edges = [round(i * size / n) for i in range(n + 1)]
    return list(zip(edges[:-1], edges[1:]))


def _cell_attn_score(head_full, y0, y1, x0, x1, M):
    """Sum of the (single) full-image attention map inside a cell. head_full is peak-normalized
    by ONE global scalar, so cross-cell sums stay comparable. The 1024-square pad puts content at
    the top-left occupying H/M x W/M of the grid, so pixel p maps to grid index p/M * G."""
    Gh, Gw = head_full.shape
    gy0, gy1 = int(y0 / M * Gh), max(int(y0 / M * Gh) + 1, int(round(y1 / M * Gh)))
    gx0, gx1 = int(x0 / M * Gw), max(int(x0 / M * Gw) + 1, int(round(x1 / M * Gw)))
    return float(head_full[gy0:gy1, gx0:gx1].sum())


def _run_split(full, rgb, prompt, n, device, precision, img_size,
               head_full=None, base_pred=None, route_topk=0, route_fill="baseline"):
    """NxN training-free split: segment each non-overlapping cell independently with the frozen
    pipeline, then stitch the cell masks back -> (1, H, W).

    route_topk == 0 (default): refine EVERY cell (the plain split). route_topk > 0: use the single
    full-image attention map (`head_full`) to pick the K cells most likely to contain the target
    and refine ONLY those, leaving the rest as the baseline prediction (route_fill='baseline') or
    background ('bg'). This removes the object-free-cell false positives that sink the plain split.
    """
    H, W = rgb.shape[:2]
    bounds = list(product(_grid_bounds(H, n), _grid_bounds(W, n)))

    keep = set(range(len(bounds)))
    if route_topk and head_full is not None:
        M = max(H, W)
        scores = [_cell_attn_score(head_full, y0, y1, x0, x1, M) for (y0, y1), (x0, x1) in bounds]
        keep = set(sorted(range(len(bounds)), key=lambda i: scores[i], reverse=True)[:route_topk])

    if route_fill == "baseline" and base_pred is not None:
        stitched = base_pred[0].clone().to(torch.int32)   # un-refined cells keep the baseline
    else:
        stitched = torch.zeros((H, W), dtype=torch.int32, device=device)

    for qi, ((y0, y1), (x0, x1)) in enumerate(bounds):
        if qi not in keep:
            continue
        cell = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        sam_t, pil_img, resized_size = _inputs_from_rgb(cell, img_size)
        pred, _ = _run_pipeline(full, sam_t, pil_img, prompt, resized_size,
                                [y1 - y0, x1 - x0], device, precision)
        stitched[y0:y1, x0:x1] = pred[0].to(stitched.dtype)
    return stitched.unsqueeze(0)  # (1, H, W)


def _bbox_of_mask(mask2d):
    """Tight (y0, y1, x0, x1) bounding box of the positive pixels, or None if the mask is empty."""
    ys, xs = torch.where(mask2d > 0)
    if ys.numel() == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _attn_peak_box(head_full, H, W, frac):
    """A fallback ROI centered on the attention peak (used when the coarse mask is empty). Maps the
    head_map argmax back to a pixel (grid g -> g/G*max(H,W)) and takes a frac*min(H,W) square."""
    Gh, Gw = head_full.shape
    flat = int(torch.argmax(head_full))
    M = max(H, W)
    cy = int((flat // Gw + 0.5) / Gh * M)
    cx = int((flat % Gw + 0.5) / Gw * M)
    half = max(1, int(frac * min(H, W) / 2))
    return (max(0, cy - half), min(H, cy + half), max(0, cx - half), min(W, cx + half))


def _run_roi(full, rgb, prompt, base_pred, head_full, device, precision, img_size,
             margin=0.5, fallback_frac=0.25):
    """Coarse-to-fine: localize the target with the full-image (coarse) prediction, then re-segment
    a single zoom-in crop around it at high resolution -> (1, H, W).

    ROI = bbox of the coarse mask expanded by `margin` on each side (keeps a little context, frees
    the target from the rigid 2x2 boundary). If the coarse mask is empty, fall back to a box on the
    attention peak. Inside the ROI we take the refined (zoomed) mask; outside stays the coarse
    prediction -- which is background anyway since the bbox bounds all coarse positives. So this is
    detect-then-zoom-then-segment, with no object-free cells and no boundary-straddling.
    """
    H, W = rgb.shape[:2]
    bb = _bbox_of_mask(base_pred[0])
    if bb is None:
        y0, y1, x0, x1 = _attn_peak_box(head_full, H, W, fallback_frac)
    else:
        y0, y1, x0, x1 = bb
        my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
        y0, y1 = max(0, y0 - my), min(H, y1 + my)
        x0, x1 = max(0, x0 - mx), min(W, x1 + mx)

    cell = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    sam_t, pil_img, resized_size = _inputs_from_rgb(cell, img_size)
    pred, _ = _run_pipeline(full, sam_t, pil_img, prompt, resized_size,
                            [y1 - y0, x1 - x0], device, precision)
    stitched = base_pred[0].clone().to(torch.int32)  # outside ROI = coarse baseline
    stitched[y0:y1, x0:x1] = pred[0].to(stitched.dtype)
    return stitched.unsqueeze(0)  # (1, H, W)


def _iou_stats(output_i, target_i):
    """Per-instance (intersection[2], union[2], gIoU@class1), matching eval.validate exactly."""
    inter, union, _ = intersectionAndUnionGPU(
        output_i.contiguous().clone(), target_i.contiguous(), 2, ignore_index=255
    )
    g = inter / (union + 1e-5)
    g[union == 0] += 1.0  # no-object target counts as a hit (LISA / eval convention)
    return inter.cpu().numpy(), union.cpu().numpy(), float(g[1].item())


def _small_object_indices(dataset, area_thresh, show):
    """Indices of ReasonSeg samples whose target area (pixels == 1) is < area_thresh of the image.

    Uses only the polygon JSON + image dimensions (no model preprocessing), so the scan is cheap.
    Requires area > 0 so degenerate all-ignore samples are not counted as 'small objects'.
    """
    images, jsons = dataset.reason_seg_data
    pairs = list(enumerate(zip(images, jsons)))
    kept = []
    for idx, (ip, jp) in (tqdm(pairs, desc="scan small-object subset") if show else pairs):
        with Image.open(ip) as im:
            w, h = im.size  # PIL .size is (W, H); get_mask_from_json only reads img.shape[:2]
        mask, _, _, _ = get_mask_from_json(jp, np.zeros((h, w, 3), dtype=np.uint8))
        area = int((mask == 1).sum())
        if area > 0 and area / float(h * w) < area_thresh:
            kept.append(idx)
    return kept


def main():
    ap = argparse.ArgumentParser(
        description="Small-object NxN-split mitigation eval on ReasonSeg (Table_R 5)."
    )
    ap.add_argument("--config", default="configs/default.yaml",
                    help="experiment YAML; its model.backbone selects the frozen backbone.")
    ap.add_argument("--checkpoint", required=True, help="FullModel checkpoint to evaluate.")
    ap.add_argument("--split", default="test",
                    help="ReasonSeg split for the subset (the experiment uses 'test').")
    ap.add_argument("--area-thresh", type=float, default=0.10,
                    help="keep instances with target area < this fraction of the image "
                         "(default 0.10 == 10%%, the tau in the table caption).")
    ap.add_argument("--split-mode", choices=["grid", "roi"], default="grid",
                    help="how the 'split' setting refines: 'grid' = NxN tiles (+ optional "
                         "--route-topk); 'roi' = coarse-to-fine -- localize the target with the "
                         "full-image (coarse) prediction, then re-segment ONE zoom-in crop around "
                         "it at high resolution (no fixed boundary, no object-free cells).")
    ap.add_argument("--roi-margin", type=float, default=0.5,
                    help="[roi] expand the coarse-mask bbox by this fraction on each side before "
                         "zooming in (keeps some context). Default 0.5.")
    ap.add_argument("--roi-fallback-frac", type=float, default=0.25,
                    help="[roi] when the coarse mask is empty, center an ROI of this fraction of "
                         "min(H,W) on the attention peak. Default 0.25.")
    ap.add_argument("--grid", type=int, default=2,
                    help="[grid] NxN split (default 2 == the 2x2 scheme described in the rebuttal).")
    ap.add_argument("--route-topk", type=int, default=0,
                    help="[grid] 0 = plain split, refine every cell (the naive scheme). >0 = refine "
                         "only the top-K cells by the single full-image attention map, keeping the "
                         "baseline elsewhere -- removes object-free-cell false positives that sink "
                         "the plain split. Recommended starting point: 1.")
    ap.add_argument("--route-fill", choices=["baseline", "bg"], default="baseline",
                    help="[grid] for un-refined cells when --route-topk>0: keep the full-image "
                         "prediction (baseline, default) or set background (bg).")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N subset samples (smoke test).")
    ap.add_argument("--dump-deltas", type=int, default=0,
                    help="after eval, print the N most-improved and N most-worsened samples "
                         "(delta gIoU, short/long, GT area, path, query) so you can see which "
                         "cases the split helps vs hurts.")
    ap.add_argument("--output", default=None,
                    help="write metrics + per-sample stats to this JSON file.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if getattr(cfg.train, "hf_endpoint", None):
        os.environ.setdefault("HF_ENDPOINT", cfg.train.hf_endpoint)

    accelerator = Accelerator()
    device = accelerator.device

    # ---- build the frozen FullModel exactly as eval.py does, then load the checkpoint ----
    conditioner = MLLM_Conditioner(cfg.model).to(device)
    net = Decoder(cfg.model.decoder)
    full = FullModel(net, conditioner, kp_thresh=cfg.model.kp_thresh).to(torch.bfloat16)
    load_full_state_dict(full, args.checkpoint, map_location="cpu")
    full = full.to(device)
    full.eval()

    precision = cfg.eval.precision
    img_size = cfg.eval.image_size

    # ---- ReasonSeg <split> + small-object subset (dataset/split fixed by the experiment) ----
    dataset = TestReasoningDataset(
        cfg.eval.dataset_dir,
        CLIPImageProcessor.from_pretrained(cfg.eval.clip_vision_model),
        img_size,
        datasetname="ReasonSeg",
        train_test_split=args.split,
        eval_only=cfg.eval.eval_only,
        conversation_records={},
    )
    kept = _small_object_indices(dataset, args.area_thresh, accelerator.is_main_process)
    if args.limit:
        kept = kept[: args.limit]
    if accelerator.is_main_process:
        print(f"[subset] ReasonSeg {args.split}: {len(kept)} small-object instances "
              f"(target area < {args.area_thresh:.0%}) out of {len(dataset)} total.", flush=True)
    if not kept:
        raise SystemExit("No samples passed the small-object filter; check --area-thresh / dataset_dir.")

    # ---- shard across processes; works single-GPU (`python ...`) and multi-GPU (`accelerate launch`) ----
    shard = kept[accelerator.process_index :: accelerator.num_processes]
    use_mm = cfg.eval.use_mm_start_end

    local = []
    with torch.inference_mode():
        for idx in tqdm(shard, disable=not accelerator.is_main_process,
                        desc=f"small-obj[{cfg.model.backbone.name}]"):
            torch.cuda.empty_cache()
            batch = collate_fn_test([dataset[idx]], tokenizer=None, use_mm_start_end=use_mm)
            image_path = batch["image_paths"][0]
            prompt = batch["conversation_list"][0]
            is_sentence = bool(batch.get("is_sentence_list", [False])[0])  # long/reasoning vs short
            sam_img = batch["images"][0]                      # full-image SAM input (3, 1024, 1024)
            gt = batch["masks_list"][0][0]                    # (H, W) GT, values {0, 1, 255}
            H, W = int(gt.shape[0]), int(gt.shape[1])
            resize_list = batch["sam_mask_shape_list"][0][0]  # sam_input_shape (H_r, W_r)
            resized_size = [resize_list[1], resize_list[0]]   # [W_r, H_r], as eval.validate swaps
            gt_i = gt.int().unsqueeze(0).to(device)           # (1, H, W)

            # (a) full-image baseline -- identical to eval.validate's per-sample path. Its global
            #     attention map (head_full) doubles as the router for the split below.
            pil_t, _, _ = data_deal(Image.open(image_path).convert("RGB"), img_size, is_mask=True)
            pred_full, head_full = _run_pipeline(full, sam_img, ToPILImage()(pil_t), prompt,
                                                 resized_size, [H, W], device, precision)
            fi, fu, fg = _iou_stats(pred_full, gt_i)

            # (b) training-free refinement (frozen backbone, no fine-tuning). 'grid' = NxN tiles
            #     (+ optional attention routing); 'roi' = coarse-to-fine zoom on the detected target.
            rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if args.split_mode == "roi":
                pred_split = _run_roi(full, rgb, prompt, pred_full, head_full, device, precision,
                                      img_size, margin=args.roi_margin,
                                      fallback_frac=args.roi_fallback_frac)
            else:
                pred_split = _run_split(full, rgb, prompt, args.grid, device, precision, img_size,
                                        head_full=head_full, base_pred=pred_full,
                                        route_topk=args.route_topk, route_fill=args.route_fill)
            si, su, sg = _iou_stats(pred_split, gt_i)

            hw = float(H * W)
            local.append({
                "id": image_path,
                "is_sentence": is_sentence,
                "query": extract_user_instruction(prompt)[:160],
                "full": {"inter": fi.tolist(), "union": fu.tolist(), "giou": fg},
                "split": {"inter": si.tolist(), "union": su.tolist(), "giou": sg},
                "area": {"gt": float((gt_i == 1).sum().item()) / hw,
                         "full": float(pred_full.sum().item()) / hw,
                         "split": float(pred_split.sum().item()) / hw},
            })

    accelerator.wait_for_everyone()
    results = gather_object(local)

    if not accelerator.is_main_process:
        return

    # gather_object can duplicate the tail when shard sizes are uneven; de-dup by image path.
    seen, deduped = set(), []
    for r in results:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        deduped.append(r)

    def agg(rows, key):
        if not rows:
            return float("nan"), float("nan")
        inter = np.sum([r[key]["inter"] for r in rows], axis=0)
        union = np.sum([r[key]["union"] for r in rows], axis=0)
        ciou = float(inter[1] / (union[1] + 1e-10))
        giou = float(np.mean([r[key]["giou"] for r in rows]))
        return giou, ciou

    fg_, fc_ = agg(deduped, "full")
    sg_, sc_ = agg(deduped, "split")
    name, n = cfg.model.backbone.name, len(deduped)

    # split-vs-no-split comparison: aggregate gain + paired per-sample breakdown (same samples,
    # same GT, same metric -- the only thing that differs is the input, full image vs quadrants).
    dg, dc = sg_ - fg_, sc_ - fc_
    deltas = [r["split"]["giou"] - r["full"]["giou"] for r in deduped]
    improved = sum(d > 1e-6 for d in deltas)
    worsened = sum(d < -1e-6 for d in deltas)
    same = n - improved - worsened
    area = {k: float(np.mean([r["area"][k] for r in deduped])) for k in ("gt", "full", "split")}

    if args.split_mode == "roi":
        mode = f"roi coarse->fine (margin={args.roi_margin})"
        split_label = "+ coarse->fine roi (ours)"
    else:
        mode = (f"grid {args.grid}x{args.grid}, top-{args.route_topk} cells rest={args.route_fill}"
                if args.route_topk else f"grid {args.grid}x{args.grid}, all cells (plain)")
        split_label = f"+ {args.grid}x{args.grid} split ({'routed' if args.route_topk else 'ours'})"
    print("=" * 78)
    print("Table_R 5 -- small-object mitigation (training-free, frozen backbone)")
    print(f"Backbone: {name}   ReasonSeg {args.split}   target area < {args.area_thresh:.0%}   "
          f"N={n}   mode={mode}")
    print("-" * 78)
    print(f"  {'Setting':<28}{'gIoU':>9}{'cIoU':>9}{'gIoU%':>9}{'cIoU%':>9}")
    print(f"  {'full image (single pass)':<28}{fg_:>9.4f}{fc_:>9.4f}{100 * fg_:>9.2f}{100 * fc_:>9.2f}")
    print(f"  {split_label:<28}{sg_:>9.4f}{sc_:>9.4f}{100 * sg_:>9.2f}{100 * sc_:>9.2f}")
    print(f"  {'delta (split - full)':<28}{dg:>+9.4f}{dc:>+9.4f}{100 * dg:>+9.2f}{100 * dc:>+9.2f}")
    print("-" * 78)
    print(f"  paired per-sample gIoU:  improved {improved} / same {same} / worsened {worsened}  (of {n})")
    # Diagnostic: if split area >> full area >> GT, the drop is object-free-cell over-segmentation
    # (each cell is forced to emit >=1 keypoint -> a mask). Use --route-topk to fix it.
    print(f"  mean predicted area (frac of image):  GT {area['gt']:.3f}   "
          f"full {area['full']:.3f}   split {area['split']:.3f}")

    # --- WHO improves? short/explicit vs long/reasoning queries (ReasonSeg's is_sentence flag).
    # Splitting trades resolution (helps small explicit targets) against global context (hurts
    # reasoning queries that need the whole scene), so the gain usually lives in the 'short' bucket.
    print("-" * 78)
    print(f"  {'subset':<18}{'N':>5}{'full gIoU':>11}{'split gIoU':>12}{'dgIoU':>9}{'dcIoU':>9}"
          f"{'imp/wrs':>10}")
    for label, rows in [("short/explicit", [r for r in deduped if not r["is_sentence"]]),
                        ("long/reasoning", [r for r in deduped if r["is_sentence"]])]:
        if not rows:
            continue
        bfg, bfc = agg(rows, "full")
        bsg, bsc = agg(rows, "split")
        imp = sum(r["split"]["giou"] - r["full"]["giou"] > 1e-6 for r in rows)
        wrs = sum(r["split"]["giou"] - r["full"]["giou"] < -1e-6 for r in rows)
        print(f"  {label:<18}{len(rows):>5}{bfg:>11.4f}{bsg:>12.4f}{bsg - bfg:>+9.4f}"
              f"{bsc - bfc:>+9.4f}{f'{imp}/{wrs}':>10}")

    # improved vs worsened groups: mean delta, mean GT-area, and long-query share -- tells you
    # whether the wins are the smaller / more-explicit objects.
    def _grp(rows):
        if not rows:
            return "  (none)"
        md = float(np.mean([r["split"]["giou"] - r["full"]["giou"] for r in rows]))
        ma = float(np.mean([r["area"]["gt"] for r in rows]))
        pl = 100.0 * sum(r["is_sentence"] for r in rows) / len(rows)
        return f"n={len(rows):>3}  mean dgIoU={md:>+.3f}  mean GT-area={ma:.3f}  long%={pl:>4.0f}"
    imp_rows = [r for r in deduped if r["split"]["giou"] - r["full"]["giou"] > 1e-6]
    wrs_rows = [r for r in deduped if r["split"]["giou"] - r["full"]["giou"] < -1e-6]
    print(f"  improved: {_grp(imp_rows)}")
    print(f"  worsened: {_grp(wrs_rows)}")
    print("=" * 78, flush=True)

    if args.dump_deltas:
        ranked = sorted(deduped, key=lambda r: r["split"]["giou"] - r["full"]["giou"])
        def _line(r):
            d = r["split"]["giou"] - r["full"]["giou"]
            kind = "long " if r["is_sentence"] else "short"
            return (f"  d={d:>+.3f} {kind} area={r['area']['gt']:.3f}  "
                    f"{os.path.basename(r['id'])} :: {r['query']}")
        print(f"\n--- {min(args.dump_deltas, len(ranked))} most WORSENED (split - full) ---")
        for r in ranked[: args.dump_deltas]:
            print(_line(r))
        print(f"--- {min(args.dump_deltas, len(ranked))} most IMPROVED ---")
        for r in ranked[::-1][: args.dump_deltas]:
            print(_line(r))
        print(flush=True)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            subsets = {}
            for key, rows in [("short", [r for r in deduped if not r["is_sentence"]]),
                              ("long", [r for r in deduped if r["is_sentence"]])]:
                bfg, bfc = agg(rows, "full")
                bsg, bsc = agg(rows, "split")
                subsets[key] = {"n": len(rows),
                                "full": {"giou": bfg, "ciou": bfc},
                                "split": {"giou": bsg, "ciou": bsc}}
            json.dump({
                "backbone": name, "model_name": cfg.model.backbone.model_name,
                "split": args.split, "area_thresh": args.area_thresh, "n": n,
                "split_mode": args.split_mode, "grid": args.grid,
                "route_topk": args.route_topk, "route_fill": args.route_fill,
                "roi_margin": args.roi_margin, "roi_fallback_frac": args.roi_fallback_frac,
                "full": {"giou": fg_, "ciou": fc_},
                "split": {"giou": sg_, "ciou": sc_},
                "delta": {"giou": dg, "ciou": dc},
                "paired_giou": {"improved": improved, "same": same, "worsened": worsened},
                "mean_area": area,
                "subsets": subsets,
                "per_sample": deduped,
            }, fh, indent=2)
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
