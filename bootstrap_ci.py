"""Bootstrap confidence intervals for LENS metrics on ReasonSeg test (rebuttal point (2):
statistical uncertainty of the reported gIoU / cIoU).

We resample the test IMAGES with replacement B times (default 10,000) and recompute the
metrics on each resample, then report the 95% CI as the [2.5, 97.5] percentiles of the
bootstrap distribution. cIoU and gIoU are the SAME estimators eval.validate uses:

    cIoU = sum_i inter_i / sum_i union_i           (pixel-pooled, class 1)
    gIoU = mean_i  inter_i / union_i               (per-image IoU, class 1; 1.0 if union==0)

so the point estimate equals the number eval.py prints (e.g. cIoU 57.3) -- confirm they match.

The expensive part (running the model over the whole test split to get per-image inter/union/
giou) runs once and is cached to --stats-out; the bootstrap itself is instant and re-runnable
on those cached stats via --from-stats (numpy only, no GPU), so you can change B / seed / the
comparison value without re-running inference.

    # 1) run the model on ReasonSeg test, cache per-image stats, and bootstrap
    accelerate launch bootstrap_ci.py --config configs/default.yaml --checkpoint /path/to/full_N.pth \
        --stats-out stats/lens_reasonseg_test.json --compare-ciou 58.6
    # 2) later: re-bootstrap from the cached stats only (no model, no GPU)
    python bootstrap_ci.py --from-stats stats/lens_reasonseg_test.json --bootstrap 10000 --compare-ciou 58.6

Standalone: does not modify eval.py / eval_small_object.py; it reuses their exact metric.
"""

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "LISA"))


def bootstrap(per_sample, n_boot=10000, seed=0, ci=95.0):
    """Percentile bootstrap over per-image (inter1, union1, giou) records -> point + CI for
    cIoU and gIoU. Resampling is at the image level (one record == one test image)."""
    import numpy as np

    inter1 = np.asarray([r["inter1"] for r in per_sample], dtype=np.float64)
    union1 = np.asarray([r["union1"] for r in per_sample], dtype=np.float64)
    giou = np.asarray([r["giou"] for r in per_sample], dtype=np.float64)
    N = len(per_sample)

    point_ciou = float(inter1.sum() / (union1.sum() + 1e-10))
    point_giou = float(giou.mean())

    rng = np.random.default_rng(seed)
    bc = np.empty(n_boot, dtype=np.float64)
    bg = np.empty(n_boot, dtype=np.float64)
    # Chunk the (n_boot x N) resampling so peak memory stays bounded for large B.
    chunk = 1000
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        idx = rng.integers(0, N, size=(b, N))
        bc[s:s + b] = inter1[idx].sum(axis=1) / (union1[idx].sum(axis=1) + 1e-10)
        bg[s:s + b] = giou[idx].mean(axis=1)

    lo_q, hi_q = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0

    def _summ(point, draws):
        return {
            "point": point,
            "boot_mean": float(draws.mean()),
            "se": float(draws.std(ddof=1)),
            "lo": float(np.percentile(draws, lo_q)),
            "hi": float(np.percentile(draws, hi_q)),
        }

    return {"n": N, "n_boot": n_boot, "seed": seed, "ci": ci,
            "ciou": _summ(point_ciou, bc), "giou": _summ(point_giou, bg)}


def collect_stats(args):
    """Run the frozen LENS baseline over ReasonSeg <split> and return per-image
    (id, is_sentence, inter1, union1, giou) -- exactly eval.validate's per-sample metric."""
    import torch
    from transformers import CLIPImageProcessor
    from accelerate import Accelerator
    from accelerate.utils import gather_object
    from PIL import Image
    from torchvision.transforms import ToPILImage
    from tqdm import tqdm

    from lens.config import load_config
    from lens.models import load_full_state_dict
    from lens.models.conditioner import MLLM_Conditioner, FullModel
    from lens.models.decoder import Decoder
    from lens.data.test_dataset import TestReasoningDataset, collate_fn_test
    from lens.utils.inference import data_deal
    # Reuse the EXACT forward + metric the small-object eval uses (which match eval.validate).
    from eval_small_object import _run_pipeline, _iou_stats

    cfg = load_config(args.config)
    if getattr(cfg.train, "hf_endpoint", None):
        os.environ.setdefault("HF_ENDPOINT", cfg.train.hf_endpoint)

    accelerator = Accelerator()
    device = accelerator.device

    conditioner = MLLM_Conditioner(cfg.model).to(device)
    net = Decoder(cfg.model.decoder)
    full = FullModel(net, conditioner, kp_thresh=cfg.model.kp_thresh).to(torch.bfloat16)
    load_full_state_dict(full, args.checkpoint, map_location="cpu")
    full = full.to(device)
    full.eval()

    precision = cfg.eval.precision
    img_size = cfg.eval.image_size
    dataset = TestReasoningDataset(
        cfg.eval.dataset_dir,
        CLIPImageProcessor.from_pretrained(cfg.eval.clip_vision_model),
        img_size,
        datasetname="ReasonSeg",
        train_test_split=args.split,
        eval_only=cfg.eval.eval_only,
        conversation_records={},
    )
    indices = list(range(len(dataset)))
    if args.limit:
        indices = indices[: args.limit]
    shard = indices[accelerator.process_index :: accelerator.num_processes]
    use_mm = cfg.eval.use_mm_start_end

    local = []
    with torch.inference_mode():
        for idx in tqdm(shard, disable=not accelerator.is_main_process,
                        desc=f"collect[{cfg.model.backbone.name}]"):
            torch.cuda.empty_cache()
            batch = collate_fn_test([dataset[idx]], tokenizer=None, use_mm_start_end=use_mm)
            image_path = batch["image_paths"][0]
            prompt = batch["conversation_list"][0]
            is_sentence = bool(batch.get("is_sentence_list", [False])[0])
            sam_img = batch["images"][0]
            gt = batch["masks_list"][0][0]
            H, W = int(gt.shape[0]), int(gt.shape[1])
            resize_list = batch["sam_mask_shape_list"][0][0]
            resized_size = [resize_list[1], resize_list[0]]
            gt_i = gt.int().unsqueeze(0).to(device)

            pil_t, _, _ = data_deal(Image.open(image_path).convert("RGB"), img_size, is_mask=True)
            pred, _ = _run_pipeline(full, sam_img, ToPILImage()(pil_t), prompt,
                                    resized_size, [H, W], device, precision)
            inter, union, giou = _iou_stats(pred, gt_i)
            local.append({"id": image_path, "is_sentence": is_sentence,
                          "inter1": float(inter[1]), "union1": float(union[1]), "giou": giou})

    accelerator.wait_for_everyone()
    gathered = gather_object(local)

    if not accelerator.is_main_process:
        return None, None

    seen, per_sample = set(), []
    for r in gathered:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        per_sample.append(r)
    return per_sample, cfg.model.backbone.name


def _print_report(name, split, res, compare_ciou):
    def line(metric, s):
        inside = ""
        if metric == "ciou" and compare_ciou is not None:
            inside = ("   [compare {:.1f}: {}]"
                      .format(compare_ciou,
                              "INSIDE CI" if s["lo"] <= compare_ciou / 100.0 <= s["hi"]
                              else "outside CI"))
        return (f"  {metric.upper():<5} point {100 * s['point']:.1f}   "
                f"{int(res['ci'])}% CI [{100 * s['lo']:.1f}, {100 * s['hi']:.1f}]   "
                f"(boot mean {100 * s['boot_mean']:.1f}, SE {100 * s['se']:.2f}){inside}")

    print("=" * 78)
    print(f"Bootstrap CIs -- {name or 'LENS'}  ReasonSeg {split}  "
          f"(N={res['n']} images, B={res['n_boot']}, seed={res['seed']})")
    print("-" * 78)
    print(line("ciou", res["ciou"]))
    print(line("giou", res["giou"]))
    print("=" * 78, flush=True)


def main():
    ap = argparse.ArgumentParser(description="Bootstrap CI for LENS gIoU/cIoU on ReasonSeg test.")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="experiment YAML (selects the backbone); needed unless --from-stats.")
    ap.add_argument("--checkpoint", default=None, help="FullModel checkpoint; needed unless --from-stats.")
    ap.add_argument("--split", default="test", help="ReasonSeg split (the reported number uses 'test').")
    ap.add_argument("--from-stats", default=None,
                    help="skip the model and bootstrap from a cached per-sample stats JSON "
                         "(this script's --stats-out, or eval_small_object's per_sample[].full).")
    ap.add_argument("--stats-out", default=None, help="cache the collected per-sample stats here.")
    ap.add_argument("--bootstrap", type=int, default=10000, help="number of bootstrap resamples.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible resampling.")
    ap.add_argument("--ci", type=float, default=95.0, help="confidence level in percent (default 95).")
    ap.add_argument("--compare-ciou", type=float, default=None,
                    help="a competitor cIoU (percent, e.g. 58.6) to check against the cIoU CI.")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (smoke test).")
    ap.add_argument("--output", default=None, help="write the CI summary JSON here.")
    args = ap.parse_args()

    name, split = None, args.split
    if args.from_stats:
        with open(args.from_stats, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        # Accept this script's stats file, a bare list, or eval_small_object's per_sample (uses
        # full.inter/full.union/full.giou) -- normalize to {inter1, union1, giou}.
        raw = blob.get("per_sample", blob) if isinstance(blob, dict) else blob
        name = blob.get("backbone") if isinstance(blob, dict) else None
        split = blob.get("split", split) if isinstance(blob, dict) else split
        per_sample = []
        for r in raw:
            if "inter1" in r:
                per_sample.append(r)
            elif "full" in r:  # eval_small_object per_sample record
                f = r["full"]
                per_sample.append({"id": r.get("id"), "is_sentence": r.get("is_sentence", False),
                                   "inter1": float(f["inter"][1]), "union1": float(f["union"][1]),
                                   "giou": float(f["giou"])})
        if not per_sample:
            raise SystemExit("--from-stats: could not find per-sample inter/union/giou records.")
        is_main = True
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --from-stats is given.")
        per_sample, name = collect_stats(args)
        is_main = per_sample is not None
        if is_main and args.stats_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.stats_out)) or ".", exist_ok=True)
            with open(args.stats_out, "w", encoding="utf-8") as fh:
                json.dump({"backbone": name, "split": args.split, "per_sample": per_sample}, fh)
            print(f"wrote per-sample stats: {args.stats_out}  ({len(per_sample)} images)", flush=True)

    if not is_main:
        return

    res = bootstrap(per_sample, n_boot=args.bootstrap, seed=args.seed, ci=args.ci)
    _print_report(name, split, res, args.compare_ciou)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"backbone": name, "split": split, "compare_ciou": args.compare_ciou,
                       **res}, fh, indent=2)
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
