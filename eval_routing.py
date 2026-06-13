"""Routing-benchmark evaluation (Table_R 2) — frozen HF MLLM, accelerate multi-GPU.

Each process loads its own copy of the frozen backbone selected by the config, evaluates a
shard of the benchmark, and the per-sample (gold, pred) tiers are gathered on the main
process to compute per-tier accuracy, overall accuracy, and false-/missed-trigger rates.

    # one row of the table per run (backbone chosen by --config):
    accelerate launch eval_routing.py --config configs/default.yaml      --benchmark data/routing_benchmark.jsonl   # LLaVA-1.5-7B
    accelerate launch eval_routing.py --config configs/qwen2vl_2b.yaml   --benchmark data/routing_benchmark.jsonl   # Qwen2-VL-2B
    accelerate launch eval_routing.py --config configs/llava_onevision_7b.yaml --benchmark data/routing_benchmark.jsonl   # LLaVA-OneVision-7B

Add --image-root if the benchmark image paths are relative. --output writes per-sample
predictions + metrics to JSON. --limit N evaluates only the first N samples (smoke test).
"""

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm

from lens.config import load_config
from lens.models.backbones import build_mllm_backbone
from lens.routing import (
    LLAVA_SYSTEM_PROMPT,
    MLLMRouter,
    compute_binary_metrics,
    compute_routing_metrics,
    load_benchmark,
    normalize_tier,
)
from lens.routing.router import TIERS


def print_routing_table(name, m, mode):
    def pct(x):
        return "  -  " if x is None else f"{100 * x:5.1f}"

    pt = m["per_tier"]
    label = ("per-tier = routed correctly (seg vs non-seg)" if mode == "binary"
             else "per-tier = exact 4-way accuracy")
    print("=" * 92)
    print(f"Routing benchmark — {name}  [mode={mode}; {label}]  (n={m['n']}, unparsed={m['unparsed']})")
    print(f"{'':<8}{'non-seg':>9}{'explicit':>9}{'reasoning':>10}{'mixed':>8}"
          f"{'overall':>9}{'false-trig':>11}{'missed-trig':>12}")
    print(f"{'acc/rate%':<8}{pct(pt['non-seg']['acc']):>9}{pct(pt['explicit']['acc']):>9}"
          f"{pct(pt['reasoning']['acc']):>10}{pct(pt['mixed']['acc']):>8}"
          f"{pct(m['overall']):>9}{pct(m['false_trigger']):>11}{pct(m['missed_trigger']):>12}")
    print(f"{'#samples':<8}{pt['non-seg']['n']:>9}{pt['explicit']['n']:>9}"
          f"{pt['reasoning']['n']:>10}{pt['mixed']['n']:>8}{m['n']:>9}")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml",
                    help="experiment YAML; its model.backbone selects the router.")
    ap.add_argument("--benchmark", default="routing_benchmark/benchmark.jsonl", help="routing benchmark JSONL / JSON.")
    ap.add_argument("--mode", choices=["binary", "tier"], default="binary",
                    help="binary = just seg-vs-non-seg (simple, default); tier = 4-way.")
    ap.add_argument("--prompt-file", default=None,
                    help="file with a custom prompt template (must contain '{question}').")
    ap.add_argument("--prompt", default=None,
                    help="inline custom prompt template (must contain '{question}').")
    ap.add_argument("--system-prompt", default=None,
                    help="system prompt: 'llava' for the built-in LLaVA-1.5 system, or any "
                         "text / a file path. Default: none (Qwen/LLaVA-OneVision add their own).")
    ap.add_argument("--image-root", default=None, help="prefix for relative image paths.")
    ap.add_argument("--output", default=None, help="write metrics + per-sample preds here.")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--image-hw", type=int, default=0,
                    help="force a square image size (0 = MLLM native preprocessing).")
    ap.add_argument("--limit", type=int, default=0, help="eval only the first N samples.")
    ap.add_argument("--tier-map", default=None,
                    help="JSON (inline or file) mapping your gold tier names to "
                         "non-seg/explicit/reasoning/mixed, e.g. '{\"mixed-intent\":\"mixed\"}'.")
    ap.add_argument("--show-tier", default=None,
                    help="after eval, print id|pred|raw for samples of this gold tier "
                         "(e.g. non-seg) to inspect what the model actually answered.")
    ap.add_argument("--show-n", type=int, default=40, help="how many --show-tier cases to print.")
    ap.add_argument("--print-prompt", action="store_true",
                    help="print the exact assembled prompt for the first sample, then continue.")
    args = ap.parse_args()

    tier_map = None
    if args.tier_map:
        raw = open(args.tier_map).read() if os.path.exists(args.tier_map) else args.tier_map
        tier_map = {str(k).strip().lower().replace("_", "-"): str(v).strip().lower()
                    for k, v in json.loads(raw).items()}

    prompt_template = None
    if args.prompt_file:
        prompt_template = open(args.prompt_file, encoding="utf-8").read()
    elif args.prompt:
        prompt_template = args.prompt
    if prompt_template is not None and "{question}" not in prompt_template:
        raise SystemExit("--prompt/--prompt-file must contain '{question}'")

    system_prompt = None
    if args.system_prompt:
        if args.system_prompt.strip().lower() == "llava":
            system_prompt = LLAVA_SYSTEM_PROMPT
        elif os.path.exists(args.system_prompt):
            system_prompt = open(args.system_prompt, encoding="utf-8").read().strip()
        else:
            system_prompt = args.system_prompt

    cfg = load_config(args.config)
    accelerator = Accelerator()
    if getattr(cfg.train, "hf_endpoint", None):
        os.environ.setdefault("HF_ENDPOINT", cfg.train.hf_endpoint)

    backbone = build_mllm_backbone(cfg.model.backbone)
    backbone.to(torch.bfloat16).to(accelerator.device)
    backbone.model.eval()
    router = MLLMRouter(
        backbone,
        mode=args.mode,
        prompt_template=prompt_template,
        max_new_tokens=args.max_new_tokens,
        image_hw=(args.image_hw or None),
        system_prompt=system_prompt,
    )

    data = load_benchmark(args.benchmark, args.image_root, tier_map=tier_map)
    if args.limit:
        data = data[: args.limit]

    if args.print_prompt and accelerator.is_main_process and data:
        print("=" * 80)
        print("ASSEMBLED PROMPT for the first sample (exactly what the model receives):")
        print("-" * 80)
        print(router.render_prompt(data[0]["question"]))
        print("=" * 80, flush=True)

    # Shard across processes (strided so tiers stay roughly balanced per shard).
    shard = data[accelerator.process_index :: accelerator.num_processes]

    local = []
    for item in tqdm(shard, disable=not accelerator.is_main_process,
                     desc=f"routing[{cfg.model.backbone.name}]"):
        try:
            pred, raw = router.predict(item["image"], item["question"])
        except Exception as e:  # keep going; a crashed sample counts as unparsed
            pred, raw = None, f"ERROR: {type(e).__name__}: {e}"
        local.append({"id": item["id"], "gold": item["tier"], "pred": pred, "raw": raw})

    accelerator.wait_for_everyone()
    all_results = gather_object(local)

    if accelerator.is_main_process:
        # gather_object can duplicate when the shard sizes are uneven; de-dup by id.
        seen, deduped = set(), []
        for r in all_results:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            deduped.append(r)

        metrics = (compute_binary_metrics(deduped) if args.mode == "binary"
                   else compute_routing_metrics(deduped))
        print_routing_table(cfg.model.backbone.name, metrics, args.mode)

        if args.show_tier:
            want = normalize_tier(args.show_tier) or args.show_tier
            cases = [r for r in deduped if r["gold"] == want]
            print(f"\n--- raw model answers for gold tier '{want}' "
                  f"(showing {min(args.show_n, len(cases))}/{len(cases)}) ---")
            for r in cases[: args.show_n]:
                raw = " ".join(str(r.get("raw", "")).split())[:120]
                print(f"[{r['id']}] pred={r['pred']!s:<8} raw={raw!r}")

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump({"backbone": cfg.model.backbone.name,
                           "model_name": cfg.model.backbone.model_name, "mode": args.mode,
                           "metrics": metrics, "results": deduped}, fh, indent=2)
            print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
