"""Build the routing benchmark (rebuild CR2).

    python -m routing_benchmark.build_benchmark \
        --dataset_dir /path/to/lisa_data/ \
        --lm_base_url http://localhost:23333/v1 --lm_model qwen3-32b \
        --out_path routing_benchmark/benchmark.jsonl

Smoke test without the LLM (template-only, a few samples):
    python -m routing_benchmark.build_benchmark --no_llm --limit 5 --out_path /tmp/bench_smoke.jsonl

Output: one JSON object per line:
    {id, tier, gold_route, query, image, image_path, source, target, distractor_route, meta}
"""

import argparse
import json
import os
import random

from .config import BenchConfig
from . import sources
from .prompts import CURATORS

# tier -> (source loader key, count attribute)
TIER_PLAN = [
    ("dialogue", "vqa", "n_dialogue"),
    ("explicit_seg", "refcoco", "n_explicit"),
    ("reasoning_seg", "reasonseg", "n_reasoning"),
    ("mixed_intent", "refcoco", "n_mixed"),
]
LOADERS = {
    "vqa": sources.load_vqa,
    "refcoco": sources.load_refcoco,
    "reasonseg": sources.load_reasonseg,
}
TIER_PREFIX = {"dialogue": "dlg", "explicit_seg": "exp", "reasoning_seg": "rsn", "mixed_intent": "mix"}

from tqdm import tqdm
def build_tier(tier, loader_key, target, cfg, lm, rng, seen):
    loader = LOADERS[loader_key]
    raw = loader(cfg, rng, int(target * cfg.oversample) + 8)
    curate = CURATORS[tier]
    accepted, rejected = [], 0
    for item in tqdm(raw):
        if len(accepted) >= target:
            break
        payload = curate(lm, item, rng, cfg.use_llm)
        if payload is None:
            rejected += 1
            continue
        qkey = payload["query"].strip().lower()
        if qkey in seen:
            continue
        seen.add(qkey)
        sample = {
            "tier": tier,
            "gold_route": payload["gold_route"],
            "query": payload["query"].strip(),
            "image": item.get("image"),
            "image_path": item.get("image_path"),
            "source": item.get("source"),
            "target": payload.get("target"),
            "distractor_route": payload.get("distractor_route"),
            "meta": {**item.get("meta", {}), **payload.get("llm_meta", {})},
        }
        accepted.append(sample)
    missing = sources.count_missing_images(accepted)
    print(f"[{tier:13s}] accepted={len(accepted):4d}/{target} rejected={rejected:4d} "
          f"from {len(raw)} raw ({loader_key}); missing images={missing}")
    if len(accepted) < target:
        print(f"  ! only {len(accepted)} (<{target}); raise --oversample or check the source.")
    return accepted


def main():
    ap = argparse.ArgumentParser(description="Build the CR2 routing benchmark.")
    ap.add_argument("--dataset_dir")
    ap.add_argument("--out_path")
    ap.add_argument("--lm_base_url")
    ap.add_argument("--lm_model")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n_dialogue", type=int)
    ap.add_argument("--n_explicit", type=int)
    ap.add_argument("--n_reasoning", type=int)
    ap.add_argument("--n_mixed", type=int)
    ap.add_argument("--oversample", type=float)
    ap.add_argument("--no_llm", dest="use_llm", action="store_false", default=None,
                    help="template-only construction (no qwen3 calls) -- for smoke tests")
    ap.add_argument("--limit", type=int, default=None,
                    help="override every tier count with this (quick smoke build)")
    args = ap.parse_args()

    cfg = BenchConfig().apply_overrides(args)
    if args.limit is not None:
        cfg.n_dialogue = cfg.n_explicit = cfg.n_reasoning = cfg.n_mixed = args.limit

    rng = random.Random(cfg.seed)
    lm = None
    if cfg.use_llm:
        from .lm_client import LMClient
        lm = LMClient(cfg)
        print(f"Using LLM router-curation via {cfg.lm_base_url} (model={cfg.lm_model}).")
    else:
        print("LLM disabled (--no_llm): template-only construction.")

    all_samples, seen = [], set()
    for tier, loader_key, count_attr in TIER_PLAN:
        target = getattr(cfg, count_attr)
        all_samples.extend(build_tier(tier, loader_key, target, cfg, lm, rng, seen))

    # stable ids per tier, then global shuffle
    per_tier_idx = {}
    for s in all_samples:
        i = per_tier_idx.get(s["tier"], 0)
        s["id"] = f"{TIER_PREFIX[s['tier']]}_{i:05d}"
        per_tier_idx[s["tier"]] = i + 1
    rng.shuffle(all_samples)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.out_path)), exist_ok=True)
    with open(cfg.out_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("-" * 60)
    print(f"Wrote {len(all_samples)} samples -> {cfg.out_path}")
    by_route = {}
    for s in all_samples:
        by_route[s["gold_route"]] = by_route.get(s["gold_route"], 0) + 1
    print("by gold_route:", by_route)


if __name__ == "__main__":
    main()
