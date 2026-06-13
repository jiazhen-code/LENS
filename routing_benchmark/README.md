# Routing / triggering benchmark (rebuttal CR2)

Builds the ~1,000-sample benchmark that grades the frozen MLLM router (Alg.1) at classifying a
query into **Dialogue / Segmentation**, across four difficulty tiers. Scoring (per-tier
accuracy + false/missed-trigger rates, **Table R2**) is done by the repo's top-level
`eval_routing.py`; this folder is the construction pipeline only.

Samples are drawn from datasets the repo already uses and **curated with an LLM**
(qwen3-32b served by lmdeploy):

| tier | source | gold route | what it tests |
|------|--------|-----------|---------------|
| `dialogue` | `llava_instruct_150k` (VQA) | dialogue | no-seg dialogue; any trigger = **false trigger** |
| `explicit_seg` | refcoco/+/g expressions | segmentation | overt "segment X" instructions |
| `reasoning_seg` | ReasonSeg (`is_sentence`) | segmentation | target only **implied** |
| `mixed_intent` | refcoco targets (LLM-composed) | segmentation | dialogue **+** seg request in one query (explicit precedence) |

## Setup

```bash
pip install openai            # talks to the lmdeploy OpenAI-compatible API
# serve your model, e.g.:
# CUDA_VISIBLE_DEVICES=0,1 lmdeploy serve api_server Qwen/Qwen3-VL-32B-Instruct  --backend pytorch   --tp 2   --server-name 0.0.0.0   --server-port 23333   --session-len 32768   --cache-max-entry-count 0.4 --model-name qwen3-32b
```

## 1. Build the benchmark

```bash
python -m routing_benchmark.build_benchmark \
    --dataset_dir /path/to/lisa_data/ \
    --lm_base_url http://localhost:23333/v1 --lm_model qwen3-32b \
    --out_path routing_benchmark/benchmark.jsonl
```

- Counts per tier via `--n_dialogue/--n_explicit/--n_reasoning/--n_mixed` (default 250 each → ~1000).
- `--no_llm --limit 5` → tiny template-only build, to smoke-test the data loaders without the LLM.
- LLM responses are cached under `routing_benchmark/.cache/`, so re-runs are cheap/resumable.

Each line of `benchmark.jsonl` (`image` is relative to `--dataset_dir`; pass it as
`--image-root` to the scorer):
```json
{"id": "exp_00012", "tier": "explicit_seg", "gold_route": "segmentation",
 "query": "Please segment the dog on the left.", "image": "refer_seg/.../COCO_train2014_000000xxxxxx.jpg",
 "source": "refcoco", "target": "the dog on the left", "distractor_route": null, "meta": {...}}
```

## 2. Score → Table R2

Scoring uses the repo's top-level `eval_routing.py`: it loads a frozen MLLM backbone, routes
every sample with the paper's router (Alg.1), and prints per-tier accuracy, overall accuracy,
and **false-trigger** (dialogue→segmentation) / **missed-trigger** (segmentation→not-
segmentation) rates. One run per backbone = one Table R2 row:

```bash
accelerate launch eval_routing.py --config configs/default.yaml \
    --benchmark routing_benchmark/benchmark.jsonl --image-root /path/to/lisa_data   # LLaVA-1.5-7B
accelerate launch eval_routing.py --config configs/qwen2vl_2b.yaml \
    --benchmark routing_benchmark/benchmark.jsonl --image-root /path/to/lisa_data   # Qwen2-VL-2B
accelerate launch eval_routing.py --config configs/llava_onevision_7b.yaml \
    --benchmark routing_benchmark/benchmark.jsonl --image-root /path/to/lisa_data   # LLaVA-OneVision-7B
```

See the top-level README ("Routing benchmark") for the optional flags (`--output`, `--limit`,
`--tier-map`).

## Files

- `config.py` — dataset paths, per-tier counts, lmdeploy endpoint (CLI-overridable).
- `sources.py` — load VQA / refcoco (`REFER`) / ReasonSeg.
- `prompts.py` — per-tier LLM prompts + parsers, with template fallbacks.
- `lm_client.py` — lmdeploy/qwen3 client (thinking off, retries, JSON parsing, disk cache).
- `build_benchmark.py` — assemble + curate → `benchmark.jsonl`.

## Notes

- qwen3-32b is text-only and is used only for **construction/curation** (operating on query
  text), never as the router under test.
- Image paths follow the LISA layout (VQA → `coco/train2017`, refcoco →
  `refer_seg/images/mscoco/images/train2014`, ReasonSeg → the jpg next to each json). The
  build prints how many resolved image paths are missing so you can fix `--dataset_dir`.
