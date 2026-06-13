"""Routing / triggering benchmark (rebuttal CR2).

Builds the ~1000-sample benchmark that grades the frozen MLLM router's ability to classify a
query into Dialogue / Segmentation, across four difficulty tiers:

    dialogue        non-segmentation dialogue/VQA   -> route: dialogue   (any trigger = false-trigger)
    explicit_seg    names the target overtly        -> route: segmentation
    reasoning_seg   target only implied             -> route: segmentation
    mixed_intent    blends dialogue + seg request   -> route: segmentation (explicit precedence)

Samples are drawn from the datasets this repo already uses (llava_instruct_150k for VQA,
refcoco/+/g for explicit targets, ReasonSeg for implicit ones) and curated with an LLM
(qwen3-32b served by lmdeploy, OpenAI-compatible API).

Entry point (run from the repo root):
    python -m routing_benchmark.build_benchmark   --help

Scoring into Table R2 is done by the repo's top-level ``eval_routing.py`` (this folder is the
construction pipeline only).
"""
