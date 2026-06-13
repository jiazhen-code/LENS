"""Configuration for the routing benchmark builder.

Defaults assume the LISA data layout used by the rest of the repo. Override the dataset
root and the lmdeploy endpoint from the CLI; see build_benchmark.py --help.
"""

from dataclasses import dataclass, field, fields
from typing import List


# Tier names and the route they must map to. Keep in sync with prompts.py / evaluate.
TIER_TO_ROUTE = {
    "dialogue": "dialogue",
    "explicit_seg": "segmentation",
    "reasoning_seg": "segmentation",
    "mixed_intent": "segmentation",
}
ROUTES = ("dialogue", "segmentation", "follow_up")


@dataclass
class BenchConfig:
    # ---- data sources (relative to dataset_dir) ----
    dataset_dir: str = "/path/to/lisa_data/"
    llava_instruct_rel: str = "llava_dataset/llava_instruct_150k.json"
    vqa_image_root_rel: str = "coco/train2017"
    refcoco_root_rel: str = "refer_seg"
    refcoco_specs: List[tuple] = field(default_factory=lambda: [
        ("refcoco", "unc"), ("refcoco+", "unc"), ("refcocog", "umd"),
    ])
    refcoco_split: str = "val"                     # which REFER split to draw from
    reasonseg_splits: List[str] = field(default_factory=lambda: ["val", "train"])

    # ---- per-tier target counts (~1000 total) ----
    n_dialogue: int = 250
    n_explicit: int = 250
    n_reasoning: int = 250
    n_mixed: int = 250
    oversample: float = 3.0                        # draw this x target before LLM filtering
    seed: int = 42

    # ---- lmdeploy / qwen3-32b (OpenAI-compatible) ----
    use_llm: bool = True                           # False -> template-only (smoke tests)
    lm_base_url: str = "http://localhost:23333/v1"
    lm_model: str = "qwen3-32b"                    # served name; check GET /v1/models
    lm_api_key: str = "EMPTY"
    lm_enable_thinking: bool = False               # Qwen3 thinking off for clean JSON
    lm_temperature: float = 0.7
    lm_top_p: float = 0.8
    lm_max_tokens: int = 1024
    lm_max_retries: int = 4
    lm_timeout: float = 120.0

    # ---- io ----
    cache_dir: str = "routing_benchmark/.cache"
    out_path: str = "routing_benchmark/benchmark.jsonl"

    def apply_overrides(self, args):
        """Override fields from an argparse Namespace (only for attrs that are not None)."""
        for f in fields(self):
            if hasattr(args, f.name) and getattr(args, f.name) is not None:
                setattr(self, f.name, getattr(args, f.name))
        return self
