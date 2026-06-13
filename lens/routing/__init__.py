"""Routing benchmark: zero-shot tier classification by a frozen MLLM backbone."""

from .benchmark import load_benchmark
from .router import (
    BINARY_ROUTING_PROMPT,
    DEFAULT_ROUTING_PROMPT,
    LLAVA_SYSTEM_PROMPT,
    SEG_TIERS,
    TIERS,
    MLLMRouter,
    compute_binary_metrics,
    compute_routing_metrics,
    normalize_tier,
    parse_binary,
    parse_tier,
)

__all__ = [
    "load_benchmark",
    "MLLMRouter",
    "compute_routing_metrics",
    "compute_binary_metrics",
    "parse_tier",
    "parse_binary",
    "normalize_tier",
    "TIERS",
    "SEG_TIERS",
    "DEFAULT_ROUTING_PROMPT",
    "BINARY_ROUTING_PROMPT",
    "LLAVA_SYSTEM_PROMPT",
]
