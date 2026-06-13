"""Pluggable backbones for LENS.

* MLLM backbones (frozen multimodal LLM read by the conditioner) live behind the
  ``BACKBONE_REGISTRY`` and are selected by name via the model config.
* SAM segmentation backbones are selected via :func:`build_sam`.

Importing this package registers all built-in MLLM backbones.
"""

from .base import (
    BACKBONE_REGISTRY,
    MLLMBackbone,
    build_mllm_backbone,
    compute_token_masks,
    extract_user_instruction,
    force_eager_attention,
    register_backbone,
)
from .generic import GenericHFVLMBackbone
from .sam import (
    DEFAULT_SAM_CHECKPOINTS,
    SAM_BUILDERS,
    SAM_URLS,
    build_sam,
    download_sam_checkpoint,
    sam_url,
)

# Import side-effect: registers built-in backbones into BACKBONE_REGISTRY.
from . import llava_backbone  # noqa: F401  (registration side-effect)
from . import qwen2vl_backbone  # noqa: F401  (registration side-effect)
from . import llava_onevision_backbone  # noqa: F401  (registration side-effect)

__all__ = [
    "BACKBONE_REGISTRY",
    "MLLMBackbone",
    "GenericHFVLMBackbone",
    "build_mllm_backbone",
    "register_backbone",
    "compute_token_masks",
    "extract_user_instruction",
    "force_eager_attention",
    "SAM_BUILDERS",
    "SAM_URLS",
    "DEFAULT_SAM_CHECKPOINTS",
    "build_sam",
    "sam_url",
    "download_sam_checkpoint",
]
