"""Qwen2-VL-2B backbone.

Reference config (Qwen/Qwen2-VL-2B-Instruct): hidden_size=1536, 28 layers, vision patch
14 with spatial-merge 2. A square image of side ``image_hw`` (multiple of 28) yields
``(image_hw/28)^2`` merged image tokens. The default 448 -> 16x16 = 256 tokens.

Loaded explicitly via ``Qwen2VLForConditionalGeneration`` (more robust than Auto across
transformers versions). Requires transformers >= 4.45.
"""

import torch

from .base import register_backbone
from .generic import GenericHFVLMBackbone


# One class handles every Qwen2-VL size: it loads by model_name and the fusion head reads
# dims from the model's own config. Registered under a generic name + per-size aliases.
@register_backbone("qwen2-vl")
@register_backbone("qwen2-vl-2b")
@register_backbone("qwen2-vl-7b")
class Qwen2VLBackbone(GenericHFVLMBackbone):
    image_token_str = "<|image_pad|>"
    # Qwen2-VL's decoder layer uses M-RoPE, so the fusion head must NOT be fed 1-D cumsum
    # position ids; pass None and let the (real) Qwen2-VL text model build its own.
    fusion_explicit_position_ids = False

    def _load(self, cfg):
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "Qwen2-VL needs a transformers version that ships "
                "Qwen2VLForConditionalGeneration (>= 4.45)."
            ) from e
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=cfg.trust_remote_code,
            attn_implementation="eager"
        )
        processor = AutoProcessor.from_pretrained(
            cfg.model_name, trust_remote_code=cfg.trust_remote_code
        )
        return model, processor

    # NOTE: no _text_model / build_fusion_model override on purpose -- the generic base
    # locates the real Qwen2-VL text decoder (model.model.language_model on transformers
    # >= 4.49, model.model on older) and rebuilds a 2-layer copy of that SAME class, so the
    # fusion head genuinely mirrors a Qwen2-VL layer (M-RoPE included) and warm-start is an
    # exact same-class weight copy.

    def _fusion_pos_capture_hook(self):
        """Grab the 3-D M-RoPE position ids the Qwen2-VL forward passes to its text model, so
        the fusion head reuses the EXACT tensor -- no recompute, immune to get_rope_index
        signature drift. Hooks the inner text transformer's forward-pre for its ``position_ids``
        kwarg; ``encode`` removes the handle and falls back to ``_compute_fusion_position_ids``
        if nothing was captured."""
        target = self._text_model()  # the inner Qwen2-VL text decoder that receives position_ids

        def _grab(module, args, kwargs):
            pos = kwargs.get("position_ids", None)
            if pos is not None:
                self._fusion_position_ids = pos.detach()

        return target.register_forward_pre_hook(_grab, with_kwargs=True)

    def _get_rope_index_fn(self):
        # get_rope_index lives on the ForConditionalGeneration (<=4.48) or the inner
        # Qwen2VLModel (>=4.49). Find whichever exposes it.
        for obj in (self.model, getattr(self.model, "model", None)):
            if obj is not None and hasattr(obj, "get_rope_index"):
                return obj.get_rope_index
        return None

    def _compute_fusion_position_ids(self, inputs):
        """Real 3-D M-RoPE position ids for the fusion head. Falls back to None (fusion
        head then builds sequential ids) if get_rope_index is missing or its signature
        changed in this transformers version."""
        fn = self._get_rope_index_fn()
        if fn is None:
            return None
        try:
            position_ids, _ = fn(
                inputs["input_ids"],
                image_grid_thw=inputs.get("image_grid_thw"),
                attention_mask=inputs.get("attention_mask"),
            )
            return position_ids  # [3, batch, seq]
        except Exception as e:
            if not getattr(self, "_warned_rope", False):
                print(f"[{self.cfg.name}] get_rope_index failed ({e}); using 1-D position "
                      f"ids for the fusion head.")
                self._warned_rope = True
            return None
