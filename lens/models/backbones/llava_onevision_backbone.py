"""LLaVA-OneVision backbone (non-anyres / fixed base grid).

LLaVA-OneVision (e.g. ``llava-hf/llava-onevision-qwen2-7b-ov-hf``) pairs a SigLIP vision
tower with a **Qwen2** language model. Despite the "LLaVA" name it does NOT reuse the
vendored LLaVA-1.5 path (``lens/models/llava.py`` + ``llava_backbone.py``), which assumes a
Llama LM. It subclasses :class:`GenericHFVLMBackbone` for the shared plumbing (fusion-head
build that mirrors a Qwen2 layer, warm-start, image squaring, text-model location), but
**overrides ``encode`` to bypass anyres**.

Why override encode (non-anyres)
--------------------------------
LLaVA-OneVision's processor runs *anyres*: for a single image it expands ``<image>`` into
``base (grid^2) + anyres_patch (grid^2) + per-row newline (grid)`` placeholders (e.g.
``729 + 729 + 27 = 1485`` at 384px) -- a non-square, resolution-dependent count, and the
model's ``pack_image_features`` always appends an ``image_newline`` even in the base-only
branch. The conditioner needs an EXACT square ``grid x grid`` image-token block, so instead
of letting the processor tile the image we drive the model the LLaVA-1.5 way:

  1. square the image to ``image_hw`` (== SigLIP base size, 384);
  2. run the SigLIP ``vision_tower`` + ``multi_modal_projector`` directly -> exactly
     ``grid^2`` (729) patch embeddings, NO base/patch duplication, NO newline;
  3. format the prompt with the model's chat template, then expand the single ``<image>``
     marker to ``num_image_tokens`` placeholders and scatter the patch embeddings in;
  4. run the inner **Qwen2** text model on the resulting ``inputs_embeds`` and read the
     hidden states (standard 1-D RoPE -> ``fusion_explicit_position_ids = True``, default).

This is structurally the LLaVA-1.5 encode with SigLIP (729, no CLS) + Qwen2. ``encode``
asserts the SigLIP token count, so if a transformers/SigLIP version yields a different
square, the error says which knob to turn -- confirm with ``scripts/smoke_backbone.py``.
"""

import numpy as np
import torch

from .base import compute_token_masks, extract_user_instruction, register_backbone
from .generic import GenericHFVLMBackbone


# One class handles every LLaVA-OneVision size: it loads by model_name and the fusion head
# reads dims from the model's own Qwen2 config. Registered under a generic name + per-size
# aliases (mirrors Qwen2VLBackbone).
@register_backbone("llava-onevision")
@register_backbone("llava-onevision-7b")
@register_backbone("llava-onevision-0.5b")
class LlavaOnevisionBackbone(GenericHFVLMBackbone):
    image_token_str = "<image>"
    # LLaVA-OneVision's Qwen2 decoder uses standard 1-D RoPE, so the fusion head IS fed
    # 1-D cumsum position ids (the inherited default). Only Qwen2-VL's M-RoPE sets this False.

    def _load(self, cfg):
        try:
            from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "LLaVA-OneVision needs a transformers version that ships "
                "LlavaOnevisionForConditionalGeneration (>= 4.45)."
            ) from e
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=cfg.trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(
            cfg.model_name, trust_remote_code=cfg.trust_remote_code
        )
        return model, processor

    # NOTE: no _text_model / build_fusion_model override -- the generic base locates the inner
    # Qwen2 text decoder (model.model.language_model) and rebuilds a same-class copy, so the
    # fusion head genuinely mirrors a Qwen2 layer and warm-start is an exact weight copy.

    # ---- non-anyres image embedding -----------------------------------------
    def _base_pixel_values(self, images):
        """SigLIP pixel values for the already-square ``image_hw`` images (a single base
        tile, no anyres). The images are exactly ``image_hw`` from ``_square_image``, so this
        is just rescale + normalize with the processor's mean/std -- no resize, so there is no
        interpolation ambiguity. Padded regions (processor mean) normalize to ~0."""
        ip = getattr(self.processor, "image_processor", self.processor)
        mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        rescale = float(getattr(ip, "rescale_factor", 1.0 / 255.0))
        arr = np.stack([np.asarray(im, dtype=np.float32) for im in images])  # (B, hw, hw, 3)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2) * rescale
        return (t - mean) / std

    def _embed_image(self, pixel_values):
        """SigLIP ``vision_tower`` + ``multi_modal_projector`` -> ``(B, grid^2, hidden)``,
        mirroring HF ``get_image_features`` but WITHOUT the anyres ``pack_image_features``
        (no base/patch duplication, no ``image_newline``)."""
        inner = getattr(self.model, "model", self.model)
        vt = getattr(inner, "vision_tower", None) or self.model.vision_tower
        proj = getattr(inner, "multi_modal_projector", None) or self.model.multi_modal_projector
        cfg = self.model.config
        layer = getattr(cfg, "vision_feature_layer", -1)
        strat = getattr(cfg, "vision_feature_select_strategy", "full")
        out = vt(pixel_values, output_hidden_states=True)
        if isinstance(layer, (list, tuple)):
            pool = [out.hidden_states[i] for i in layer]
            if strat == "default":
                pool = [h[:, 1:] for h in pool]
            feats = torch.cat(pool, dim=-1)
        else:
            feats = out.hidden_states[layer]
            if strat == "default":  # SigLIP has no CLS -> strategy is 'full', no drop
                feats = feats[:, 1:]
        return proj(feats)

    # ---- encode -------------------------------------------------------------
    @torch.no_grad
    def encode(self, image, text):
        if self._image_token_id is None:
            raise ValueError(
                f"[{self.cfg.name}] could not resolve the image token id from the model "
                f"config/processor. Set `image_token_str` on the backbone class."
            )
        n_img = self.num_image_tokens
        device = self.model.device

        images = self._to_pil(image)
        images = [self._square_image(im) for im in images]  # image_hw x image_hw (SigLIP base)

        # 1-2. single base tile -> exactly n_img SigLIP patch embeddings (no anyres)
        pixel_values = self._base_pixel_values(images).to(device, self.model.dtype)
        image_embeds = self._embed_image(pixel_values)  # (B, n_img, hidden)
        produced = image_embeds.shape[1]
        if produced != n_img:
            raise ValueError(
                f"[{self.cfg.name}] SigLIP produced {produced} patch tokens but "
                f"num_image_tokens={n_img}. Set model.backbone.num_image_tokens={produced} "
                f"(must be a perfect square) and image_hw to the SigLIP base size."
            )

        # 3. prompt via chat template, expand the single <image> marker -> n_img placeholders
        prompts = [self._build_prompt(extract_user_instruction(t)) for t in text]
        tok = self.processor.tokenizer
        rows = tok(prompts, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
        expanded = []
        for ids in rows:
            if ids.count(self._image_token_id) != 1:
                raise ValueError(
                    f"[{self.cfg.name}] expected exactly one <image> marker per prompt from the "
                    f"chat template, got {ids.count(self._image_token_id)}."
                )
            out = []
            for tid in ids:
                out.extend([self._image_token_id] * n_img if tid == self._image_token_id else [tid])
            expanded.append(out)
        padded = tok.pad({"input_ids": expanded}, padding=True, return_tensors="pt")
        input_ids = padded["input_ids"].to(device)
        attention_mask = padded["attention_mask"].to(device)

        # 4. scatter image embeds into the text embedding sequence, run the Qwen2 LM directly
        inputs_embeds = self.model.get_input_embeddings()(input_ids).clone()
        img_mask = input_ids == self._image_token_id
        inputs_embeds[img_mask] = image_embeds.reshape(-1, image_embeds.shape[-1]).to(inputs_embeds.dtype)

        self._fusion_position_ids = None  # 1-D RoPE -> grounding head uses cumsum positions
        outputs = self._text_model()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        multimodal_features = outputs.hidden_states[self.selected_layer_id]
        last_multimodal_features = outputs.hidden_states[-1]

        image_token_mask, text_token_mask, ids = compute_token_masks(
            input_ids, attention_mask, self._image_token_id
        )
        return (
            multimodal_features,
            attention_mask,
            image_token_mask,
            text_token_mask,
            last_multimodal_features,
            ids,
        )
