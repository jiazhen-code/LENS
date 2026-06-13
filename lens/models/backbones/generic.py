"""Generic Hugging Face vision-language backbone.

This base implements the chat-template ``encode`` used by **Qwen2-VL**: format the user
instruction with the model's own chat template, let the processor expand the image
placeholder, run the model with ``output_hidden_states=True``, and read the hidden states at
a chosen layer. Only a few per-model constants differ (repo name, hidden size, image-token
count, image size). **LLaVA-OneVision** also subclasses this base (for the fusion-head build,
image squaring and text-model location) but OVERRIDES ``encode`` to bypass its anyres
tokeniser -- see ``llava_onevision_backbone.py``.

This path is deliberately separate from the vendored LLaVA-1.5 backbone (whose ``encode`` is
kept byte-for-byte). Unlike LLaVA-1.5, a chat-template backbone here:
  * rebuild the prompt via ``apply_chat_template`` (each model has its own image markers);
  * square the image to a fixed ``grid x grid`` token grid -- by default aspect-preserving
    resize + bottom/right pad (``image_square_mode``), matching the SAM / heatmap-GT frame;
  * skip the grounding-head warm-start (their LLM weights are not Llama-compatible) --
    set ``model.backbone.grounding_init_start_layer: null`` in the config.

NOTE: the exact image-token count depends on the processor/version. ``encode`` asserts the
count matches ``num_image_tokens`` and raises a clear error telling you which knob to turn,
so a wrong assumption fails fast instead of silently corrupting the heatmap grid.
"""

import numpy as np
import torch
from PIL import Image

from .base import MLLMBackbone, compute_token_masks, extract_user_instruction


class GenericHFVLMBackbone(MLLMBackbone):
    # Chat-template content entry for the image part; overridable per model if needed.
    image_content = {"type": "image"}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model, self.processor = self._load(cfg)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.hidden_size = cfg.hidden_size
        # Fail fast on a wrong hidden_size (the #1 mistake when adding a new model size)
        # instead of a confusing shape error deep in the forward pass.
        try:
            actual_hidden = self._text_model().config.hidden_size
        except Exception:
            actual_hidden = None
        if actual_hidden is not None and actual_hidden != self.hidden_size:
            raise ValueError(
                f"[{cfg.name}] backbone.hidden_size={self.hidden_size} but the loaded "
                f"model's hidden size is {actual_hidden}. Set "
                f"model.backbone.hidden_size: {actual_hidden}."
            )

        self.num_image_tokens = cfg.num_image_tokens
        self.image_grid_size = int(cfg.num_image_tokens ** 0.5)
        self.selected_layer_id = cfg.selected_layer_id
        self.grounding_init_start_layer = cfg.grounding_init_start_layer
        self.image_hw = cfg.image_hw
        # 'pad' (aspect-preserving resize + bottom/right pad, the SAM / heatmap-GT frame) or
        # 'stretch' (old resize-to-square). Only used when image_hw is set.
        self.image_square_mode = getattr(cfg, "image_square_mode", "pad")
        self._image_token_id = self._resolve_image_token_id()

    # ---- loading (override in subclass to pin a specific HF class) ----------
    def _load(self, cfg):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=cfg.trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(
            cfg.model_name, trust_remote_code=cfg.trust_remote_code
        )
        return model, processor

    def _resolve_image_token_id(self):
        for attr in ("image_token_id", "image_token_index"):
            tok = getattr(self.model.config, attr, None)
            if tok is not None:
                return tok
        # Fall back to a name lookup if the subclass declares the token string.
        token_str = getattr(self, "image_token_str", None)
        if token_str is not None:
            return self.processor.tokenizer.convert_tokens_to_ids(token_str)
        return None

    @property
    def dtype(self):
        return self.model.dtype

    def _text_model(self):
        """Locate the inner text transformer across transformers layouts. Override in a
        subclass if your version nests it elsewhere (the error message lists what to do)."""
        m = self.model
        candidates = (
            lambda: m.model.language_model,  # native VLMs: ForCausalLM.model.language_model
            lambda: m.language_model,        # some expose the text model directly
            lambda: m.model,                 # older: .model is the text stack
        )
        for get in candidates:
            try:
                obj = get()
            except AttributeError:
                continue
            if hasattr(obj, "layers") and hasattr(obj, "config"):
                return obj
        raise AttributeError(
            f"[{self.cfg.name}] could not locate the inner text transformer for fusion-head "
            f"construction. Override _text_model() in the backbone class for your "
            f"transformers version."
        )

    # ---- helpers ------------------------------------------------------------
    def _to_pil(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(image, np.ndarray):
            return [Image.fromarray(img.astype("uint8"), "RGB") for img in image]
        return [
            Image.open(it).convert("RGB") if isinstance(it, str) else it.convert("RGB")
            for it in image
        ]

    def _pad_color(self):
        """Per-channel processor mean in 0-255, used to pad so the padded region normalizes
        to ~0 (the neutral value SAM's zero-pad-after-normalize also produces)."""
        ip = getattr(self.processor, "image_processor", self.processor)
        mean = getattr(ip, "image_mean", None)
        if not mean:
            return (124, 116, 104)
        if max(mean) <= 1.0:
            mean = [m * 255.0 for m in mean]
        return tuple(int(round(m)) for m in mean)

    def _square_image(self, im):
        """Square ``im`` to ``image_hw`` x ``image_hw`` for the fixed grid x grid tokeniser.

        'pad' (default): aspect-preserving resize to longest side == image_hw, then pad
        bottom/right with the processor mean -- the SAME frame SAM uses (ResizeLongestSide +
        bottom/right pad) and that the heatmap GT (data_deal) uses, so the attention heatmap,
        its GT, and the SAM point/dense prompts share ONE normalized space. 'stretch'
        reproduces the old aspect-distorting resize-to-square.
        """
        hw = self.image_hw
        if self.image_square_mode == "stretch":
            return im.resize((hw, hw))
        w, h = im.size
        scale = hw / max(w, h)
        new_w = min(hw, max(1, round(w * scale)))
        new_h = min(hw, max(1, round(h * scale)))
        resized = im.resize((new_w, new_h))
        canvas = Image.new("RGB", (hw, hw), self._pad_color())
        canvas.paste(resized, (0, 0))  # content top-left, pad on bottom/right (matches SAM)
        return canvas

    def _build_prompt(self, instruction):
        messages = [{"role": "user",
                     "content": [self.image_content, {"type": "text", "text": instruction}]}]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _fusion_pos_capture_hook(self):
        """Install a forward hook that captures the position ids the backbone forward passes
        to its text model (stored in ``self._fusion_position_ids``); return the handle so
        ``encode`` can remove it. Default: no hook (None). Only backbones whose fusion head
        needs non-1-D positions (Qwen2-VL M-RoPE) override this."""
        return None

    # ---- encode -------------------------------------------------------------
    @torch.no_grad
    def encode(self, image, text):
        images = self._to_pil(image)
        if self.image_hw is not None:
            images = [self._square_image(im) for im in images]

        prompts = [self._build_prompt(extract_user_instruction(t)) for t in text]
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        )
        inputs = {
            k: (
                v.to(self.model.device).to(self.model.dtype)
                if v.is_floating_point()
                else v.to(self.model.device)
            )
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }

        # Capture (NOT recompute) the position ids the backbone forward itself uses for the
        # fusion head, by hooking the inner text model -> the fusion head reuses the EXACT
        # tensor. A backbone that needs them (Qwen2-VL M-RoPE) installs the hook; others leave
        # it None (-> 1-D / sequential downstream).
        self._fusion_position_ids = None
        _pos_hook = self._fusion_pos_capture_hook()
        try:
            outputs = self.model(**inputs, output_hidden_states=True)
        finally:
            if _pos_hook is not None:
                _pos_hook.remove()
        # Fallback: if the hook captured nothing (e.g. transformers moved where position_ids
        # flow) but the fusion head needs non-1-D ids, recompute via get_rope_index and warn
        # loudly -- never silently degrade to sequential positions.
        if self._fusion_position_ids is None and self.fusion_explicit_position_ids is False:
            self._fusion_position_ids = self._compute_fusion_position_ids(inputs)
            if not getattr(self, "_warned_pos_hook", False):
                print(f"[{self.cfg.name}] fusion position-id hook captured nothing; fell back "
                      f"to get_rope_index. Check _fusion_pos_capture_hook for this transformers "
                      f"version.")
                self._warned_pos_hook = True
        multimodal_features = outputs.hidden_states[self.selected_layer_id]
        last_multimodal_features = outputs.hidden_states[-1]

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if self._image_token_id is None:
            raise ValueError(
                f"[{self.cfg.name}] could not resolve the image token id from the model "
                f"config/processor. Set `image_token_str` on the backbone class."
            )

        image_token_mask, text_token_mask, ids = compute_token_masks(
            input_ids, attention_mask, self._image_token_id
        )

        counts = image_token_mask.sum(dim=1)
        if not bool((counts == self.num_image_tokens).all()):
            raise ValueError(
                f"[{self.cfg.name}] expected {self.num_image_tokens} image tokens per "
                f"sample but got {counts.tolist()}. The keypoint grid must be a square of "
                f"size sqrt(num_image_tokens). Adjust model.backbone.image_hw and/or "
                f"num_image_tokens so the processor emits a square token grid."
            )

        return (
            multimodal_features,
            attention_mask,
            image_token_mask,
            text_token_mask,
            last_multimodal_features,
            ids,
        )
