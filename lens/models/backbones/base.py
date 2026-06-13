"""Backbone registry for LENS.

A *backbone* here is the frozen multimodal LLM that the conditioner reads
language->image attention from. The conditioner is written against the small interface
below, so adding a new backbone (e.g. Qwen-VL, LLaVA-OneVision, a different LLaVA size) only
means writing one ``MLLMBackbone`` subclass and registering it -- no edits to
``conditioner.py``, ``decoder.py`` or the training loop.

Interface a subclass must provide
---------------------------------
Attributes (set in ``__init__``):
    model                    : the underlying nn.Module (kept frozen)
    processor                : the HF processor / tokenizer wrapper used by ``encode``
    hidden_size              : width of the multimodal hidden states tapped
    num_image_tokens         : number of image tokens in the sequence
    image_grid_size          : int(sqrt(num_image_tokens))
    selected_layer_id        : hidden-state layer index used as multimodal features
    grounding_init_start_layer : backbone decoder layer the GroundingHead is seeded from
    dtype (property)         : parameter dtype of the backbone

Methods:
    encode(image, text) -> (multimodal_features, attention_mask, image_token_mask,
                            text_token_mask, last_multimodal_features, ids)
    get_init_source_layers() -> indexable list of decoder layers for weight copy
"""

import torch
import torch.nn as nn


BACKBONE_REGISTRY = {}


def force_eager_attention(model):
    """Force eager attention on a model (and any sub-model configs).

    The grounding head needs ``output_attentions=True`` (to read language->image attention),
    which transformers only supports under eager attention. Setting ``_attn_implementation =
    'eager'`` on the config BEFORE construction is not enough: ``_autoset_attn_implementation``
    runs during ``__init__`` and may reset it to ``sdpa``. The unified attention dispatch
    reads ``config._attn_implementation`` at forward time, so re-applying it post-init on the
    (shared) config objects makes ``output_attentions`` work. Returns ``model``.
    """
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
    return model


def extract_user_instruction(text: str, image_token: str = "<image>") -> str:
    """Pull the raw user question out of a llava_v1-formatted conversation string.

    The datasets emit prompts like
    ``"<system> USER: <image>\\n{question} ASSISTANT:"``. New backbones rebuild the
    prompt with their own chat template, so they only need the bare ``{question}``.
    """
    t = text.split("ASSISTANT:")[0]
    if "USER:" in t:
        t = t.split("USER:")[-1]
    t = t.replace(image_token, "").replace("<image>", "")
    return t.strip()

def compute_token_masks(input_ids, attention_mask, image_token_id):
    """Image / text token masks + ``ids``, identical in spirit to LLaVA's encode().

    * image_token_mask : positions equal to the image placeholder id
    * text_token_mask  : non-image, non-pad tokens that come *after* the last image token
    * ids              : input_ids with pad positions set to -100
    """
    device = input_ids.device
    image_token_mask = (input_ids == image_token_id)
    initial_text_mask = (input_ids != image_token_id) & (attention_mask == 1)

    last_img_indices = image_token_mask.cumsum(dim=1).argmax(dim=1)
    seq_len = input_ids.shape[1]
    indices = torch.arange(seq_len, device=device).unsqueeze(0)
    after_image_mask = indices > last_img_indices.unsqueeze(1)
    text_token_mask = initial_text_mask & after_image_mask

    ids = input_ids.clone()
    ids[attention_mask == 0] = -100
    return image_token_mask, text_token_mask, ids


def register_backbone(name):
    """Class decorator that records a backbone under ``name`` in the registry."""
    def _decorator(cls):
        if name in BACKBONE_REGISTRY and BACKBONE_REGISTRY[name] is not cls:
            raise KeyError(f"Backbone '{name}' is already registered to {BACKBONE_REGISTRY[name]!r}")
        BACKBONE_REGISTRY[name] = cls
        return cls
    return _decorator


def build_mllm_backbone(backbone_cfg):
    """Instantiate the backbone selected by ``backbone_cfg.name``."""
    name = backbone_cfg.name
    if name not in BACKBONE_REGISTRY:
        raise KeyError(
            f"Unknown backbone '{name}'. Registered backbones: {sorted(BACKBONE_REGISTRY)}"
        )
    return BACKBONE_REGISTRY[name](backbone_cfg)


class MLLMBackbone(nn.Module):
    """Base class documenting the contract used by ``MLLM_Conditioner``.

    Beyond ``encode``, a backbone also OWNS the construction of the grounding-head fusion
    transformer. Per CR1(a), the 2-layer fusion head is re-instantiated to mirror ONE of
    the backbone's own transformer layers (same hidden size + attention config) and is
    warm-started from the backbone's middle layers -- so swapping the backbone swaps the
    fusion-head architecture with it, instead of forcing a LLaVA-shaped LlamaModel.
    """

    # Whether the grounding head feeds the fusion model explicit 1-D position ids
    # (cumsum of the attention mask, as LLaVA-1.5/Qwen2 expect). Backbones whose layer
    # uses a non-1-D scheme (e.g. Qwen2-VL's M-RoPE) set this False so the head passes
    # position_ids=None and the model builds its own (degenerate -> sequential) ids.
    fusion_explicit_position_ids = True

    def encode(self, image, text):
        raise NotImplementedError

    # ---- fusion-head position ids -------------------------------------------
    def fusion_position_ids(self):
        """Position ids to feed the fusion model for the MOST RECENT encode() call, or
        None. A backbone whose layer needs non-1-D positions (e.g. Qwen2-VL M-RoPE) stashes
        them during encode via ``_compute_fusion_position_ids``; otherwise the grounding
        head falls back to 1-D (cumsum) or sequential ids.
        """
        return getattr(self, "_fusion_position_ids", None)

    def _compute_fusion_position_ids(self, inputs):
        """Override to return backbone-specific position ids (e.g. 3-D M-RoPE) from the
        processed ``inputs`` dict. Default: None (1-D / sequential downstream)."""
        return None

    # ---- fusion-head construction / warm-start ------------------------------
    def _text_model(self):
        """Return the backbone's inner text transformer.

        Must expose ``.layers`` (a list of decoder layers), ``.config`` (the per-layer
        config), and a ``forward(inputs_embeds=..., attention_mask=..., position_ids=...,
        output_attentions=True)`` returning ``last_hidden_state`` + ``attentions``.
        """
        raise NotImplementedError

    def get_init_source_layers(self):
        """Decoder layers used to warm-start the fusion head."""
        return self._text_model().layers

    def build_fusion_model(self, num_layers):
        """Build a ``num_layers`` transformer mirroring ONE of this backbone's decoder
        layers: same class and per-layer config, ``embed_tokens`` removed, eager attention
        (so ``attentions`` are returned). Subclasses override when the inner text model
        carries a composite config (e.g. Qwen2-VL) and a plain text model must be built.
        """
        import copy
        text_model = self._text_model()
        config = copy.deepcopy(text_model.config)
        config.num_hidden_layers = num_layers
        # Set eager FIRST: a config deep-copied from a loaded model carries
        # _attn_implementation="sdpa", and the `output_attentions` SETTER raises under sdpa.
        # We don't set config.output_attentions at all -- the grounding head passes
        # output_attentions=True per forward call; force_eager_attention re-applies eager
        # post-init (autoset may flip it back to sdpa during __init__).
        config._attn_implementation = "eager"
        if hasattr(config, "vocab_size"):
            config.vocab_size = 1  # embed_tokens is deleted; avoid a large transient alloc
        fusion = type(text_model)(config)
        if hasattr(fusion, "embed_tokens"):
            del fusion.embed_tokens
        return force_eager_attention(fusion)

    def warmstart_fusion_layers(self, fusion_model, start_layer):
        """Copy decoder layers ``[start_layer : start_layer + L]`` of this backbone into
        ``fusion_model.layers`` (``L`` = number of fusion layers). The fusion layers are
        the same class as the source, so the state-dict keys line up.
        """
        source = self.get_init_source_layers()
        target = fusion_model.layers
        n = len(target)
        if start_layer + n > len(source):
            start_layer = max(0, len(source) - n)
        print(f"Warm-starting {n} fusion layer(s) from backbone layers "
              f"[{start_layer}:{start_layer + n}] of {len(source)}.")
        for i in range(n):
            target[i].load_state_dict(source[start_layer + i].state_dict())
        print("Fusion warm-start complete. ✅")
