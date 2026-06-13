"""Central configuration for LENS.

All hyper-parameters that used to be hard-coded inside ``train.py`` / ``eval.py`` /
``lens.models.*`` live here as plain dataclasses. Every default below is *exactly* the
value that was previously hard-coded, so running with the default config reproduces the
original behaviour bit-for-bit -- the config layer only changes *where* the numbers come
from, never the logic that consumes them.

Usage
-----
    from lens.config import load_config
    cfg = load_config("configs/default.yaml")      # or load_config(None) for pure defaults

A YAML file only needs to list the keys it wants to override; everything else falls back
to the dataclass defaults. Nested sections map 1:1 to the nested dataclasses, e.g.::

    train:
      base_lr: 1.0e-4
    model:
      backbone:
        name: llava-1.5-7b
"""

from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Model sub-configs
# ---------------------------------------------------------------------------
@dataclass
class BackboneConfig:
    """Selects and parameterises the (frozen) MLLM backbone of the conditioner.

    ``name`` is looked up in ``lens.models.backbones`` registry. To add a new backbone,
    register a class there and point ``name`` at it -- no other code needs to change.
    """
    name: str = "llava-1.5-7b"
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    hidden_size: int = 4096
    num_image_tokens: int = 576
    # Hidden-state layer tapped for the multimodal features (conditioner.selected_layer_id).
    selected_layer_id: int = 14
    # Start index of the backbone layers used to WARM-START the fusion head. The fusion
    # head mirrors the backbone's own layer (see backbone.build_fusion_model), so warm-start
    # is valid for every backbone -- set this to ~num_layers//2 (the middle). null = skip
    # (train the fusion head from scratch). Originally 14 for LLaVA-1.5-7B.
    grounding_init_start_layer: Optional[int] = 14
    # Load HF model/processor with trust_remote_code (some VLMs ship custom modeling).
    trust_remote_code: bool = False
    # Force every input image to this square size so the image-token grid is a fixed
    # ``grid x grid`` (== num_image_tokens). None => let the processor decide (LLaVA).
    image_hw: Optional[int] = None
    # How a forced-square backbone (image_hw set) makes the image square before the fixed
    # grid x grid tokeniser:
    #   'pad'     -> aspect-preserving resize (longest side == image_hw) + bottom/right pad
    #                with the processor mean. SAME frame SAM uses (ResizeLongestSide + pad) and
    #                that the heatmap GT (data_deal) uses, so the attention heatmap, its GT,
    #                and the SAM point/dense prompts all share ONE normalized space.
    #   'stretch' -> old resize-to-square; distorts the aspect ratio and misaligns those
    #                frames for non-square images. Set this only to eval a stretch-trained ckpt.
    # Ignored when image_hw is None (e.g. LLaVA, which has its own processor).
    image_square_mode: str = "pad"


@dataclass
class GroundingHeadConfig:
    num_attention_layers: int = 2
    num_heads: int = 32
    intermediate_size: int = 11008


@dataclass
class KeypointConfig:
    num_keypoints: int = 16
    patch_size: int = 3
    thre_ratio: float = 0.4
    suppression_radius: int = 4


@dataclass
class DecoderConfig:
    sam_variant: str = "vit_h"
    sam_checkpoint: str = "./sam_vit_h_4b8939.pth"
    # Segmentation loss weights: (2 * bce + 4 * dice).
    bce_weight: float = 4.0
    dice_weight: float = 2.0
    # Optionally feed the attention heatmap (projected to SAM's 256-d embedding space) into
    # the mask decoder as the DENSE prompt, instead of SAM's learned no-mask embedding.
    # Default False -> original behaviour (no_mask_embed).
    use_attn_dense_prompt: bool = False


@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    grounding_head: GroundingHeadConfig = field(default_factory=GroundingHeadConfig)
    keypoint: KeypointConfig = field(default_factory=KeypointConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    # Keypoint threshold passed to the conditioner during the forward pass (FullModel).
    kp_thresh: float = 0.7


# ---------------------------------------------------------------------------
# Data sub-config (shared by train / eval)
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    dataset_dir: str = "/path/to/lisa_data/"

    image_size: int = 1024
    clip_vision_model: str = "openai/clip-vit-large-patch14"
    tokenizer_name: str = "liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    model_max_length: int = 512
    num_classes_per_sample: int = 3
    sample_rates: str = "3,3,0,1"
    dataset: str = "sem_seg||refer_seg||vqa||reason_seg"
    sem_seg_data: str = "ade20k||cocostuff||pascal_part||paco_lvis||mapillary"
    refer_seg_data: str = "refclef||refcoco||refcoco+||refcocog"
    reason_seg_data: str = "ReasonSeg|train"
    vqa_data: str = "llava_instruct_150k"
    explanatory: int = -1
    exclude_val: bool = True


# ---------------------------------------------------------------------------
# Train sub-config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    batch_size: int = 8
    num_workers: int = 4
    train_per_steps: int = 100
    num_epochs: int = 500
    base_lr: float = 5e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eta_min: float = 1e-6
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    save_dir: str = "./save_dir"
    # Checkpoint loaded before training (was ``./save_dir/full_10.pth``, strict=False).
    # Set to null/empty in YAML to train from scratch.
    resume_checkpoint: Optional[str] = None
    save_every_epochs: int = 1
    # Local visualization during training: dump up to max_viz samples/epoch (image +
    # attention overlay + predicted keypoints, pred mask vs GT) + instruction/attention
    # sidecars to viz_train/epoch_<ep>/. Mirrors EvalConfig.save_viz/max_viz; main proc only.
    save_viz: bool = True
    max_viz: int = 20
    # Save only trainable tensors. The frozen MLLM backbone (~billions of params) and the
    # two frozen SAM copies are rebuilt from from_pretrained / build_sam on load, so writing
    # them every epoch just fills the disk. The tolerant loader restores the rest. Set False
    # to save the full state_dict (original behaviour).
    save_trainable_only: bool = True
    hf_endpoint: str = "https://hf-mirror.com"
    # The SAM checkpoint is downloaded to model.decoder.sam_checkpoint using the URL for
    # model.decoder.sam_variant (see lens.models.backbones.sam.SAM_URLS).
    # ---- logging / Weights & Biases (training; default off reproduces the original) ----
    # When use_wandb is true, train.py inits one run on the main process and logs per
    # optimizer-step train losses + lr (and per-epoch eval gIoU/cIoU) to it. Keep
    # eval.use_wandb FALSE while training so eval_model does not start a second run.
    use_wandb: bool = False
    wandb_project: str = "lens"
    wandb_run_name: Optional[str] = None      # None -> wandb auto-generates a name
    wandb_entity: Optional[str] = None        # team/user; None -> your default entity
    wandb_mode: Optional[str] = None          # None -> online; "offline" | "disabled" also valid
    # Cadence (in optimizer steps) for the detailed console line; wandb logs EVERY step.
    log_every: int = 10
    # ---- logging / TensorBoard (training; pure-local, no network or account needed) ----
    # When use_tensorboard is true (default), train.py creates a SummaryWriter on the main
    # process and logs the SAME per-step train losses + lr (and per-epoch means) as wandb,
    # into a timestamped run dir under tb_log_dir. View the curves locally with:
    #     tensorboard --logdir <tb_log_dir>      (then open http://localhost:6006)
    # Writing these scalars is observation-only and does not affect training numerics.
    use_tensorboard: bool = True
    tb_log_dir: str = "./logs"


# ---------------------------------------------------------------------------
# Eval sub-config
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    val_dataset: str = "ReasonSeg"
    val_split: str = "val"
    dataset_dir: str = "/path/to/lisa_data/"
    clip_vision_model: str = "openai/clip-vit-large-patch14"
    image_size: int = 1024
    eval_only: bool = True
    val_batch_size: int = 1
    use_wandb: bool = False
    precision: str = "fp32"
    use_mm_start_end: bool = False
    exp_name: str = "eval_exp"
    log_dir: str = "./flows/logs"
    workers: int = 4
    save_viz: bool = True
    max_viz: int = 20


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _build(dc_type, data):
    """Recursively instantiate a (possibly nested) dataclass, overriding only the
    keys present in ``data`` and keeping dataclass defaults for the rest."""
    if data is None:
        return dc_type()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {dc_type.__name__}, got {type(data).__name__}")

    valid = {f.name: f for f in fields(dc_type)}
    unknown = set(data) - set(valid)
    if unknown:
        raise KeyError(
            f"Unknown config key(s) for {dc_type.__name__}: {sorted(unknown)}. "
            f"Allowed: {sorted(valid)}"
        )

    kwargs = {}
    for name, f in valid.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _build(f.type, value)
        elif isinstance(f.default, tuple) and isinstance(value, list):
            # YAML has no tuple type; honour the dataclass default's tuple-ness (e.g. betas).
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return dc_type(**kwargs)


def load_config(path: Optional[str] = None) -> Config:
    """Load a :class:`Config` from a YAML file. ``path=None`` returns pure defaults."""
    if path is None:
        return Config()

    import yaml  # local import so the package imports cleanly even without PyYAML

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _build(Config, raw)


__all__ = [
    "Config",
    "ModelConfig",
    "BackboneConfig",
    "GroundingHeadConfig",
    "KeypointConfig",
    "DecoderConfig",
    "DataConfig",
    "TrainConfig",
    "EvalConfig",
    "load_config",
]
