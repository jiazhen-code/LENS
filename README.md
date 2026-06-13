# LENS

Reasoning segmentation with **LENS**: an LLaVA-based multimodal conditioner reads the text
prompt and image, turns the language→image attention into a heatmap and a set of
**keypoints**, and feeds those keypoints as point prompts to a frozen SAM mask decoder to
produce the segmentation mask.

## Layout

```
.
├── train.py              # training entry point (main()) — trains LENS
├── eval.py               # evaluation entry point (eval_model / validate) + local visualization
├── configs/
│   ├── default.yaml      # all hyper-parameters (= the old hard-coded values, LLaVA default); copy & edit per experiment
│   ├── qwen2vl_2b.yaml   # Qwen2-VL-2B backbone experiment
│   ├── llava_onevision_7b.yaml # LLaVA-OneVision-7B backbone experiment
│   ├── sam_vit_l.yaml    # SAM ViT-L decoder experiment
│   └── sam_vit_b.yaml    # SAM ViT-B decoder experiment
├── lens/
│   ├── config.py             # dataclass config schema + load_config(yaml)
│   ├── models/
│   │   ├── __init__.py       # load_full_state_dict (shape-tolerant resume)
│   │   ├── conditioner.py    # MLLM_Conditioner, GroundingHead, KeypointExtractor, Point2Vec, FullModel
│   │   ├── decoder.py        # SAM-based mask Decoder + segmentation losses
│   │   ├── llava.py          # LLaVA-1.5 model def (exposes attention / hidden states)
│   │   └── backbones/        # pluggable backbone registry
│   │       ├── base.py            # MLLMBackbone interface + registry + encode helpers
│   │       ├── generic.py        # GenericHFVLMBackbone (chat-template encode, shared)
│   │       ├── llava_backbone.py  # LLaVA-1.5 backbone (frozen model + encode)
│   │       ├── qwen2vl_backbone.py# Qwen2-VL-2B backbone
│   │       ├── llava_onevision_backbone.py# LLaVA-OneVision backbone
│   │       └── sam.py            # SAM variant selector + checkpoint download (vit_h/l/b)
│   ├── data/                 # dataset loaders (sem / refer / reason seg, VQA, train/val/test)
│   └── utils/
│       ├── inference.py      # image preprocessing (data_deal) + single-image helpers
│       └── visualize.py      # save_seg_visualization — local PNG dumps
└── LISA/                 # third-party LISA code (SAM, LLaVA vendor, datasets) — UNMODIFIED
```

`LISA/` is third-party and is intentionally left untouched. `segment_anything` is expected to
be importable in the environment (it is installed on the training server). The entry points
add `LISA/` and the repo root to `sys.path` so the loaders' `model_lisa` imports resolve.

## Configuration

All hyper-parameters live in a YAML file ([configs/default.yaml](configs/default.yaml)). Every
default there equals the value that used to be hard-coded, so the default config reproduces the
original behaviour exactly. To run a new experiment, copy the file, change only the keys you
need, and pass it with `--config`:

```bash
cp configs/default.yaml configs/my_exp.yaml      # edit lr, data, backbone, …
accelerate launch train.py --config configs/my_exp.yaml
```

A YAML file only needs to list the keys it overrides; everything else falls back to the
dataclass defaults in [lens/config.py](lens/config.py). Unknown keys raise a clear error.

## Experiments: backbones and SAM sizes

Built-in MLLM backbones (registry keys, set as `model.backbone.name`):

| name           | model_name (HF)              | hidden | img tokens | fusion head (CR1a/b) |
|----------------|------------------------------|--------|-----------|----------------------|
| `llava-1.5-7b` | `llava-hf/llava-1.5-7b-hf`   | 4096   | 576       | Llama layer |
| `qwen2-vl-2b`  | `Qwen/Qwen2-VL-2B-Instruct`  | 1536   | 256       | Qwen2 layer|
| `llava-onevision-7b` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | 3584 | 729 | Qwen2 layer |


```bash
accelerate launch train.py --config configs/qwen2vl_2b.yaml
accelerate launch train.py --config configs/llava_onevision_7b.yaml
```

Per **CR1**, the keypoint pipeline, descriptor module, objectives (`L_attn + L_seg`),
keypoint→SAM-prompt interface, and protocol are **identical** across backbones. What adapts
to each backbone: **(a)** the 2-layer fusion head is re-instantiated to mirror *that
backbone's own* transformer layer (`backbone.build_fusion_model`) and is warm-started from
its middle layers (`backbone.warmstart_fusion_layers`); **(b)** `selected_layer_id` is
re-centred to the middle of the backbone (14 was LLaVA-specific); **(c)** the attention map
is reshaped to the backbone's native patch grid (`num_image_tokens` → `grid×grid`).

SAM segmentation backbone (LLaVA unchanged) via `model.decoder.sam_variant` (`vit_h/l/b`);
the checkpoint is auto-downloaded:

```bash
accelerate launch train.py --config configs/sam_vit_l.yaml
accelerate launch train.py --config configs/sam_vit_b.yaml
```

The SAM-size configs warm-start the trained conditioner + mask decoder from the resume
checkpoint and load the frozen SAM image encoder fresh (shape-mismatched encoder tensors
in the checkpoint are skipped by `load_full_state_dict`). Set `train.resume_checkpoint:
null` to train a variant from scratch.

> The two new MLLM backbones force a fixed square image size so the image-token grid is a
> known `grid × grid` (= `num_image_tokens`). `encode()` asserts the produced token count
> matches and raises a clear error (telling you to adjust `image_hw` / `num_image_tokens`)
> if your transformers/processor version emits a different count — so validate each new
> backbone with a quick 1-step run on the GPU box before a full launch.

## Adding another backbone

1. Subclass `GenericHFVLMBackbone` ([generic.py](lens/models/backbones/generic.py)) — usually
   you only override `_load` and set `image_token_str` (see
   [qwen2vl_backbone.py](lens/models/backbones/qwen2vl_backbone.py)) — or subclass
   `MLLMBackbone` directly for full control of `encode()` (see
   [llava_backbone.py](lens/models/backbones/llava_backbone.py)). Decorate with
   `@register_backbone("my-backbone")`. The base class already builds + warm-starts a
   fusion head that mirrors your backbone's layer; override `_text_model()` /
   `build_fusion_model()` only if your transformers layout needs it (Qwen2-VL is the
   example — it builds a plain `Qwen2Model` to avoid M-RoPE).
2. Import it in [lens/models/backbones/__init__.py](lens/models/backbones/__init__.py) so the
   registration runs.
3. Point a config at it (centre the two layer indices on the backbone's middle):

   ```yaml
   model:
     backbone:
       name: my-backbone
       model_name: org/my-vlm
       hidden_size: 3584
       num_image_tokens: 256
       selected_layer_id: 14              # ~ num_layers // 2
       grounding_init_start_layer: 14     # ~ num_layers // 2 (null = no warm-start)
       image_hw: 448
   ```

No edits to `conditioner.py`, `decoder.py`, or the training loop are needed.

To choose `selected_layer_id` / `grounding_init_start_layer` for a new backbone (instead of
guessing "the middle"), run the layer probe — it ranks decoder layers by how well their
last-text-token→image attention localizes the GT mask (pointing-game), no training. Two
steps (export a probe set from your val split, then probe each backbone):

```bash
# 1) export {image, question, mask} from the eval val split (renders masks)
python scripts/make_probe_set.py --config configs/default.yaml --out data/layer_probe.jsonl --limit 200
# 2) rank layers for each backbone (image/mask paths are absolute, no --image-root needed)
python scripts/probe_layers.py --config configs/default.yaml      --probe-set data/layer_probe.jsonl   # sanity: LLaVA should peak ~14
python scripts/probe_layers.py --config configs/qwen2vl_2b.yaml   --probe-set data/layer_probe.jsonl
python scripts/probe_layers.py --config configs/llava_onevision_7b.yaml --probe-set data/layer_probe.jsonl
```

Then confirm the top-1/2 layers with a short training run (compare val gIoU).

## Train

```bash
accelerate launch train.py --config configs/default.yaml
```

`main()` builds the model from the config (`FullModel(Decoder(cfg.model.decoder),
MLLM_Conditioner(cfg.model))`), trains on the LISA hybrid dataset, and **after every epoch**
runs evaluation (`eval_model`) and writes visualizations locally — so testing and
visualization happen during training.

## Evaluate (standalone)

```bash
accelerate launch eval.py --config configs/default.yaml --checkpoint /path/to/full_N.pth
```

## Routing benchmark (Table_R 2)

Zero-shot evaluation of a **frozen** MLLM as a router that classifies each (image, query)
into a difficulty tier — `non-seg` / `explicit` / `reasoning` / `mixed` — where
`trigger = tier != non-seg`. Reports per-tier accuracy, overall accuracy, and false-/
missed-trigger rates. Uses the HF backbone weights directly (no training) and runs
`accelerate` multi-GPU (each process evaluates a shard; results are gathered on rank 0).

Benchmark format — JSONL (or a JSON array), one sample per line. The released benchmark
([routing_benchmark/benchmark.jsonl](routing_benchmark/benchmark.jsonl)) looks like:

```json
{"id": "exp_00234", "tier": "explicit_seg", "query": "Segment the man in the image.", "image": "refer_seg/images/mscoco/images/train2014/COCO_train2014_000000149996.jpg"}
```

Field names are read flexibly: question from `query`/`question`/`text`/…, image from
`image_path`(absolute)/`image`/…, tier from `tier`/`label`/…. Tier labels are normalized
onto `non-seg`/`explicit`/`reasoning`/`mixed` (e.g. `dialogue`→non-seg, `explicit_seg`→
explicit, `mixed_intent`→mixed); use `--tier-map '{"my_label":"mixed"}'` for anything that
doesn't auto-resolve.

One run = one table row (the backbone is whatever `model.backbone` the config selects):

```bash
accelerate launch eval_routing.py --config configs/default.yaml      --benchmark routing_benchmark/benchmark.jsonl --image-root /path/to/lisa_data   # LLaVA-1.5-7B
accelerate launch eval_routing.py --config configs/qwen2vl_2b.yaml   --benchmark routing_benchmark/benchmark.jsonl --image-root /path/to/lisa_data   # Qwen2-VL-2B
accelerate launch eval_routing.py --config configs/llava_onevision_7b.yaml --benchmark routing_benchmark/benchmark.jsonl --image-root /path/to/lisa_data   # LLaVA-OneVision-7B
```

`--output results.json` dumps per-sample predictions + metrics; `--limit N` runs a quick
smoke; `--image-hw N` forces a square input (default: the MLLM's native preprocessing). The
routing prompt lives in `lens/routing/router.py` (`DEFAULT_ROUTING_PROMPT`) — tune it there.
The harness lives under [lens/routing/](lens/routing/) (`MLLMRouter`, `parse_tier`,
`compute_routing_metrics`, `load_benchmark`).

## Visualizations

During evaluation (standalone or training-time) PNGs are written to:

```
viz/step_<epoch-or-step>/rank_<rank>/<index>.png
viz/step_<epoch-or-step>/rank_<rank>/<index>.txt   # instruction + acc_iou + image path
viz/step_<epoch-or-step>/rank_<rank>/<index>.npy   # raw attention heatmap (head_maps)
```

The PNG is a row of panels: raw image, attention heatmap (the instruction-token
attention the keypoints are read off) overlaid on the image, predicted mask, and
ground-truth mask — with the predicted keypoints overlaid (red) on all but the GT
panel and the instruction shown as the title. The full instruction (titles truncate)
and the raw attention map are also dumped as the `.txt` / `.npy` sidecars.
Controlled by `MyArgs.save_viz` (default on) and `MyArgs.max_viz` (default 20 per run).

The same panels are written **during training** to:

```
viz_train/epoch_<ep>/<index>.png   (+ .txt / .npy sidecars)
```

on the main process only, for the first `max_viz` batches of each epoch. Controlled by
`TrainConfig.save_viz` / `TrainConfig.max_viz`. (Over many epochs this can add up — lower
`max_viz` if disk is tight.)

