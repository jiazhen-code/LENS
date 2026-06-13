"""Single-process smoke test for a backbone config (no training, no accelerate).

    python scripts/smoke_backbone.py --config configs/qwen2vl_2b.yaml

Validates, in order, the things that tend to break when porting LENS to a new backbone:
  1. backbone + processor load;
  2. the fusion head is the backbone's own layer class, with the right #layers, and the
     warm-start copy succeeded (printed during build);
  3. encode() produces exactly num_image_tokens image tokens (the square-grid assertion);
  4. one forward through encode + the grounding head (fusion attention -> keypoints).

Run it from the repo root (Point2Vec loads ./sam_vit_*.pth from the cwd).
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "LISA"))

import torch
from PIL import Image

from lens.config import load_config
from lens.models.conditioner import MLLM_Conditioner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="experiment YAML (e.g. configs/qwen2vl_2b.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"== building MLLM_Conditioner from {args.config} on {device} ==")
    cond = MLLM_Conditioner(cfg.model).to(torch.bfloat16).to(device)

    fm = cond.conditioning_module.fusion_model
    print("backbone class      :", type(cond.backbone).__name__)
    print("fusion model class  :", type(fm).__name__, "(should mirror the backbone's text model)")
    print("fusion #layers      :", len(fm.layers), "(expected", cfg.model.grounding_head.num_attention_layers, ")")
    print("explicit_position_ids:", cond.conditioning_module.explicit_position_ids)
    print("hidden / img_tokens :", cfg.model.backbone.hidden_size, "/", cfg.model.backbone.num_image_tokens)

    # One dummy sample in the dataset's llava_v1 prompt format.
    img = Image.new("RGB", (640, 480), (123, 116, 103))
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "USER: <image>\nWhat is the object in this image? ASSISTANT:"
    )

    with torch.no_grad():
        coord, indicator, img_f, keypoints, loss, ind_train, pad = cond([img], [prompt])

    # Confirm which fusion position-id path is live (Qwen should be real 3-D M-RoPE).
    fpi = cond.backbone.fusion_position_ids()
    if fpi is None:
        print("fusion_position_ids : None  -> 1-D/sequential (M-RoPE NOT active; expected "
              "for LLaVA-1.5 / LLaVA-OneVision, but a WARNING for Qwen = get_rope_index fell back)")
    else:
        ok = (tuple(fpi.shape)[0] == 3)
        print(f"fusion_position_ids : {tuple(fpi.shape)} {fpi.dtype} "
              f"-> {'real 3-D M-RoPE ACTIVE' if ok else 'unexpected shape'}")

    print("heatmap shape       :", tuple(coord.shape))
    print("keypoints shape     :", tuple(keypoints.shape))
    print("indicator shape     :", tuple(indicator.shape))
    print("== SMOKE OK ==")


if __name__ == "__main__":
    main()
