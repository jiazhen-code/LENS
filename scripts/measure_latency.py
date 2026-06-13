"""Measure per-image inference latency of LENS (MLLM conditioner + SAM decoder).

    python scripts/measure_latency.py --config configs/qwen2vl_2b.yaml \
        --checkpoint /path/full_N.pth --num 100 --warmup 10
    # (single-GPU is enough; latency is per-image so no need for multi-GPU)

Reports mean +/- std ms/image over the val set, batch=1, bf16, with CUDA synchronised
around each forward and the first --warmup images discarded (kernel autotune / allocator).
GT masks are NOT passed (mask_raw=None), so this is the pure inference path (no loss).

Fair-comparison note: LENS runs ONE MLLM forward (a single prefill — it reads attention,
it does NOT autoregressively generate a [SEG] token), then SAM encode+decode. Methods that
generate [SEG] pay prefill + K decode steps on the 7B, so LENS's MLLM cost is a strict
subset -> LENS latency should be <= theirs, not just "comparable".
"""

import argparse
import os
import sys
import time
from functools import partial

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "LISA"))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from transformers import CLIPImageProcessor

from lens.config import load_config
from lens.utils.inference import data_deal
from lens.models.decoder import Decoder
from lens.models.conditioner import MLLM_Conditioner, FullModel
from lens.models import load_full_state_dict
from lens.data.test_dataset import TestReasoningDataset, TestReferDataset
from lens.data.trainval_dataset import collate_fn_val


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_val_loader(ev):
    proc = CLIPImageProcessor.from_pretrained(ev.clip_vision_model)
    if ev.val_dataset in ("ReasonSeg",):
        ds = TestReasoningDataset(ev.dataset_dir, proc, ev.image_size, datasetname=ev.val_dataset,
                                  train_test_split=ev.val_split, eval_only=ev.eval_only,
                                  conversation_records={})
    else:
        ds = TestReferDataset(ev.dataset_dir, proc, ev.image_size, datasetname=ev.val_dataset,
                              train_test_split=ev.val_split, conversation_records={})
    return torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=partial(collate_fn_val, tokenizer=None, use_mm_start_end=ev.use_mm_start_end))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num", type=int, default=100, help="number of timed images")
    ap.add_argument("--warmup", type=int, default=10, help="leading images to discard")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = Decoder(cfg.model.decoder)
    cond = MLLM_Conditioner(cfg.model)
    full = FullModel(net, cond, kp_thresh=cfg.model.kp_thresh).to(torch.bfloat16).to(device)
    load_full_state_dict(full, args.checkpoint, map_location="cpu")
    full.eval()

    loader = build_val_loader(cfg.eval)

    lat = []
    for input_dict in loader:
        if len(lat) >= args.num + args.warmup:
            break
        # ---- input prep, mirroring eval.validate() (GT masks not needed for timing) ----
        imgs = input_dict["images"].to(device).bfloat16()
        offset = input_dict["offset"]
        imgs = torch.cat(
            [imgs[j].unsqueeze(0).expand(offset[j + 1] - offset[j], -1, -1, -1).contiguous()
             for j in range(len(offset) - 1)], dim=0)
        resized_size, ori_size, raw_img = [], [], []
        for s, m in enumerate(input_dict["masks_list"]):
            resize_list = input_dict["sam_mask_shape_list"][s][0]
            for mm in m:
                ori_size.append([mm.shape[0], mm.shape[1]])
                resized_size.append([resize_list[1], resize_list[0]])
                img_r, _, _ = data_deal(Image.open(input_dict["image_paths"][s]).convert("RGB"),
                                        imgs.shape[-1], is_mask=True)
                raw_img.append(img_r)
        prompts = input_dict["conversation_list"]
        imgs_pil = [ToPILImage()(x) for x in raw_img]

        _sync()
        t0 = time.perf_counter()
        full(imgs, imgs_pil, prompts, None, resized_size, ori_size)  # mask_raw=None => inference only
        _sync()
        lat.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(lat[args.warmup:])
    if lat.size:
        print(f"[{cfg.model.backbone.name}] latency: {lat.mean():.1f} +/- {lat.std():.1f} "
              f"ms/image  (n={lat.size}, warmup {args.warmup} dropped, bs=1, bf16)")
    else:
        print("not enough images timed; lower --warmup or check the val set")


if __name__ == "__main__":
    main()
