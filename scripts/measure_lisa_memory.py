"""Estimate a LISA/READ-style baseline's TRAINING memory + trainable params (single GPU).

It reproduces their trainable set on LLaVA-1.5-7B:
  * LoRA (r=8, alpha=16, target q_proj/v_proj on the LLM only),
  * full embed_tokens + lm_head (LISA marks these trainable for the [SEG] token),
  * the SAM mask decoder (the same decoder you use),
  * a text->SAM-prompt projection (text_hidden_fcs),
runs ONE training step = CE language loss + SAM mask BCE, backward, optimizer step, and
prints trainable params (M) and peak GPU memory (GB).

    pip install peft
    python scripts/measure_lisa_memory.py --batch 1                 # per-image footprint
    python scripts/measure_lisa_memory.py --batch 12 --grad-ckpt    # mimic READ's batch+ckpt

FAIR-COMPARISON NOTES
  * Measure LENS the SAME way (single GPU, bf16, same --batch, no ZeRO) so the Peak-Mem
    column is apples-to-apples. READ's paper number uses DeepSpeed ZeRO-2 (optimizer states
    sharded across GPUs) + grad-checkpointing + batch 12; this script does NOT shard the
    optimizer, so it reports the true *per-replica* memory (what one GPU must hold).
  * The win you should see: LISA/READ back-prop THROUGH the 7B (LoRA) and store gradients +
    optimizer states for the 262M embed/lm_head; LENS freezes the whole backbone, so its
    activation + optimizer memory is far smaller despite a larger head.
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "LISA"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

from lens.models.llava import LlavaForConditionalGeneration
from lens.models.backbones import build_sam


def n_params(params):
    return sum(p.numel() for p in params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf",
                    help="architecturally equivalent to LISA's LLaVA-1.5-7B for memory.")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-target", default="q_proj,v_proj")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (like READ)")
    ap.add_argument("--no-embed-head", action="store_true",
                    help="do NOT train embed_tokens/lm_head (test the 'LoRA-only' lower bound)")
    ap.add_argument("--sam-variant", default="vit_h")
    ap.add_argument("--sam-ckpt", default="./sam_vit_h_4b8939.pth")
    ap.add_argument("--image-size", type=int, default=1024, help="SAM input size")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need a GPU to measure training memory"
    device = "cuda"

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise SystemExit("peft not installed -> pip install peft")

    # ---- LLaVA-1.5-7B + LoRA (restricted to the language model, like LISA) ----
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    for p in model.parameters():
        p.requires_grad_(False)

    mods = "|".join(t.strip() for t in args.lora_target.split(","))
    lora_regex = rf".*language_model.*\.({mods})"   # excludes the CLIP vision tower
    model = get_peft_model(
        model,
        LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                   target_modules=lora_regex, bias="none", task_type="CAUSAL_LM"),
    )
    if not args.no_embed_head:
        for n, p in model.named_parameters():
            if "embed_tokens" in n or "lm_head" in n:
                p.requires_grad_(True)
    model.to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # ---- SAM: frozen encoder/prompt-encoder, trainable mask decoder + text->prompt proj ----
    sam = build_sam(args.sam_variant, args.sam_ckpt).to(device).to(torch.bfloat16)
    for p in sam.parameters():
        p.requires_grad_(False)
    for p in sam.mask_decoder.parameters():
        p.requires_grad_(True)
    text_proj = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(),
                              nn.Linear(4096, 256)).to(device).to(torch.bfloat16)

    # ---- trainable param report ----
    lora_p = n_params(p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    emb_p = n_params(p for n, p in model.named_parameters()
                     if p.requires_grad and ("embed_tokens" in n or "lm_head" in n))
    dec_p = n_params(p for p in sam.mask_decoder.parameters() if p.requires_grad)
    proj_p = n_params(text_proj.parameters())
    total_tr = lora_p + emb_p + dec_p + proj_p
    print("=" * 70)
    print(f"LISA/READ-style trainable params: {total_tr/1e6:.1f} M")
    print(f"  LoRA(r{args.lora_r}, {args.lora_target}) {lora_p/1e6:.2f}M | "
          f"embed+lm_head {emb_p/1e6:.1f}M | mask_decoder {dec_p/1e6:.2f}M | proj {proj_p/1e6:.2f}M")
    if lora_p == 0:
        print("WARNING: 0 LoRA params matched -- your peft is likely too old for regex "
              "target_modules. Upgrade peft (>=0.6), or set target_modules=['q_proj','v_proj'] "
              "in the script (it will then also LoRA the CLIP tower, a negligible ~0.8M).")

    trainable = [p for p in model.parameters() if p.requires_grad] \
        + [p for p in sam.mask_decoder.parameters() if p.requires_grad] \
        + list(text_proj.parameters())
    opt = torch.optim.AdamW(trainable, lr=3e-4)

    # ---- one training step ----
    torch.cuda.reset_peak_memory_stats()
    B = args.batch
    img = Image.fromarray(np.random.randint(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8))
    prompt = "USER: <image>\nWhat object should be segmented? ASSISTANT: The object.</s>"
    inputs = processor(text=[prompt] * B, images=[img] * B, return_tensors="pt", padding=True)
    inputs = {k: (v.to(device).to(torch.bfloat16) if v.is_floating_point() else v.to(device))
              for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    labels = inputs["input_ids"].clone()

    out = model(**inputs, labels=labels, output_hidden_states=True)
    ce_loss = out.loss
    prompt_emb = text_proj(out.hidden_states[-1][:, -1]).unsqueeze(1)  # [B,1,256]

    sam_img = torch.randn(B, 3, args.image_size, args.image_size, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        img_emb = sam.image_encoder(sam_img)                          # [B,256,64,64] frozen
    h, w = img_emb.shape[-2:]
    dense = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(B, -1, h, w)
    low_res, _ = sam.mask_decoder(
        image_embeddings=img_emb,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=prompt_emb.to(torch.bfloat16),
        dense_prompt_embeddings=dense.to(torch.bfloat16),
        multimask_output=False,
    )
    target = (torch.rand_like(low_res.float()) > 0.5).float()
    mask_loss = F.binary_cross_entropy_with_logits(low_res.float(), target)

    (ce_loss + mask_loss).backward()
    opt.step()
    opt.zero_grad()

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print("-" * 70)
    print(f"peak training memory: {peak:.1f} GB   "
          f"(batch={B}, bf16, grad_ckpt={args.grad_ckpt}, single-GPU, no ZeRO)")
    print("=" * 70)


if __name__ == "__main__":
    main()
