import os
import sys

# Make the third-party LISA tree importable (it provides the top-level `model_lisa`
# package used by lens.data) and keep the repo root on sys.path (for `lens` / `eval`).
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "LISA"))

from eval import eval_model, _attn_pad_frac

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import urllib.request
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
# SAM dependencies
from lens.models.decoder import Decoder, sigmoid_ce_loss
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from lens.utils.visualize import save_seg_visualization
from torch.utils.tensorboard import SummaryWriter


# ----------------------------------------------------------------------------
# 0. Ensure SAM checkpoint (only on main process)
# ----------------------------------------------------------------------------
from lens.models.backbones import download_sam_checkpoint


# ==================== NEW TRAINING FUNCTION WITH MULTI-T SAMPLING ====================
def train_epoch_multi_t(
        accelerator: Accelerator,
        loader: DataLoader,
        full,
        opt: torch.optim.Optimizer,
        scheduler=None,
        ep=0,
        save_viz=False,
        max_viz=20,
        cfg=None,
        global_step=0,
        writer=None,
):
    full.train()
    total_loss = 0.0
    # ---- logging setup (main process only) ----
    use_wandb = bool(cfg is not None and cfg.train.use_wandb and accelerator.is_main_process)
    log_every = max(1, cfg.train.log_every) if cfg is not None else 10
    if use_wandb:
        import wandb
    # Proper per-epoch means of the REAL losses (heatmap + seg), independent of the legacy
    # total_loss aggregate below (which folds in the always-(-1) loss_key placeholder).
    epoch_total = epoch_seg = epoch_heatmap = 0.0
    n_opt_steps = 0
    pbar = tqdm(loader, disable=not accelerator.is_main_process, desc="Training (Multi-T)")

    for i, batch in enumerate(pbar):
        imgs = batch['images']
        offset = batch['offset']
        images_list = []
        for j in range(len(offset) - 1):
            start_i, end_i = offset[j], offset[j + 1]
            images_i = (
                imgs[j]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_list.append(images_i)
        imgs = torch.cat(images_list, dim=0).to(torch.bfloat16)

        mask_raw = batch['masks_list']
        mask_raw_ = []
        ori_size = []
        resized_size = []
        raw_img = []
        mask_trans = []
        imgs_raw_path = []
        is_vqa = []
        for s, m in enumerate(mask_raw):
            for mm in m:
                mask_raw_.append((mm.int()).to(imgs.device))
                mask_tensor, _, _ = data_deal(mm.cpu().numpy(), 1024, is_mask=True)
                ori_size.append([mm.shape[0], mm.shape[1]])
                rl = [batch['resize_list'][s][1], batch['resize_list'][s][0]]
                resized_size.append(rl)
                img_r = Image.open(batch['image_paths'][s]).convert('RGB')
                img_r, _, _ = data_deal(img_r, 1024, is_mask=True)
                raw_img.append(img_r)
                imgs_raw_path.append(batch['image_paths'][s])
                mask_trans.append(((mask_tensor // 255).int()).to(imgs.device))
                is_vqa.append(False)
        mask_raw = mask_raw_
        prompts = batch['conversation_list']

        mask_trans = torch.stack(mask_trans)

        ss = i

        B = imgs.size(0)  # Original batch size

        # 1. Encode text prompts (once per batch)
        imgs_pil = [ToPILImage()(i) for i in raw_img]
        head_maps, keypoint_coords, pred_masks, loss_seg = full(imgs, imgs_pil, prompts, mask_raw, resized_size, ori_size, is_train=True)
        ## resized_size h w; ori_size w h

        # Save a few training-time visualizations (image + attention overlay + predicted
        # keypoints, pred mask vs GT) + instruction/attention sidecars. Main process only,
        # capped at max_viz batches/epoch. MUST run before keypoint_coords is rescaled below
        # (save_seg_visualization expects coords normalized to [0, 1]).
        if save_viz and i < max_viz and accelerator.is_main_process:
            with torch.no_grad():
                p0 = (pred_masks[0].sigmoid() > 0.6).int()[0]
                g0 = mask_raw[0].int()
                inter = ((p0 == 1) & (g0 == 1)).sum().item()
                uni = ((p0 == 1) | (g0 == 1)).sum().item()
                iou0 = inter / (uni + 1e-6)
                viz_image = np.array(Image.open(imgs_raw_path[0]).convert("RGB"))
                save_seg_visualization(
                    pred_mask=p0.cpu().numpy(),
                    gt_mask=(g0.cpu().numpy() == 1),
                    keypoint_coords=keypoint_coords[0:1].detach().clone(),
                    acc_iou=round(iou0, 4),
                    save_path=f"viz_train/epoch_{ep}/{i}.png",
                    attention_map=head_maps[0].detach(),
                    prompt=prompts[0],
                    image=viz_image,
                    meta={
                        "image_path": imgs_raw_path[0], "epoch": ep, "iter": i,
                        # Shapes should all be the original (H, W). If pred_shape is square
                        # (e.g. 1024x1024) while gt/image are not, postprocess_masks did not
                        # crop the SAM padding -> check resize_list / ori_size.
                        "pred_shape": tuple(p0.shape),
                        "gt_shape": tuple(g0.shape),
                        "image_shape": viz_image.shape,
                    },
                    attn_pad_frac=_attn_pad_frac(full, viz_image),
                )

        idd = -1
        keypoint_coords = keypoint_coords.reshape(B, -1, 2)
        labels_keypoints_list = []
        for idd in range(len(head_maps)):
            h, w = pred_masks[idd].shape[-2:]
            keypoint_coords[idd, :, 0] *= max(h, w)
            keypoint_coords[idd, :, 1] *= max(h, w)
            ind_keypoint = keypoint_coords[idd].to(torch.long)
            
            labels_keypoints = torch.zeros(ind_keypoint.shape[0], dtype=torch.long, device=ind_keypoint.device)

            # 2. Get the x and y coordinates
            x_coords = ind_keypoint[:, 0]
            y_coords = ind_keypoint[:, 1]

            # 3. Create a boolean mask to find keypoints that are within the valid (h, w) range
            valid_indices = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)

            # 4. If there are any valid keypoints, proceed to check their values in the mask
            if valid_indices.any():
                # Get the coordinates of only the valid keypoints
                valid_x = x_coords[valid_indices]
                valid_y = y_coords[valid_indices]

                # Check the value in the prediction mask at these valid locations.
                # pred_masks[idd] has shape [C, H, W], so we index with [0, y, x].
                mask_values = mask_raw[idd][valid_y, valid_x]

                # 5. Update the labels_keypoints tensor.
                # Set the label to 1 only for those valid keypoints where the mask value is also 1.
                labels_keypoints[valid_indices] = (mask_values == 1).long()
            labels_keypoints_list.append(labels_keypoints)

        sp = head_maps.shape[-2:]
        # gt_masks_trans = mask_trans
        head_maps_trans = head_maps
        gt_masks_trans = F.interpolate(
                        mask_trans.float(),
                        size=sp,
                        mode='nearest') > 0.5

        # # 使用归一化后的结果计算损失
        # squeeze(1) drops only the channel dim; plain .squeeze() also eats the batch dim
        # when B == 1 (single-conversation sample) -> 2D -> sigmoid_ce_loss.flatten(1,2) fails.
        loss_heatmap = sigmoid_ce_loss(head_maps_trans.squeeze(1), gt_masks_trans.squeeze(1).float(), num_masks=len(head_maps), need_logits=False)
        loss = loss_seg + loss_heatmap
        # ----------------------------------------------

        # 6. Accelerator handles backward pass and optimizer step
        with accelerator.accumulate(full):
            accelerator.backward(loss)

            # if accelerator.sync_gradients:
            #     # 可选的梯度裁剪
            #     params = itertools.chain(net.parameters(), clip_encoder.parameters())
            #     accelerator.clip_grad_norm_(params, max_norm=1.0)
            # parameters_to_clip = itertools.chain(net.parameters(), clip_encoder.parameters())
            # total_grad_norm = accelerator.clip_grad_norm_(parameters_to_clip, max_norm=10) 

            opt.step()
            opt.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()

            if accelerator.sync_gradients:
                try:
                    avg_loss1 = accelerator.gather_for_metrics(loss_heatmap.detach()).mean()
                except:
                    avg_loss1 = torch.tensor(-1.0, device=accelerator.device)
                try:
                    avg_loss2 = accelerator.gather_for_metrics(loss_key.detach()).mean()
                except:
                    avg_loss2 = torch.tensor(-1.0, device=accelerator.device)
                try:
                    avg_loss3 = accelerator.gather_for_metrics(loss_seg.detach()).mean()
                except:
                    avg_loss3 = torch.tensor(-1.0, device=accelerator.device)

                B = imgs.size(0)
                total_loss += (avg_loss1.item() + avg_loss2.item() + avg_loss3.item()) * B / 3

                # Real total = heatmap + seg (the loss actually back-propagated); loss_key is
                # the always-(-1) placeholder, kept out of the reported metrics.
                l_heatmap = avg_loss1.item()
                l_seg = avg_loss3.item()
                l_total = l_heatmap + l_seg
                global_step += 1
                n_opt_steps += 1
                epoch_total += l_total
                epoch_seg += l_seg
                epoch_heatmap += l_heatmap

                if accelerator.is_main_process:
                    cur_lr = (scheduler.get_last_lr()[0] if scheduler is not None
                              else opt.param_groups[0]["lr"])
                    pbar.set_postfix(loss=f"{l_total:.4f}", seg=f"{l_seg:.4f}",
                                     heatmap=f"{l_heatmap:.4f}", lr=f"{cur_lr:.2e}")
                    if use_wandb:
                        wandb.log({
                            "train/loss": l_total,
                            "train/loss_seg": l_seg,
                            "train/loss_heatmap": l_heatmap,
                            "train/lr": cur_lr,
                            "epoch": ep + 1,
                        }, step=global_step)
                    if writer is not None:
                        writer.add_scalar("train/loss", l_total, global_step)
                        writer.add_scalar("train/loss_seg", l_seg, global_step)
                        writer.add_scalar("train/loss_heatmap", l_heatmap, global_step)
                        writer.add_scalar("train/lr", cur_lr, global_step)
                    if (n_opt_steps % log_every) == 0:
                        tqdm.write(
                            f"[ep {ep + 1} | step {n_opt_steps} | gstep {global_step}] "
                            f"loss={l_total:.4f}  seg={l_seg:.4f}  heatmap={l_heatmap:.4f}  "
                            f"lr={cur_lr:.3e}"
                        )

    # ---- epoch summary (proper means of the real losses) ----
    if accelerator.is_main_process and n_opt_steps > 0:
        ep_loss = epoch_total / n_opt_steps
        ep_seg = epoch_seg / n_opt_steps
        ep_heat = epoch_heatmap / n_opt_steps
        tqdm.write(f"[ep {ep + 1}] train means over {n_opt_steps} steps: "
                   f"loss={ep_loss:.4f}  seg={ep_seg:.4f}  heatmap={ep_heat:.4f}")
        if use_wandb:
            wandb.log({
                "train/epoch_loss": ep_loss,
                "train/epoch_loss_seg": ep_seg,
                "train/epoch_loss_heatmap": ep_heat,
                "epoch": ep + 1,
            }, step=global_step)
        if writer is not None:
            writer.add_scalar("train/epoch_loss", ep_loss, global_step)
            writer.add_scalar("train/epoch_loss_seg", ep_seg, global_step)
            writer.add_scalar("train/epoch_loss_heatmap", ep_heat, global_step)
    return total_loss / len(loader.dataset), global_step


from accelerate.utils import DistributedDataParallelKwargs
from lens.models.conditioner import MLLM_Conditioner, FullModel
from lens.models import load_full_state_dict
from lens.config import load_config


from LISA.utils.dataset import HybridDataset, collate_fn
import transformers
from torch.utils.data import DataLoader
from lens.utils.inference import data_deal

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen2vl_2b.yaml",
                        help="Path to the experiment YAML config.")
    cli_args = parser.parse_args()
    cfg = load_config(cli_args.config)

    os.environ['HF_ENDPOINT'] = cfg.train.hf_endpoint

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Pass the ddp_kwargs object to the Accelerator
    # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision,
    )
    if accelerator.is_main_process:
        SAM_CHECKPOINT = download_sam_checkpoint(
            cfg.model.decoder.sam_variant, cfg.model.decoder.sam_checkpoint
        )
    accelerator.wait_for_everyone()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.data.tokenizer_name,
        cache_dir=None,
        model_max_length=cfg.data.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    sample_rates = cfg.data.sample_rates
    sem_seg_data = cfg.data.sem_seg_data
    refer_seg_data = cfg.data.refer_seg_data
    reason_seg_data = cfg.data.reason_seg_data
    vqa_data = cfg.data.vqa_data
    train_per_steps = cfg.train.train_per_steps

    BATCH_SIZE = cfg.train.batch_size  # 根据你的 GPU 显存调整
    NUM_WORKERS = cfg.train.num_workers # 使用多个进程来加载数据，加快速度
    samples_per_epoch = train_per_steps*8*BATCH_SIZE*accelerator.gradient_accumulation_steps
    train_dataset = HybridDataset(
            cfg.data.dataset_dir,
            tokenizer,
            cfg.data.clip_vision_model,
            samples_per_epoch=samples_per_epoch,
            image_size=cfg.data.image_size,
            num_classes_per_sample=cfg.data.num_classes_per_sample,
            exclude_val=cfg.data.exclude_val,
            dataset=cfg.data.dataset,
            sample_rate=[float(x) for x in sample_rates.split(",")],
            sem_seg_data=sem_seg_data,
            refer_seg_data=refer_seg_data,
            vqa_data=vqa_data,
            reason_seg_data=reason_seg_data,
            explanatory=cfg.data.explanatory,
        )


    # 创建 DataLoader 实例
    dl = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 在训练时打乱数据
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True, # 如果使用 GPU，可以加速数据转移
    )

    # Recommended "Balanced" model configuration
    net = Decoder(cfg.model.decoder).to(torch.bfloat16).to(accelerator.device)
    conditioner = MLLM_Conditioner(cfg.model).to(torch.bfloat16).to(accelerator.device)

    conditioner = conditioner.to(torch.bfloat16)

    num_epoch = cfg.train.num_epochs
    full = FullModel(net, conditioner, kp_thresh=cfg.model.kp_thresh)
    if cfg.train.resume_checkpoint:
        load_full_state_dict(full, cfg.train.resume_checkpoint, map_location='cpu')

    base_lr = cfg.train.base_lr
    opt = torch.optim.AdamW(
        full.parameters(),
        weight_decay=cfg.train.weight_decay, betas=tuple(cfg.train.betas), lr=base_lr
    )
    total_steps = train_per_steps * 10000000
    scheduler = CosineAnnealingLR(opt, T_max=total_steps, eta_min=cfg.train.eta_min)
    full.to(torch.bfloat16)
    full, opt, dl, scheduler = accelerator.prepare(
        full, opt, dl, scheduler
    )

    # ---- Weights & Biases: one run for the whole training (main process only) ----
    use_wandb = bool(cfg.train.use_wandb and accelerator.is_main_process)
    if use_wandb:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "train.use_wandb=true but wandb is not installed. Run `pip install wandb` "
                "(and `wandb login`), or set train.use_wandb: false."
            ) from e
        from dataclasses import asdict
        wandb.init(
            project=cfg.train.wandb_project,
            name=cfg.train.wandb_run_name,
            entity=cfg.train.wandb_entity,
            mode=cfg.train.wandb_mode,
            config=asdict(cfg),
        )
        print(f"[wandb] logging to project='{cfg.train.wandb_project}' run='{wandb.run.name}'")

    # ---- TensorBoard: pure-local loss curves (main process only; no network/account) ----
    writer = None
    if cfg.train.use_tensorboard and accelerator.is_main_process:
        from datetime import datetime
        run_dir = os.path.join(
            cfg.train.tb_log_dir, datetime.now().strftime("run_%Y%m%d_%H%M%S")
        )
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(run_dir)
        print(f"[tensorboard] logging to '{run_dir}'  ->  "
              f"view with: tensorboard --logdir {cfg.train.tb_log_dir}")

    global_step = 0
    for ep in range(num_epoch):
        if accelerator.is_main_process:
            print(f"--- Starting Epoch {ep + 1}/{num_epoch} ---")

        # accelerator.wait_for_everyone()
        # giou, ciou, nums, cr = eval_model(accelerator, full, ep, eval_cfg=cfg.eval)
        # if accelerator.is_main_process:
        #     print(f"Epoch {ep + 1} Evaluation Results:")
        #     print(f"nums: {nums}, gIoU: {giou:.4f}, cIoU: {ciou:.4f}")

        # Call the new training function with the multi-t sampling parameter
        avg_loss, global_step = train_epoch_multi_t(
            accelerator=accelerator,
            loader=dl,
            full=full,
            opt=opt,
            scheduler=scheduler,
            ep=ep,
            save_viz=cfg.train.save_viz,
            max_viz=cfg.train.max_viz,
            cfg=cfg,
            global_step=global_step,
            writer=writer,
        )

        # accelerator.wait_for_everyone()
        # giou, ciou, nums, cr = eval_model(accelerator, full, ep, eval_cfg=cfg.eval)
        # if accelerator.is_main_process:
        #     print(f"Epoch {ep + 1} Evaluation Results:")
        #     print(f"nums: {nums}, gIoU: {giou:.4f}, cIoU: {ciou:.4f}")
        #     if use_wandb:
        #         wandb.log({"eval/gIoU": giou, "eval/cIoU": ciou,
        #                    "eval/nums": nums, "epoch": ep + 1}, step=global_step)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print(f"Epoch {ep + 1}: avg loss={avg_loss:.4f}")
            # Save the model every 5 epochs. makedirs (not mkdir) so nested save_dir paths
            # like ./ssd1/save_dir_qwen are created with their parents; exist_ok is a no-op
            # when it already exists.
            os.makedirs(cfg.train.save_dir, exist_ok=True)
            if (ep + 1) % cfg.train.save_every_epochs == 0:
                unwrapped_full = accelerator.unwrap_model(full)
                # accelerator.wait_for_everyone()
                save_sd = unwrapped_full.state_dict()
                if cfg.train.save_trainable_only:
                    # Skip frozen tensors (MLLM backbone + the two SAM copies); they are
                    # rebuilt on load. Massively shrinks each checkpoint.
                    trainable = {n for n, p in unwrapped_full.named_parameters() if p.requires_grad}
                    save_sd = {k: v for k, v in save_sd.items() if k in trainable}
                torch.save(save_sd, f"{cfg.train.save_dir}/full_{ep + 1}.pth")
                print(f"Model saved at epoch {ep + 1} ({len(save_sd)} tensors)")

    if use_wandb:
        wandb.finish()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
