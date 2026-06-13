import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Optional
import types # 导入 types 模块
from ..config import DecoderConfig
from .backbones import build_sam

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
    need_logits=False
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if need_logits:
        inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)

    # l = torch.nn.functional.softplus(inputs.sum() - targets.sum()) / (inputs.sum() + targets.sum() + 1e-8)

    return loss

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    need_logits=False,
    alpha: float = 0.25,  # α参数，控制正负样本的权重
    gamma: float = 2.0,   # γ参数，控制难易样本的衰减
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The raw predictions (logits) for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: Number of masks, used for normalizing the loss.
        alpha: Weighting factor for balancing positive and negative samples.
        gamma: Focusing parameter to decrease the relative loss for well-classified examples.

    Returns:
        Loss tensor
    """
    if need_logits:
        prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)  
    focal_factor = (1 - p_t) ** gamma 
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_factor * ce_loss
    loss = loss.flatten(1, 2) 
    loss = loss.mean(1).sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    need_logits=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    if need_logits:
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    else:
        loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def get_batch_size(
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text_embeds: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif text_embeds is not None:
            return text_embeds.shape[0]
        else:
            return 1

def new_prompt_encoder_forward(
    self, # This 'self' is a placeholder for the prompt_encoder instance it will be bound to.
    points: Optional[Tuple[torch.Tensor, torch.Tensor]],
    boxes: Optional[torch.Tensor],
    masks: Optional[torch.Tensor],
    text_embeds: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This is the new forward function implementation. 'self' will correctly refer to
    the prompt_encoder instance once it's bound.
    """
    # print("--- NEW custom forward (standalone function) called ---")

    bs = get_batch_size(points, boxes, masks, text_embeds)
    sparse_embeddings = torch.empty(
        (bs, 0, self.embed_dim), device=self._get_device()
    )
    if points is not None:
        coords, labels = points
        point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
        sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
    if boxes is not None:
        box_embeddings = self._embed_boxes(boxes)
        sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

    if text_embeds is not None:
        sparse_embeddings = torch.cat([sparse_embeddings, text_embeds], dim=1)

    if masks is not None:
        dense_embeddings = F.interpolate(
            masks,
            size=(64, 64),
            mode='bilinear',
            align_corners=True
        )
        # dense_embeddings = self._embed_masks(masks)
    else:
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        )
    return sparse_embeddings, dense_embeddings

def dtype_safe_pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
    """SAM ``PositionEmbeddingRandom._pe_encoding`` made dtype-safe.

    ``PositionEmbeddingRandom.forward`` builds its coordinate grid as float32, but after the
    model is cast to bf16 the ``positional_encoding_gaussian_matrix`` buffer is bf16 -> the
    original ``coords @ matrix`` raises "mat1 and mat2 have the same dtype". We cast the
    (frozen, random) matrix up to the coords' dtype so the matmul + sin/cos run in float32,
    exactly matching the original numerics; the caller (`get_dense_pe`) casts the result to
    bf16 afterwards. Math is otherwise identical to SAM's implementation.
    """
    coords = 2 * coords - 1
    matrix = self.positional_encoding_gaussian_matrix
    if coords.dtype != matrix.dtype:
        matrix = matrix.to(coords.dtype)
    coords = coords @ matrix
    coords = 2 * np.pi * coords
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)


class Decoder(nn.Module):

    def __init__(self, cfg: Optional[DecoderConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = DecoderConfig()
        self.cfg = cfg
        self.bce_weight = cfg.bce_weight
        self.dice_weight = cfg.dice_weight

        self.visual_model = build_sam(cfg.sam_variant, cfg.sam_checkpoint)
        for param in self.visual_model.parameters():
            param.requires_grad = False
    
        self.visual_model.mask_decoder.train()
        # # 1. 先把整个 mask_decoder 设置为需要梯度
        self.visual_model.mask_decoder.requires_grad_(True)
        # # 2. 然后把不需要梯度的部分单独关闭
        self.visual_model.mask_decoder.iou_prediction_head.requires_grad_(False)

        self.visual_model.prompt_encoder.forward = types.MethodType(
            new_prompt_encoder_forward,
            self.visual_model.prompt_encoder
        )
        # Make the image positional encoding (get_dense_pe -> _pe_encoding) dtype-safe so it
        # works after the model is cast to bf16 (float32 grid @ bf16 matrix would crash).
        self.visual_model.prompt_encoder.pe_layer._pe_encoding = types.MethodType(
            dtype_safe_pe_encoding, self.visual_model.prompt_encoder.pe_layer
        )
        self.not_a_point_embed = self.visual_model.prompt_encoder.not_a_point_embed

        # Optional: project the (B,1,grid,grid) attention heatmap to SAM's 256-d dense
        # embedding space; the prompt encoder then resizes it to the 64x64 grid. Only built
        # when enabled, so default checkpoints keep their exact keys.
        self.use_attn_dense_prompt = cfg.use_attn_dense_prompt
        if self.use_attn_dense_prompt:
            self.heatmap_proj = nn.Sequential(
                nn.Conv2d(1, 256, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(256, 256, kernel_size=1),
            )
            # Zero-init the last conv so the dense prompt starts at 0 (no perturbation) and
            # the heatmap signal is learned in gradually.
            nn.init.zeros_(self.heatmap_proj[-1].weight)
            nn.init.zeros_(self.heatmap_proj[-1].bias)


    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                # torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0).bfloat16()
                )
                image_embeddings_list.append(image_embeddings)
            # torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def decode_mask(self, x, sparse_embeddings_list, dense_embeddings_list, resize_list, original_size_list, padding_mask):
        """Decode the mask from the latent space."""
        pred_masks = []
        # Decode masks using the visual model's mask decoder
        for i in range(x.shape[0]):
            sparse_embeddings = sparse_embeddings_list[i].to(x.device)
            dense_embeddings = dense_embeddings_list[i].to(x.device)
            padding_mask_ = padding_mask[i]
            # padding_mask_ = torch.cat([padding_mask_, torch.tensor([0], device=padding_mask_.device, dtype=padding_mask_.dtype)])

            # sparse_embeddings[padding_mask_] = self.not_a_point_embed.weight
            sparse_embeddings = sparse_embeddings[~padding_mask_]
            device = x.device
            low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                    image_embeddings=x[i].unsqueeze(0).bfloat16(),
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe().bfloat16(),
                    sparse_prompt_embeddings=sparse_embeddings.unsqueeze(0).bfloat16(),
                    dense_prompt_embeddings=dense_embeddings.bfloat16(),
                    multimask_output=False,
                )
            resize_in = (resize_list[i][1], resize_list[i][0])
            original_in = (original_size_list[i][0], original_size_list[i][1])
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_in,
                original_size=original_in
            )
            pred_masks.append(pred_mask[:, 0].to(x.dtype))
        
        return pred_masks

    def forward(self, pixel_values, text_embeds, attn_maps, ground_mask=None, resize_list=None, original_size_list=None, is_vqa=None, padding_mask=None) -> torch.Tensor:
        image_sam_features = self.get_visual_embs(pixel_values)
        # Optionally turn the attention heatmap into the dense prompt (else SAM's no-mask
        # embedding is used, the original behaviour).
        if self.use_attn_dense_prompt and attn_maps is not None:
            proj_dtype = next(self.heatmap_proj.parameters()).dtype
            dense_mask = self.heatmap_proj(attn_maps.to(proj_dtype))
        else:
            dense_mask = None
        sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=dense_mask,
            text_embeds=text_embeds,
        )

        pred_masks = self.decode_mask(
            image_sam_features, sparse_embeddings, dense_embeddings, resize_list, original_size_list, padding_mask
        )
        if ground_mask is None:
            return pred_masks, None
        else:
            mask_bce_loss = mask_dice_loss = 0
            mask_raw = ground_mask
            nums = 0
            for i in range(len(pred_masks)):
                if is_vqa is not None:
                    is_valid_ = is_vqa[i] == 0
                else:
                    is_valid_ = 1
                mask_raw_ = mask_raw[i]
                mask_raw_ = F.interpolate(
                    mask_raw_.float().unsqueeze(0).unsqueeze(0),
                    size=pred_masks[i].shape[-2:],
                    mode="bilinear", align_corners=False
                )[:, 0]

                mask_bce_loss += (
                    sigmoid_ce_loss(pred_masks[i], mask_raw_, num_masks=1, need_logits=True)
                ) * is_valid_
                mask_dice_loss += (
                    dice_loss(pred_masks[i], mask_raw_, num_masks=1, need_logits=True)
                ) * is_valid_
                nums += is_valid_
            pred_masks = [p.detach() for p in pred_masks]
            return pred_masks, (self.bce_weight*mask_bce_loss + self.dice_weight*mask_dice_loss) / (nums + 1e-6)


