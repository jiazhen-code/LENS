import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Union, List, Tuple
import requests
from io import BytesIO
import torch.nn.functional as F
# 确保必要的库已安装
from transformers import AutoProcessor, LlamaConfig
# LlamaDecoderLayer 是 Llama 1, 2, 和 3 的通用核心模块
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig
from torch.nn import CrossEntropyLoss
from typing import Tuple, Optional
import types
from ..config import ModelConfig, KeypointConfig
from .backbones import build_mllm_backbone, build_sam, force_eager_attention

class FullModel(nn.Module):
    def __init__(self, net, conditioner, kp_thresh: float = 0.4):
        super().__init__()
        self.net = net
        self.conditioner = conditioner
        self.kp_thresh = kp_thresh

    def forward(self, imgs, imgs_pil, prompts, mask_raw=None, resized_size=None, ori_size=None, is_train=False):

        head_maps, text_token_features, image_token_features, keypoint_coords, loss_auto, indicator_for_train, padding_mask = self.conditioner(imgs_pil, prompts, is_train, self.kp_thresh)
        pred_masks, loss_seg = self.net(imgs, text_token_features, head_maps, mask_raw, resized_size, ori_size, None, padding_mask)

        return head_maps, keypoint_coords, pred_masks, loss_seg


class Point2Vec(nn.Module):

    def __init__(self, sam_variant: str = "vit_h", sam_checkpoint: str = "./sam_vit_h_4b8939.pth"):
        super().__init__()
        visual_model = build_sam(sam_variant, sam_checkpoint)
        for param in visual_model.parameters():
            param.requires_grad = False
    
        # self.visual_model.mask_decoder.train()
        # # 1. 先把整个 mask_decoder 设置为需要梯度
        # self.visual_model.mask_decoder.requires_grad_(True)
        # # 2. 然后把不需要梯度的部分单独关闭
        # self.visual_model.mask_decoder.iou_prediction_head.requires_grad_(False)

        self.pe_layer = visual_model.prompt_encoder.pe_layer
        self.pos_emb = visual_model.prompt_encoder.point_embeddings
        self.not_a_point_embed = visual_model.prompt_encoder.not_a_point_embed


        self.pe_layer.forward_with_coords = types.MethodType(
            new_forward_with_coords, self.pe_layer
        )

        self.pe_layer._pe_encoding = types.MethodType(
            new_pe_encoding, self.pe_layer
        )

    def get_point_emb(self, sampled_coords, lb_in=None, pad=False):

        point_embedding = self.pe_layer.forward_with_coords(
            sampled_coords, [1, 1]
        ) 
        if lb_in is None:
            emb = point_embedding
        else:
            point_embedding[lb_in==1] = point_embedding[lb_in==1] + self.pos_emb[1].weight
            point_embedding[lb_in==0] = point_embedding[lb_in==0] + self.pos_emb[0].weight

            emb = point_embedding

        if not pad:
            return emb

        # padding_point = torch.zeros((sampled_coords.shape[0], 1, 2), device=point_embedding.device)
        # padding_point_emb = self.pe_layer.forward_with_coords(
        #     padding_point, [1, 1]
        # )
        # padding_point_emb[:] = 0
        # padding_point_emb = padding_point_emb + self.not_a_point_embed.weight

        
        # emb = torch.cat((emb, padding_point_emb), 1)

        return emb

def new_pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
    """Positionally encode points that are normalized to [0,1]."""
    # Use a new variable to prevent ambiguity
    processed_coords = 2 * coords - 1

    if processed_coords.dtype != self.positional_encoding_gaussian_matrix.dtype:
        processed_coords = processed_coords.to(self.positional_encoding_gaussian_matrix.dtype)

    # Use a new variable for the matrix multiplication output
    projected_coords = processed_coords @ self.positional_encoding_gaussian_matrix.clone()

    # Use another new variable for the scaled output
    scaled_coords = 2 * np.pi * projected_coords

    return torch.cat([torch.sin(scaled_coords), torch.cos(scaled_coords)], dim=-1)

def new_forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Positionally encode points that are not normalized to [0,1].
    (Corrected version to be 'autograd-safe').
    """
    x_coords = coords_input[:, :, 0]
    y_coords = coords_input[:, :, 1]
    x_coords_normalized = x_coords / image_size[1]
    y_coords_normalized = y_coords / image_size[0]
    normalized_coords = torch.stack(
        [x_coords_normalized, y_coords_normalized], dim=2
    )
    # Note: it calls self._pe_encoding, which must exist on the instance
    return self._pe_encoding(normalized_coords.float()).bfloat16()

class KeypointExtractor(nn.Module):
    def __init__(self, num_keypoints=4, patch_size=4,
                 grid_size=16, num_image_tokens=256, suppression_radius=4, thre_ratio=0.9,
                 sam_variant="vit_h", sam_checkpoint="./sam_vit_h_4b8939.pth"):
        super().__init__()
        # self.N = n_query_tokens
        self.K = num_keypoints
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_image_tokens = num_image_tokens
        self.poe_emb_convertor = Point2Vec(sam_variant, sam_checkpoint)
        self.suppression_radius = suppression_radius
        self.thre_ratio = thre_ratio


    def _find_keypoints_nms(self, heatmaps, K, suppression_radius):
        """
        在所有N个类别的热图中执行跨类别的非最大值抑制(NMS)，
        为每个批次项贪婪地寻找K个空间上分离的最佳关键点。

        Args:
        - heatmaps (torch.Tensor): 输入热图 (B, N, H, W)。
        - K (int): 要寻找的关键点总数。
        - suppression_radius (float): NMS的抑制半径。

        Returns:
        - final_coords (torch.Tensor): 找到的关键点整数坐标 (B, K, 2)。
        - final_scores (torch.Tensor): 找到的关键点分数 (B, K)。
        - final_class_indices (torch.Tensor): 每个关键点对应的类别索引 (B, K)。
        - num_found (torch.Tensor): 每个样本实际找到的有效关键点数量 (B,)。
        """
        B, N, H, W = heatmaps.shape
        device = heatmaps.device

        # 用于抑制操作的副本
        heatmaps_suppressed = heatmaps.clone()
        
        # 用于存储最终结果的张量
        final_coords = torch.zeros(B, K, 2, dtype=torch.long, device=device)
        final_scores = torch.zeros(B, K, dtype=heatmaps.dtype, device=device)
        final_class_indices = torch.zeros(B, K, dtype=torch.long, device=device)
        
        # 创建坐标网格用于高效计算抑制区域
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        for k in range(K):
            # 1. 在当前所有热图中找到全局最大值及其扁平化索引
            #    将 (N, H, W) 维度展平为一维
            flat_heatmaps = heatmaps_suppressed.view(B, -1)
            scores, flat_indices = torch.max(flat_heatmaps, dim=1)

            # 如果所有剩余点的最大分数都为0，可以提前退出
            if torch.all(scores == 0):
                break

            # 2. 将扁平化索引转换回 (n, y, x) 坐标
            n_indices = flat_indices // (H * W)
            y_coords = (flat_indices % (H * W)) // W
            x_coords = (flat_indices % (H * W)) % W
            
            # 3. 存储找到的坐标、分数和类别索引
            final_coords[:, k, 0] = x_coords
            final_coords[:, k, 1] = y_coords
            final_scores[:, k] = scores
            final_class_indices[:, k] = n_indices

            # 4. 抑制该点邻域，为下一次迭代做准备
            #    计算每个像素到新找到的关键点的距离的平方
            dist_sq = (xx - x_coords.view(B, 1, 1))**2 + (yy - y_coords.view(B, 1, 1))**2
            
            #    创建抑制掩码 (B, H, W)
            suppression_mask = dist_sq <= suppression_radius**2

            #    应用掩码: 将掩码扩展到N个通道，并将这些区域的值设为0
            #    suppression_mask.unsqueeze(1) -> (B, 1, H, W)
            heatmaps_suppressed[suppression_mask.unsqueeze(1).expand(-1, N, -1, -1)] = 0.0

        # 计算每个样本找到的有效关键点数量（分数大于0）
        num_found = (final_scores > 0).sum(dim=1)

        return final_coords, final_scores, final_class_indices, num_found

    def forward(self, final_attn_weights, text_token_mask, image_token_mask, thresh=None):
        device = final_attn_weights.device
        
        n_query_indices = self._select_query_tokens(text_token_mask, device)
        n_heatmaps = self._get_heatmaps(final_attn_weights, n_query_indices, text_token_mask, image_token_mask)
        # 调用更新后的函数
        peak_values = n_heatmaps.amax(dim=(-2, -1))
        # 2. Calculate the dynamic threshold relative to the peak
        use_zero_padding = True
        if thresh is None:
            thresh = self.thre_ratio
            use_zero_padding = True
        dynamic_thresholds = peak_values * thresh

        keypoint_coords, local_patches, _, padding_mask = self._find_keypoints_and_extract_patches_vectorized(
            n_heatmaps, dynamic_thresholds, use_zero_padding
        )
        # local_patches = local_patches / (local_patches + 1e-6)
        with torch.no_grad():
            # B, N, S, _ = keypoint_coords.shape
            pos_embs = self.poe_emb_convertor.get_point_emb(keypoint_coords)
        return keypoint_coords, pos_embs, local_patches, n_heatmaps, padding_mask

    def _select_query_tokens(self, text_token_mask, device):
        # (保留了您对 start 的修改)
        batch_size = text_token_mask.shape[0]
        n_query_indices = []
        for i in range(batch_size):
            text_indices = torch.where(text_token_mask[i])[0]
            # if len(text_indices) == 0:
            #     n_query_indices.append(torch.zeros(self.N, dtype=torch.long, device=device))
            #     continue
            # num_text_tokens = len(text_indices)
            # linspace_indices = torch.linspace(
            #     start=num_text_tokens - 10, end=num_text_tokens - 1, steps=self.N, device=device
            # ).long()
            selected_indices = text_indices[[-1]]
            n_query_indices.append(selected_indices)
        return torch.stack(n_query_indices)

    def _get_heatmaps(self, final_attn_weights, n_query_indices, text_token_mask, image_token_mask):
        # (此函数保持不变)
        batch_size, seq_len, _ = final_attn_weights.shape
        device = final_attn_weights.device
        idx = n_query_indices.unsqueeze(-1).expand(-1, -1, seq_len)
        n_attention_vectors = torch.gather(final_attn_weights, 1, idx)  ###########
        all_normalized_heatmaps = []
        for i in range(batch_size):
            sample_attn = n_attention_vectors[i][:, image_token_mask[i]]
            if sample_attn.shape[1] != self.num_image_tokens:
                delta = self.num_image_tokens - sample_attn.shape[1]
                sample_attn = F.pad(sample_attn, (0, delta))
            # sample_attn = F.softmax(sample_attn, dim=-1)
            heatmaps = sample_attn.reshape(1, self.grid_size, self.grid_size)
            # Per-sample peak normalization with a DETACHED scale: rescale so the map's peak
            # is 1 (background ~0) WITHOUT moving the peaks. This makes the downstream
            # BCE-vs-mask loss reachable (a softmax distribution summing to <=1 can never hit
            # 1 on a whole region), while keypoint NMS and the peak*ratio threshold stay
            # scale-invariant -> keypoint selection is unchanged. Clamp to (0,1) so
            # F.binary_cross_entropy is numerically safe (its grad blows up at exactly 0/1).
            peak = heatmaps.amax(dim=(-2, -1), keepdim=True).detach().clamp_min(1e-8)
            heatmaps = (heatmaps / peak).clamp(1e-6, 1.0 - 1e-6)
            all_normalized_heatmaps.append(heatmaps)
        return torch.stack(all_normalized_heatmaps)

    def _refine_coords_subpixel(self, heatmaps, coords_int):
        B, K, H, W = heatmaps.shape
        device = heatmaps.device
        
        x_int = coords_int[..., 0]
        y_int = coords_int[..., 1]

        # --- 1. 批量采样9个点 (中心点及其8邻域) ---
        # 使用 F.grid_sample 高效地一次性获取所有需要的值
        norm_x = 2 * (x_int.float() / (W - 1)) - 1
        norm_y = 2 * (y_int.float() / (H - 1)) - 1
        dx = 2.0 / (W - 1)
        dy = 2.0 / (H - 1)
        
        grid = torch.stack([
            torch.stack([norm_x, norm_y], -1),           # center (0)
            torch.stack([norm_x + dx, norm_y], -1),      # x+1, y   (1)
            torch.stack([norm_x - dx, norm_y], -1),      # x-1, y   (2)
            torch.stack([norm_x, norm_y + dy], -1),      # x,   y+1 (3)
            torch.stack([norm_x, norm_y - dy], -1),      # x,   y-1 (4)
            torch.stack([norm_x + dx, norm_y + dy], -1), # x+1, y+1 (5)
            torch.stack([norm_x - dx, norm_y - dy], -1), # x-1, y-1 (6)
            torch.stack([norm_x - dx, norm_y + dy], -1), # x-1, y+1 (7)
            torch.stack([norm_x + dx, norm_y - dy], -1), # x+1, y-1 (8)
        ], dim=-2) # Shape: (B, K, 9, 2)
        
        # Reshape for grid_sample
        heatmaps_reshaped = heatmaps.reshape(B * K, 1, H, W)
        grid_reshaped = grid.view(B * K, 9, 1, 2)
        
        sampled_values = F.grid_sample(
            heatmaps_reshaped, grid_reshaped.to(heatmaps_reshaped.dtype), mode='bilinear', align_corners=True
        ).squeeze().view(B, K, 9)

        # --- 2. 使用有限差分计算梯度和Hessian矩阵的元素 ---
        v = sampled_values
        Dx = 0.5 * (v[..., 1] - v[..., 2])
        Dy = 0.5 * (v[..., 3] - v[..., 4])
        Dxx = v[..., 1] - 2 * v[..., 0] + v[..., 2]
        Dyy = v[..., 3] - 2 * v[..., 0] + v[..., 4]
        # 混合偏导数 Dxy，这是精确版的关键
        Dxy = 0.25 * (v[..., 5] + v[..., 6] - v[..., 7] - v[..., 8])

        # --- 3. 构建梯度向量 g 和 Hessian 矩阵 H ---
        # g: (B, K, 2, 1), H: (B, K, 2, 2)
        g = torch.stack([Dx, Dy], dim=-1).unsqueeze(-1)
        H = torch.stack([Dxx, Dxy, Dxy, Dyy], dim=-1).view(B, K, 2, 2)
        # 为了数值稳定性，给Hessian的对角线增加一个极小值
        H = H + torch.eye(2, device=device, dtype=H.dtype).view(1, 1, 2, 2) * 1e-6

        # --- 4. 求解亚像素偏移量 delta = -H^{-1} * g ---
        try:
            # 批量求解2x2矩阵的逆
            H_inv = torch.linalg.inv(H)
            delta = -torch.matmul(H_inv, g).squeeze(-1)
        except torch.linalg.LinAlgError:
            # 如果矩阵是奇异的（非常罕见），则退化为原来的简化方法作为备用
            delta_x = -Dx / (Dxx + 1e-6)
            delta_y = -Dy / (Dyy + 1e-6)
            delta = torch.stack([delta_x, delta_y], dim=-1)

        # 将偏移量限制在合理范围内（+/- 1个像素），防止数值不稳定导致的异常值
        delta = torch.clamp(delta, -1, 1)
        
        coords_subpixel = coords_int.float() + delta
        return coords_subpixel.to(heatmaps_reshaped.dtype)

    def _find_keypoints_and_extract_patches_vectorized(self, heatmaps, threshold, use_zero_padding=True):
        """
        Finds a total of K best keypoints across all N classes in the heatmaps,
        extracts their neighborhood patches, and generates a separate sparse heatmap for each patch.
        This version implements cross-category NMS and a robust padding strategy.

        Args:
        - heatmaps (torch.Tensor): Input heatmaps (B, N, H, W)
        - threshold (float): Threshold to filter low-scoring keypoints.
        - use_zero_padding (bool): If True, use index-0 padding; otherwise use random coords in [0,1].

        Returns:
        - final_coords_relative: Normalized keypoint coordinates (B, K, 2).
        - sparse_heatmaps_collection: A collection of individual sparse heatmaps for each keypoint (B, K, H, W).
        - final_scores: The final scores for each keypoint (B, K).
        - needs_padding_mask: Bool mask indicating which slots were padded (B, K).
        """
        import torch
        import torch.nn.functional as F

        B, N, H, W = heatmaps.shape
        device = heatmaps.device
        ps = self.patch_size

        # --- Step 1: cross-category NMS to find K candidates ---
        # k_coords_int: (B, K, 2), k_scores: (B, K), k_class_indices: (B, K)
        k_coords_int, k_scores, k_class_indices, _ = self._find_keypoints_nms(
            heatmaps, self.K, self.suppression_radius
        )

        # --- Step 2: validity + padding indices ---
        valid_mask = k_scores > threshold                          # (B, K)
        num_valid = valid_mask.sum(dim=1)                           # (B,)
        needs_padding_mask = torch.arange(self.K, device=device).view(1, -1) >= num_valid.view(-1, 1)  # (B, K)
        base_indices = torch.arange(self.K, device=device).view(1, -1).expand(B, -1)                    # (B, K)

        if use_zero_padding:
            # Original: pad by copying index 0
            padding_idx = torch.zeros((B, 1), dtype=torch.long, device=device)
            final_indices = torch.where(needs_padding_mask, padding_idx, base_indices)                   # (B, K)

            final_indices_expanded = final_indices.unsqueeze(-1).expand(-1, -1, 2)                      # (B, K, 2)
            final_coords_int = torch.gather(k_coords_int, 1, final_indices_expanded)                    # (B, K, 2)
            final_scores = torch.gather(k_scores, 1, final_indices)                                     # (B, K)
            final_class_indices = torch.gather(k_class_indices, 1, final_indices)                       # (B, K)

            # Safer validity: true only for the first num_valid slots
            final_is_valid_keypoint = ~needs_padding_mask                                               # (B, K)

        else:
            # Random padding: mark padding slots with -1, then clamp for gather, then overwrite coords
            final_indices = torch.where(needs_padding_mask, torch.full_like(base_indices, -1), base_indices)  # (B, K)
            final_indices_clamped = final_indices.clamp(min=0)

            final_indices_expanded = final_indices_clamped.unsqueeze(-1).expand(-1, -1, 2)              # (B, K, 2)
            final_coords_int = torch.gather(k_coords_int, 1, final_indices_expanded)                    # (B, K, 2)
            final_scores = torch.gather(k_scores, 1, final_indices_clamped)                             # (B, K)
            final_class_indices = torch.gather(k_class_indices, 1, final_indices_clamped)               # (B, K)

            # For downstream logic, only original valid slots are True
            #                                             # (B, K)

            # --- Overwrite coords of padded slots with random coords in [0,1] (then to pixel grid) ---
            # sample relative coords in [0,1], then scale to pixel indices
            rand_rel = torch.rand((B, self.K, 2), device=device)                                        # (B, K, 2), [0,1]
            rand_pix = torch.empty_like(rand_rel)
            # x in [0, W-1], y in [0, H-1]
            rand_pix[..., 0] = rand_rel[..., 0] * (W - 1)
            rand_pix[..., 1] = rand_rel[..., 1] * (H - 1)
            rand_pix = torch.round(rand_pix).long()                                                     # integer pixels
            final_coords_int[needs_padding_mask] = rand_pix[needs_padding_mask]
            # needs_padding_mask[:] = 0  
            final_is_valid_keypoint = ~needs_padding_mask 
            
            # NOTE: keeping scores/class_indices gathered from clamped index (0) is fine since
            # final_is_valid_keypoint marks these as invalid and they won't affect downstream
            # If you prefer random scores/classes for padded slots, add similar overwrites here.

        # --- Step 3: sub-pixel refinement (valid points only) ---
        b_idx = torch.arange(B, device=device).view(B, 1)
        relevant_heatmaps = heatmaps[b_idx, final_class_indices]                                        # (B, K, H, W)

        coords_subpixel = self._refine_coords_subpixel(relevant_heatmaps, final_coords_int)             # (B, K, 2)

        # Use refined coords for valid points; padded use integer coords
        final_coords_absolute = torch.where(
            final_is_valid_keypoint.unsqueeze(-1), coords_subpixel, final_coords_int.float()
        )                                                                                               # (B, K, 2)

        # --- Step 4: normalize coords to [0,1] ---
        relative_x = final_coords_absolute[..., 0] / (W - 1)
        relative_y = final_coords_absolute[..., 1] / (H - 1)
        final_coords_relative = torch.stack([relative_x, relative_y], dim=-1)                           # (B, K, 2)

        # --- Step 5: patch extraction via grid_sample ---
        norm_coords_x = 2 * (final_coords_absolute[..., 0] / (W - 1)) - 1
        norm_coords_y = 2 * (final_coords_absolute[..., 1] / (H - 1)) - 1
        norm_coords = torch.stack([norm_coords_x, norm_coords_y], dim=-1)                               # (B, K, 2)

        offset_range = torch.linspace(-1, 1, ps, device=device)
        grid_y, grid_x = torch.meshgrid(offset_range, offset_range, indexing='ij')
        scale_x, scale_y = (ps / W), (ps / H)
        base_grid = torch.stack([grid_x * scale_x, grid_y * scale_y], dim=-1)                           # (ps, ps, 2)

        sampling_grid = norm_coords.view(B, self.K, 1, 1, 2) + base_grid.view(1, 1, ps, ps, 2)          # (B,K,ps,ps,2)

        patches = F.grid_sample(
            relevant_heatmaps.reshape(B * self.K, 1, H, W),
            sampling_grid.reshape(B * self.K, ps, ps, 2),
            mode='bilinear',
            align_corners=False
        )
        final_patches = patches.view(B, self.K, ps, ps)                                                 # (B, K, ps, ps)

        # --- Step 6: build sparse heatmaps for each patch (valid only) ---
        sparse_heatmaps_collection = torch.zeros(B, self.K, H, W, device=device, dtype=heatmaps.dtype)
        ps_half = ps // 2

        top_left_coords = torch.round(final_coords_absolute).long() - ps_half                           # (B, K, 2)
        patch_y_offsets, patch_x_offsets = torch.meshgrid(
            torch.arange(ps, device=device), torch.arange(ps, device=device), indexing='ij'
        )

        dest_y = top_left_coords[..., 1].unsqueeze(-1).unsqueeze(-1) + patch_y_offsets
        dest_x = top_left_coords[..., 0].unsqueeze(-1).unsqueeze(-1) + patch_x_offsets

        in_bounds_mask = (dest_x >= 0) & (dest_x < W) & (dest_y >= 0) & (dest_y < H)
        write_mask = in_bounds_mask & final_is_valid_keypoint.view(B, self.K, 1, 1)

        b_idx_m, k_idx_m, _, _ = torch.meshgrid(
            torch.arange(B, device=device),
            torch.arange(self.K, device=device),
            torch.arange(ps, device=device),
            torch.arange(ps, device=device),
            indexing='ij'
        )

        b_idx_flat = b_idx_m[write_mask]
        k_idx_flat = k_idx_m[write_mask]
        y_flat = dest_y[write_mask]
        x_flat = dest_x[write_mask]
        values_flat = final_patches[write_mask]

        sparse_heatmaps_collection[b_idx_flat, k_idx_flat, y_flat, x_flat] = values_flat

        return final_coords_relative.to(sparse_heatmaps_collection.dtype), sparse_heatmaps_collection, final_scores, needs_padding_mask



class GroundingHeadWithLlamaModel(nn.Module):
    """
    An alternative implementation using the high-level LlamaModel.
    """
    def __init__(self,
                 hidden_dim: int = 1024,
                 num_attention_layers: int = 3,
                 num_heads: int = 32,
                 intermediate_size: int = 11008,
                 num_image_tokens: int = 576,
                 keypoint_cfg: Optional[KeypointConfig] = None,
                 sam_variant: str = "vit_h",
                 sam_checkpoint: str = "./sam_vit_h_4b8939.pth",
                 fusion_model: Optional[nn.Module] = None,
                 explicit_position_ids: bool = True):
        super().__init__()
        if keypoint_cfg is None:
            keypoint_cfg = KeypointConfig()

        self.hidden_dim = hidden_dim
        self.num_image_tokens = num_image_tokens
        self.image_grid_size = int(num_image_tokens ** 0.5)
        # Feed the fusion model explicit 1-D position ids (LLaVA-1.5/Qwen2) or let it build
        # its own (Qwen2-VL M-RoPE -> pass None).
        self.explicit_position_ids = explicit_position_ids

        # The 2-layer fusion "head" mirrors ONE transformer layer of the backbone (same
        # hidden size + attention config). When the conditioner builds it via
        # backbone.build_fusion_model(...) it is passed in here; otherwise (standalone use)
        # fall back to the original LLaVA-shaped LlamaModel so behaviour is unchanged.
        if fusion_model is None:
            # 1. Create a single LlamaConfig
            config = LlamaConfig(
                hidden_size=hidden_dim,
                num_attention_heads=num_heads,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_attention_layers,
                output_attentions=True,
            )
            config._attn_implementation = "eager"  # 关键:只有 eager 会返回 attention weights

            # self.i_proj = nn.Linear(256, 1)
            # 2. Instantiate the entire LlamaModel instead of a list of layers
            fusion_model = LlamaModel(config)
            del fusion_model.embed_tokens
            force_eager_attention(fusion_model)  # autoset may reset eager->sdpa during init
        self.fusion_model = fusion_model

                # 1. Create a single LlamaConfig
        # config = LlamaConfig(
        #     hidden_size=hidden_dim,
        #     num_attention_heads=num_heads,
        #     intermediate_size=int(hidden_dim * mlp_ratio),
        #     num_hidden_layers=4,
        # )
        
        # # 2. Instantiate the entire LlamaModel instead of a list of layers
        # self.senmatic_model = LlamaModel(config)
        # del self.senmatic_model.embed_tokens
        
        # self.upsampler = HeatmapUpsampler()
        
        input_dim = 4096
        # self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_proj = nn.Identity()

        # self.input_proj_sen = nn.Linear(input_dim, hidden_dim)

        self.keypoint_map_extract = KeypointExtractor(
            num_keypoints=keypoint_cfg.num_keypoints,
            patch_size=keypoint_cfg.patch_size,
            grid_size=self.image_grid_size,
            num_image_tokens=self.num_image_tokens,
            suppression_radius=keypoint_cfg.suppression_radius,
            thre_ratio=keypoint_cfg.thre_ratio,
            sam_variant=sam_variant,
            sam_checkpoint=sam_checkpoint,
        )

        # self.fusion_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim // 4, 256),
        #     # nn.LayerNorm(256),
        #     nn.Dropout(0.),
        # )
        in_dim = hidden_dim
        out_dim = 256
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            # nn.LayerNorm(out_dim), 
            nn.Dropout(0.),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        self.cls_token_pos = nn.Parameter(torch.randn(1, 1, out_dim))

        self.feature_aggregator = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Transformer 中通常还包含 LayerNorm 和 FeedForward 网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.encoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)
    
    def forward(self,
                multimodal_features: torch.Tensor,
                high_level_multimodal_features: torch.Tensor,
                attention_mask: torch.Tensor,
                image_token_mask: torch.Tensor,
                text_token_mask: torch.Tensor,
                thresh_keypoint = None,
                fusion_position_ids = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, seq_len, _ = multimodal_features.shape
        features = self.input_proj(multimodal_features)
        # features_sen = self.input_proj_sen(high_level_multimodal_features)

        if fusion_position_ids is not None:
            # Backbone-provided ids, e.g. Qwen2-VL's real 3-D M-RoPE [3, B, seq].
            position_ids = fusion_position_ids
        elif self.explicit_position_ids:
            position_ids = (torch.cumsum(attention_mask, dim=1).long() - 1).clamp(min=0)
        else:
            # e.g. a backbone that wants the text model to build its own (sequential) ids.
            position_ids = None

        # The mask creation is the same
        causal_mask = torch.full((seq_len, seq_len), torch.finfo(features.dtype).min, device=features.device)
        causal_mask = causal_mask.triu(diagonal=1)
        padding_mask = attention_mask.view(batch_size, 1, 1, seq_len)
        padding_mask = (1.0 - padding_mask) * torch.finfo(features.dtype).min
        final_hf_mask = causal_mask + padding_mask

        # 3. Call the LlamaModel once, requesting attention outputs
        outputs = self.fusion_model(
            inputs_embeds=features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True,
        )
        
        # The final hidden states are the first element of the output
        features = outputs.last_hidden_state
        # features_sen = outputs_sen.last_hidden_state
        # `outputs.attentions` is a tuple of all layer attentions. We want the last one.
        final_attn_weights = outputs.attentions[0]
        
        # The rest of the logic is identical
        final_attn_weights = final_attn_weights.mean(dim=1)

        keypoint_coords, pos_embs, local_patches, n_heatmaps, padding_masks = self.keypoint_map_extract(
            final_attn_weights.float(), text_token_mask, image_token_mask, thresh_keypoint
        )

        features_img = features[image_token_mask].reshape(batch_size, -1, features.shape[-1])
        p = int(features_img.shape[1] ** 0.5)
        k = 3
        features_img = features_img.view(batch_size, p, p, -1)
        # 将通道维度提前，以满足 grid_sample 的要求 (B, C, H, W)
        features_channels_first = features_img.permute(0, 3, 1, 2)
        B, C, H, W = features_channels_first.shape
        N = keypoint_coords.shape[1]

        center_coords = (keypoint_coords * 2.0 - 1.0) # 形状: (B, N, 2)
        k_half = (k - 1) / 2.0
        pixel_step = 2.0 / (H - 1) if H > 1 else 0

        # 生成从 -k_half 到 +k_half 的坐标范围
        k_range = torch.arange(-k_half, k_half + 1.0, device=features_channels_first.device) * pixel_step
        grid_y, grid_x = torch.meshgrid(k_range, k_range, indexing='ij') # 形状都为 (k, k)
        grid_offsets = torch.stack([grid_x, grid_y], dim=-1)
        final_sampling_grid = center_coords.view(B, N, 1, 1, 2) + grid_offsets.view(1, 1, k, k, 2)
        features_expanded = features_channels_first.unsqueeze(1).expand(-1, N, -1, -1, -1) # 形状: (B, N, C, H, W)
        features_reshaped = features_expanded.reshape(B * N, C, H, W)
        grid_reshaped = final_sampling_grid.reshape(B * N, k, k, 2)

        # 6. 执行网格采样
        sampled_patches_flat = F.grid_sample(
            features_reshaped,
            grid_reshaped.to(features_reshaped.dtype),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ) # 输出形状: (B * N, C, k, k)
        sampled_features = sampled_patches_flat.view(B, N, C, -1).permute(0, 1, 3, 2)

        last_text_token_indices = text_token_mask.cumsum(dim=1).argmax(dim=1)
        batch_indexer = torch.arange(batch_size, device=features.device)
        last_text_token = features[batch_indexer, last_text_token_indices, :]
        last_text_token_ = last_text_token.unsqueeze(1).repeat(1, N, 1)

        last_text_token_ = last_text_token_.view(B*N, 1, C)
        sampled_img_features = sampled_features.view(B*N, -1, C)

        attn_output, attn_weights = self.feature_aggregator(
            query=last_text_token_,
            key=sampled_img_features,
            value=sampled_img_features
        ) 
        agg_features = self.norm1(last_text_token_ + attn_output)

        # 3. 通过一个前馈网络 (Feed-Forward Network)
        ffn_output = self.ffn(agg_features)
        agg_features = self.norm2(agg_features + ffn_output)

        # 4. 得到最终聚合后的特征
        # agg_features 的形状仍然是 (B*N, 1, C)
        # 你可以将其 reshape 回 (B, N, C) 以便后续处理
        final_aggregated_features = agg_features.view(B, N, C)
        # padding_masks[:] = 1
        padding_masks = torch.cat([padding_masks, torch.tensor([0], device=padding_masks.device, dtype=padding_masks.dtype).view(1, 1).repeat([len(padding_masks), 1])], dim=1)
        

        last_text_token = last_text_token.unsqueeze(1)
        final_aggregated_features_self = torch.cat((final_aggregated_features,last_text_token), dim=1)
        final_aggregated_features_self = self.encoder_layer(final_aggregated_features_self, src_key_padding_mask=padding_masks.bool())
        final_aggregated_features_self = self.text_hidden_fcs[0](final_aggregated_features_self)
        last_text_token = final_aggregated_features_self[:, -1].unsqueeze(1)
        final_aggregated_features = final_aggregated_features_self[:, :-1]
        # indicators = torch.cat((agg_img_features, last_text_token_), dim=-1)
        # indicators = self.fusion_mlp(final_aggregated_features)
        indicators = final_aggregated_features
        # indicator_for_train = self.i_proj(indicators)
        indicator_for_train = indicators
        # neg_pos = self.keypoint_map_extract.poe_emb_convertor.pos_emb[0].weight
        # pos_pos = self.keypoint_map_extract.poe_emb_convertor.pos_emb[1].weight
        # indicators = torch.where(
        #     (torch.norm(indicators - pos_pos, dim=-1, keepdim=True) < torch.norm(indicators - neg_pos, dim=-1, keepdim=True)),
        #     pos_pos,
        #     neg_pos
        # )

        indicators = indicators + pos_embs.view(batch_size, -1, pos_embs.shape[-1])

        # last_text_token = self.text_hidden_fcs[0](last_text_token).unsqueeze(1) + self.cls_token_pos.expand(last_text_token.shape[0], -1, -1)
        last_text_token = last_text_token + self.cls_token_pos.expand(last_text_token.shape[0], -1, -1)
        indicators = torch.cat((indicators, last_text_token), dim=1)

        heatmap = n_heatmaps.mean(1).unsqueeze(1)

        heatmap = F.interpolate(
            heatmap,
            size=(128, 128),
            mode='bilinear',
            align_corners=True
        )
        upsampled_heatmap = heatmap
        
        updated_image_features = features[image_token_mask].reshape(
            batch_size, self.num_image_tokens, self.hidden_dim
        )

        updated_image_features_map = updated_image_features.view(
            batch_size, self.image_grid_size, self.image_grid_size, -1
        ).permute(0, 3, 1, 2)
        
        return upsampled_heatmap, indicators, updated_image_features_map, keypoint_coords, features, indicator_for_train, padding_masks
import random

class MLLM_Conditioner(nn.Module):
    """
    一个使用 LLaVA-1.5 的 Conditioner 类。
    """
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # The frozen multimodal backbone is now pluggable; selected by cfg.backbone.name.
        self.backbone = build_mllm_backbone(cfg.backbone)

        # self.aggregator = ChunkedMLPAggregator(embed_dim=32, num_chunks=6)

        self.selected_layer_id = cfg.backbone.selected_layer_id

        # CR1(a): the 2-layer fusion head mirrors ONE of the BACKBONE's own transformer
        # layers (Llama for LLaVA-1.5, Qwen2 for Qwen2-VL / LLaVA-OneVision). The
        # backbone builds it so its hidden size + attention config follow that backbone.
        fusion_model = self.backbone.build_fusion_model(cfg.grounding_head.num_attention_layers)

        self.conditioning_module = GroundingHeadWithLlamaModel(
            hidden_dim=cfg.backbone.hidden_size,
            num_attention_layers=cfg.grounding_head.num_attention_layers,
            num_heads=cfg.grounding_head.num_heads,
            intermediate_size=cfg.grounding_head.intermediate_size,
            num_image_tokens=cfg.backbone.num_image_tokens,
            keypoint_cfg=cfg.keypoint,
            # The Point2Vec positional embeddings come from the same SAM the decoder uses,
            # so a SAM-size experiment needs only one checkpoint.
            sam_variant=cfg.decoder.sam_variant,
            sam_checkpoint=cfg.decoder.sam_checkpoint,
            fusion_model=fusion_model,
            explicit_position_ids=self.backbone.fusion_explicit_position_ids,
        ).to(self.backbone.dtype)

        # CR1(b): warm-start the fusion layers from the backbone's middle layers
        # (grounding_init_start_layer ~ num_backbone_layers // 2). Now valid for every
        # backbone since the fusion head shares its architecture. Set to null to skip.
        if cfg.backbone.grounding_init_start_layer is not None:
            self.backbone.warmstart_fusion_layers(
                self.conditioning_module.fusion_model,
                cfg.backbone.grounding_init_start_layer,
            )
        else:
            print("Skipping grounding-head warm-start (grounding_init_start_layer is null); "
                  "fusion layers train from scratch.")

        # self.lm_head = nn.Linear(self.model.config.text_config.hidden_size, self.model.config.vocab_size, bias=False)


    def forward(self,
            image: Union[np.ndarray, List[str], List[Image.Image]],
            text: List[str],
            is_train = False,
            kp_thresh = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        multimodal_features, attention_mask, image_token_mask, text_token_mask, last_multimodal_features, input_ids = self.backbone.encode(image, text)
        # Backbone-provided fusion position ids (e.g. Qwen2-VL 3-D M-RoPE), else None.
        fusion_position_ids = self.backbone.fusion_position_ids()

        coord, indicator, img_f, keypoints, features, indicator_for_train, padding_masks = self.conditioning_module(
            multimodal_features,
            last_multimodal_features,
            attention_mask,
            image_token_mask,
            text_token_mask,
            kp_thresh,
            fusion_position_ids,
        )
        loss = 0
        # indicator_for_train = None
        # if is_vqa is not None:
        if is_train:
            # labels = input_ids.clone()
            # labels[input_ids == 32000] = -100
            # logits = self.lm_head(features)
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # # Flatten the tokens
            # b = shift_labels.shape[0]
            # loss_fct = CrossEntropyLoss(reduction='none')
            # shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
            # # Enable model/pipeline parallelism
            # shift_labels = shift_labels.to(shift_logits.device)
            # loss = loss_fct(shift_logits, shift_labels).view(b, -1).mean(1)
            # is_vqa = torch.tensor(is_vqa).to(loss.device)
            # loss = (loss * is_vqa).sum() / (is_vqa.sum() + 1e-6)
            
            indicator_new = indicator.clone()
            for s, i in enumerate(indicator):
                if random.random() < 0.9:
                    padding_masks[s, :-1] = 1

                # if random.random() < 0.5:
                #     padding_masks[s, -1] = 1
                    
            a = 1
            indicator = indicator_new

        # coord_p = (coord.view(len(coord), 1, -1)).softmax(-1)
        # coord = F.interpolate(
        #     coord,
        #     size=(1024, 1024),
        #     mode='bilinear',
        #     align_corners=True
        # )
        return coord, indicator, img_f, keypoints, loss, indicator_for_train, padding_masks
