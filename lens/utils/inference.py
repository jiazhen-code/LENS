import numpy as np
import torch.nn.functional as F
import sys 
from PIL import Image
import torch

from segment_anything.utils.transforms import ResizeLongestSide

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    is_mask=False
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    if not is_mask:
        x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def data_deal(image, img_size = 1024, is_mask=False):
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    raw_mask = None
    if is_mask:
        img = torch.from_numpy(image_np / 255.)
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        raw_mask = img
        
    transform = ResizeLongestSide(img_size)

    image = transform.apply_image(image_np)
    new_size = image.shape[:2]
    img = torch.from_numpy(image)
    if len(img.shape) == 2:
        img = img.unsqueeze(-1)
    img_tensor = (
        preprocess(img.permute(2, 0, 1).contiguous(), img_size=img_size, is_mask=is_mask)
    )
    return img_tensor, new_size, raw_mask

def decode_tensor(sam_backbone, img_tensor, point=None):
    sam_backbone.set_image(img_tensor)
    x0 = sam_backbone.bottom_feature()
    mask_pred_tensor0 = sam_backbone.decode_mask(x0, img_tensor, point)

    return mask_pred_tensor0
    

def run_one_mask(sam_backbone, img_path, img_size = 1024):
    if isinstance(img_path, str):
        img = Image.open(img_path)
    else:
        img = img_path
    
    img_tensor, original_size_list, resize_list = data_deal(img, img_size)
    img_tensor = img_tensor.unsqueeze(0).to(sam_backbone.device)
    mask_pred_tensor0 = decode_tensor(sam_backbone, img_tensor)
    
    pred_mask = sam_backbone.sam.postprocess_masks(
                    mask_pred_tensor0,
                    input_size=resize_list[0],
                    original_size=original_size_list[0]
                )

    return pred_mask, img_tensor
