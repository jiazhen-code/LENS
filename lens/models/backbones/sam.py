"""SAM segmentation backbone selection.

The mask decoder uses a frozen SAM image encoder + prompt encoder. The variant and
checkpoint are config-driven through this tiny registry so a different SAM size can be
swapped in without touching ``decoder.py``. Default is ``vit_h`` with the original
checkpoint path, i.e. identical to the previous hard-coded ``build_sam_vit_h(...)`` call.

Note: SAM's prompt encoder (used by Point2Vec for positional embeddings) is 256-dim for
*every* variant, so the keypoint machinery is unaffected by which size is chosen.
"""

import os
import urllib.request

from segment_anything import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l


SAM_BUILDERS = {
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

# Official download URLs per variant.
SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

# Conventional local filenames per variant.
DEFAULT_SAM_CHECKPOINTS = {
    "vit_h": "./sam_vit_h_4b8939.pth",
    "vit_l": "./sam_vit_l_0b3195.pth",
    "vit_b": "./sam_vit_b_01ec64.pth",
}


def sam_url(variant: str) -> str:
    if variant not in SAM_URLS:
        raise KeyError(f"Unknown SAM variant '{variant}'. Available: {sorted(SAM_URLS)}")
    return SAM_URLS[variant]


def build_sam(variant: str = "vit_h", checkpoint: str = "./sam_vit_h_4b8939.pth"):
    if variant not in SAM_BUILDERS:
        raise KeyError(f"Unknown SAM variant '{variant}'. Available: {sorted(SAM_BUILDERS)}")
    return SAM_BUILDERS[variant](checkpoint)


def download_sam_checkpoint(variant: str = "vit_h", checkpoint: str = None) -> str:
    """Download the SAM checkpoint for ``variant`` to ``checkpoint`` if missing."""
    if checkpoint is None:
        checkpoint = DEFAULT_SAM_CHECKPOINTS[variant]
    if not os.path.exists(checkpoint):
        url = sam_url(variant)
        print(f"SAM checkpoint not found at '{checkpoint}'. Downloading {variant} from {url} ...")
        urllib.request.urlretrieve(url, checkpoint)
    return checkpoint
