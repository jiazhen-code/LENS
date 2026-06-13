"""Local visualization helpers for LENS evaluation / training-time eval."""
import os
import textwrap

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe: render straight to file, no display needed
import matplotlib.pyplot as plt


def _to_numpy(x):
    """Best-effort tensor/array -> float numpy on CPU (handles bf16 / autograd)."""
    if x is None:
        return None
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _prompt_to_text(prompt):
    """Flatten whatever the caller passes for the instruction into one string."""
    if prompt is None:
        return None
    if isinstance(prompt, (list, tuple)):
        return "\n".join(str(p) for p in prompt)
    return str(prompt)


def save_seg_visualization(
    pred_mask,
    gt_mask,
    keypoint_coords,
    acc_iou,
    save_path,
    attention_map=None,
    prompt=None,
    image=None,
    meta=None,
    attn_pad_frac=None,
):
    """Save a visualization of a single prediction.

    Always drawn:
        - predicted mask with the predicted keypoints overlaid (red dots)
        - ground-truth mask

    Drawn when provided (the test-time call passes all three):
        - the raw image with the predicted keypoints
        - the attention heatmap (``head_maps`` — the instruction-token attention the
          keypoints are read off) overlaid on the image
        - the instruction text, shown as the figure title

    Sidecar files are written next to the PNG so nothing is lost to title truncation
    or colormap rescaling:
        - ``<save_path>.txt``  : the full instruction + acc_iou + any ``meta``
        - ``<save_path>.npy``  : the raw attention heatmap (when given)

    Args:
        pred_mask: 2D array ``(H, W)`` — thresholded predicted mask.
        gt_mask: 2D array ``(H, W)`` — ground-truth mask.
        keypoint_coords: array/tensor ``(B, N, 2)`` with coords normalized to ``[0, 1]``.
            Only the first sample (``[0]``) is drawn — pass the sample shown here.
        acc_iou: per-sample IoU, shown in the title and the sidecar.
        save_path: destination PNG path; parent directories are created if missing.
        attention_map: optional array ``(h, w)`` (extra leading dims are squeezed) —
            the attention heatmap to visualize and dump.
        prompt: optional str / list[str] — the instruction fed to the model.
        image: optional ``(H, W, 3)`` array / PIL image — raw image for context + overlay.
        meta: optional dict of extra key/value info to write into the ``.txt`` sidecar
            (e.g. ``{"image_path": ...}``).
        attn_pad_frac: optional ``(fx, fy)`` — content fraction of the attention's padded
            square (Qwen/LLaVA-OneVision 'pad' mode). When given, the heatmap is cropped to its
            top-left content region and keypoints are divided by ``(fx, fy)`` to map them back
            to the original-image frame. ``None`` => attention already fills the image
            (LLaVA / 'stretch').
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    pred_mask = _to_numpy(pred_mask)
    gt_mask = _to_numpy(gt_mask)
    kp = _to_numpy(keypoint_coords)
    attn = _to_numpy(attention_map)
    if attn is not None:
        attn = np.squeeze(attn)  # (1, h, w) / (1, 1, h, w) -> (h, w)
    img = np.asarray(image) if image is not None else None

    # If the attention lives in an aspect-preserve+pad square (Qwen/LLaVA-OneVision 'pad' mode), its
    # real content is only the top-left (fx, fy) fraction of the grid; crop that for display
    # and divide the padded-frame keypoints by (fx, fy) to map them back to the original-image
    # frame -- otherwise the heatmap/keypoints sit compressed toward the top-left.
    fx, fy = attn_pad_frac if attn_pad_frac else (1.0, 1.0)
    attn_disp = attn
    if attn is not None and attn_pad_frac is not None:
        gh, gw = attn.shape[:2]
        attn_disp = attn[: max(1, round(fy * gh)), : max(1, round(fx * gw))]

    # (name, data, overlay-background, draw-keypoints?)
    panels = []
    if img is not None:
        panels.append(("image", img, None, True))
    if attn is not None:
        panels.append(("attention", attn_disp, img, True))
    panels.append(("prediction", pred_mask, None, True))
    panels.append(("ground truth", gt_mask, None, False))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, data, bg, draw_kp) in zip(axes, panels):
        if name == "attention" and bg is not None:
            disp_h, disp_w = bg.shape[:2]
            ax.imshow(bg)
            ax.imshow(
                data,
                cmap="jet",
                alpha=0.6,
                extent=(0, disp_w, disp_h, 0),
                interpolation="bilinear",
            )
        elif name == "attention":
            disp_h, disp_w = data.shape[:2]
            ax.imshow(data, cmap="jet", interpolation="bilinear")
        else:
            disp_h, disp_w = data.shape[:2]
            ax.imshow(data)

        # keypoints are normalized over the (possibly padded) attention frame -> divide by the
        # content fraction so they land correctly on this original-frame panel.
        if draw_kp and kp is not None:
            ax.scatter(kp[0, :, 0] * disp_w / fx, kp[0, :, 1] * disp_h / fy, c="r", s=2)

        ax.set_title(name)
        ax.axis("off")

    title = f"acc_iou: {acc_iou}"
    text = _prompt_to_text(prompt)
    if text is not None:
        wrapped = "\n".join(
            textwrap.wrap(text, width=90, max_lines=4, placeholder=" ...")
        )
        title = f"{title}\n{wrapped}"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(save_path)
    plt.close(fig)

    # Sidecars: full instruction (titles truncate) + raw attention (colormap rescales).
    sidecar = os.path.splitext(save_path)[0]
    if text is not None or meta:
        with open(sidecar + ".txt", "w", encoding="utf-8") as f:
            f.write(f"acc_iou: {acc_iou}\n")
            for k, v in (meta or {}).items():
                f.write(f"{k}: {v}\n")
            if text is not None:
                f.write(f"instruction:\n{text}\n")
    if attn is not None:
        np.save(sidecar + ".npy", attn)
