import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import SEGFORMER


def segformer_loss(logits, labels, weights=None):
    """
    logits : (B, C, H, W) — raw model output (not yet upsampled)
    labels : (B, H, W)    — integer class labels, 255 = ignore
    weights: list[float] | None — per-class weight
    """
    up = F.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    w = None
    if weights is not None:
        w = torch.tensor(weights, dtype=torch.float32, device=logits.device)

    from utils.config import SEGFORMER
    ignore = SEGFORMER["ignore_index"]

    # guard: if no valid pixels exist, loss would be nan — return 0 instead
    if (labels != ignore).sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return F.cross_entropy(
        up, labels.long(),
        weight=w,
        ignore_index=ignore,
    )


def compute_iou(preds, labels, num_classes, ignore_index=255):
    """Per-class IoU. Returns list of floats length num_classes."""
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        gt_c   = (labels == c) & (labels != ignore_index)
        inter  = (pred_c & gt_c).sum().float()
        union  = (pred_c | gt_c).sum().float()
        ious.append((inter / (union + 1e-6)).item())
    return ious