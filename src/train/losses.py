import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import SEGFORMER, CLASSES, TRAIN


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

    if (labels != ignore).sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return F.cross_entropy(
        up, labels.long(),
        weight=w,
        ignore_index=ignore,
    )


def bev_loss(pred_logits, bev_gt, pos_weights=None):
    """
    Per-class binary cross-entropy loss in BEV space.

    BEV classes are NOT mutually exclusive — a pixel can be both
    road=1 and vehicle=0 simultaneously (vehicle on drivable surface).
    So we use independent BCE per channel, not softmax CE.

    pred_logits : (B, C, H, W) — raw logits (no sigmoid)
    bev_gt      : (B, C, H, W) — binary float GT per class
    pos_weights : list[float] | None — upweight rare classes (vehicle, ped)

    Returns scalar loss (mean over classes).
    """
    n_cls = pred_logits.shape[1]
    loss  = 0.0
    for c in range(n_cls):
        pw = None
        if pos_weights is not None:
            pw = torch.tensor(pos_weights[c], dtype=torch.float32,
                              device=pred_logits.device)
        loss += F.binary_cross_entropy_with_logits(
            pred_logits[:, c],
            bev_gt[:, c],
            pos_weight=pw,
        )
    return loss / n_cls


def compute_iou(preds, labels, num_classes, ignore_index=255):
    """Per-class IoU for front-view segmentation. Returns list of floats."""
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        gt_c   = (labels == c) & (labels != ignore_index)
        inter  = (pred_c & gt_c).sum().float()
        union  = (pred_c | gt_c).sum().float()
        ious.append((inter / (union + 1e-6)).item())
    return ious


class BEVIoUMetric:
    """
    Accumulates TP/FP/FN across batches then computes IoU.
    Accumulating then computing is more stable than averaging per-batch IoU.

    Usage:
        metric = BEVIoUMetric()
        for batch in loader:
            metric.update(logits, gt)
        results = metric.compute()   # {"drivable_area": 0.45, ..., "mIoU": 0.38}
    """

    def __init__(self, threshold=0.5):
        self.threshold  = threshold
        self.class_names = CLASSES["names"]
        self.n_cls      = CLASSES["num_classes"]
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.n_cls)
        self.fp = torch.zeros(self.n_cls)
        self.fn = torch.zeros(self.n_cls)

    def update(self, pred_logits, gt_masks):
        """
        pred_logits : (B, C, H, W) raw logits
        gt_masks    : (B, C, H, W) binary float GT
        """
        pred = (pred_logits.sigmoid() > self.threshold).float().cpu()
        gt   = gt_masks.float().cpu()
        for c in range(self.n_cls):
            p = pred[:, c].bool()
            g = gt[:, c].bool()
            self.tp[c] += (p & g).sum().float()
            self.fp[c] += (p & ~g).sum().float()
            self.fn[c] += (~p & g).sum().float()

    def compute(self):
        iou     = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        results = {name: iou[i].item()
                   for i, name in enumerate(self.class_names)}
        results["mIoU"] = iou.mean().item()
        return results