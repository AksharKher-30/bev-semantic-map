from train.losses import BEVIoUMetric
from utils.config import CLASSES


def make_metric():
    """Return a fresh BEVIoUMetric instance."""
    return BEVIoUMetric()


def compute_iou_from_logits(pred_logits, gt_masks, threshold=0.5):
    """
    One-shot IoU computation for a single batch.

    pred_logits : (B, C, H, W) raw logits
    gt_masks    : (B, C, H, W) binary float GT

    Returns dict: {"drivable_area": ..., "vehicle": ..., "mIoU": ...}
    """
    m = BEVIoUMetric(threshold=threshold)
    m.update(pred_logits, gt_masks)
    return m.compute()