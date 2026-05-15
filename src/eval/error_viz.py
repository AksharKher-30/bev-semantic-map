import numpy as np
import torch
from pathlib import Path
from utils.config import CLASSES, PATHS, EVAL


# Colour scheme: green=TP, red=FP, blue=FN, black=TN
TP_COLOR  = (0,   200, 0)
FP_COLOR  = (200, 0,   0)
FN_COLOR  = (0,   0,   200)
TN_COLOR  = (20,  20,  20)


def make_error_map(pred_logits, gt_mask, class_idx=0, threshold=None):
    """
    Generate a TP/FP/FN colour map for one class.

    pred_logits : (C, H, W) raw logits — single sample (no batch dim)
    gt_mask     : (C, H, W) binary float GT
    class_idx   : which class to visualise

    Returns
    -------
    error_map : (H, W, 3) uint8 RGB
    """
    thr  = threshold or EVAL["iou_threshold"]
    pred = (pred_logits[class_idx].sigmoid() > thr).numpy().astype(bool)
    gt   = gt_mask[class_idx].numpy().astype(bool)

    H, W    = pred.shape
    err_map = np.full((H, W, 3), TN_COLOR, dtype=np.uint8)
    err_map[pred &  gt]  = TP_COLOR
    err_map[pred & ~gt]  = FP_COLOR
    err_map[~pred &  gt] = FN_COLOR

    return err_map


def make_all_class_error_maps(pred_logits, gt_mask, threshold=None):
    """
    Generate error maps for every class.

    Returns
    -------
    maps : dict {class_name: (H, W, 3) uint8}
    """
    return {
        name: make_error_map(pred_logits, gt_mask, c, threshold)
        for c, name in enumerate(CLASSES["names"])
    }


def save_error_maps(pred_logits, gt_mask, sample_token, model_type="lss"):
    """
    Save error maps for all classes to outputs/results/error_maps/.

    Parameters
    ----------
    pred_logits  : (C, H, W) single-sample logits (no batch dim)
    gt_mask      : (C, H, W) single-sample GT
    sample_token : str — used as filename prefix
    model_type   : 'lss' or 'ipm'
    """
    import cv2

    out_dir = PATHS["results"] / "error_maps" / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = make_all_class_error_maps(pred_logits, gt_mask)
    for cls_name, err_map in maps.items():
        fname = out_dir / f"{sample_token[:8]}_{cls_name}.png"
        bgr   = cv2.cvtColor(err_map, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(fname), bgr)

    return out_dir