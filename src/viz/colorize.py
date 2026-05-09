import numpy as np
from utils.config import CLASSES

# front-view 5-class palette: road, vehicle, pedestrian, sky, background
SEG_COLORS = np.array([
    [100, 160, 100],   # 0 road        green
    [200,  80,  50],   # 1 vehicle     orange-red
    [ 50, 120, 220],   # 2 pedestrian  blue
    [200, 200, 200],   # 3 sky         light grey
    [ 30,  30,  30],   # 4 background  dark
], dtype=np.uint8)

# BEV 3-class palette matches channels in CLASSES["names"]
BEV_COLORS = np.array([
    [100, 160, 100],   # 0 drivable_area
    [200,  80,  50],   # 1 vehicle
    [ 50, 120, 220],   # 2 pedestrian
], dtype=np.uint8)


def colorize_seg(mask):
    """mask: (H, W) uint8 with values 0-4  →  (H, W, 3) RGB"""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(SEG_COLORS):
        out[mask == i] = color
    return out


def colorize_bev(bev_gt):
    """
    bev_gt : (C, H, W) float32 binary  OR  (H, W) uint8 class labels
    returns: (H, W, 3) RGB
    """
    if bev_gt.ndim == 3:
        # multi-channel binary
        out = np.zeros((bev_gt.shape[1], bev_gt.shape[2], 3), dtype=np.uint8)
        for c in range(bev_gt.shape[0]):
            out[bev_gt[c] > 0.5] = BEV_COLORS[c]
    else:
        # single-channel label map
        out = np.zeros((*bev_gt.shape, 3), dtype=np.uint8)
        for i, color in enumerate(BEV_COLORS):
            out[bev_gt == i] = color
    return out


def overlay_seg(image_rgb, mask, alpha=0.45):
    """Blend segmentation colour map onto raw RGB image. Handles size mismatch."""
    seg_rgb = colorize_seg(mask)
    # resize seg to match image if they differ (e.g. 900×1600 image vs 512×1024 mask)
    if seg_rgb.shape[:2] != image_rgb.shape[:2]:
        import cv2
        seg_rgb = cv2.resize(seg_rgb,
                             (image_rgb.shape[1], image_rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    return (image_rgb * (1 - alpha) + seg_rgb * alpha).astype(np.uint8)