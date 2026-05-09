import cv2
import numpy as np
from utils.config import BEV, CLASSES


# background fill value for pixels outside the warp boundary
_BACKGROUND = 255   # ignore index — consistent with pseudo label convention


def apply_ipm(seg_mask, H, bev_size=None):
    """
    Warp a front-view segmentation mask into BEV using homography H.

    Uses INTER_NEAREST to preserve discrete class labels — bilinear
    interpolation would create spurious in-between class values.

    Parameters
    ----------
    seg_mask : (H, W) uint8 — class labels from SegFormer (values 0-4 or 0-2)
    H        : (3,3) homography from compute_ipm_homography()
    bev_size : int — output grid side (default from config)

    Returns
    -------
    bev_mask : (bev_size, bev_size) uint8 — warped class labels
    """
    sz = bev_size or BEV["size"]

    bev_mask = cv2.warpPerspective(
        seg_mask.astype(np.uint8),
        H,
        (sz, sz),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_BACKGROUND,
    )
    return bev_mask


def bev_mask_to_channels(bev_mask, num_classes=None):
    """
    Convert single-channel label BEV mask → multi-channel binary tensor.
    Consistent with the format used by BEV GT and LSS output.

    Parameters
    ----------
    bev_mask   : (H, W) uint8 — class label per pixel
    num_classes: int — number of foreground classes (default from config)

    Returns
    -------
    channels : (num_classes, H, W) float32 — binary mask per class
               ignore pixels (255) → all-zero across channels
    """
    nc = num_classes or CLASSES["num_classes"]
    H, W = bev_mask.shape
    channels = np.zeros((nc, H, W), dtype=np.float32)
    for c in range(nc):
        channels[c] = (bev_mask == c).astype(np.float32)
    return channels