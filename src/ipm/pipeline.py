import cv2
import numpy as np
import torch

from utils.config import BEV, SEGFORMER
from utils.device import get_device
from ipm.homography import compute_ipm_homography, get_camera_height_from_nusc
from ipm.warp import apply_ipm, bev_mask_to_channels


def run_ipm_pipeline(image_bgr, K, sample_token=None, model=None, device=None,
                     bev_size=None, bev_res=None):
    """
    Full IPM pipeline for one frame.

    Steps:
        1. Run zero-shot SegFormer → 3-class front-view mask
        2. Compute IPM homography from K + camera height
        3. Warp mask to BEV
        4. Convert to multi-channel binary tensor

    Parameters
    ----------
    image_bgr    : (H, W, 3) numpy BGR image — raw from cv2.imread
    K            : (3,3) camera intrinsic matrix
    sample_token : str | None — if given, reads camera height from calibration
    model        : zero-shot SegFormer model | None (loads fresh if None)
    device       : torch.device | None
    bev_size     : int (default from config)
    bev_res      : float (default from config)

    Returns
    -------
    dict with:
        seg_mask_3  : (img_h, img_w) uint8 — 3-class front-view mask
        bev_mask    : (bev_size, bev_size) uint8 — warped class labels
        bev_channels: (num_classes, bev_size, bev_size) float32 — binary per class
        H           : (3,3) homography used
    """
    from models.segformer.inference import run_inference_3class

    if device is None:
        device = get_device(verbose=False)

    if model is None:
        from models.segformer.model import build_segformer_zero_shot
        model = build_segformer_zero_shot().to(device)

    sz  = bev_size or BEV["size"]
    res = bev_res  or BEV["resolution"]

    # step 1 — segmentation
    seg_mask = run_inference_3class(model, image_bgr, device)   # (img_h, img_w)

    # step 2 — homography
    if sample_token is not None:
        cam_h = get_camera_height_from_nusc(sample_token)
    else:
        cam_h = BEV["camera_height"]

    H = compute_ipm_homography(K, camera_height=cam_h,
                               bev_size=sz, bev_res=res)

    # step 3 — warp
    bev_mask = apply_ipm(seg_mask, H, bev_size=sz)

    # step 4 — multi-channel tensor
    bev_channels = bev_mask_to_channels(bev_mask)

    return {
        "seg_mask_3"  : seg_mask,
        "bev_mask"    : bev_mask,
        "bev_channels": bev_channels,
        "H"           : H,
    }


def run_ipm_batch(model, batch, device):
    """
    Run IPM pipeline on a DataLoader batch.
    Used by eval/evaluate.py for quantitative IoU scoring.

    Parameters
    ----------
    model : zero-shot SegFormer (19-class, remapped at inference)
    batch : dict from NuScenesBEVDataset — must have 'image', 'K', 'sample_token'
    device: torch.device

    Returns
    -------
    bev_logits : (B, num_classes, bev_size, bev_size) float32 tensor
                 Values are 0.0 or 1.0 (binary) — compatible with BCEWithLogitsLoss
                 after logit inversion: positive=large positive, negative=large negative
    """
    from models.segformer.inference import run_inference_3class
    from utils.config import CLASSES

    B         = batch["image"].shape[0]
    sz        = BEV["size"]
    nc        = CLASSES["num_classes"]
    out       = np.zeros((B, nc, sz, sz), dtype=np.float32)

    # convert model-input normalised tensor back to BGR for SegFormer inference
    imgs_np = (batch["image"].numpy() * 255).astype(np.uint8)   # (B,3,H,W)

    for i in range(B):
        rgb     = imgs_np[i].transpose(1, 2, 0)                 # (H,W,3) RGB
        bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        K_np    = batch["K"][i].numpy()
        token   = batch["sample_token"][i] if "sample_token" in batch else None

        result  = run_ipm_pipeline(bgr, K_np, sample_token=token,
                                   model=model, device=device)
        out[i]  = result["bev_channels"]

    # convert binary mask to pseudo-logits:
    # 1.0 → +10 (confident positive), 0.0 → -10 (confident negative)
    # this makes sigmoid(logit) ≈ label, compatible with eval threshold 0.5
    logits = (out * 20.0) - 10.0
    return torch.from_numpy(logits).float()