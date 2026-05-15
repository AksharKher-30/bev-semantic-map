import torch
import numpy as np
from tqdm import tqdm

from utils.config import EVAL, CLASSES, PATHS
from utils.device import get_device, move_batch
from data.nuscenes_dataset import build_dataloader
from train.losses import BEVIoUMetric


def evaluate_lss(model, split="val", device=None, verbose=True):
    """
    Evaluate trained LSS model on a dataset split.

    Parameters
    ----------
    model  : LSSModel — loaded with trained weights
    split  : 'val' | 'train' | 'all'
    device : torch.device

    Returns
    -------
    dict: {"drivable_area": float, "vehicle": float, "pedestrian": float, "mIoU": float}
    """
    if device is None:
        device = get_device(verbose=False)

    loader = build_dataloader(split, batch_size=2, shuffle=False)
    metric = BEVIoUMetric()
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval LSS ({split})", disable=not verbose):
            batch  = move_batch(batch, device)
            logits, _ = model(batch["image"], batch["K"], batch["T_cam2ego"])
            metric.update(logits.cpu(), batch["bev_gt"].cpu())

    return metric.compute()


def evaluate_ipm(seg_model, split="val", device=None, verbose=True):
    """
    Evaluate IPM baseline on a dataset split.

    Runs zero-shot SegFormer → 3-class mask → IPM warp → BEV channels.
    Converts binary BEV channels to pseudo-logits for BEVIoUMetric.

    Parameters
    ----------
    seg_model : zero-shot SegFormer (19-class)
    split     : 'val' | 'train' | 'all'

    Returns
    -------
    dict: {"drivable_area": float, "vehicle": float, "pedestrian": float, "mIoU": float}
    """
    import cv2
    from ipm.pipeline import run_ipm_pipeline

    if device is None:
        device = get_device(verbose=False)

    loader = build_dataloader(split, batch_size=1, shuffle=False)
    metric = BEVIoUMetric()
    seg_model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval IPM ({split})", disable=not verbose):
            imgs_np = (batch["image"].numpy() * 255).astype(np.uint8)   # (B,3,H,W)
            B = imgs_np.shape[0]

            for i in range(B):
                rgb = imgs_np[i].transpose(1, 2, 0)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                K   = batch["K"][i].numpy()
                tok = batch["sample_token"][i]

                result = run_ipm_pipeline(
                    bgr, K, sample_token=tok,
                    model=seg_model, device=device,
                )
                # bev_channels: (C, H, W) binary → pseudo-logits
                ch      = torch.from_numpy(result["bev_channels"]).float()
                logits  = (ch * 20.0) - 10.0   # 1→+10, 0→-10
                logits  = logits.unsqueeze(0)   # (1, C, H, W)
                gt      = batch["bev_gt"][i].unsqueeze(0)   # (1, C, H, W)
                metric.update(logits, gt)

    return metric.compute()


def evaluate_both(lss_model, seg_model, split="val", device=None):
    """
    Evaluate both models and return combined results dict.

    Returns
    -------
    dict: {
        "lss": {"drivable_area": ..., "mIoU": ...},
        "ipm": {"drivable_area": ..., "mIoU": ...},
    }
    """
    if device is None:
        device = get_device(verbose=False)

    print("Evaluating LSS model...")
    lss_results = evaluate_lss(lss_model, split=split, device=device)

    print("Evaluating IPM baseline...")
    ipm_results = evaluate_ipm(seg_model, split=split, device=device)

    return {"lss": lss_results, "ipm": ipm_results}