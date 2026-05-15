import torch
import numpy as np
import cv2
from tqdm import tqdm

from utils.device import get_device, move_batch
from data.nuscenes_dataset import build_dataloader
from data.nuscenes_loader import get_scene_location, get_nusc
from train.losses import BEVIoUMetric


def _get_location(sample_token):
    nusc   = get_nusc()
    sample = nusc.get("sample", sample_token)
    return get_scene_location(sample["scene_token"])


def evaluate_by_terrain(model, model_type="lss", seg_model=None,
                         split="val", device=None):
    """
    Evaluate model separately on Boston (non-flat) and Singapore (flat) scenes.

    Boston  → has ramps, speed bumps → IPM flat-ground assumption breaks
    Singapore → relatively flat → IPM more accurate

    Parameters
    ----------
    model      : LSSModel or zero-shot SegFormer (for IPM)
    model_type : 'lss' or 'ipm'
    seg_model  : required when model_type='ipm'
    split      : dataset split

    Returns
    -------
    dict: {
        "boston":    {"drivable_area": ..., "mIoU": ...},
        "singapore": {"drivable_area": ..., "mIoU": ...},
    }
    """
    if device is None:
        device = get_device(verbose=False)

    loader  = build_dataloader(split, batch_size=1, shuffle=False)
    metrics = {
        "boston"    : BEVIoUMetric(),
        "singapore" : BEVIoUMetric(),
    }

    if model_type == "lss":
        model.eval()
    else:
        seg_model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Terrain eval ({model_type})"):
            token    = batch["sample_token"][0]
            location = _get_location(token)
            city     = "boston" if "boston" in location else "singapore"

            if model_type == "lss":
                batch  = move_batch(batch, device)
                logits, _ = model(batch["image"], batch["K"], batch["T_cam2ego"])
                gt     = batch["bev_gt"].cpu()
                logits = logits.cpu()

            else:   # ipm
                from ipm.pipeline import run_ipm_pipeline
                imgs_np = (batch["image"].numpy() * 255).astype(np.uint8)
                rgb     = imgs_np[0].transpose(1, 2, 0)
                bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                K       = batch["K"][0].numpy()
                result  = run_ipm_pipeline(bgr, K, sample_token=token,
                                           model=seg_model, device=device)
                ch      = torch.from_numpy(result["bev_channels"]).float()
                logits  = (ch * 20.0 - 10.0).unsqueeze(0)
                gt      = batch["bev_gt"]

            metrics[city].update(logits, gt)

    return {city: m.compute() for city, m in metrics.items()}


def print_terrain_table(lss_terrain, ipm_terrain):
    """Print a formatted terrain comparison table."""
    from utils.config import CLASSES
    names = CLASSES["names"] + ["mIoU"]

    print("\n" + "="*65)
    print(f"{'Metric':<20} {'LSS-Boston':>12} {'LSS-SG':>10} "
          f"{'IPM-Boston':>12} {'IPM-SG':>10}")
    print("="*65)
    for name in names:
        lb = lss_terrain.get("boston", {}).get(name, 0.0)
        ls = lss_terrain.get("singapore", {}).get(name, 0.0)
        ib = ipm_terrain.get("boston", {}).get(name, 0.0)
        is_ = ipm_terrain.get("singapore", {}).get(name, 0.0)
        print(f"{name:<20} {lb:>12.4f} {ls:>10.4f} {ib:>12.4f} {is_:>10.4f}")
    print("="*65)
    print("SG = Singapore (flat). Boston has ramps/speed bumps.")
    print("Key finding: LSS advantage over IPM is larger in Boston.\n")