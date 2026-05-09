import torch
from pathlib import Path
from utils.config import PATHS


def save(model, optimizer, epoch, miou, name="segformer_nuscenes.pth"):
    path = PATHS["checkpoints"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch":     epoch,
        "best_miou": miou,
    }, path)
    print(f"[ckpt] saved → {path}  (epoch={epoch}, mIoU={miou:.4f})")


def load(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)


def best_path(name="segformer_nuscenes.pth"):
    return PATHS["checkpoints"] / name