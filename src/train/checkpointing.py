import torch
from pathlib import Path
from utils.config import PATHS

LSS_CKPT_NAME       = "lss_best.pth"
SEGFORMER_CKPT_NAME = "segformer_nuscenes.pth"


def save(model, optimizer, epoch, miou, name=SEGFORMER_CKPT_NAME):
    path = PATHS["checkpoints"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch":     epoch,
        "best_miou": miou,
    }, path)
    print(f"[ckpt] saved → {path}  (epoch={epoch}, mIoU={miou:.4f})")


def save_lss(model, optimizer, scheduler, epoch, miou):
    path = PATHS["checkpoints"] / LSS_CKPT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch":     epoch,
        "best_miou": miou,
    }, path)
    print(f"[ckpt] LSS saved → {path}  (epoch={epoch}, mIoU={miou:.4f})")


def load(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)


def load_lss(model, optimizer=None, scheduler=None, device="cpu"):
    path = PATHS["checkpoints"] / LSS_CKPT_NAME
    if not path.exists():
        return 0, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer  is not None and "optimizer"  in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler  is not None and "scheduler"  in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch = ckpt.get("epoch", 0)
    miou  = ckpt.get("best_miou", 0.0)
    print(f"[ckpt] LSS resumed from epoch {epoch}, best mIoU={miou:.4f}")
    return epoch, miou


def best_path(name=SEGFORMER_CKPT_NAME):
    return PATHS["checkpoints"] / name


def lss_path():
    return PATHS["checkpoints"] / LSS_CKPT_NAME