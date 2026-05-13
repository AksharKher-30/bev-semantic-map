import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from utils.config import TRAIN, CLASSES, PATHS
from utils.device import get_device, move_batch
from data.nuscenes_dataset import build_dataloader
from models.lss.lss_model import LSSModel
from train.losses import bev_loss, BEVIoUMetric
from train.optimizers import build_lss_optimizer
from train.checkpointing import save_lss, load_lss, lss_path
from train.augmentations import augment_lss
from train.logging_utils import get_writer, log_train_step, log_val_epoch, print_epoch


def _apply_augmentation(batch):
    """Apply augment_lss to every sample in a batch (in-place on clones)."""
    imgs = batch["image"].clone()
    gts  = batch["bev_gt"].clone()
    for i in range(imgs.shape[0]):
        imgs[i], gts[i] = augment_lss(imgs[i], gts[i])
    batch["image"]   = imgs
    batch["bev_gt"]  = gts
    return batch


def validate(model, loader, device):
    model.eval()
    metric = BEVIoUMetric()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits, _ = model(batch["image"], batch["K"], batch["T_cam2ego"])
            metric.update(logits.cpu(), batch["bev_gt"].cpu())
    return metric.compute()


def train(resume=False):
    device  = get_device()
    cfg     = TRAIN["lss"]

    train_loader = build_dataloader("train", batch_size=cfg["batch_size"],
                                    shuffle=True)
    val_loader   = build_dataloader("val",   batch_size=cfg["batch_size"],
                                    shuffle=False)

    model = LSSModel().to(device)
    optimizer, scheduler = build_lss_optimizer(model)

    start_epoch = 0
    best_miou   = 0.0
    if resume and lss_path().exists():
        start_epoch, best_miou = load_lss(model, optimizer, scheduler, device)

    writer     = get_writer("lss_nuscenes")
    pos_weights = cfg["pos_weights"]
    total_epochs = cfg["epochs"]

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader,
                                          desc=f"Epoch {epoch+1}/{total_epochs}")):
            batch  = _apply_augmentation(batch)
            batch  = move_batch(batch, device)

            logits, _ = model(batch["image"], batch["K"], batch["T_cam2ego"])
            loss      = bev_loss(logits, batch["bev_gt"], pos_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            epoch_loss  += loss.item()
            global_step  = epoch * len(train_loader) + step
            lr = optimizer.param_groups[0]["lr"]
            log_train_step(writer, loss.item(), global_step, lr)

        avg_loss = epoch_loss / len(train_loader)
        results  = validate(model, val_loader, device)

        scheduler.step()
        log_val_epoch(writer, results, epoch)
        print_epoch(epoch + 1, total_epochs, avg_loss, results)

        if results["mIoU"] > best_miou:
            best_miou = results["mIoU"]
            save_lss(model, optimizer, scheduler, epoch + 1, best_miou)

    writer.close()
    print(f"\nTraining done. Best val mIoU: {best_miou:.4f}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)