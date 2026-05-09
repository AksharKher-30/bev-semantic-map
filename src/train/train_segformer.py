import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Add the 'src' directory to the python path
src_dir = Path(__file__).resolve().parents[1] 
sys.path.append(str(src_dir))

import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.config import SEGFORMER, TRAIN, PATHS, CLASSES
from utils.device import get_device
from data.nuscenes_dataset import get_sample_tokens
from data.nuscenes_loader import get_camera_data
from data.pseudo_labels import generate_pseudo_label
from data.class_mappings import num_classes
from models.segformer.model import build_segformer
from train.losses import segformer_loss, compute_iou
from train.optimizers import build_segformer_optimizer
from train.checkpointing import save, load, best_path
from data.nuscenes_dataset import NuScenesBEVDataset


# ── class weights: road is dense, vehicle/ped are rare ────────
# road=0.5, vehicle=8.0, pedestrian=12.0, sky=0.5, background=0.3
# heavy upweighting of rare classes forces the model to learn them
LOSS_WEIGHTS = [1.0, 2.5, 8.0, 1.0, 0.5]


def augment(image, label):
    """
    On-the-fly augmentation applied consistently to image + label.
    Only geometric augmentations applied to both; colour only to image.
    """
    # horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        label = torch.flip(label, dims=[-1])

    # colour jitter (image only)
    image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
    image = TF.adjust_contrast(image,   random.uniform(0.85, 1.15))
    image = TF.adjust_saturation(image, random.uniform(0.85, 1.15))
    return image, label


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, split, pseudo_labels, augment_fn=None):
        self.base         = NuScenesBEVDataset(split=split)
        self.labels       = pseudo_labels
        self.augment_fn   = augment_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item  = self.base[idx]
        token = item["sample_token"]
        label = torch.from_numpy(self.labels[token]).long()

        if self.augment_fn is not None:
            item["image"], label = self.augment_fn(item["image"], label)

        item["seg_label"] = label
        return item


def build_pseudo_labels(split):
    tokens = get_sample_tokens(split)
    print(f"Generating pseudo labels for {split} ({len(tokens)} samples)...")
    labels = {}
    for t in tqdm(tokens, desc="pseudo-labels"):
        labels[t] = generate_pseudo_label(t)

    # quick sanity: count labeled pixels
    total = labeled = veh = ped = 0
    for lbl in labels.values():
        total   += lbl.size
        labeled += (lbl != 255).sum()
        veh     += (lbl == 1).sum()
        ped     += (lbl == 2).sum()

    density = labeled / total * 100
    print(f"  labeled: {density:.1f}%   vehicle px: {veh:,}   pedestrian px: {ped:,}")

    if veh == 0:
        print("  WARNING: zero vehicle pixels — check pseudo_labels.py box projection")
    return labels


def validate(model, loader, device):
    model.eval()
    n_cls    = num_classes()
    all_iou  = [[] for _ in range(n_cls)]

    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["seg_label"].to(device)

            logits = model(pixel_values=imgs).logits
            up     = F.interpolate(logits, size=labels.shape[-2:],
                                   mode="bilinear", align_corners=False)
            preds  = up.argmax(dim=1)
            ious   = compute_iou(preds, labels, n_cls,
                                 ignore_index=SEGFORMER["ignore_index"])
            for c, v in enumerate(ious):
                all_iou[c].append(v)

    per_class = [float(np.mean(v)) for v in all_iou]
    return float(np.mean(per_class)), per_class


def train(resume=False):
    device = get_device()
    cfg    = SEGFORMER

    train_labels = build_pseudo_labels("train")
    val_labels   = build_pseudo_labels("val")

    train_ds = SegDataset("train", train_labels, augment_fn=augment)
    val_ds   = SegDataset("val",   val_labels,   augment_fn=None)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=TRAIN["num_workers"], drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=TRAIN["num_workers"],
    )

    model = build_segformer(pretrained=True).to(device)
    optimizer, scheduler = build_segformer_optimizer(model, len(train_loader))

    # freeze backbone for first 2 epochs — stabilise head training first
    # FREEZE_EPOCHS = 2
    # for p in model.segformer.parameters():
    #     p.requires_grad = False
    # print(f"Backbone frozen for first {FREEZE_EPOCHS} epochs.")

    start_epoch = 0
    best_miou   = 0.0
    if resume and best_path("segformer_nuscenes.pth").exists():
        start_epoch, best_miou = load(
            best_path("segformer_nuscenes.pth"), model, optimizer, device
        )
        print(f"Resumed from epoch {start_epoch}, best mIoU={best_miou:.4f}")

    writer = SummaryWriter(str(PATHS["runs"] / "segformer_finetune"))

    for epoch in range(start_epoch, cfg["epochs"]):

        # unfreeze backbone after FREEZE_EPOCHS
        # if epoch == FREEZE_EPOCHS:
        #     for p in model.segformer.parameters():
        #         p.requires_grad = True
        #     print(f"Epoch {epoch+1}: backbone unfrozen.")

        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")):
            imgs   = batch["image"].to(device)
            labels = batch["seg_label"].to(device)

            logits = model(pixel_values=imgs).logits
            loss   = segformer_loss(logits, labels, LOSS_WEIGHTS)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("train/loss", loss.item(),
                               epoch * len(train_loader) + step)

        avg_loss = epoch_loss / len(train_loader)
        miou, per_class = validate(model, val_loader, device)
        scheduler.step()

        writer.add_scalar("val/mIoU", miou, epoch)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}  val_mIoU={miou:.4f}")
        for name, iou in zip(CLASSES["names"], per_class):
            print(f"  {name}: {iou:.4f}")

        if miou > best_miou:
            best_miou = miou
            save(model, optimizer, epoch + 1, best_miou, "segformer_nuscenes.pth")

    writer.close()
    print(f"\nTraining done. Best mIoU: {best_miou:.4f}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)