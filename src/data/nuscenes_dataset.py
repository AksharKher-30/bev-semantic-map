# ============================================================
# src/data/nuscenes_dataset.py
# ============================================================
# PyTorch Dataset - one item per nuScenes sample.
#
# Each __getitem__ returns a dict with ALL tensors needed by
# every downstream phase:
#   Phase 2  (SegFormer): 'image'
#   Phase 3  (IPM):       'image', 'K', 'T_cam2ego'
#   Phase 4-5 (LSS):      'image', 'K', 'T_cam2ego', 'T_ego2world'
#   Phase 6  (eval):      'bev_gt', 'sample_token', 'location'
#   Phase 7  (video):     all of the above
#
# HOW TO CHANGE:
#   • Train/val split ratio → edit SPLIT_RATIO below.
#   • Image resize         → edit IMG_H / IMG_W in config.py SEGFORMER.
#   • Add augmentation     → import from train/augmentations.py and
#                            pass transform=... to the constructor.
#   • Use all 6 cameras    → extend get_camera_data() calls (Phase 4+).
# ============================================================

from __future__ import annotations

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

from utils.config import NUSCENES, SEGFORMER, BEV
from data.nuscenes_loader import (
    get_nusc,
    get_all_scene_tokens,
    iterate_scene_samples,
    get_camera_data,
    get_scene_location,
)
from data.bev_gt_generator import generate_bev_gt
from data.calibration import validate_intrinsics, validate_transform


# ── Split config ──────────────────────────────────────────────
# nuScenes mini has 10 scenes → 8 train, 2 val (no official mini split)
SPLIT_RATIO = 0.8    # change here only - affects both loaders


def _get_scene_split() -> tuple[list[str], list[str]]:
    """
    Split scenes into train / val by SPLIT_RATIO.
    Returns (train_scene_tokens, val_scene_tokens).
    Scene order is deterministic (sorted by token string).
    """
    all_scenes = sorted(get_all_scene_tokens())
    n_train    = int(len(all_scenes) * SPLIT_RATIO)
    return all_scenes[:n_train], all_scenes[n_train:]


def get_sample_tokens(split: str) -> list[str]:
    """
    Return all sample tokens for a given split ('train' | 'val' | 'all').

    Parameters
    ----------
    split : str - 'train', 'val', or 'all'
    """
    train_scenes, val_scenes = _get_scene_split()
    if split == "train":
        scene_tokens = train_scenes
    elif split == "val":
        scene_tokens = val_scenes
    elif split == "all":
        scene_tokens = sorted(get_all_scene_tokens())
    else:
        raise ValueError(f"split must be 'train', 'val', or 'all'. Got: '{split}'")

    tokens = []
    for sc in scene_tokens:
        tokens.extend(iterate_scene_samples(sc))
    return tokens


# ── Image preprocessing ───────────────────────────────────────

def preprocess_image(
    image_path: Path,
    img_h: int = SEGFORMER["img_h"],
    img_w: int = SEGFORMER["img_w"],
) -> torch.Tensor:
    """
    Load, resize, and normalise a camera image.

    Resize: 900×1600 → img_h × img_w (default 512×1024 for SegFormer).
    Normalise: divide by 255.0 → [0.0, 1.0] float32.
    Layout: HWC (OpenCV) → CHW (PyTorch).

    Parameters
    ----------
    image_path : Path
    img_h, img_w : int - target height / width in pixels

    Returns
    -------
    tensor : (3, img_h, img_w) float32 in [0, 1]
    """
    bgr   = cv2.imread(str(image_path))
    assert bgr is not None, f"Could not read image: {image_path}"
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb   = cv2.resize(rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0   # (3,H,W)
    return tensor


# ── Dataset class ─────────────────────────────────────────────

class NuScenesBEVDataset(Dataset):
    """
    PyTorch Dataset for nuScenes BEV semantic mapping.

    One item per keyframe sample. BEV GT is generated on-the-fly
    (no pre-caching needed for 400-sample mini split).

    Parameters
    ----------
    split : str
        'train', 'val', or 'all'.
    transform : callable | None
        Optional augmentation function accepting and returning
        (image_tensor, bev_gt_array). Applied only when split='train'.
        Implementation lives in train/augmentations.py (Phase 5).
    img_h, img_w : int
        Resize target for front-camera image.
    bev_size, bev_res : int / float
        BEV grid parameters (default from config).
    """

    def __init__(
        self,
        split:    str   = "train",
        transform        = None,
        img_h:    int   = SEGFORMER["img_h"],
        img_w:    int   = SEGFORMER["img_w"],
        bev_size: int   = BEV["size"],
        bev_res:  float = BEV["resolution"],
    ):
        self.split        = split
        self.transform    = transform
        self.img_h        = img_h
        self.img_w        = img_w
        self.bev_size     = bev_size
        self.bev_res      = bev_res
        self.sample_tokens = get_sample_tokens(split)

        print(f"[Dataset] split={split}  samples={len(self.sample_tokens)}")

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        dict with keys:
            image        : (3, H, W)       float32 tensor  - normalised RGB
            K            : (3, 3)          float32 tensor  - camera intrinsics
            T_cam2ego    : (4, 4)          float32 tensor  - sensor→ego
            T_ego2world  : (4, 4)          float32 tensor  - ego→world
            bev_gt       : (C, bev_size, bev_size) float32 - binary GT per class
            sample_token : str             - nuScenes token (for traceability)
            location     : str             - 'boston-seaport' | 'singapore-*'
        """
        token    = self.sample_tokens[idx]
        cam_data = get_camera_data(token)

        # ── Image ─────────────────────────────────────────────
        image = preprocess_image(
            cam_data["image_path"], self.img_h, self.img_w
        )

        # ── Calibration tensors ────────────────────────────────
        K           = torch.from_numpy(cam_data["K"]).float()
        T_cam2ego   = torch.from_numpy(cam_data["T_cam2ego"]).float()
        T_ego2world = torch.from_numpy(cam_data["T_ego2world"]).float()

        # ── BEV ground truth ───────────────────────────────────
        bev_gt_np = generate_bev_gt(token, self.bev_size, self.bev_res)
        bev_gt    = torch.from_numpy(bev_gt_np).float()   # (C,H,W)

        # ── Optional augmentation (Phase 5 attaches this) ──────
        if self.transform is not None and self.split == "train":
            image, bev_gt = self.transform(image, bev_gt)

        # ── Location (for terrain-stratified eval in Phase 6) ──
        nusc     = get_nusc()
        sample   = nusc.get("sample", token)
        location = get_scene_location(sample["scene_token"])

        return {
            "image"        : image,
            "K"            : K,
            "T_cam2ego"    : T_cam2ego,
            "T_ego2world"  : T_ego2world,
            "bev_gt"       : bev_gt,
            "sample_token" : token,
            "location"     : location,
        }


# ── DataLoader factory ────────────────────────────────────────

def build_dataloader(
    split:      str   = "train",
    batch_size: int   = 2,
    shuffle:    bool  = True,
    transform          = None,
) -> torch.utils.data.DataLoader:
    """
    Convenience factory used by train scripts and eval scripts.

    MPS-safe defaults:
        num_workers=0  (MPS + multiprocessing = crash on macOS)
        pin_memory=False

    Parameters
    ----------
    split      : 'train' | 'val' | 'all'
    batch_size : int
    shuffle    : bool - True for train, False for val
    transform  : optional augmentation callable

    Returns
    -------
    DataLoader
    """
    from utils.config import TRAIN
    dataset = NuScenesBEVDataset(split=split, transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = TRAIN["num_workers"],    # 0 for MPS
        pin_memory  = TRAIN["pin_memory"],     # False for MPS
        drop_last   = (split == "train"),
    )