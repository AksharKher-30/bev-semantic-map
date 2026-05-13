#!/usr/bin/env python
# ============================================================
# scripts/verify_phase5.py
# Run AFTER Phase 5 training completes (or at least 1 epoch).
# Run BEFORE starting Phase 6 (evaluation).
#
# Usage: python scripts/verify_phase5.py
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"
failures = []

def check(label, fn):
    try:
        result = fn()
        tag = FAIL if result is False else PASS
        if result is False:
            failures.append(label)
        print(f"{tag}  {label}")
    except Exception as e:
        print(f"{FAIL}  {label}  →  {e}")
        failures.append(label)


# ── imports ───────────────────────────────────────────────────
print("\n── Module imports ───────────────────────────────────────")

def _imports():
    from train.losses       import bev_loss, BEVIoUMetric
    from train.augmentations import augment_lss
    from train.logging_utils import get_writer, log_train_step, log_val_epoch
    from train.checkpointing import save_lss, load_lss, lss_path, LSS_CKPT_NAME
    from train.optimizers    import build_lss_optimizer
check("all Phase 5 modules importable", _imports)


# ── bev_loss ─────────────────────────────────────────────────
print("\n── BEV loss ─────────────────────────────────────────────")

def _bev_loss_scalar():
    from train.losses import bev_loss
    from utils.config import CLASSES, BEV
    B, C = 2, CLASSES["num_classes"]
    logits = torch.randn(B, C, BEV["size"], BEV["size"])
    gt     = torch.zeros(B, C, BEV["size"], BEV["size"])
    loss   = bev_loss(logits, gt)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
check("bev_loss: scalar, finite, positive", _bev_loss_scalar)

def _bev_loss_pos_weights():
    from train.losses import bev_loss
    from utils.config import CLASSES, BEV, TRAIN
    B, C = 1, CLASSES["num_classes"]
    logits = torch.randn(B, C, BEV["size"], BEV["size"])
    gt     = torch.zeros(B, C, BEV["size"], BEV["size"])
    pw     = TRAIN["lss"]["pos_weights"]
    loss   = bev_loss(logits, gt, pw)
    assert torch.isfinite(loss)
check("bev_loss with pos_weights: finite", _bev_loss_pos_weights)


# ── BEVIoUMetric ─────────────────────────────────────────────
print("\n── BEVIoUMetric ─────────────────────────────────────────")

def _metric_perfect():
    from train.losses import BEVIoUMetric
    from utils.config import CLASSES, BEV
    C  = CLASSES["num_classes"]
    m  = BEVIoUMetric()
    gt = torch.ones(1, C, BEV["size"], BEV["size"])
    logits = torch.ones(1, C, BEV["size"], BEV["size"]) * 10.0  # large → sigmoid≈1
    m.update(logits, gt)
    r = m.compute()
    assert abs(r["mIoU"] - 1.0) < 1e-3, f"Perfect pred mIoU={r['mIoU']:.4f} ≠ 1.0"
check("BEVIoUMetric: perfect prediction → mIoU=1.0", _metric_perfect)

def _metric_zero():
    from train.losses import BEVIoUMetric
    from utils.config import CLASSES, BEV
    C  = CLASSES["num_classes"]
    m  = BEVIoUMetric()
    gt     = torch.ones (1, C, BEV["size"], BEV["size"])
    logits = torch.ones (1, C, BEV["size"], BEV["size"]) * -10.0  # all neg
    m.update(logits, gt)
    r = m.compute()
    assert abs(r["mIoU"] - 0.0) < 1e-3, f"All-miss mIoU={r['mIoU']:.4f} ≠ 0.0"
check("BEVIoUMetric: all-miss → mIoU=0.0", _metric_zero)

def _metric_has_class_names():
    from train.losses import BEVIoUMetric
    from utils.config import CLASSES
    m  = BEVIoUMetric()
    m.reset()
    r  = m.compute()
    assert "mIoU" in r
    for name in CLASSES["names"]:
        assert name in r, f"Missing class '{name}' in metric output"
check("BEVIoUMetric result has all class keys + mIoU", _metric_has_class_names)


# ── augmentations ─────────────────────────────────────────────
print("\n── Augmentations ────────────────────────────────────────")

def _augment_shapes():
    from train.augmentations import augment_lss
    from utils.config import SEGFORMER, CLASSES, BEV
    img = torch.randn(3, SEGFORMER["img_h"], SEGFORMER["img_w"])
    gt  = torch.zeros(CLASSES["num_classes"], BEV["size"], BEV["size"])
    img2, gt2 = augment_lss(img, gt)
    assert img2.shape == img.shape
    assert gt2.shape  == gt.shape
check("augment_lss: output shapes unchanged", _augment_shapes)

def _augment_image_range():
    from train.augmentations import augment_lss
    from utils.config import SEGFORMER, CLASSES, BEV
    img = torch.rand(3, SEGFORMER["img_h"], SEGFORMER["img_w"])
    gt  = torch.zeros(CLASSES["num_classes"], BEV["size"], BEV["size"])
    img2, _ = augment_lss(img, gt)
    assert img2.min() >= 0.0 and img2.max() <= 1.0 + 1e-4
check("augment_lss: image values stay in [0,1]", _augment_image_range)


# ── checkpoint ────────────────────────────────────────────────
print("\n── Checkpointing ────────────────────────────────────────")

def _lss_path_correct():
    from train.checkpointing import lss_path, LSS_CKPT_NAME
    assert lss_path().name == LSS_CKPT_NAME
check("lss_path() returns correct filename", _lss_path_correct)

def _save_and_load_lss(tmp_path=None):
    import tempfile, os
    from train.checkpointing import save_lss, load_lss, lss_path
    from train.optimizers import build_lss_optimizer
    from models.lss.lss_model import LSSModel

    model = LSSModel()
    opt, sch = build_lss_optimizer(model)
    # temporarily redirect checkpoint path
    from utils.config import PATHS
    orig = PATHS["checkpoints"]
    with tempfile.TemporaryDirectory() as tmp:
        PATHS["checkpoints"] = Path(tmp)
        save_lss(model, opt, sch, epoch=3, miou=0.25)
        model2 = LSSModel()
        opt2, sch2 = build_lss_optimizer(model2)
        epoch, miou = load_lss(model2, opt2, sch2, device="cpu")
        assert epoch == 3
        assert abs(miou - 0.25) < 1e-5
    PATHS["checkpoints"] = orig
check("save_lss + load_lss round-trip correct", _save_and_load_lss)


# ── LSS training smoke test (1 step) ─────────────────────────
print("\n── LSS training smoke test ──────────────────────────────")

from utils.config import PATHS, NUSCENES
_dataset_ok = (PATHS["dataroot"] / NUSCENES["version"] / "scene.json").exists()

if not _dataset_ok:
    print(f"{WARN}  Dataset not found — skipping training smoke test")
else:
    def _one_step():
        from data.nuscenes_dataset import build_dataloader
        from models.lss.lss_model import LSSModel
        from train.losses import bev_loss
        from train.optimizers import build_lss_optimizer
        from utils.config import TRAIN

        device = torch.device("cpu")
        loader = build_dataloader("val", batch_size=1, shuffle=False)
        batch  = next(iter(loader))

        model = LSSModel().to(device)
        opt, _ = build_lss_optimizer(model)

        model.train()
        imgs   = batch["image"].to(device)
        K      = batch["K"].to(device)
        T      = batch["T_cam2ego"].to(device)
        gt     = batch["bev_gt"].to(device)

        logits, _ = model(imgs, K, T)
        loss = bev_loss(logits, gt, TRAIN["lss"]["pos_weights"])
        loss.backward()
        opt.step()

        assert torch.isfinite(loss)
        assert logits.shape[1:] == (3, 200, 200)
        print(f"       1-step loss={loss.item():.4f}  logits={tuple(logits.shape)}")
    check("LSS 1-step forward+backward on real data", _one_step)


# ── checkpoint file (post-training) ──────────────────────────
print("\n── Checkpoint file ──────────────────────────────────────")

def _checkpoint_exists():
    from train.checkpointing import lss_path
    path = lss_path()
    if not path.exists():
        print(f"{WARN}  No checkpoint yet: {path}")
        print(f"       Run:  bash scripts/run_training_lss.sh")
        return True   # warn only — training may not have started
    size_mb = path.stat().st_size / 1e6
    print(f"       checkpoint size: {size_mb:.1f} MB")
    assert size_mb > 1.0, "Checkpoint suspiciously small"
check("LSS checkpoint exists (warn-only if missing)", _checkpoint_exists)


# ── summary ───────────────────────────────────────────────────
print("\n────────────────────────────────────────────────────────")
if failures:
    print(f"FAILED — {len(failures)} check(s):")
    for f in failures:
        print(f"  • {f}")
    print("Fix before proceeding to Phase 6.\n")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Phase 5 complete. Safe to start Phase 6.\n")
    sys.exit(0)