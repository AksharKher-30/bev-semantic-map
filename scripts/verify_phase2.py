#!/usr/bin/env python
# ============================================================
# scripts/verify_phase2.py
# Run AFTER all Phase 2 code is written.
# Run BEFORE starting Phase 3 (IPM baseline).
#
# Usage: python scripts/verify_phase2.py
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

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

def _model_imports():
    from models.segformer.model import build_segformer, load_checkpoint
    from models.segformer.inference import run_inference, run_inference_tensor
check("models.segformer importable", _model_imports)

def _train_imports():
    from train.losses import segformer_loss, compute_iou
    from train.optimizers import build_segformer_optimizer
    from train.checkpointing import save, load, best_path
check("train modules importable", _train_imports)

def _viz_imports():
    from viz.colorize import colorize_seg, colorize_bev, overlay_seg
check("viz.colorize importable", _viz_imports)

def _pseudo_imports():
    from data.pseudo_labels import generate_pseudo_label, IGNORE
check("data.pseudo_labels importable", _pseudo_imports)


# ── model ─────────────────────────────────────────────────────
print("\n── SegFormer model ──────────────────────────────────────")

def _build_model():
    from models.segformer.model import build_segformer
    from utils.config import SEGFORMER
    model = build_segformer(pretrained=False)
    out_ch = model.decode_head.classifier.out_channels
    assert out_ch == SEGFORMER["num_classes"]
check("build_segformer: head has correct output channels", _build_model)

def _forward_shape():
    from models.segformer.model import build_segformer
    from utils.config import SEGFORMER
    model = build_segformer(pretrained=False).eval()
    x = torch.randn(1, 3, SEGFORMER["img_h"], SEGFORMER["img_w"])
    with torch.no_grad():
        logits = model(pixel_values=x).logits
    assert logits.shape[1] == SEGFORMER["num_classes"]
    assert torch.isfinite(logits).all()
check("forward pass: correct channels + finite values", _forward_shape)

def _inference_shape():
    from models.segformer.model import build_segformer
    from models.segformer.inference import run_inference_tensor
    from utils.config import SEGFORMER
    model  = build_segformer(pretrained=False)
    tensor = torch.randn(3, SEGFORMER["img_h"], SEGFORMER["img_w"])
    mask   = run_inference_tensor(model, tensor, device=torch.device("cpu"))
    assert mask.shape == (SEGFORMER["img_h"], SEGFORMER["img_w"])
    assert mask.dtype == np.uint8
    assert mask.max() < SEGFORMER["num_classes"]
check("run_inference_tensor: shape + dtype + value range", _inference_shape)


# ── losses ────────────────────────────────────────────────────
print("\n── Losses & metrics ─────────────────────────────────────")

def _loss_scalar():
    from train.losses import segformer_loss
    from utils.config import SEGFORMER
    logits = torch.randn(2, SEGFORMER["num_classes"], 64, 128)
    labels = torch.zeros(2, SEGFORMER["img_h"], SEGFORMER["img_w"], dtype=torch.long)
    loss   = segformer_loss(logits, labels)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
check("segformer_loss: scalar, finite, positive", _loss_scalar)

def _iou_perfect():
    from train.losses import compute_iou
    from utils.config import SEGFORMER
    p = torch.zeros(2, 4, 4, dtype=torch.long)
    l = torch.zeros(2, 4, 4, dtype=torch.long)
    ious = compute_iou(p, l, SEGFORMER["num_classes"])
    assert abs(ious[0] - 1.0) < 1e-4
check("compute_iou: perfect pred → IoU=1.0", _iou_perfect)

def _iou_zero():
    from train.losses import compute_iou
    from utils.config import SEGFORMER
    p = torch.zeros(2, 4, 4, dtype=torch.long)
    l = torch.ones(2, 4, 4, dtype=torch.long)
    ious = compute_iou(p, l, SEGFORMER["num_classes"])
    assert abs(ious[0] - 0.0) < 1e-4
check("compute_iou: no overlap → IoU=0.0", _iou_zero)


# ── colorize ──────────────────────────────────────────────────
print("\n── Visualization ────────────────────────────────────────")

def _colorize_seg():
    from viz.colorize import colorize_seg
    mask = np.zeros((512, 1024), dtype=np.uint8)
    out  = colorize_seg(mask)
    assert out.shape == (512, 1024, 3) and out.dtype == np.uint8
check("colorize_seg: shape + dtype", _colorize_seg)

def _colorize_bev():
    from viz.colorize import colorize_bev
    from utils.config import CLASSES
    bev = np.zeros((CLASSES["num_classes"], 200, 200), dtype=np.float32)
    assert colorize_bev(bev).shape == (200, 200, 3)
check("colorize_bev: multi-channel input → (H,W,3)", _colorize_bev)

def _overlay_seg():
    from viz.colorize import overlay_seg
    img  = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    mask = np.zeros((512, 1024), dtype=np.uint8)
    out  = overlay_seg(img, mask)
    assert out.shape == (512, 1024, 3)
check("overlay_seg: blended output shape correct", _overlay_seg)


# ── pseudo labels (dataset required) ─────────────────────────
print("\n── Pseudo labels ────────────────────────────────────────")

from utils.config import PATHS, NUSCENES, SEGFORMER
_dataset_ok = (PATHS["dataroot"] / NUSCENES["version"] / "scene.json").exists()

if not _dataset_ok:
    print(f"{WARN}  Dataset not found — skipping pseudo label checks")
else:
    def _pseudo_shape():
        from data.pseudo_labels import generate_pseudo_label
        from data.nuscenes_loader import get_all_sample_tokens
        token = get_all_sample_tokens()[0]
        label = generate_pseudo_label(token)
        assert label.shape == (SEGFORMER["img_h"], SEGFORMER["img_w"])
        assert label.dtype == np.uint8
    check("pseudo label shape + dtype", _pseudo_shape)

    def _pseudo_values():
        from data.pseudo_labels import generate_pseudo_label, IGNORE
        from data.nuscenes_loader import get_all_sample_tokens
        token  = get_all_sample_tokens()[0]
        label  = generate_pseudo_label(token)
        unique = set(np.unique(label).tolist())
        assert unique.issubset({0, 1, 2, IGNORE})
    check("pseudo label values in {0,1,2,255}", _pseudo_values)

    def _pseudo_not_empty():
        from data.pseudo_labels import generate_pseudo_label, IGNORE
        from data.nuscenes_loader import get_all_sample_tokens
        token = get_all_sample_tokens()[0]
        label = generate_pseudo_label(token)
        labeled = (label != IGNORE).sum()
        assert labeled > 0, "All pixels are ignore — BEV projection failed"
        print(f"       labeled pixels: {labeled} / {label.size}")
    check("pseudo label has non-ignore pixels", _pseudo_not_empty)


# ── pytest ────────────────────────────────────────────────────
print("\n── Pytest ───────────────────────────────────────────────")

def _run_pytest():
    import subprocess
    root   = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_segformer.py",
         "-v", "--tb=short", "-q"],
        cwd=str(root), capture_output=True, text=True
    )
    out = result.stdout
    print(out[-3000:] if len(out) > 3000 else out)
    if result.returncode != 0:
        print(result.stderr[-500:])
        raise AssertionError("pytest failures")
check("pytest test_segformer all pass", _run_pytest)


# ── summary ───────────────────────────────────────────────────
print("\n────────────────────────────────────────────────────────")
if failures:
    print(f"FAILED — {len(failures)} check(s):")
    for f in failures:
        print(f"  • {f}")
    print("Fix before proceeding to Phase 3.\n")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Phase 2 complete. Safe to start Phase 3.\n")
    sys.exit(0)