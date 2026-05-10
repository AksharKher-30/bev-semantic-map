#!/usr/bin/env python
# ============================================================
# scripts/verify_phase4.py
# Run AFTER all Phase 4 code is written.
# Run BEFORE starting Phase 5 (LSS training).
#
# Usage: python scripts/verify_phase4.py
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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

def _lss_imports():
    from models.lss.depth_bins import get_depth_bins, get_depth_bin_size
    from models.lss.lift       import LiftModule
    from models.lss.splat      import SplatModule
    from models.lss.shoot      import ShootModule
    from models.lss.lss_model  import LSSModel
check("all LSS modules importable", _lss_imports)


# ── depth bins ────────────────────────────────────────────────
print("\n── Depth bins ───────────────────────────────────────────")

def _bins_shape():
    from models.lss.depth_bins import get_depth_bins
    from utils.config import LSS, BEV
    bins = get_depth_bins()
    assert bins.shape == (BEV["d_bins"],)
check("depth bins shape == (D,)", _bins_shape)

def _bins_range():
    from models.lss.depth_bins import get_depth_bins
    from utils.config import LSS, BEV
    bins = get_depth_bins()
    assert abs(bins[0].item()  - BEV["d_min"]) < 1e-4
    assert abs(bins[-1].item() - BEV["d_max"]) < 1e-4
check("depth bins span [d_min, d_max]", _bins_range)


# ── lift ──────────────────────────────────────────────────────
print("\n── Lift module ──────────────────────────────────────────")

def _lift_shape():
    from models.lss.lift import LiftModule
    from utils.config import LSS, BEV, SEGFORMER
    model = LiftModule().eval()
    x     = torch.randn(1, 3, SEGFORMER["img_h"], SEGFORMER["img_w"])
    with torch.no_grad():
        feats, depth = model(x)
    assert feats.ndim  == 5
    assert feats.shape[1] == BEV["d_bins"]
    assert feats.shape[2] == LSS["feature_channels"]
    assert depth.shape[1] == BEV["d_bins"]
    print(f"       frustum: {tuple(feats.shape)}  depth: {tuple(depth.shape)}")
check("Lift output shapes correct", _lift_shape)

def _lift_depth_sums_one():
    from models.lss.lift import LiftModule
    from utils.config import SEGFORMER
    model = LiftModule().eval()
    x     = torch.randn(1, 3, SEGFORMER["img_h"], SEGFORMER["img_w"])
    with torch.no_grad():
        _, depth = model(x)
    err = (depth.sum(dim=1) - 1.0).abs().max().item()
    assert err < 1e-4, f"depth dist sum error: {err:.2e}"
check("Lift depth distribution sums to 1.0", _lift_depth_sums_one)


# ── splat ─────────────────────────────────────────────────────
print("\n── Splat module ─────────────────────────────────────────")

def _splat_shape():
    from models.lss.lift  import LiftModule
    from models.lss.splat import SplatModule
    from utils.config import LSS, BEV, SEGFORMER
    lift  = LiftModule().eval()
    splat = SplatModule().eval()
    B = 2
    x     = torch.randn(B, 3, SEGFORMER["img_h"], SEGFORMER["img_w"])
    K     = torch.eye(3).unsqueeze(0).repeat(B, 1, 1) * 1266.0
    T     = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    with torch.no_grad():
        feats, _ = lift(x)
        bev      = splat(feats, K, T)
    expected = (B, LSS["feature_channels"], BEV["size"], BEV["size"])
    assert tuple(bev.shape) == expected, f"Got {tuple(bev.shape)}, expected {expected}"
    assert torch.isfinite(bev).all()
    print(f"       bev feats: {tuple(bev.shape)}")
check("Splat output shape + finite values", _splat_shape)


# ── shoot ─────────────────────────────────────────────────────
print("\n── Shoot module ─────────────────────────────────────────")

def _shoot_shape():
    from models.lss.shoot import ShootModule
    from utils.config import LSS, BEV, CLASSES
    shoot  = ShootModule().eval()
    bev    = torch.randn(2, LSS["feature_channels"], BEV["size"], BEV["size"])
    with torch.no_grad():
        logits = shoot(bev)
    expected = (2, CLASSES["num_classes"], BEV["size"], BEV["size"])
    assert tuple(logits.shape) == expected
    assert torch.isfinite(logits).all()
    print(f"       logits: {tuple(logits.shape)}")
check("Shoot output shape + finite values", _shoot_shape)


# ── full model ────────────────────────────────────────────────
print("\n── Full LSSModel ────────────────────────────────────────")

def _full_forward():
    from models.lss.lss_model import LSSModel
    from utils.config import CLASSES, BEV, SEGFORMER
    model  = LSSModel().eval()
    B      = 2
    x      = torch.randn(B, 3, SEGFORMER["img_h"], SEGFORMER["img_w"])
    K      = torch.eye(3).unsqueeze(0).repeat(B, 1, 1) * 1266.0
    T      = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    with torch.no_grad():
        logits, depth = model(x, K, T)
    expected = (B, CLASSES["num_classes"], BEV["size"], BEV["size"])
    assert tuple(logits.shape) == expected, \
        f"Critical shape contract failed: got {tuple(logits.shape)}, need {expected}"
    assert torch.isfinite(logits).all()
    assert logits.abs().max().item() > 0.0
    print(f"       logits: {tuple(logits.shape)}  ✓  (B, {CLASSES['num_classes']}, {BEV['size']}, {BEV['size']})")
check("LSSModel full forward → correct output shape", _full_forward)

def _backward_pass():
    from models.lss.lss_model import LSSModel
    from utils.config import CLASSES, BEV, SEGFORMER
    model  = LSSModel()
    B      = 1
    x      = torch.randn(B, 3, SEGFORMER["img_h"], SEGFORMER["img_w"])
    K      = torch.eye(3).unsqueeze(0) * 1266.0
    T      = torch.eye(4).unsqueeze(0)
    logits, _ = model(x, K, T)
    gt     = torch.zeros(B, CLASSES["num_classes"], BEV["size"], BEV["size"])
    loss   = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters()
                  if p.grad is not None and p.requires_grad]
    assert len(grad_norms) > 0, "No gradients computed"
    assert all(not (g != g) for g in grad_norms), "NaN gradients"
check("Backward pass: gradients flow, no NaN", _backward_pass)


# ── pytest ────────────────────────────────────────────────────
print("\n── Pytest ───────────────────────────────────────────────")

def _run_pytest():
    import subprocess
    root   = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_lss_forward.py",
         "-v", "--tb=short", "-q"],
        cwd=str(root), capture_output=True, text=True
    )
    out = result.stdout
    print(out[-3000:] if len(out) > 3000 else out)
    if result.returncode != 0:
        print(result.stderr[-500:])
        raise AssertionError("pytest failures")
check("pytest test_lss_forward all pass", _run_pytest)


# ── summary ───────────────────────────────────────────────────
print("\n────────────────────────────────────────────────────────")
if failures:
    print(f"FAILED — {len(failures)} check(s):")
    for f in failures:
        print(f"  • {f}")
    print("Fix before proceeding to Phase 5.\n")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Phase 4 complete. Safe to start Phase 5.\n")
    sys.exit(0)