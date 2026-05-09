#!/usr/bin/env python
# ============================================================
# scripts/verify_phase3.py
# Run AFTER all Phase 3 code is written.
# Run BEFORE starting Phase 4 (LSS architecture).
#
# Usage: python scripts/verify_phase3.py
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import cv2

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

def _ipm_imports():
    from ipm.homography import compute_ipm_homography, get_camera_height_from_nusc
    from ipm.warp       import apply_ipm, bev_mask_to_channels
    from ipm.pipeline   import run_ipm_pipeline, run_ipm_batch
check("ipm package all functions importable", _ipm_imports)


# ── homography ────────────────────────────────────────────────
print("\n── Homography ───────────────────────────────────────────")

K = np.array([
    [1266.417,    0.0,    816.267],
    [   0.0,  1266.417,  491.507],
    [   0.0,     0.0,      1.0  ],
], dtype=np.float64)

def _H_shape():
    from ipm.homography import compute_ipm_homography
    H = compute_ipm_homography(K)
    assert H.shape == (3, 3) and H.dtype == np.float32
check("H shape (3,3) float32", _H_shape)

def _H_finite():
    from ipm.homography import compute_ipm_homography
    H = compute_ipm_homography(K)
    assert np.isfinite(H).all()
check("H values all finite", _H_finite)

def _H_invertible():
    from ipm.homography import compute_ipm_homography
    H     = compute_ipm_homography(K)
    H_inv = np.linalg.inv(H.astype(np.float64))
    err   = np.abs(H.astype(np.float64) @ H_inv - np.eye(3)).max()
    assert err < 1e-4, f"H @ H_inv max error: {err:.2e}"
check("H @ inv(H) = I  (invertible)", _H_invertible)

def _H_forward_point():
    from ipm.homography import compute_ipm_homography
    from utils.config import SEGFORMER, BEV
    H  = compute_ipm_homography(K, bev_size=200, bev_res=0.5)
    h  = BEV["camera_height"]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    sx = SEGFORMER["img_w"] / 1600.0
    sy = SEGFORMER["img_h"] / 900.0
    x, y = 0.0, 20.0
    u = (fx * (x/y) + cx) * sx
    v = (fy * (-h/y) + cy) * sy
    pt = np.array([[[u, v]]], dtype=np.float32)
    pt_bev = cv2.perspectiveTransform(pt, H)[0, 0]
    assert pt_bev[1] < 100, f"20m-ahead point row={pt_bev[1]:.1f} should be < 100 (centre)"
check("20m forward point maps above BEV centre", _H_forward_point)


# ── warp ──────────────────────────────────────────────────────
print("\n── Warp ─────────────────────────────────────────────────")

def _warp_shape():
    from ipm.homography import compute_ipm_homography
    from ipm.warp import apply_ipm
    from utils.config import BEV, SEGFORMER
    H    = compute_ipm_homography(K)
    mask = np.zeros((SEGFORMER["img_h"], SEGFORMER["img_w"]), dtype=np.uint8)
    out  = apply_ipm(mask, H)
    assert out.shape == (BEV["size"], BEV["size"]) and out.dtype == np.uint8
check("warp output shape + dtype", _warp_shape)

def _warp_discrete_values():
    from ipm.homography import compute_ipm_homography
    from ipm.warp import apply_ipm
    from utils.config import SEGFORMER
    H    = compute_ipm_homography(K)
    mask = np.zeros((SEGFORMER["img_h"], SEGFORMER["img_w"]), dtype=np.uint8)
    mask[300:, :] = 0
    mask[150:300, 300:700] = 1
    out  = apply_ipm(mask, H)
    unique = set(np.unique(out).tolist())
    assert unique.issubset({0, 1, 2, 255}), f"Non-discrete values: {unique}"
check("warp preserves discrete class labels (INTER_NEAREST)", _warp_discrete_values)

def _channels_shape():
    from ipm.warp import bev_mask_to_channels
    from utils.config import CLASSES, BEV
    mask = np.zeros((BEV["size"], BEV["size"]), dtype=np.uint8)
    mask[50:100, 50:100] = 1
    ch   = bev_mask_to_channels(mask)
    assert ch.shape == (CLASSES["num_classes"], BEV["size"], BEV["size"])
    assert ch.dtype == np.float32
    assert set(np.unique(ch).tolist()).issubset({0.0, 1.0})
check("bev_mask_to_channels shape + binary values", _channels_shape)

def _channels_correct_class():
    from ipm.warp import bev_mask_to_channels
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[70:90, 70:90] = 1   # vehicle patch
    ch   = bev_mask_to_channels(mask)
    assert ch[1, 80, 80] == 1.0   # vehicle channel
    assert ch[0, 80, 80] == 0.0   # road channel
check("bev_mask_to_channels maps to correct channel", _channels_correct_class)


# ── pipeline (dataset required) ───────────────────────────────
print("\n── Pipeline (dataset) ───────────────────────────────────")

from utils.config import PATHS, NUSCENES, BEV, SEGFORMER, CLASSES
_dataset_ok = (PATHS["dataroot"] / NUSCENES["version"] / "scene.json").exists()

if not _dataset_ok:
    print(f"{WARN}  Dataset not found — skipping pipeline checks")
else:
    def _pipeline_keys():
        import torch
        from data.nuscenes_loader import get_all_sample_tokens, get_camera_data
        from models.segformer.model import build_segformer_zero_shot
        from ipm.pipeline import run_ipm_pipeline
        token  = get_all_sample_tokens()[0]
        cam    = get_camera_data(token)
        bgr    = cv2.imread(str(cam["image_path"]))
        model  = build_segformer_zero_shot()
        result = run_ipm_pipeline(bgr, cam["K"], sample_token=token,
                                  model=model, device=torch.device("cpu"))
        for key in ["seg_mask_3", "bev_mask", "bev_channels", "H"]:
            assert key in result
    check("run_ipm_pipeline returns all required keys", _pipeline_keys)

    def _pipeline_shapes():
        import torch
        from data.nuscenes_loader import get_all_sample_tokens, get_camera_data
        from models.segformer.model import build_segformer_zero_shot
        from ipm.pipeline import run_ipm_pipeline
        token  = get_all_sample_tokens()[0]
        cam    = get_camera_data(token)
        bgr    = cv2.imread(str(cam["image_path"]))
        model  = build_segformer_zero_shot()
        result = run_ipm_pipeline(bgr, cam["K"], sample_token=token,
                                  model=model, device=torch.device("cpu"))
        assert result["seg_mask_3"].shape   == (SEGFORMER["img_h"], SEGFORMER["img_w"])
        assert result["bev_mask"].shape     == (BEV["size"], BEV["size"])
        assert result["bev_channels"].shape == (CLASSES["num_classes"], BEV["size"], BEV["size"])
        assert result["H"].shape            == (3, 3)
    check("run_ipm_pipeline output shapes correct", _pipeline_shapes)

    def _pipeline_bev_values():
        import torch
        from data.nuscenes_loader import get_all_sample_tokens, get_camera_data
        from models.segformer.model import build_segformer_zero_shot
        from ipm.pipeline import run_ipm_pipeline
        token  = get_all_sample_tokens()[0]
        cam    = get_camera_data(token)
        bgr    = cv2.imread(str(cam["image_path"]))
        model  = build_segformer_zero_shot()
        result = run_ipm_pipeline(bgr, cam["K"], sample_token=token,
                                  model=model, device=torch.device("cpu"))
        unique = set(np.unique(result["bev_mask"]).tolist())
        assert unique.issubset({0, 1, 2, 255})
        ch = result["bev_channels"]
        assert ch.min() >= 0.0 and ch.max() <= 1.0
    check("BEV mask values in {0,1,2,255} and channels in [0,1]", _pipeline_bev_values)

    def _pipeline_camera_height():
        from data.nuscenes_loader import get_all_sample_tokens
        from ipm.homography import get_camera_height_from_nusc
        token = get_all_sample_tokens()[0]
        h     = get_camera_height_from_nusc(token)
        assert 0.5 < h < 3.0, f"Camera height {h:.2f}m outside expected 0.5-3.0m range"
        print(f"       camera height = {h:.3f}m")
    check("get_camera_height_from_nusc returns plausible value", _pipeline_camera_height)


# ── pytest ────────────────────────────────────────────────────
print("\n── Pytest ───────────────────────────────────────────────")

def _run_pytest():
    import subprocess
    root   = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_ipm.py",
         "-v", "--tb=short", "-q"],
        cwd=str(root), capture_output=True, text=True
    )
    out = result.stdout
    print(out[-3000:] if len(out) > 3000 else out)
    if result.returncode != 0:
        print(result.stderr[-500:])
        raise AssertionError("pytest failures")
check("pytest test_ipm all pass", _run_pytest)


# ── summary ───────────────────────────────────────────────────
print("\n────────────────────────────────────────────────────────")
if failures:
    print(f"FAILED — {len(failures)} check(s):")
    for f in failures:
        print(f"  • {f}")
    print("Fix before proceeding to Phase 4.\n")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Phase 3 complete. Safe to start Phase 4.\n")
    sys.exit(0)