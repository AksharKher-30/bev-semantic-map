#!/usr/bin/env python
# ============================================================
# scripts/verify_phase1.py  -  Phase 1 integrity check
# Run AFTER all Phase 1 code is written.
# Run BEFORE starting Phase 2 (SegFormer fine-tuning).
#
# Usage: python scripts/verify_phase1.py
# Exit 0 = all pass. Exit 1 = fix required.
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


# ─────────────────────────────────────────────────────────────
# 1. Module imports
# ─────────────────────────────────────────────────────────────
print("\n── Module imports ───────────────────────────────────────")

def _import_geometry():
    from utils.geometry import (
        quat_to_rotation_matrix, build_transform_matrix,
        invert_transform, transform_points,
        project_to_image, ego_to_bev_pixels,
    )
check("utils.geometry all functions importable", _import_geometry)

def _import_calibration():
    from data.calibration import (
        build_cam_to_world, build_world_to_cam, build_world_to_ego,
        project_world_points_to_image, project_world_points_to_ego_bev,
        validate_intrinsics, validate_transform,
    )
check("data.calibration all functions importable", _import_calibration)

def _import_class_mappings():
    from data.class_mappings import (
        category_to_bev_class, num_classes,
        DRIVABLE, VEHICLE, PEDESTRIAN,
    )
check("data.class_mappings importable", _import_class_mappings)

def _import_nuscenes_loader():
    from data.nuscenes_loader import (
        get_nusc, get_all_scene_tokens, iterate_scene_samples,
        get_camera_data, get_sample_annotations, get_ego_pose,
    )
check("data.nuscenes_loader importable", _import_nuscenes_loader)

def _import_bev_gt_generator():
    from data.bev_gt_generator import (
        generate_map_bev_mask, generate_box_bev_masks, generate_bev_gt,
    )
check("data.bev_gt_generator importable", _import_bev_gt_generator)

def _import_nuscenes_dataset():
    from data.nuscenes_dataset import (
        NuScenesBEVDataset, build_dataloader, get_sample_tokens,
    )
check("data.nuscenes_dataset importable", _import_nuscenes_dataset)


# ─────────────────────────────────────────────────────────────
# 2. Geometry unit tests (no dataset needed)
# ─────────────────────────────────────────────────────────────
print("\n── Geometry (no dataset) ────────────────────────────────")

def _quat_identity():
    from utils.geometry import quat_to_rotation_matrix
    R = quat_to_rotation_matrix([1,0,0,0])
    np.testing.assert_allclose(R, np.eye(3), atol=1e-8)
check("identity quaternion → identity R", _quat_identity)

def _transform_orthonormal():
    from utils.geometry import quat_to_rotation_matrix
    import math
    q = [math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)]
    R = quat_to_rotation_matrix(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
check("R orthonormal: R @ R.T = I", _transform_orthonormal)

def _transform_invertible():
    from utils.geometry import build_transform_matrix, invert_transform
    import math
    q = [math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)]
    T = build_transform_matrix([3.0, 1.0, 0.5], q)
    T_inv = invert_transform(T)
    np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-10)
check("T @ inv(T) = I", _transform_invertible)

def _ego_origin_bev_centre():
    from utils.geometry import ego_to_bev_pixels
    pts = np.array([[0.0, 0.0]])
    pix, valid = ego_to_bev_pixels(pts, bev_size=200, bev_resolution=0.5)
    assert valid[0]
    assert pix[0,0] == 100 and pix[0,1] == 100
check("ego origin → BEV centre pixel (100,100)", _ego_origin_bev_centre)

def _class_mapping_vehicle():
    from data.class_mappings import category_to_bev_class, VEHICLE
    assert category_to_bev_class("vehicle.car") == VEHICLE
check("category_to_bev_class: vehicle.car → VEHICLE", _class_mapping_vehicle)

def _class_mapping_none():
    from data.class_mappings import category_to_bev_class
    assert category_to_bev_class("static_object.bicycle_rack") is None
check("category_to_bev_class: static_object → None", _class_mapping_none)


# ─────────────────────────────────────────────────────────────
# 3. Dataset tests (require nuScenes download)
# ─────────────────────────────────────────────────────────────
print("\n── Dataset pipeline ─────────────────────────────────────")

from utils.config import PATHS, NUSCENES, BEV, CLASSES
_dataset_ok = (PATHS["dataroot"] / NUSCENES["version"] / "scene.json").exists()

if not _dataset_ok:
    print(f"{WARN}  Dataset not found - skipping pipeline checks.")
    print(f"       Download nuScenes mini and re-run.")
else:
    def _loader_scene_count():
        from data.nuscenes_loader import get_all_scene_tokens
        tokens = get_all_scene_tokens()
        assert len(tokens) == 10, f"Expected 10 scenes, got {len(tokens)}"
    check("get_all_scene_tokens() → 10 scenes", _loader_scene_count)

    def _loader_sample_count():
        from data.nuscenes_loader import get_all_sample_tokens
        tokens = get_all_sample_tokens()
        assert len(tokens) == 404, f"Expected 404 samples, got {len(tokens)}"
    check("get_all_sample_tokens() → 404 samples", _loader_sample_count)

    def _camera_data_keys():
        from data.nuscenes_loader import get_all_sample_tokens, get_camera_data
        token = get_all_sample_tokens()[0]
        d = get_camera_data(token)
        for k in ["image_path","K","T_cam2ego","T_ego2world"]:
            assert k in d, f"Missing key: {k}"
        assert d["image_path"].exists(), f"Image not found: {d['image_path']}"
    check("get_camera_data() returns valid keys + image exists", _camera_data_keys)

    def _calibration_valid():
        from data.nuscenes_loader import get_all_sample_tokens, get_camera_data
        from data.calibration import validate_intrinsics, validate_transform
        token = get_all_sample_tokens()[0]
        d = get_camera_data(token)
        validate_intrinsics(d["K"])
        validate_transform(d["T_cam2ego"],   "T_cam2ego")
        validate_transform(d["T_ego2world"], "T_ego2world")
    check("K intrinsics + T matrices pass validation", _calibration_valid)

    def _map_mask_shape():
        from data.nuscenes_loader import get_all_sample_tokens
        from data.bev_gt_generator import generate_map_bev_mask
        token = get_all_sample_tokens()[0]
        mask = generate_map_bev_mask(token)
        assert mask.shape == (BEV["size"], BEV["size"])
        assert mask.dtype == np.uint8
    check(f"map mask shape = ({BEV['size']},{BEV['size']}) uint8", _map_mask_shape)

    def _map_mask_has_drivable():
        from data.nuscenes_loader import get_all_sample_tokens
        from data.bev_gt_generator import generate_map_bev_mask
        from data.class_mappings import DRIVABLE
        token = get_all_sample_tokens()[0]
        mask  = generate_map_bev_mask(token)
        assert (mask == DRIVABLE).sum() > 0
    check("map mask contains drivable area pixels", _map_mask_has_drivable)

    def _box_masks_shape():
        from data.nuscenes_loader import get_all_sample_tokens
        from data.bev_gt_generator import generate_box_bev_masks
        token = get_all_sample_tokens()[0]
        masks = generate_box_bev_masks(token)
        assert masks.shape == (CLASSES["num_classes"], BEV["size"], BEV["size"])
        assert set(np.unique(masks).tolist()).issubset({0,1})
    check("box masks shape + binary values", _box_masks_shape)

    def _bev_gt_full():
        from data.nuscenes_loader import get_all_sample_tokens
        from data.bev_gt_generator import generate_bev_gt
        token = get_all_sample_tokens()[0]
        gt = generate_bev_gt(token)
        assert gt.shape  == (CLASSES["num_classes"], BEV["size"], BEV["size"])
        assert gt.dtype  == np.float32
        assert gt.min()  >= 0.0
        assert gt.max()  <= 1.0
        assert gt.sum()  > 0, "All GT channels are zero - check data pipeline"
    check("generate_bev_gt: shape, dtype, range, non-empty", _bev_gt_full)

    def _dataset_item():
        import torch
        from data.nuscenes_dataset import NuScenesBEVDataset
        from utils.config import SEGFORMER
        ds   = NuScenesBEVDataset(split="train")
        assert len(ds) > 0
        item = ds[0]
        assert item["image"].shape    == (3, SEGFORMER["img_h"], SEGFORMER["img_w"])
        assert item["K"].shape        == (3, 3)
        assert item["T_cam2ego"].shape == (4, 4)
        assert item["bev_gt"].shape   == (CLASSES["num_classes"], BEV["size"], BEV["size"])
        assert item["image"].min() >= 0.0 and item["image"].max() <= 1.0
        assert isinstance(item["sample_token"], str)
        assert isinstance(item["location"], str)
    check("NuScenesBEVDataset[0] all keys + shapes correct", _dataset_item)

    def _train_val_no_overlap():
        from data.nuscenes_dataset import get_sample_tokens
        train = set(get_sample_tokens("train"))
        val   = set(get_sample_tokens("val"))
        assert len(train & val) == 0, "Train/val token overlap!"
        assert train | val == set(get_sample_tokens("all"))
    check("train/val split: no overlap, union=all", _train_val_no_overlap)

    def _dataloader_batch():
        import torch
        from data.nuscenes_dataset import build_dataloader
        loader = build_dataloader(split="val", batch_size=2, shuffle=False)
        batch  = next(iter(loader))
        assert batch["image"].shape[0]   <= 2       # batch dim
        assert batch["bev_gt"].ndim      == 4        # (B, C, H, W)
        assert batch["K"].shape[-2:]     == (3, 3)
    check("build_dataloader batch shapes correct", _dataloader_batch)


# ─────────────────────────────────────────────────────────────
# 4. Run pytest unit tests
# ─────────────────────────────────────────────────────────────
print("\n── Pytest unit tests ────────────────────────────────────")

def _run_pytest():
    import subprocess
    # Auto-install pytest if missing
    try:
        import pytest
    except ImportError:
        print(f"{WARN}  pytest not found - installing...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pytest", "-q"],
            check=True
        )
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_calibration.py",
         "tests/test_bev_gt.py",
         "-v", "--tb=short", "-q"],
        cwd=str(root),
        capture_output=True, text=True
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(result.stderr[-500:])
        raise AssertionError("pytest reported failures")
check("pytest test_calibration + test_bev_gt all pass", _run_pytest)


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n────────────────────────────────────────────────────────")
if failures:
    print(f"FAILED - {len(failures)} check(s) did not pass:")
    for f in failures:
        print(f"  • {f}")
    print("Fix the above before proceeding to Phase 2.\n")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED - Phase 1 complete. Safe to start Phase 2.\n")
    sys.exit(0)