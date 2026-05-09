# ============================================================
# tests/test_bev_gt.py  -  Phase 1
# ============================================================
# Tests that require nuScenes dataset to be present.
# Skipped automatically if dataset is not downloaded.
#
# Run: python -m pytest tests/test_bev_gt.py -v
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from utils.config import PATHS, NUSCENES, BEV, CLASSES

# ── Dataset availability guard ────────────────────────────────
_DATASET_AVAILABLE = (
    PATHS["dataroot"] / NUSCENES["version"] / "scene.json"
).exists()

skip_if_no_data = pytest.mark.skipif(
    not _DATASET_AVAILABLE,
    reason="nuScenes dataset not found - download to run these tests"
)

# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def first_sample_token():
    """Return the first sample token from the dataset."""
    from data.nuscenes_loader import get_all_sample_tokens
    tokens = get_all_sample_tokens()
    assert len(tokens) > 0, "No samples found"
    return tokens[0]

@pytest.fixture(scope="module")
def first_scene_token():
    from data.nuscenes_loader import get_all_scene_tokens
    return get_all_scene_tokens()[0]


# ── class_mappings tests (no dataset needed) ──────────────────

class TestClassMappings:
    def test_vehicle_cat_maps_to_vehicle(self):
        from data.class_mappings import category_to_bev_class, VEHICLE
        assert category_to_bev_class("vehicle.car") == VEHICLE

    def test_pedestrian_cat_maps_to_pedestrian(self):
        from data.class_mappings import category_to_bev_class, PEDESTRIAN
        assert category_to_bev_class("human.pedestrian.adult") == PEDESTRIAN

    def test_static_object_returns_none(self):
        from data.class_mappings import category_to_bev_class
        assert category_to_bev_class("static_object.bicycle_rack") is None

    def test_unknown_category_returns_none(self):
        from data.class_mappings import category_to_bev_class
        assert category_to_bev_class("unknown.category.xyz") is None

    def test_unknown_vehicle_subtype_still_maps(self):
        """Prefix fallback: new vehicle sub-types → VEHICLE."""
        from data.class_mappings import category_to_bev_class, VEHICLE
        assert category_to_bev_class("vehicle.future_type") == VEHICLE

    def test_num_classes_matches_config(self):
        from data.class_mappings import num_classes
        assert num_classes() == CLASSES["num_classes"]


# ── Map mask tests (dataset required) ────────────────────────

class TestMapMask:
    @skip_if_no_data
    def test_map_mask_shape(self, first_sample_token):
        from data.bev_gt_generator import generate_map_bev_mask
        mask = generate_map_bev_mask(first_sample_token)
        assert mask.shape == (BEV["size"], BEV["size"]), \
            f"Expected ({BEV['size']},{BEV['size']}), got {mask.shape}"

    @skip_if_no_data
    def test_map_mask_dtype(self, first_sample_token):
        from data.bev_gt_generator import generate_map_bev_mask
        mask = generate_map_bev_mask(first_sample_token)
        assert mask.dtype == np.uint8

    @skip_if_no_data
    def test_map_mask_values_in_valid_range(self, first_sample_token):
        from data.bev_gt_generator import generate_map_bev_mask
        mask = generate_map_bev_mask(first_sample_token)
        unique_vals = set(np.unique(mask).tolist())
        valid_vals  = set(range(CLASSES["num_classes"])) | {255}
        assert unique_vals.issubset(valid_vals), \
            f"Unexpected values in map mask: {unique_vals - valid_vals}"

    @skip_if_no_data
    def test_map_mask_has_drivable_pixels(self, first_sample_token):
        """At least some drivable area should exist in any driving scene."""
        from data.bev_gt_generator import generate_map_bev_mask
        from data.class_mappings import DRIVABLE
        mask = generate_map_bev_mask(first_sample_token)
        assert (mask == DRIVABLE).sum() > 0, \
            "No drivable area pixels found - check map API or map expansion install"


# ── Box mask tests ────────────────────────────────────────────

class TestBoxMasks:
    @skip_if_no_data
    def test_box_masks_shape(self, first_sample_token):
        from data.bev_gt_generator import generate_box_bev_masks
        masks = generate_box_bev_masks(first_sample_token)
        assert masks.shape == (CLASSES["num_classes"], BEV["size"], BEV["size"]), \
            f"Expected ({CLASSES['num_classes']},{BEV['size']},{BEV['size']}), got {masks.shape}"

    @skip_if_no_data
    def test_box_masks_binary(self, first_sample_token):
        from data.bev_gt_generator import generate_box_bev_masks
        masks = generate_box_bev_masks(first_sample_token)
        unique_vals = set(np.unique(masks).tolist())
        assert unique_vals.issubset({0, 1}), \
            f"Box masks must be binary (0/1), got: {unique_vals}"

    @skip_if_no_data
    def test_vehicle_channel_has_annotations(self, first_sample_token):
        """nuScenes mini scenes always contain at least one vehicle."""
        from data.bev_gt_generator import generate_box_bev_masks
        from data.class_mappings import VEHICLE
        masks = generate_box_bev_masks(first_sample_token)
        assert masks[VEHICLE].sum() > 0, \
            "No vehicle pixels found - check annotation loading or BEV transform"


# ── Combined GT tests ─────────────────────────────────────────

class TestBEVGT:
    @skip_if_no_data
    def test_bev_gt_shape(self, first_sample_token):
        from data.bev_gt_generator import generate_bev_gt
        gt = generate_bev_gt(first_sample_token)
        assert gt.shape == (CLASSES["num_classes"], BEV["size"], BEV["size"])

    @skip_if_no_data
    def test_bev_gt_dtype_float32(self, first_sample_token):
        from data.bev_gt_generator import generate_bev_gt
        gt = generate_bev_gt(first_sample_token)
        assert gt.dtype == np.float32

    @skip_if_no_data
    def test_bev_gt_values_in_0_1(self, first_sample_token):
        from data.bev_gt_generator import generate_bev_gt
        gt = generate_bev_gt(first_sample_token)
        assert gt.min() >= 0.0 and gt.max() <= 1.0, \
            f"GT values out of [0,1]: min={gt.min()}, max={gt.max()}"

    @skip_if_no_data
    def test_bev_gt_channels_independent(self, first_sample_token):
        """A pixel can be 1 in multiple channels (vehicle on drivable area)."""
        from data.bev_gt_generator import generate_bev_gt
        gt = generate_bev_gt(first_sample_token)
        # Channel sum > 1 at any pixel = multi-label (valid, expected)
        channel_sum = gt.sum(axis=0)
        # Just assert it's not all zeros everywhere (empty GT)
        assert channel_sum.sum() > 0, "All BEV GT channels are zero"


# ── Dataset class tests ───────────────────────────────────────

class TestNuScenesBEVDataset:
    @skip_if_no_data
    def test_dataset_len_positive(self):
        from data.nuscenes_dataset import NuScenesBEVDataset
        ds = NuScenesBEVDataset(split="train")
        assert len(ds) > 0

    @skip_if_no_data
    def test_dataset_item_keys(self):
        from data.nuscenes_dataset import NuScenesBEVDataset
        ds    = NuScenesBEVDataset(split="train")
        item  = ds[0]
        required_keys = {"image", "K", "T_cam2ego", "T_ego2world",
                         "bev_gt", "sample_token", "location"}
        assert required_keys.issubset(set(item.keys())), \
            f"Missing keys: {required_keys - set(item.keys())}"

    @skip_if_no_data
    def test_dataset_image_shape(self):
        import torch
        from data.nuscenes_dataset import NuScenesBEVDataset
        from utils.config import SEGFORMER
        ds   = NuScenesBEVDataset(split="train")
        item = ds[0]
        assert item["image"].shape == (3, SEGFORMER["img_h"], SEGFORMER["img_w"]), \
            f"Unexpected image shape: {item['image'].shape}"

    @skip_if_no_data
    def test_dataset_image_range(self):
        from data.nuscenes_dataset import NuScenesBEVDataset
        ds   = NuScenesBEVDataset(split="train")
        item = ds[0]
        assert item["image"].min() >= 0.0 and item["image"].max() <= 1.0

    @skip_if_no_data
    def test_dataset_K_shape(self):
        from data.nuscenes_dataset import NuScenesBEVDataset
        ds   = NuScenesBEVDataset(split="train")
        item = ds[0]
        assert item["K"].shape == (3, 3)

    @skip_if_no_data
    def test_dataset_transforms_shape(self):
        from data.nuscenes_dataset import NuScenesBEVDataset
        ds   = NuScenesBEVDataset(split="train")
        item = ds[0]
        assert item["T_cam2ego"].shape   == (4, 4)
        assert item["T_ego2world"].shape == (4, 4)

    @skip_if_no_data
    def test_dataset_bev_gt_shape(self):
        from data.nuscenes_dataset import NuScenesBEVDataset
        from utils.config import CLASSES, BEV
        ds   = NuScenesBEVDataset(split="train")
        item = ds[0]
        expected = (CLASSES["num_classes"], BEV["size"], BEV["size"])
        assert tuple(item["bev_gt"].shape) == expected, \
            f"Expected {expected}, got {tuple(item['bev_gt'].shape)}"

    @skip_if_no_data
    def test_train_val_split_no_overlap(self):
        """Train and val sample token sets must be disjoint."""
        from data.nuscenes_dataset import get_sample_tokens
        train = set(get_sample_tokens("train"))
        val   = set(get_sample_tokens("val"))
        overlap = train & val
        assert len(overlap) == 0, \
            f"Train/val overlap: {len(overlap)} shared tokens"

    @skip_if_no_data
    def test_all_split_is_union(self):
        from data.nuscenes_dataset import get_sample_tokens
        train = set(get_sample_tokens("train"))
        val   = set(get_sample_tokens("val"))
        all_  = set(get_sample_tokens("all"))
        assert train | val == all_, "train ∪ val ≠ all"


# ── Standalone runner ─────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["python", "-m", "pytest", __file__, "-v"]))