# ============================================================
# src/data/bev_gt_generator.py
# ============================================================
# Generates BEV ground-truth semantic masks for each sample.
#
# Two sources combined:
#   A) nuScenes Map API  → static road structure (drivable area, lanes)
#   B) 3D bounding boxes → dynamic objects (vehicles, pedestrians)
#
# Output: (num_classes, bev_size, bev_size) binary float32 tensor
#   Channel 0 = drivable_area
#   Channel 1 = vehicle
#   Channel 2 = pedestrian
#
# HOW TO CHANGE:
#   • Add a class → update config.py + class_mappings.py only.
#   • Change BEV resolution → edit config.py BEV dict only.
#   • Nothing in this file needs editing for those changes.
# ============================================================

from __future__ import annotations

import numpy as np
import cv2

from utils.config import BEV, CLASSES
from utils.geometry import ego_to_bev_pixels
from data.class_mappings import (
    MAP_LAYERS,
    NUSCENES_MAP_LAYER_TO_BEV,
    category_to_bev_class,
    DRIVABLE, VEHICLE, PEDESTRIAN,
)
from data.nuscenes_loader import (
    get_nusc,
    get_nusc_map,
    get_scene_location,
    get_sample_annotations,
    get_ego_pose,
)
from data.calibration import project_world_points_to_ego_bev


# ── Internal constants (derived from config) ──────────────────
_BEV_SIZE = BEV["size"]
_BEV_RES  = BEV["resolution"]
_N_CLS    = CLASSES["num_classes"]


# ── Source A: Map API rasterisation ──────────────────────────

def _get_map_name_for_sample(sample_token: str) -> str:
    """Resolve map name from sample → scene → log → location."""
    nusc   = get_nusc()
    sample = nusc.get("sample", sample_token)
    scene  = nusc.get("scene",  sample["scene_token"])
    return get_scene_location(scene["token"])


def generate_map_bev_mask(
    sample_token: str,
    bev_size: int   = _BEV_SIZE,
    bev_res:  float = _BEV_RES,
) -> np.ndarray:
    """
    Rasterise nuScenes HD map layers into a BEV label mask.

    The nuScenes Map API returns binary masks aligned to a patch
    centred on the ego vehicle. We map each layer to our class index
    and stack into a single-channel label mask.

    Parameters
    ----------
    sample_token : str
    bev_size     : int    - grid side in pixels
    bev_res      : float  - metres per pixel

    Returns
    -------
    mask : (bev_size, bev_size) np.uint8
        Each pixel = class index (0=drivable, or 0 for background).
        Background pixels are 255 (ignore index, same as seg training).
    """
    map_name  = _get_map_name_for_sample(sample_token)
    nusc_map  = get_nusc_map(map_name)
    ego_pose  = get_ego_pose(sample_token)

    cx, cy    = ego_pose["translation"][:2]
    patch_m   = bev_size * bev_res                    # total coverage in metres
    patch_box = (
        cx - patch_m / 2, cy - patch_m / 2,
        cx + patch_m / 2, cy + patch_m / 2,
    )

    # get_map_mask returns (num_layers, H, W) binary arrays
    # canvas_size must be (rows, cols) = (bev_size, bev_size)
    layer_masks = nusc_map.get_map_mask(
        patch_box   = patch_box,
        patch_angle = 0.0,           # ego heading = up in BEV
        layer_names = MAP_LAYERS,
        canvas_size = (bev_size, bev_size),
    )                                # shape: (len(MAP_LAYERS), bev_size, bev_size)

    # Combine layers: later layers in MAP_LAYERS overwrite earlier ones
    mask = np.zeros((bev_size, bev_size), dtype=np.uint8)
    for i, layer_name in enumerate(MAP_LAYERS):
        cls_idx = NUSCENES_MAP_LAYER_TO_BEV.get(layer_name)
        if cls_idx is not None:
            mask[layer_masks[i].astype(bool)] = cls_idx

    return mask


# ── Source B: 3D bounding boxes ───────────────────────────────

def generate_box_bev_masks(
    sample_token: str,
    bev_size: int   = _BEV_SIZE,
    bev_res:  float = _BEV_RES,
) -> np.ndarray:
    """
    Draw 3D annotation bounding boxes onto BEV class channels.

    For each annotation:
        1. Transform box centre from world frame → ego frame.
        2. Convert ego (x,y) → BEV pixel indices.
        3. Draw filled rotated rectangle using box size.

    Parameters
    ----------
    sample_token : str
    bev_size     : int
    bev_res      : float

    Returns
    -------
    masks : (num_classes, bev_size, bev_size) np.uint8
        Binary mask per class. 1 where that class is present.
    """
    from utils.geometry import quat_to_rotation_matrix

    nusc      = get_nusc()
    ego_pose  = get_ego_pose(sample_token)
    from data.calibration import build_world_to_ego
    from utils.geometry import build_transform_matrix
    T_ego2world = build_transform_matrix(
        ego_pose["translation"], ego_pose["rotation"]
    )
    T_world2ego = build_world_to_ego(T_ego2world)

    masks = np.zeros((_N_CLS, bev_size, bev_size), dtype=np.uint8)
    anns  = get_sample_annotations(sample_token)

    for ann in anns:
        cls_idx = category_to_bev_class(ann["category_name"])
        if cls_idx is None:
            continue

        # Box centre: world → ego → BEV pixels
        centre_world = np.array(ann["translation"][:3]).reshape(1, 3)
        centre_ego   = project_world_points_to_ego_bev(centre_world, T_ego2world)
        # centre_ego: (1, 2)

        bev_pix, valid = ego_to_bev_pixels(centre_ego, bev_size, bev_res)
        if not valid[0]:
            continue   # box centre outside BEV grid

        cx_px, cy_px = int(bev_pix[0, 0]), int(bev_pix[0, 1])

        # Box dimensions: size = [width, length, height] in metres
        w_m, l_m = ann["size"][0], ann["size"][1]
        w_px = max(1, int(w_m / bev_res))
        l_px = max(1, int(l_m / bev_res))

        # Rotation angle in BEV (yaw from quaternion)
        R = quat_to_rotation_matrix(ann["rotation"])
        # Extract yaw from world-frame rotation, then transform to ego frame
        # R_ego = T_world2ego[:3,:3] @ R
        R_ego = T_world2ego[:3, :3] @ R
        yaw_rad = float(np.arctan2(R_ego[1, 0], R_ego[0, 0]))
        yaw_deg = float(np.degrees(yaw_rad))

        # Draw rotated rectangle on the correct class channel
        box = cv2.boxPoints((
            (cx_px, cy_px),          # centre
            (l_px,  w_px),           # (length, width) in pixels
            yaw_deg,
        ))
        box = np.intp(box)
        cv2.fillPoly(masks[cls_idx], [box], 1)

    return masks


# ── Combined GT mask ──────────────────────────────────────────

def generate_bev_gt(
    sample_token: str,
    bev_size: int   = _BEV_SIZE,
    bev_res:  float = _BEV_RES,
) -> np.ndarray:
    """
    Generate the full BEV ground-truth mask for one sample.

    Combines map-layer mask (static) and box masks (dynamic).

    Parameters
    ----------
    sample_token : str
    bev_size     : int
    bev_res      : float

    Returns
    -------
    bev_gt : (num_classes, bev_size, bev_size) np.float32
        Multi-channel binary mask. Each channel is independent:
        a pixel can be 1 in multiple channels simultaneously
        (e.g. a vehicle on a drivable area).

    Notes
    -----
    We use float32 (not uint8) so it can be directly used as a
    PyTorch training target with BCEWithLogitsLoss without casting.
    """
    # Map mask: (bev_size, bev_size) - single channel label
    map_mask = generate_map_bev_mask(sample_token, bev_size, bev_res)

    # Box masks: (num_classes, bev_size, bev_size)
    box_masks = generate_box_bev_masks(sample_token, bev_size, bev_res)

    # Assemble multi-channel GT
    gt = np.zeros((_N_CLS, bev_size, bev_size), dtype=np.float32)

    # Channel 0: drivable_area - from map
    gt[DRIVABLE] = (map_mask == DRIVABLE).astype(np.float32)

    # Channels 1,2: vehicle, pedestrian - from boxes
    gt[VEHICLE]     = box_masks[VEHICLE].astype(np.float32)
    gt[PEDESTRIAN]  = box_masks[PEDESTRIAN].astype(np.float32)

    return gt   # (3, bev_size, bev_size) float32