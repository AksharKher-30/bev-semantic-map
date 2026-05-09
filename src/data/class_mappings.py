# All category-to-class-index mappings live HERE.
# Used by bev_gt_generator.py (3D box labels) and
# pseudo_labels.py (Phase 2 front-view supervision).
#
# HOW TO CHANGE:
#   Add/rename a class → update CLASSES in config.py first,
#   then update the dicts below to match.
#   Nothing else in the codebase needs to change.
# ============================================================

import numpy as np
from utils.config import CLASSES

# Our 3 BEV class indices (matches CLASSES["names"] order)
DRIVABLE = 0    # drivable_area, road_divider, lane_divider, ped_crossing
VEHICLE  = 1    # all vehicle types
PEDESTRIAN = 2  # person, rider

# ── nuScenes 23 annotation categories → BEV class index ──────
# Full list: https://www.nuscenes.org/nuscenes#taxonomy
# None = ignore (not mapped to any BEV class)
NUSCENES_CATEGORY_TO_BEV: dict[str, int | None] = {
    # Vehicles
    "vehicle.car"                   : VEHICLE,
    "vehicle.truck"                 : VEHICLE,
    "vehicle.bus.rigid"             : VEHICLE,
    "vehicle.bus.bendy"             : VEHICLE,
    "vehicle.motorcycle"            : VEHICLE,
    "vehicle.bicycle"               : VEHICLE,
    "vehicle.trailer"               : VEHICLE,
    "vehicle.construction"          : VEHICLE,
    "vehicle.emergency.ambulance"   : VEHICLE,
    "vehicle.emergency.police"      : VEHICLE,

    # Pedestrians
    "human.pedestrian.adult"        : PEDESTRIAN,
    "human.pedestrian.child"        : PEDESTRIAN,
    "human.pedestrian.wheelchair"   : PEDESTRIAN,
    "human.pedestrian.stroller"     : PEDESTRIAN,
    "human.pedestrian.personal_mobility": PEDESTRIAN,
    "human.pedestrian.police_officer"   : PEDESTRIAN,
    "human.pedestrian.construction_worker": PEDESTRIAN,

    # Static / ignored in BEV (no label assigned)
    "static_object.bicycle_rack"    : None,
    "movable_object.barrier"        : None,
    "movable_object.trafficcone"    : None,
    "movable_object.pushable_pullable": None,
    "movable_object.debris"         : None,
    "animal"                        : None,
}

# ── nuScenes map layer → BEV class index ─────────────────────
# Used in bev_gt_generator.py when rasterising the HD map.
NUSCENES_MAP_LAYER_TO_BEV: dict[str, int] = {
    "drivable_area" : DRIVABLE,
    "road_divider"  : DRIVABLE,
    "lane_divider"  : DRIVABLE,
    "ped_crossing"  : DRIVABLE,
}

# All map layers we rasterise (order = layer priority - later overwrites earlier)
MAP_LAYERS = list(NUSCENES_MAP_LAYER_TO_BEV.keys())


def category_to_bev_class(category_name: str) -> int | None:
    """
    Map a nuScenes annotation category string to a BEV class index.

    Returns None if the category should be ignored (not drawn on BEV GT).
    Uses substring prefix matching so new sub-categories added by nuScenes
    still resolve correctly.

    Parameters
    ----------
    category_name : str
        e.g. "vehicle.car", "human.pedestrian.adult"

    Returns
    -------
    int | None
    """
    # Exact match first
    if category_name in NUSCENES_CATEGORY_TO_BEV:
        return NUSCENES_CATEGORY_TO_BEV[category_name]

    # Prefix match fallback (catches unknown sub-types)
    if category_name.startswith("vehicle."):
        return VEHICLE
    if category_name.startswith("human.pedestrian."):
        return PEDESTRIAN

    return None   # ignore


def num_classes() -> int:
    """Return the number of BEV semantic classes (from config)."""
    return CLASSES["num_classes"]

# ── Cityscapes 19-class → our class mappings ──────────────────
# Used for zero-shot inference (Option C) — no fine-tuning needed.

# Full 19-class Cityscapes label names (for visualization)
CITYSCAPES_19_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

# 19 → 5 mapping (road, vehicle, pedestrian, sky, background)
# Used for front-view segmentation visualization
CITYSCAPES_19_TO_5 = np.array([
    0,   # road       → road
    4,   # sidewalk   → background
    4,   # building   → background
    4,   # wall       → background
    4,   # fence      → background
    4,   # pole       → background
    4,   # t-light    → background
    4,   # t-sign     → background
    4,   # vegetation → background
    0,   # terrain    → road
    3,   # sky        → sky
    2,   # person     → pedestrian
    2,   # rider      → pedestrian
    1,   # car        → vehicle
    1,   # truck      → vehicle
    1,   # bus        → vehicle
    1,   # train      → vehicle
    1,   # motorcycle → vehicle
    1,   # bicycle    → vehicle
], dtype=np.uint8)

# 19 → 3 mapping (road, vehicle, pedestrian only)
# Used for IPM warp + BEV evaluation
CITYSCAPES_19_TO_3 = np.array([
    0,   # road       → road
    255, # sidewalk   → ignore
    255, # building   → ignore
    255, # wall       → ignore
    255, # fence      → ignore
    255, # pole       → ignore
    255, # t-light    → ignore
    255, # t-sign     → ignore
    255, # vegetation → ignore
    0,   # terrain    → road
    255, # sky        → ignore
    2,   # person     → pedestrian
    2,   # rider      → pedestrian
    1,   # car        → vehicle
    1,   # truck      → vehicle
    1,   # bus        → vehicle
    1,   # train      → vehicle
    1,   # motorcycle → vehicle
    1,   # bicycle    → vehicle
], dtype=np.uint8)


def remap_19_to_5(mask_19):
    """(H, W) uint8 with values 0-18 → (H, W) uint8 with values 0-4"""
    return CITYSCAPES_19_TO_5[mask_19]

def remap_19_to_3(mask_19):
    """(H, W) uint8 with values 0-18 → (H, W) uint8 with values 0-2 or 255"""
    return CITYSCAPES_19_TO_3[mask_19]