# Thin wrapper around the nuScenes devkit.
# Provides typed, project-specific helpers so the rest of the
# codebase never imports nuscenes.NuScenes directly.
#
# HOW TO CHANGE:
#   • Switch dataset split (mini → trainval): edit config.py NUSCENES["version"]
#   • Add a new camera: change NUSCENES["camera"] in config.py
#   • Nothing in this file needs editing for those changes.

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from utils.config import PATHS, NUSCENES


# ── Singleton loader ──────────────────────────────────────────
# Cached NuScenes instance - expensive to load (~2 s), load once.
_nusc_cache: NuScenes | None = None
_nusc_map_cache: dict[str, NuScenesMap] = {}


def get_nusc(verbose: bool = False) -> NuScenes:
    """
    Return (and cache) the NuScenes devkit instance.

    Parameters
    ----------
    verbose : bool
        Print loading progress from devkit if True.

    Returns
    -------
    NuScenes
    """
    global _nusc_cache
    if _nusc_cache is None:
        dataroot = str(PATHS["dataroot"])
        version  = NUSCENES["version"]
        _nusc_cache = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    return _nusc_cache


def get_nusc_map(map_name: str) -> NuScenesMap:
    """
    Return (and cache) a NuScenesMap for the given city.

    Parameters
    ----------
    map_name : str
        One of: 'boston-seaport', 'singapore-onenorth',
                'singapore-hollandvillage', 'singapore-queenstown'.

    Returns
    -------
    NuScenesMap
    """
    global _nusc_map_cache
    if map_name not in _nusc_map_cache:
        _nusc_map_cache[map_name] = NuScenesMap(
            dataroot=str(PATHS["dataroot"]),
            map_name=map_name,
        )
    return _nusc_map_cache[map_name]


# ── Scene / Sample helpers ────────────────────────────────────

def get_all_scene_tokens() -> list[str]:
    """Return tokens for all scenes in the loaded split."""
    nusc = get_nusc()
    return [s["token"] for s in nusc.scene]


def get_scene_location(scene_token: str) -> str:
    """
    Return the city name for a scene token.

    Derived from the log record. Used for terrain-stratified evaluation
    (Boston scenes have non-flat terrain; Singapore is flat).

    Returns
    -------
    str  - e.g. 'boston-seaport', 'singapore-onenorth'
    """
    nusc = get_nusc()
    scene = nusc.get("scene", scene_token)
    log   = nusc.get("log", scene["log_token"])
    return log["location"]


def iterate_scene_samples(scene_token: str) -> Iterator[str]:
    """
    Yield sample tokens in temporal order for a scene.

    Parameters
    ----------
    scene_token : str

    Yields
    ------
    str  - sample token
    """
    nusc = get_nusc()
    scene = nusc.get("scene", scene_token)
    token = scene["first_sample_token"]
    while token:
        yield token
        sample = nusc.get("sample", token)
        token  = sample["next"]   # "" for the last sample


def get_all_sample_tokens() -> list[str]:
    """Return all sample tokens across all scenes (order: scene order)."""
    tokens = []
    for scene_token in get_all_scene_tokens():
        tokens.extend(iterate_scene_samples(scene_token))
    return tokens


# ── Per-sample data accessors ─────────────────────────────────

def get_camera_data(sample_token: str, camera: str | None = None) -> dict:
    """
    Load raw camera data for one sample.

    Parameters
    ----------
    sample_token : str
    camera : str | None
        Camera name. Defaults to NUSCENES["camera"] from config.
        Options: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
                 CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.

    Returns
    -------
    dict with keys:
        image_path  : Path to the JPG file
        K           : (3,3) np.ndarray - camera intrinsic matrix
        T_cam2ego   : (4,4) np.ndarray - sensor→ego rigid transform
        T_ego2world : (4,4) np.ndarray - ego→world rigid transform
        cam_token   : str - sample_data token for this camera
        ego_token   : str - ego_pose token
    """
    from utils.geometry import build_transform_matrix

    nusc   = get_nusc()
    camera = camera or NUSCENES["camera"]

    sample     = nusc.get("sample", sample_token)
    cam_token  = sample["data"][camera]
    cam_data   = nusc.get("sample_data", cam_token)

    # ── Calibration (static - where sensor sits on the car) ───
    calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    K     = np.array(calib["camera_intrinsic"], dtype=np.float64)  # (3,3)
    T_cam2ego = build_transform_matrix(calib["translation"], calib["rotation"])

    # ── Ego pose (dynamic - where car is in the world) ────────
    ego_record  = nusc.get("ego_pose", cam_data["ego_pose_token"])
    T_ego2world = build_transform_matrix(
        ego_record["translation"], ego_record["rotation"]
    )

    image_path = PATHS["dataroot"] / cam_data["filename"]

    return {
        "image_path"  : image_path,
        "K"           : K,
        "T_cam2ego"   : T_cam2ego,
        "T_ego2world" : T_ego2world,
        "cam_token"   : cam_token,
        "ego_token"   : cam_data["ego_pose_token"],
    }


def get_sample_annotations(sample_token: str) -> list[dict]:
    """
    Return all 3D bounding box annotations for a sample.

    Each annotation dict contains:
        token          : str
        category_name  : str   (e.g. "vehicle.car")
        translation    : [x, y, z]  centre in world frame (metres)
        size           : [width, length, height]  (metres)
        rotation       : [w, x, y, z]  quaternion

    Parameters
    ----------
    sample_token : str

    Returns
    -------
    list[dict]
    """
    nusc   = get_nusc()
    sample = nusc.get("sample", sample_token)
    anns   = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        # Resolve category name from instance → category chain
        instance = nusc.get("instance", ann["instance_token"])
        category = nusc.get("category", instance["category_token"])
        anns.append({
            "token"         : ann_token,
            "category_name" : category["name"],
            "translation"   : ann["translation"],
            "size"          : ann["size"],
            "rotation"      : ann["rotation"],
        })
    return anns


def get_ego_pose(sample_token: str) -> dict:
    """
    Return ego pose dict for the primary camera at this sample.

    Returns
    -------
    dict with keys:
        translation : [x, y, z] in world frame
        rotation    : [w, x, y, z] quaternion
        token       : ego_pose token
    """
    nusc     = get_nusc()
    sample   = nusc.get("sample", sample_token)
    cam_data = nusc.get("sample_data", sample["data"][NUSCENES["camera"]])
    ego      = nusc.get("ego_pose", cam_data["ego_pose_token"])
    return ego