# ============================================================
# src/data/calibration.py
# ============================================================
# Camera calibration helpers.
# Handles the 3-frame coordinate chain nuScenes uses:
#
#   sensor frame  →  ego frame  →  world frame
#   (camera)         (car body)    (global map)
#
# All raw quaternion/translation data comes from nuscenes_loader.
# All math primitives live in utils/geometry.py.
# This file only orchestrates the calibration chain.
#
# HOW TO CHANGE:
#   • Add a new camera: no changes needed - pass camera name to loader.
#   • Change BEV coverage: edit config.py BEV dict only.
# ============================================================

from __future__ import annotations

import numpy as np
from utils.geometry import (
    build_transform_matrix,
    invert_transform,
    transform_points,
    project_to_image,
)


# ── Coordinate chain builders ─────────────────────────────────

def build_cam_to_world(T_cam2ego: np.ndarray,
                       T_ego2world: np.ndarray) -> np.ndarray:
    """
    Compose camera→ego and ego→world into one camera→world transform.

    P_world = T_cam2world @ P_cam   (homogeneous coords)

    Parameters
    ----------
    T_cam2ego   : (4,4) rigid body transform - sensor frame to ego frame
    T_ego2world : (4,4) rigid body transform - ego frame to world frame

    Returns
    -------
    T_cam2world : (4,4)
    """
    return T_ego2world @ T_cam2ego


def build_world_to_cam(T_cam2ego: np.ndarray,
                       T_ego2world: np.ndarray) -> np.ndarray:
    """
    Compose the inverse chain: world → camera frame.

    Used when projecting world-frame LiDAR / annotation points
    onto the camera image plane.

    Returns
    -------
    T_world2cam : (4,4)
    """
    T_cam2world = build_cam_to_world(T_cam2ego, T_ego2world)
    return invert_transform(T_cam2world)


def build_world_to_ego(T_ego2world: np.ndarray) -> np.ndarray:
    """
    Invert ego→world to get world→ego.

    Used when placing world-frame annotation bounding boxes
    into the ego-centric BEV grid.

    Returns
    -------
    T_world2ego : (4,4)
    """
    return invert_transform(T_ego2world)


# ── Projection helpers ────────────────────────────────────────

def project_world_points_to_image(
    points_world: np.ndarray,
    K: np.ndarray,
    T_cam2ego: np.ndarray,
    T_ego2world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D world-frame points onto the camera image.

    Full chain:
        P_cam = T_world2cam @ P_world
        [u, v] = K @ P_cam / z

    Parameters
    ----------
    points_world : (N, 3)  - 3D points in world frame
    K            : (3, 3)  - camera intrinsic matrix
    T_cam2ego    : (4, 4)
    T_ego2world  : (4, 4)

    Returns
    -------
    pixels : (N, 2)  - [u, v] pixel coordinates
    depths : (N,)    - depth (z) in camera frame
    valid  : (N,)    - bool mask: in front of camera
    """
    T_world2cam = build_world_to_cam(T_cam2ego, T_ego2world)
    points_cam  = transform_points(points_world, T_world2cam)
    pixels, depths, valid = project_to_image(points_cam, K)
    return pixels, depths, valid


def project_world_points_to_ego_bev(
    points_world: np.ndarray,
    T_ego2world: np.ndarray,
) -> np.ndarray:
    """
    Transform 3D world-frame points into the ego-centric BEV plane.

    Returns only (x, y) - z (height) is discarded because BEV is
    a top-down projection onto z=0.

    Parameters
    ----------
    points_world : (N, 3)
    T_ego2world  : (4, 4)

    Returns
    -------
    xy_ego : (N, 2)  - [x_lateral, y_forward] in ego frame (metres)
    """
    T_world2ego = build_world_to_ego(T_ego2world)
    pts_ego     = transform_points(points_world, T_world2ego)
    return pts_ego[:, :2]   # discard z


# ── Calibration validation ────────────────────────────────────

def validate_intrinsics(K: np.ndarray) -> None:
    """
    Assert that a camera intrinsic matrix has the expected structure.

    K must be:
        [ fx   0  cx ]
        [  0  fy  cy ]
        [  0   0   1 ]

    Raises
    ------
    AssertionError if any constraint is violated.
    """
    assert K.shape == (3, 3),          f"K must be (3,3), got {K.shape}"
    assert K[2, 2] == 1.0,             f"K[2,2] must be 1.0, got {K[2,2]}"
    assert K[0, 1] == 0.0,             f"K[0,1] (skew) must be 0, got {K[0,1]}"
    assert K[1, 0] == 0.0,             f"K[1,0] must be 0, got {K[1,0]}"
    assert K[2, 0] == 0.0,             f"K[2,0] must be 0, got {K[2,0]}"
    assert K[2, 1] == 0.0,             f"K[2,1] must be 0, got {K[2,1]}"
    assert K[0, 0] > 0,                f"fx must be positive, got {K[0,0]}"
    assert K[1, 1] > 0,                f"fy must be positive, got {K[1,1]}"


def validate_transform(T: np.ndarray, name: str = "T") -> None:
    """
    Assert that a 4×4 matrix is a valid rigid-body transform.

    Checks:
        • Shape (4,4)
        • Bottom row = [0, 0, 0, 1]
        • Rotation block R is orthonormal: R @ R.T ≈ I, det(R) ≈ +1

    Raises
    ------
    AssertionError
    """
    assert T.shape == (4, 4),    f"{name} must be (4,4), got {T.shape}"
    np.testing.assert_allclose(
        T[3], [0, 0, 0, 1], atol=1e-6,
        err_msg=f"{name} bottom row must be [0,0,0,1]"
    )
    R = T[:3, :3]
    np.testing.assert_allclose(
        R @ R.T, np.eye(3), atol=1e-5,
        err_msg=f"{name} rotation block is not orthonormal"
    )
    det = np.linalg.det(R)
    assert abs(det - 1.0) < 1e-5, \
        f"{name} rotation determinant must be +1 (got {det:.6f})"