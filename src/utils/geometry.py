import numpy as np
from pyquaternion import Quaternion


# ── Quaternion ────────────────────────────────────────────────

def quat_to_rotation_matrix(q_wxyz) -> np.ndarray:
    """
    Convert a quaternion [w, x, y, z] to a 3×3 rotation matrix.

    nuScenes stores all rotations as [w, x, y, z] quaternions.
    pyquaternion expects the same order.

    Parameters
    ----------
    q_wxyz : array-like, shape (4,)
        Quaternion components [w, x, y, z].

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Orthonormal rotation matrix satisfying R @ R.T == I.
    """
    q = Quaternion(q_wxyz)
    return q.rotation_matrix


def build_transform_matrix(translation, rotation_quat) -> np.ndarray:
    """
    Build a 4×4 rigid-body transform matrix from translation + quaternion.

    Layout:
        [ R  | t ]
        [ 0  | 1 ]

    This is the standard homogeneous transform used to move points
    from one coordinate frame to another:
        P_dest = T @ P_src   (P in homogeneous coords [x,y,z,1])

    Parameters
    ----------
    translation : array-like, shape (3,)
        [x, y, z] translation vector.
    rotation_quat : array-like, shape (4,)
        [w, x, y, z] quaternion.

    Returns
    -------
    T : np.ndarray, shape (4, 4)
    """
    R = quat_to_rotation_matrix(rotation_quat)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = np.array(translation, dtype=np.float64)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Efficiently invert a rigid-body 4×4 transform.

    For orthonormal R:  inv(T) = [ R.T | -R.T @ t ]
                                  [  0  |     1    ]

    This is faster and more numerically stable than np.linalg.inv
    for rotation+translation matrices.

    Parameters
    ----------
    T : np.ndarray, shape (4, 4)

    Returns
    -------
    T_inv : np.ndarray, shape (4, 4)
    """
    R = T[:3, :3]
    t = T[:3,  3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] =  R.T
    T_inv[:3,  3] = -R.T @ t
    return T_inv


# ── Point transforms ──────────────────────────────────────────

def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply a 4×4 transform matrix to an array of 3D points.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Array of N 3D points.
    T : np.ndarray, shape (4, 4)
        Rigid-body transform.

    Returns
    -------
    transformed : np.ndarray, shape (N, 3)
    """
    N = points.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.hstack([points.astype(np.float64), ones])   # (N, 4)
    out_h = (T @ pts_h.T).T                                  # (N, 4)
    return out_h[:, :3]


def project_to_image(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 3D points in camera frame to 2D pixel coordinates.

    Uses the standard pinhole camera model:
        u = fx * (x/z) + cx
        v = fy * (y/z) + cy

    Parameters
    ----------
    points_cam : np.ndarray, shape (N, 3)
        Points in camera coordinate frame [x, y, z].
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    pixels : np.ndarray, shape (N, 2)  - [u, v] per point
    depths  : np.ndarray, shape (N,)   - z depth per point
    valid   : np.ndarray, shape (N,)   - bool, True if in front of camera
    """
    z = points_cam[:, 2]
    valid = z > 0.1   # only points in front of the camera

    # Avoid division by zero
    z_safe = np.where(valid, z, 1.0)

    u = K[0, 0] * (points_cam[:, 0] / z_safe) + K[0, 2]
    v = K[1, 1] * (points_cam[:, 1] / z_safe) + K[1, 2]

    pixels = np.stack([u, v], axis=1)   # (N, 2)
    return pixels, z, valid


# ── BEV coordinate helpers ────────────────────────────────────

def ego_to_bev_pixels(
    xy_ego: np.ndarray,
    bev_size: int,
    bev_resolution: float,
) -> np.ndarray:
    """
    Convert ego-frame (x, y) coordinates [metres] to BEV pixel indices.

    BEV convention used throughout this project:
        • Ego vehicle always at pixel (bev_size/2, bev_size/2).
        • +x = right (increasing column index).
        • +y = forward (decreasing row index - forward is UP in the image).
        • Resolution = metres per pixel.

    Parameters
    ----------
    xy_ego : np.ndarray, shape (N, 2)
        Points in ego frame [x, y] in metres.
    bev_size : int
        BEV grid side length in pixels (e.g. 200).
    bev_resolution : float
        Metres per pixel (e.g. 0.5 → 100 m × 100 m coverage).

    Returns
    -------
    pixels : np.ndarray, shape (N, 2), dtype int
        [col (ix), row (iy)] pixel indices.
    valid : np.ndarray, shape (N,), dtype bool
        True if the point falls within the BEV grid.
    """
    center = bev_size / 2.0
    ix = (xy_ego[:, 0] / bev_resolution + center).astype(int)   # lateral  → col
    iy = (center - xy_ego[:, 1] / bev_resolution).astype(int)   # forward  → row (flipped)
    pixels = np.stack([ix, iy], axis=1)
    valid = (ix >= 0) & (ix < bev_size) & (iy >= 0) & (iy < bev_size)
    return pixels, valid