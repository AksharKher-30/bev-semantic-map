import cv2
import numpy as np
from utils.config import BEV, SEGFORMER


def compute_ipm_homography(K, camera_height=None, bev_size=None, bev_res=None):
    """
    Compute the homography matrix H that maps front-view image pixels
    to BEV (bird's eye view) pixels, assuming a flat ground plane (z=0).

    Math:
        For a point on the ground plane at (x_ego, y_ego, z=0):
            u = fx * (x_ego / y_ego) + cx     (image column)
            v = fy * (-h   / y_ego) + cy      (image row, h = camera height)

        Four such ground-truth point pairs define H uniquely via
        cv2.getPerspectiveTransform (solves 8-DOF system).

    Parameters
    ----------
    K             : (3,3) camera intrinsic matrix from nuScenes calibration
    camera_height : metres above ground (default from config)
    bev_size      : BEV grid side in pixels (default from config)
    bev_res       : metres per pixel (default from config)

    Returns
    -------
    H : (3,3) float32 homography matrix
    """
    h   = camera_height or BEV["camera_height"]
    sz  = bev_size      or BEV["size"]
    res = bev_res       or BEV["resolution"]

    fx, fy   = K[0, 0], K[1, 1]
    cx, cy   = K[0, 2], K[1, 2]

    # scale factors: nuScenes native 1600×900 → model input size
    sx = SEGFORMER["img_w"] / 1600.0
    sy = SEGFORMER["img_h"] / 900.0

    # ground truth world points (x_ego=lateral, y_ego=forward, metres)
    world_pts = np.float32([
        [-3.0,  8.0],
        [ 3.0,  8.0],
        [-6.0, 40.0],
        [ 6.0, 40.0],
    ])

    # project world points → image pixels
    img_pts = []
    for x, y in world_pts:
        u = (fx * (x / y) + cx) * sx
        v = (fy * (-h / y) + cy) * sy
        img_pts.append([u, v])
    img_pts = np.float32(img_pts)

    # project world points → BEV pixels
    # ego at centre of grid; +x=right (col), +y=forward (row decreases)
    centre = sz / 2.0
    bev_pts = []
    for x, y in world_pts:
        bev_u = centre + x / res
        bev_v = centre - y / res
        bev_pts.append([bev_u, bev_v])
    bev_pts = np.float32(bev_pts)

    H = cv2.getPerspectiveTransform(img_pts, bev_pts)
    return H.astype(np.float32)


def get_camera_height_from_nusc(sample_token):
    """
    Read actual camera height from nuScenes calibration for a given sample.
    More accurate than the config default for samples with non-standard mounting.

    Returns height in metres (z-component of T_cam2ego translation).
    """
    from data.nuscenes_loader import get_camera_data
    cam = get_camera_data(sample_token)
    # T_cam2ego[:3,3] = [tx, ty, tz] — tz is the height of the sensor above ego
    return float(cam["T_cam2ego"][2, 3])