import numpy as np
import cv2

from utils.config import SEGFORMER, BEV
from utils.geometry import build_transform_matrix, quat_to_rotation_matrix
from data.nuscenes_loader import get_camera_data, get_ego_pose, get_sample_annotations
from data.calibration import build_world_to_cam
from data.class_mappings import category_to_bev_class, DRIVABLE, VEHICLE, PEDESTRIAN
from data.bev_gt_generator import generate_map_bev_mask

IGNORE = 255


def _build_camera_context(sample_token):
    cam         = get_camera_data(sample_token)
    K           = cam["K"]
    T_cam2ego   = cam["T_cam2ego"]
    T_ego2world = cam["T_ego2world"]
    T_w2c       = build_world_to_cam(T_cam2ego, T_ego2world)
    return K, T_w2c, T_ego2world


def _scale_factors(img_h, img_w):
    return img_w / 1600.0, img_h / 900.0


def _project_point(pt_world, K, T_w2c, sx, sy):
    R, t = T_w2c[:3, :3], T_w2c[:3, 3]
    pt   = R @ pt_world + t
    if pt[2] < 0.3:
        return None
    u = K[0, 0] * (pt[0] / pt[2]) + K[0, 2]
    v = K[1, 1] * (pt[1] / pt[2]) + K[1, 2]
    return int(u * sx), int(v * sy)


def _box_corners_world(ann):
    cx, cy, cz = ann["translation"]
    w, l, h    = ann["size"]
    half = np.array([
        [-l/2, -w/2, 0], [l/2, -w/2, 0], [l/2, w/2, 0], [-l/2, w/2, 0],
        [-l/2, -w/2, h], [l/2, -w/2, h], [l/2, w/2, h], [-l/2, w/2, h],
    ])
    R      = quat_to_rotation_matrix(ann["rotation"])
    centre = np.array([cx, cy, cz])
    return (R @ half.T).T + centre


def _draw_road_labels(label, sample_token, K, T_w2c, T_ego2world, img_h, img_w):
    sx, sy      = _scale_factors(img_h, img_w)
    fx, fy      = K[0, 0], K[1, 1]
    cx_k, cy_k  = K[0, 2], K[1, 2]

    map_mask = generate_map_bev_mask(sample_token)
    ys, xs   = np.where(map_mask == DRIVABLE)
    if len(xs) == 0:
        return

    bev_size = BEV["size"]
    bev_res  = BEV["resolution"]
    centre   = bev_size / 2.0

    x_ego = (xs - centre) * bev_res
    y_ego = -(ys - centre) * bev_res

    R_e2w = T_ego2world[:3, :3]
    t_e2w = T_ego2world[:3, 3]
    pts_ego_3d = np.stack([x_ego, y_ego, np.zeros(len(xs))], axis=1)
    pts_world  = (R_e2w @ pts_ego_3d.T).T + t_e2w

    R_w2c   = T_w2c[:3, :3]
    t_w2c_v = T_w2c[:3, 3]
    pts_cam = (R_w2c @ pts_world.T).T + t_w2c_v

    valid = pts_cam[:, 2] > 0.3
    if valid.sum() == 0:
        return

    pc  = pts_cam[valid]
    u_i = np.round((fx * (pc[:, 0] / pc[:, 2]) + cx_k) * sx).astype(int)
    v_i = np.round((fy * (pc[:, 1] / pc[:, 2]) + cy_k) * sy).astype(int)

    in_b = (u_i >= 0) & (u_i < img_w) & (v_i >= 0) & (v_i < img_h)
    label[v_i[in_b], u_i[in_b]] = DRIVABLE

    road_ch = (label == DRIVABLE).astype(np.uint8)
    # morphological closing fills internal gaps, then dilate outward slightly
    # closing = dilate then erode — fills holes without expanding boundary much
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dil_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed  = cv2.morphologyEx(road_ch, cv2.MORPH_CLOSE, close_k)
    dilated = cv2.dilate(closed, dil_k)
    label[dilated > 0] = DRIVABLE


def _draw_box_labels(label, sample_token, K, T_w2c, img_h, img_w):
    sx, sy = _scale_factors(img_h, img_w)
    anns   = get_sample_annotations(sample_token)

    for ann in anns:
        bev_cls = category_to_bev_class(ann["category_name"])
        if bev_cls not in (VEHICLE, PEDESTRIAN):
            continue

        corners = _box_corners_world(ann)
        pts_img = []
        for corner in corners:
            px = _project_point(corner, K, T_w2c, sx, sy)
            if px is not None:
                pts_img.append(px)

        if len(pts_img) < 3:
            continue

        pts_arr = np.array(pts_img, dtype=np.int32)
        pts_arr[:, 0] = np.clip(pts_arr[:, 0], 0, img_w - 1)
        pts_arr[:, 1] = np.clip(pts_arr[:, 1], 0, img_h - 1)

        hull = cv2.convexHull(pts_arr)
        cv2.fillPoly(label, [hull], int(bev_cls))


def generate_pseudo_label(sample_token, img_h=None, img_w=None):
    """
    Two-source pseudo label strategy:
      Road        → BEV map ground-plane projection (z=0, geometrically correct)
      Vehicle/Ped → 3D box corner convex hull in image space (covers full body)
    """
    h = img_h or SEGFORMER["img_h"]
    w = img_w or SEGFORMER["img_w"]

    K, T_w2c, T_ego2world = _build_camera_context(sample_token)
    label = np.full((h, w), IGNORE, dtype=np.uint8)

    _draw_road_labels(label, sample_token, K, T_w2c, T_ego2world, h, w)
    _draw_box_labels(label, sample_token, K, T_w2c, h, w)

    return label


def generate_pseudo_label_batch(sample_tokens):
    return [generate_pseudo_label(t) for t in sample_tokens]