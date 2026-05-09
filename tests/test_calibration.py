# ============================================================
# tests/test_calibration.py  -  Phase 1
# ============================================================
# Run: python -m pytest tests/test_calibration.py -v
#      OR: python tests/test_calibration.py
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from utils.geometry import (
    quat_to_rotation_matrix,
    build_transform_matrix,
    invert_transform,
    transform_points,
    ego_to_bev_pixels,
)
from data.calibration import (
    build_cam_to_world,
    build_world_to_cam,
    build_world_to_ego,
    validate_intrinsics,
    validate_transform,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def identity_transform():
    return np.eye(4, dtype=np.float64)

@pytest.fixture
def sample_K():
    """Realistic nuScenes-like intrinsic matrix."""
    return np.array([
        [1266.417,    0.0,    816.267],
        [   0.0,   1266.417, 491.507],
        [   0.0,      0.0,     1.0  ],
    ], dtype=np.float64)

@pytest.fixture
def sample_quaternion_identity():
    return [1.0, 0.0, 0.0, 0.0]   # identity rotation

@pytest.fixture
def sample_quaternion_yaw90():
    """90-degree yaw rotation around z-axis."""
    import math
    return [math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)]


# ── Quaternion tests ──────────────────────────────────────────

class TestQuaternion:
    def test_identity_quaternion_gives_identity_R(self, sample_quaternion_identity):
        R = quat_to_rotation_matrix(sample_quaternion_identity)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-8)

    def test_rotation_matrix_is_3x3(self, sample_quaternion_identity):
        R = quat_to_rotation_matrix(sample_quaternion_identity)
        assert R.shape == (3, 3)

    def test_rotation_matrix_is_orthonormal(self, sample_quaternion_yaw90):
        R = quat_to_rotation_matrix(sample_quaternion_yaw90)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_rotation_determinant_is_one(self, sample_quaternion_yaw90):
        R = quat_to_rotation_matrix(sample_quaternion_yaw90)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_yaw90_rotates_x_to_y(self, sample_quaternion_yaw90):
        """A 90° yaw should map [1,0,0] → [0,1,0]."""
        R  = quat_to_rotation_matrix(sample_quaternion_yaw90)
        x_hat = np.array([1.0, 0.0, 0.0])
        result = R @ x_hat
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-6)


# ── Build transform matrix ────────────────────────────────────

class TestBuildTransform:
    def test_shape(self, sample_quaternion_identity):
        T = build_transform_matrix([1, 2, 3], sample_quaternion_identity)
        assert T.shape == (4, 4)

    def test_bottom_row(self, sample_quaternion_identity):
        T = build_transform_matrix([0, 0, 0], sample_quaternion_identity)
        np.testing.assert_array_equal(T[3], [0, 0, 0, 1])

    def test_translation_stored_correctly(self, sample_quaternion_identity):
        T = build_transform_matrix([5.0, -3.0, 1.5], sample_quaternion_identity)
        np.testing.assert_allclose(T[:3, 3], [5.0, -3.0, 1.5])

    def test_identity_quaternion_gives_identity_R_block(self, sample_quaternion_identity):
        T = build_transform_matrix([0, 0, 0], sample_quaternion_identity)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-8)


# ── Invert transform ──────────────────────────────────────────

class TestInvertTransform:
    def test_inverse_of_identity_is_identity(self):
        T_inv = invert_transform(np.eye(4))
        np.testing.assert_allclose(T_inv, np.eye(4), atol=1e-10)

    def test_T_times_Tinv_is_identity(self, sample_quaternion_yaw90):
        T     = build_transform_matrix([1.0, 2.0, 0.5], sample_quaternion_yaw90)
        T_inv = invert_transform(T)
        product = T @ T_inv
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_Tinv_times_T_is_identity(self, sample_quaternion_yaw90):
        T     = build_transform_matrix([1.0, 2.0, 0.5], sample_quaternion_yaw90)
        T_inv = invert_transform(T)
        product = T_inv @ T
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)


# ── Transform points ──────────────────────────────────────────

class TestTransformPoints:
    def test_identity_transform_leaves_points_unchanged(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = transform_points(pts, np.eye(4))
        np.testing.assert_allclose(out, pts)

    def test_pure_translation(self, sample_quaternion_identity):
        T   = build_transform_matrix([10, 20, 30], sample_quaternion_identity)
        pts = np.array([[0.0, 0.0, 0.0]])
        out = transform_points(pts, T)
        np.testing.assert_allclose(out, [[10.0, 20.0, 30.0]])

    def test_output_shape_preserved(self, sample_quaternion_identity):
        T   = build_transform_matrix([1, 0, 0], sample_quaternion_identity)
        pts = np.random.randn(100, 3)
        out = transform_points(pts, T)
        assert out.shape == (100, 3)


# ── Calibration chain ─────────────────────────────────────────

class TestCalibrationChain:
    @pytest.fixture
    def sample_transforms(self, sample_quaternion_yaw90):
        T_cam2ego   = build_transform_matrix([0.0, 0.0, 1.5], sample_quaternion_yaw90)
        T_ego2world = build_transform_matrix([10.0, 20.0, 0.0], [1,0,0,0])
        return T_cam2ego, T_ego2world

    def test_cam2world_shape(self, sample_transforms):
        T_c2e, T_e2w = sample_transforms
        T = build_cam_to_world(T_c2e, T_e2w)
        assert T.shape == (4, 4)

    def test_world2cam_inverts_cam2world(self, sample_transforms):
        T_c2e, T_e2w = sample_transforms
        T_c2w = build_cam_to_world(T_c2e, T_e2w)
        T_w2c = build_world_to_cam(T_c2e, T_e2w)
        np.testing.assert_allclose(T_c2w @ T_w2c, np.eye(4), atol=1e-10)

    def test_round_trip_point(self, sample_transforms):
        """Point transformed cam→world→cam should equal original."""
        T_c2e, T_e2w = sample_transforms
        T_c2w = build_cam_to_world(T_c2e, T_e2w)
        T_w2c = build_world_to_cam(T_c2e, T_e2w)
        pt = np.array([[3.0, 1.0, 10.0]])
        pt_world = transform_points(pt, T_c2w)
        pt_back  = transform_points(pt_world, T_w2c)
        np.testing.assert_allclose(pt_back, pt, atol=1e-9)


# ── Validate intrinsics / transforms ─────────────────────────

class TestValidation:
    def test_valid_K_passes(self, sample_K):
        validate_intrinsics(sample_K)   # no assertion error

    def test_invalid_K_wrong_shape(self):
        with pytest.raises(AssertionError):
            validate_intrinsics(np.eye(4))

    def test_invalid_K_wrong_bottom_row(self, sample_K):
        bad_K = sample_K.copy()
        bad_K[2, 2] = 2.0
        with pytest.raises(AssertionError):
            validate_intrinsics(bad_K)

    def test_valid_transform_passes(self, sample_quaternion_yaw90):
        T = build_transform_matrix([1, 2, 3], sample_quaternion_yaw90)
        validate_transform(T, "T_test")   # no assertion error

    def test_invalid_transform_non_orthogonal(self):
        T = np.eye(4)
        T[0, 0] = 2.0   # breaks orthonormality
        with pytest.raises(AssertionError):
            validate_transform(T, "T_bad")


# ── BEV pixel conversion ──────────────────────────────────────

class TestBEVPixels:
    def test_ego_origin_maps_to_grid_centre(self):
        pts  = np.array([[0.0, 0.0]])   # ego position
        pix, valid = ego_to_bev_pixels(pts, bev_size=200, bev_resolution=0.5)
        assert valid[0]
        assert pix[0, 0] == 100   # col = centre
        assert pix[0, 1] == 100   # row = centre

    def test_forward_point_maps_above_centre(self):
        """A point 10m ahead should have smaller row index (row = up in image)."""
        pts  = np.array([[0.0, 10.0]])   # 10m forward
        pix, valid = ego_to_bev_pixels(pts, bev_size=200, bev_resolution=0.5)
        assert valid[0]
        assert pix[0, 1] < 100   # row should be above centre

    def test_out_of_range_point_is_invalid(self):
        pts  = np.array([[999.0, 999.0]])   # way outside 100m coverage
        _, valid = ego_to_bev_pixels(pts, bev_size=200, bev_resolution=0.5)
        assert not valid[0]


# ── Standalone runner ─────────────────────────────────────────

if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.call(["python", "-m", "pytest", __file__, "-v"]))