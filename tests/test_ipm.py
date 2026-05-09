import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import cv2
import pytest

from utils.config import BEV, SEGFORMER, CLASSES, PATHS, NUSCENES

_dataset_ok = (PATHS["dataroot"] / NUSCENES["version"] / "scene.json").exists()
skip_no_data = pytest.mark.skipif(not _dataset_ok, reason="nuScenes not found")

# ── sample K for tests (realistic nuScenes values) ────────────
SAMPLE_K = np.array([
    [1266.417,    0.0,    816.267],
    [   0.0,  1266.417,  491.507],
    [   0.0,     0.0,      1.0  ],
], dtype=np.float64)


# ── homography ────────────────────────────────────────────────

class TestHomography:
    def test_returns_3x3(self):
        from ipm.homography import compute_ipm_homography
        H = compute_ipm_homography(SAMPLE_K)
        assert H.shape == (3, 3)

    def test_dtype_float32(self):
        from ipm.homography import compute_ipm_homography
        H = compute_ipm_homography(SAMPLE_K)
        assert H.dtype == np.float32

    def test_finite_values(self):
        from ipm.homography import compute_ipm_homography
        H = compute_ipm_homography(SAMPLE_K)
        assert np.isfinite(H).all()

    def test_invertible(self):
        from ipm.homography import compute_ipm_homography
        H     = compute_ipm_homography(SAMPLE_K)
        H_inv = np.linalg.inv(H)
        prod  = H.astype(np.float64) @ H_inv
        np.testing.assert_allclose(prod, np.eye(3), atol=1e-4)

    def test_different_heights_give_different_H(self):
        from ipm.homography import compute_ipm_homography
        H1 = compute_ipm_homography(SAMPLE_K, camera_height=1.5)
        H2 = compute_ipm_homography(SAMPLE_K, camera_height=2.0)
        assert not np.allclose(H1, H2)

    def test_forward_point_maps_to_upper_bev(self):
        """
        A point directly ahead (x=0, y=20m) should map to
        above-centre in BEV (smaller row index than centre).
        """
        from ipm.homography import compute_ipm_homography
        H  = compute_ipm_homography(SAMPLE_K, bev_size=200, bev_res=0.5)
        sz = 200
        h  = BEV["camera_height"]
        fx, fy = SAMPLE_K[0,0], SAMPLE_K[1,1]
        cx, cy = SAMPLE_K[0,2], SAMPLE_K[1,2]
        sx = SEGFORMER["img_w"] / 1600.0
        sy = SEGFORMER["img_h"] / 900.0

        # project (x=0, y=20) to image
        x, y = 0.0, 20.0
        u = (fx * (x / y) + cx) * sx
        v = (fy * (-h / y) + cy) * sy

        pt_img = np.array([[[u, v]]], dtype=np.float32)
        pt_bev = cv2.perspectiveTransform(pt_img, H)[0, 0]

        # forward point → row < centre (forward = up in BEV image)
        assert pt_bev[1] < sz / 2, \
            f"Forward point row {pt_bev[1]:.1f} should be < centre {sz/2}"


# ── warp ──────────────────────────────────────────────────────

class TestWarp:
    def test_output_shape(self):
        from ipm.homography import compute_ipm_homography
        from ipm.warp import apply_ipm
        H    = compute_ipm_homography(SAMPLE_K)
        mask = np.zeros((SEGFORMER["img_h"], SEGFORMER["img_w"]), dtype=np.uint8)
        out  = apply_ipm(mask, H)
        assert out.shape == (BEV["size"], BEV["size"])

    def test_output_dtype_uint8(self):
        from ipm.homography import compute_ipm_homography
        from ipm.warp import apply_ipm
        H    = compute_ipm_homography(SAMPLE_K)
        mask = np.zeros((SEGFORMER["img_h"], SEGFORMER["img_w"]), dtype=np.uint8)
        out  = apply_ipm(mask, H)
        assert out.dtype == np.uint8

    def test_class_labels_preserved(self):
        """Class labels must not be interpolated — only discrete values allowed."""
        from ipm.homography import compute_ipm_homography
        from ipm.warp import apply_ipm
        H    = compute_ipm_homography(SAMPLE_K)
        mask = np.zeros((SEGFORMER["img_h"], SEGFORMER["img_w"]), dtype=np.uint8)
        mask[400:, :] = 0    # road in lower half
        mask[200:400, 400:800] = 1   # vehicle patch
        out  = apply_ipm(mask, H)
        unique = set(np.unique(out).tolist())
        assert unique.issubset({0, 1, 2, 255}), \
            f"Non-discrete values in warp output: {unique - {0,1,2,255}}"

    def test_custom_bev_size(self):
        from ipm.homography import compute_ipm_homography
        from ipm.warp import apply_ipm
        H    = compute_ipm_homography(SAMPLE_K, bev_size=100)
        mask = np.zeros((SEGFORMER["img_h"], SEGFORMER["img_w"]), dtype=np.uint8)
        out  = apply_ipm(mask, H, bev_size=100)
        assert out.shape == (100, 100)

    def test_bev_mask_to_channels_shape(self):
        from ipm.warp import bev_mask_to_channels
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:100, 50:100] = 1
        out  = bev_mask_to_channels(mask)
        assert out.shape == (CLASSES["num_classes"], 200, 200)
        assert out.dtype == np.float32

    def test_bev_mask_to_channels_binary(self):
        from ipm.warp import bev_mask_to_channels
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:100, 50:100] = 1
        out  = bev_mask_to_channels(mask)
        assert set(np.unique(out).tolist()).issubset({0.0, 1.0})

    def test_bev_mask_to_channels_correct_class(self):
        from ipm.warp import bev_mask_to_channels
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:100, 50:100] = 1   # vehicle region
        out  = bev_mask_to_channels(mask)
        assert out[1, 75, 75] == 1.0   # vehicle channel hot
        assert out[0, 75, 75] == 0.0   # road channel cold


# ── pipeline (dataset required) ───────────────────────────────

class TestIPMPipeline:
    @skip_no_data
    def test_pipeline_output_keys(self):
        import cv2
        from data.nuscenes_loader import get_all_sample_tokens, get_camera_data
        from models.segformer.model import build_segformer_zero_shot
        from ipm.pipeline import run_ipm_pipeline
        import torch

        token    = get_all_sample_tokens()[0]
        cam      = get_camera_data(token)
        bgr      = cv2.imread(str(cam["image_path"]))
        model    = build_segformer_zero_shot()
        device   = torch.device("cpu")
        result   = run_ipm_pipeline(bgr, cam["K"], sample_token=token,
                                    model=model, device=device)

        for key in ["seg_mask_3", "bev_mask", "bev_channels", "H"]:
            assert key in result, f"Missing key: {key}"

    @skip_no_data
    def test_pipeline_shapes(self):
        import cv2
        from data.nuscenes_loader import get_all_sample_tokens, get_camera_data
        from models.segformer.model import build_segformer_zero_shot
        from ipm.pipeline import run_ipm_pipeline
        import torch

        token  = get_all_sample_tokens()[0]
        cam    = get_camera_data(token)
        bgr    = cv2.imread(str(cam["image_path"]))
        model  = build_segformer_zero_shot()
        result = run_ipm_pipeline(bgr, cam["K"], sample_token=token,
                                  model=model, device=torch.device("cpu"))

        assert result["seg_mask_3"].shape == (SEGFORMER["img_h"], SEGFORMER["img_w"])
        assert result["bev_mask"].shape   == (BEV["size"], BEV["size"])
        assert result["bev_channels"].shape == (CLASSES["num_classes"],
                                                BEV["size"], BEV["size"])
        assert result["H"].shape == (3, 3)

    @skip_no_data
    def test_pipeline_bev_values(self):
        import cv2
        from data.nuscenes_loader import get_all_sample_tokens, get_camera_data
        from models.segformer.model import build_segformer_zero_shot
        from ipm.pipeline import run_ipm_pipeline
        import torch

        token  = get_all_sample_tokens()[0]
        cam    = get_camera_data(token)
        bgr    = cv2.imread(str(cam["image_path"]))
        model  = build_segformer_zero_shot()
        result = run_ipm_pipeline(bgr, cam["K"], sample_token=token,
                                  model=model, device=torch.device("cpu"))

        unique = set(np.unique(result["bev_mask"]).tolist())
        assert unique.issubset({0, 1, 2, 255}), \
            f"Unexpected BEV values: {unique}"

        ch = result["bev_channels"]
        assert ch.min() >= 0.0 and ch.max() <= 1.0


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["python", "-m", "pytest", __file__, "-v"]))