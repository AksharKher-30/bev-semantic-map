import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import pytest
from utils.config import SEGFORMER, CLASSES, BEV, PATHS, NUSCENES


_dataset_ok = (PATHS["dataroot"] / NUSCENES["version"] / "scene.json").exists()
skip_no_data = pytest.mark.skipif(not _dataset_ok, reason="nuScenes not found")


# ── model ─────────────────────────────────────────────────────

class TestSegFormerModel:
    def test_builds_without_error(self):
        from models.segformer.model import build_segformer
        model = build_segformer(pretrained=False)
        assert model is not None

    def test_head_output_channels(self):
        from models.segformer.model import build_segformer
        model = build_segformer(pretrained=False)
        out_ch = model.decode_head.classifier.out_channels
        assert out_ch == SEGFORMER["num_classes"], \
            f"Expected {SEGFORMER['num_classes']} out channels, got {out_ch}"

    def test_forward_pass_shape(self):
        from models.segformer.model import build_segformer
        model = build_segformer(pretrained=False).eval()
        x     = torch.randn(2, 3, SEGFORMER["img_h"], SEGFORMER["img_w"])
        with torch.no_grad():
            out = model(pixel_values=x)
        # SegFormer outputs at 1/4 resolution
        expected_h = SEGFORMER["img_h"] // 4
        expected_w = SEGFORMER["img_w"] // 4
        assert out.logits.shape == (2, SEGFORMER["num_classes"], expected_h, expected_w)

    def test_forward_produces_finite_values(self):
        from models.segformer.model import build_segformer
        model = build_segformer(pretrained=False).eval()
        x     = torch.randn(1, 3, SEGFORMER["img_h"], SEGFORMER["img_w"])
        with torch.no_grad():
            logits = model(pixel_values=x).logits
        assert torch.isfinite(logits).all()


# ── inference ─────────────────────────────────────────────────

class TestSegFormerInference:
    def test_inference_tensor_shape(self):
        from models.segformer.model import build_segformer
        from models.segformer.inference import run_inference_tensor
        model  = build_segformer(pretrained=False)
        tensor = torch.randn(3, SEGFORMER["img_h"], SEGFORMER["img_w"])
        mask   = run_inference_tensor(model, tensor, device=torch.device("cpu"))
        assert mask.shape == (SEGFORMER["img_h"], SEGFORMER["img_w"])

    def test_inference_values_in_valid_range(self):
        from models.segformer.model import build_segformer
        from models.segformer.inference import run_inference_tensor
        model  = build_segformer(pretrained=False)
        tensor = torch.randn(3, SEGFORMER["img_h"], SEGFORMER["img_w"])
        mask   = run_inference_tensor(model, tensor, device=torch.device("cpu"))
        assert mask.min() >= 0
        assert mask.max() < SEGFORMER["num_classes"]

    def test_inference_dtype_uint8(self):
        from models.segformer.model import build_segformer
        from models.segformer.inference import run_inference_tensor
        model  = build_segformer(pretrained=False)
        tensor = torch.randn(3, SEGFORMER["img_h"], SEGFORMER["img_w"])
        mask   = run_inference_tensor(model, tensor, device=torch.device("cpu"))
        assert mask.dtype == np.uint8


# ── losses ────────────────────────────────────────────────────

class TestLosses:
    def test_segformer_loss_scalar(self):
        from train.losses import segformer_loss
        logits = torch.randn(2, SEGFORMER["num_classes"], 128, 256)
        labels = torch.zeros(2, SEGFORMER["img_h"], SEGFORMER["img_w"], dtype=torch.long)
        loss   = segformer_loss(logits, labels)
        assert loss.ndim == 0          # scalar
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_ignore_index_excluded(self):
        from train.losses import segformer_loss
        logits = torch.randn(1, SEGFORMER["num_classes"], 128, 256)
        # all labels = ignore → loss should still be finite (no valid pixels)
        labels = torch.full(
            (1, SEGFORMER["img_h"], SEGFORMER["img_w"]),
            SEGFORMER["ignore_index"],
            dtype=torch.long
        )
        loss = segformer_loss(logits, labels)
        assert torch.isfinite(loss)

    def test_iou_perfect_prediction(self):
        from train.losses import compute_iou
        preds  = torch.zeros(2, 4, 4, dtype=torch.long)
        labels = torch.zeros(2, 4, 4, dtype=torch.long)
        ious   = compute_iou(preds, labels, num_classes=SEGFORMER["num_classes"])
        assert ious[0] == pytest.approx(1.0, abs=1e-4)

    def test_iou_no_overlap_is_zero(self):
        from train.losses import compute_iou
        preds  = torch.zeros(2, 4, 4, dtype=torch.long)
        labels = torch.ones(2, 4, 4, dtype=torch.long)
        ious   = compute_iou(preds, labels, num_classes=SEGFORMER["num_classes"])
        assert ious[0] == pytest.approx(0.0, abs=1e-4)


# ── colorize ──────────────────────────────────────────────────

class TestColorize:
    def test_colorize_seg_shape(self):
        from viz.colorize import colorize_seg
        mask = np.zeros((512, 1024), dtype=np.uint8)
        out  = colorize_seg(mask)
        assert out.shape == (512, 1024, 3)
        assert out.dtype == np.uint8

    def test_colorize_bev_multichannel(self):
        from viz.colorize import colorize_bev
        bev = np.zeros((CLASSES["num_classes"], 200, 200), dtype=np.float32)
        out = colorize_bev(bev)
        assert out.shape == (200, 200, 3)

    def test_colorize_bev_label_map(self):
        from viz.colorize import colorize_bev
        bev = np.zeros((200, 200), dtype=np.uint8)
        out = colorize_bev(bev)
        assert out.shape == (200, 200, 3)

    def test_overlay_seg_shape(self):
        from viz.colorize import overlay_seg
        img  = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        mask = np.zeros((512, 1024), dtype=np.uint8)
        out  = overlay_seg(img, mask)
        assert out.shape == (512, 1024, 3)


# ── pseudo labels ─────────────────────────────────────────────

class TestPseudoLabels:
    @skip_no_data
    def test_pseudo_label_shape(self):
        from data.pseudo_labels import generate_pseudo_label
        from data.nuscenes_loader import get_all_sample_tokens
        token = get_all_sample_tokens()[0]
        label = generate_pseudo_label(token)
        assert label.shape == (SEGFORMER["img_h"], SEGFORMER["img_w"])

    @skip_no_data
    def test_pseudo_label_dtype(self):
        from data.pseudo_labels import generate_pseudo_label
        from data.nuscenes_loader import get_all_sample_tokens
        token = get_all_sample_tokens()[0]
        label = generate_pseudo_label(token)
        assert label.dtype == np.uint8

    @skip_no_data
    def test_pseudo_label_values(self):
        from data.pseudo_labels import generate_pseudo_label, IGNORE
        from data.nuscenes_loader import get_all_sample_tokens
        token  = get_all_sample_tokens()[0]
        label  = generate_pseudo_label(token)
        unique = set(np.unique(label).tolist())
        valid  = {0, 1, 2, IGNORE}
        assert unique.issubset(valid), f"Unexpected values: {unique - valid}"

    @skip_no_data
    def test_pseudo_label_not_all_ignore(self):
        from data.pseudo_labels import generate_pseudo_label, IGNORE
        from data.nuscenes_loader import get_all_sample_tokens
        token = get_all_sample_tokens()[0]
        label = generate_pseudo_label(token)
        assert (label != IGNORE).sum() > 0, "All pixels are ignore — projection failed"


# ── checkpointing ─────────────────────────────────────────────

class TestCheckpointing:
    def test_best_path_returns_pth(self):
        from train.checkpointing import best_path
        assert str(best_path()).endswith(".pth")

    def test_save_and_load(self, tmp_path):
        from models.segformer.model import build_segformer
        from train.checkpointing import save, load
        from torch.optim import AdamW
        import os

        model     = build_segformer(pretrained=False)
        optimizer = AdamW(model.parameters(), lr=1e-4)

        ckpt_path = tmp_path / "test.pth"
        torch.save({
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch":     5,
            "best_miou": 0.42,
        }, ckpt_path)

        model2 = build_segformer(pretrained=False)
        epoch, miou = load(str(ckpt_path), model2, device="cpu")
        assert epoch == 5
        assert abs(miou - 0.42) < 1e-5


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["python", "-m", "pytest", __file__, "-v"]))