import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import pytest
from utils.config import CLASSES, BEV


def _make_logits(value, C=None, size=None):
    C    = C    or CLASSES["num_classes"]
    size = size or BEV["size"]
    return torch.full((1, C, size, size), value)


def _make_gt(value, C=None, size=None):
    C    = C    or CLASSES["num_classes"]
    size = size or BEV["size"]
    return torch.full((1, C, size, size), float(value))


class TestBEVIoUMetric:
    def test_perfect_prediction_miou_is_one(self):
        from train.losses import BEVIoUMetric
        m  = BEVIoUMetric()
        gt = _make_gt(1)
        # large positive logit → sigmoid ≈ 1 → predicted positive
        m.update(_make_logits(10.0), gt)
        r  = m.compute()
        assert abs(r["mIoU"] - 1.0) < 1e-3, f"mIoU={r['mIoU']:.4f} ≠ 1.0"

    def test_all_miss_miou_is_zero(self):
        from train.losses import BEVIoUMetric
        m  = BEVIoUMetric()
        gt = _make_gt(1)
        # large negative logit → sigmoid ≈ 0 → predicted negative
        m.update(_make_logits(-10.0), gt)
        r  = m.compute()
        assert abs(r["mIoU"] - 0.0) < 1e-3

    def test_all_false_positive_miou_is_zero(self):
        from train.losses import BEVIoUMetric
        m  = BEVIoUMetric()
        gt = _make_gt(0)   # GT all negative
        m.update(_make_logits(10.0), gt)
        r  = m.compute()
        assert abs(r["mIoU"] - 0.0) < 1e-3

    def test_result_has_all_class_keys(self):
        from train.losses import BEVIoUMetric
        m = BEVIoUMetric()
        m.update(_make_logits(0.0), _make_gt(0))
        r = m.compute()
        assert "mIoU" in r
        for name in CLASSES["names"]:
            assert name in r, f"Missing class '{name}'"

    def test_miou_is_mean_of_class_ious(self):
        from train.losses import BEVIoUMetric
        m = BEVIoUMetric()
        m.update(_make_logits(10.0), _make_gt(1))
        r = m.compute()
        class_mean = sum(r[n] for n in CLASSES["names"]) / CLASSES["num_classes"]
        assert abs(r["mIoU"] - class_mean) < 1e-5

    def test_accumulation_across_batches(self):
        """Results should be identical whether computed per-batch or accumulated."""
        from train.losses import BEVIoUMetric
        m = BEVIoUMetric()
        for _ in range(5):
            m.update(_make_logits(10.0), _make_gt(1))
        r = m.compute()
        assert abs(r["mIoU"] - 1.0) < 1e-3

    def test_reset_clears_state(self):
        from train.losses import BEVIoUMetric
        m = BEVIoUMetric()
        m.update(_make_logits(10.0), _make_gt(1))
        m.reset()
        r = m.compute()
        # after reset with no updates, tp=fp=fn=0 → iou = 0/(0+0+0+eps) ≈ 0
        assert r["mIoU"] < 1e-3

    def test_partial_overlap(self):
        """Left half predicted positive, right half missed → IoU ≈ 0.5."""
        from train.losses import BEVIoUMetric
        sz = 10
        C  = CLASSES["num_classes"]   # must match metric's n_cls
        m  = BEVIoUMetric()

        # all GT positive across all channels
        gt = torch.ones(1, C, sz, sz)

        # left half positive, right half negative across all channels
        logits = torch.full((1, C, sz, sz), -10.0)
        logits[:, :, :, :sz//2] = 10.0

        m.update(logits, gt)
        r = m.compute()
        # TP = sz*(sz//2), FP = 0, FN = sz*(sz//2) → IoU = 0.5 per class
        assert abs(r["mIoU"] - 0.5) < 0.05, f"Expected ~0.5, got {r['mIoU']:.4f}"

    def test_threshold_effect(self):
        """Lower threshold → more positives → different IoU."""
        from train.losses import BEVIoUMetric
        # logit = 0.0 → sigmoid = 0.5
        # threshold 0.5: NOT predicted (0.5 is NOT > 0.5)
        # threshold 0.4: predicted (0.5 > 0.4)
        gt = _make_gt(1)

        m_strict = BEVIoUMetric(threshold=0.5)
        m_strict.update(_make_logits(0.0), gt)
        r_strict = m_strict.compute()

        m_loose = BEVIoUMetric(threshold=0.4)
        m_loose.update(_make_logits(0.0), gt)
        r_loose = m_loose.compute()

        assert r_loose["mIoU"] > r_strict["mIoU"]


class TestComputeIoUFromLogits:
    def test_returns_dict(self):
        from eval.iou_metric import compute_iou_from_logits
        logits = _make_logits(10.0)
        gt     = _make_gt(1)
        r = compute_iou_from_logits(logits, gt)
        assert isinstance(r, dict)
        assert "mIoU" in r

    def test_perfect_prediction(self):
        from eval.iou_metric import compute_iou_from_logits
        r = compute_iou_from_logits(_make_logits(10.0), _make_gt(1))
        assert abs(r["mIoU"] - 1.0) < 1e-3


class TestErrorViz:
    def test_error_map_shape(self):
        from eval.error_viz import make_error_map
        logits = _make_logits(10.0).squeeze(0)   # (C, H, W)
        gt     = _make_gt(1).squeeze(0)
        err    = make_error_map(logits, gt, class_idx=0)
        assert err.shape == (BEV["size"], BEV["size"], 3)

    def test_error_map_dtype(self):
        from eval.error_viz import make_error_map
        import numpy as np
        logits = _make_logits(10.0).squeeze(0)
        gt     = _make_gt(1).squeeze(0)
        err    = make_error_map(logits, gt, class_idx=0)
        assert err.dtype == np.uint8

    def test_all_tp_gives_green(self):
        from eval.error_viz import make_error_map, TP_COLOR
        logits = _make_logits(10.0).squeeze(0)
        gt     = _make_gt(1).squeeze(0)
        err    = make_error_map(logits, gt, class_idx=0)
        # all pixels should be green (TP)
        assert (err == TP_COLOR).all()

    def test_all_fp_gives_red(self):
        from eval.error_viz import make_error_map, FP_COLOR
        logits = _make_logits(10.0).squeeze(0)   # predicted positive
        gt     = _make_gt(0).squeeze(0)           # GT negative
        err    = make_error_map(logits, gt, class_idx=0)
        assert (err == FP_COLOR).all()

    def test_all_fn_gives_blue(self):
        from eval.error_viz import make_error_map, FN_COLOR
        logits = _make_logits(-10.0).squeeze(0)   # predicted negative
        gt     = _make_gt(1).squeeze(0)            # GT positive
        err    = make_error_map(logits, gt, class_idx=0)
        assert (err == FN_COLOR).all()

    def test_all_class_error_maps_keys(self):
        from eval.error_viz import make_all_class_error_maps
        logits = _make_logits(0.0).squeeze(0)
        gt     = _make_gt(0).squeeze(0)
        maps   = make_all_class_error_maps(logits, gt)
        for name in CLASSES["names"]:
            assert name in maps


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["python", "-m", "pytest", __file__, "-v"]))