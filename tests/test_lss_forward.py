import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import pytest
from utils.config import LSS, BEV, CLASSES, SEGFORMER


# ── helpers ───────────────────────────────────────────────────

def make_batch(B=2, device=None):
    """Create a minimal batch matching the DataLoader output format."""
    dev = device or torch.device("cpu")
    H, W = SEGFORMER["img_h"], SEGFORMER["img_w"]
    return {
        "image"      : torch.randn(B, 3, H, W, device=dev),
        "K"          : torch.eye(3, device=dev).unsqueeze(0).repeat(B, 1, 1) * 1266.0,
        "T_cam2ego"  : torch.eye(4, device=dev).unsqueeze(0).repeat(B, 1, 1),
    }


# ── depth bins ────────────────────────────────────────────────

class TestDepthBins:
    def test_shape(self):
        from models.lss.depth_bins import get_depth_bins
        bins = get_depth_bins()
        assert bins.shape == (BEV["d_bins"],)

    def test_range(self):
        from models.lss.depth_bins import get_depth_bins
        bins = get_depth_bins()
        assert abs(bins[0].item()  - BEV["d_min"]) < 1e-4
        assert abs(bins[-1].item() - BEV["d_max"]) < 1e-4

    def test_monotone_increasing(self):
        from models.lss.depth_bins import get_depth_bins
        bins = get_depth_bins()
        assert (bins[1:] > bins[:-1]).all()


# ── lift module ───────────────────────────────────────────────

class TestLiftModule:
    @pytest.fixture(scope="class")
    def model(self):
        from models.lss.lift import LiftModule
        return LiftModule().eval()

    def test_output_types(self, model):
        batch = make_batch(B=1)
        with torch.no_grad():
            feats, depth = model(batch["image"])
        assert feats.ndim == 5    # (B, D, C, H', W')
        assert depth.ndim == 4   # (B, D, H', W')

    def test_frustum_depth_dim(self, model):
        batch = make_batch(B=1)
        with torch.no_grad():
            feats, depth = model(batch["image"])
        assert feats.shape[1]  == BEV["d_bins"]
        assert feats.shape[2]  == LSS["feature_channels"]
        assert depth.shape[1]  == BEV["d_bins"]

    def test_depth_dist_sums_to_one(self, model):
        batch = make_batch(B=1)
        with torch.no_grad():
            _, depth = model(batch["image"])
        col_sums = depth.sum(dim=1)   # sum over D → (B, H', W')
        assert (col_sums - 1.0).abs().max().item() < 1e-4

    def test_frustum_feats_finite(self, model):
        batch = make_batch(B=1)
        with torch.no_grad():
            feats, _ = model(batch["image"])
        assert torch.isfinite(feats).all()

    def test_batch_dim_preserved(self, model):
        batch = make_batch(B=2)
        with torch.no_grad():
            feats, depth = model(batch["image"])
        assert feats.shape[0]  == 2
        assert depth.shape[0]  == 2


# ── splat module ──────────────────────────────────────────────

class TestSplatModule:
    def test_output_shape(self):
        from models.lss.lift  import LiftModule
        from models.lss.splat import SplatModule
        lift  = LiftModule().eval()
        splat = SplatModule().eval()
        batch = make_batch(B=2)
        with torch.no_grad():
            feats, _ = lift(batch["image"])
            bev      = splat(feats, batch["K"], batch["T_cam2ego"])
        assert bev.shape == (2, LSS["feature_channels"],
                             BEV["size"], BEV["size"])

    def test_bev_feats_finite(self):
        from models.lss.lift  import LiftModule
        from models.lss.splat import SplatModule
        lift  = LiftModule().eval()
        splat = SplatModule().eval()
        batch = make_batch(B=1)
        with torch.no_grad():
            feats, _ = lift(batch["image"])
            bev      = splat(feats, batch["K"], batch["T_cam2ego"])
        assert torch.isfinite(bev).all()


# ── shoot module ──────────────────────────────────────────────

class TestShootModule:
    def test_output_shape(self):
        from models.lss.shoot import ShootModule
        shoot = ShootModule().eval()
        bev   = torch.randn(2, LSS["feature_channels"],
                            BEV["size"], BEV["size"])
        with torch.no_grad():
            logits = shoot(bev)
        assert logits.shape == (2, CLASSES["num_classes"],
                                BEV["size"], BEV["size"])

    def test_output_finite(self):
        from models.lss.shoot import ShootModule
        shoot  = ShootModule().eval()
        bev    = torch.randn(1, LSS["feature_channels"],
                             BEV["size"], BEV["size"])
        with torch.no_grad():
            logits = shoot(bev)
        assert torch.isfinite(logits).all()


# ── full lss model ────────────────────────────────────────────

class TestLSSModel:
    @pytest.fixture(scope="class")
    def model(self):
        from models.lss.lss_model import LSSModel
        return LSSModel().eval()

    def test_forward_output_shape(self, model):
        """
        THE critical contract:
        (B, 3, H, W) image → (B, num_classes, bev_size, bev_size) logits
        """
        batch = make_batch(B=2)
        with torch.no_grad():
            logits, depth = model(batch["image"],
                                  batch["K"],
                                  batch["T_cam2ego"])
        assert logits.shape == (2, CLASSES["num_classes"],
                                BEV["size"], BEV["size"]), \
            f"Expected (2,{CLASSES['num_classes']},{BEV['size']},{BEV['size']}), got {logits.shape}"

    def test_depth_dist_shape(self, model):
        batch = make_batch(B=1)
        with torch.no_grad():
            _, depth = model(batch["image"], batch["K"], batch["T_cam2ego"])
        assert depth.shape[0] == 1
        assert depth.shape[1] == BEV["d_bins"]

    def test_logits_finite(self, model):
        batch = make_batch(B=1)
        with torch.no_grad():
            logits, _ = model(batch["image"], batch["K"], batch["T_cam2ego"])
        assert torch.isfinite(logits).all(), "LSS logits contain NaN or Inf"

    def test_logits_not_all_zero(self, model):
        batch = make_batch(B=1)
        with torch.no_grad():
            logits, _ = model(batch["image"], batch["K"], batch["T_cam2ego"])
        assert logits.abs().max().item() > 0.0, "All logits are zero — model may not be initialised"

    def test_single_batch_works(self, model):
        batch = make_batch(B=1)
        with torch.no_grad():
            logits, depth = model(batch["image"], batch["K"], batch["T_cam2ego"])
        assert logits.shape[0] == 1

    def test_get_bev_features(self, model):
        batch = make_batch(B=1)
        with torch.no_grad():
            bev_feats, _ = model.get_bev_features(
                batch["image"], batch["K"], batch["T_cam2ego"]
            )
        assert bev_feats.shape == (1, LSS["feature_channels"],
                                   BEV["size"], BEV["size"])

    def test_depth_sums_to_one(self, model):
        batch = make_batch(B=1)
        with torch.no_grad():
            _, depth = model(batch["image"], batch["K"], batch["T_cam2ego"])
        sums = depth.sum(dim=1)
        assert (sums - 1.0).abs().max().item() < 1e-3


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["python", "-m", "pytest", __file__, "-v"]))