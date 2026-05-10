import torch
import torch.nn as nn
from timm import create_model
from utils.config import LSS, BEV
from models.lss.depth_bins import get_depth_bins


class LiftModule(nn.Module):
    """
    LIFT: 2D image features → 3D frustum representation.

    For every pixel (u,v) in the feature map:
      1. Predict a D-way depth distribution (softmax over D depth bins)
      2. Predict a C-dim context feature vector
      3. Outer product: depth_dist (D,) × context (C,) = (D, C)
         → one weighted feature vector per depth bin per pixel

    This gives a (D, C, H', W') frustum volume where each entry
    is the feature weighted by how likely that depth bin is.

    Parameters from config (utils/config.py LSS dict):
      d_bins  : D  — number of discrete depth bins (default 41)
      feature_channels : C — context feature channels (default 64)
      backbone : EfficientNet-B0
    """

    def __init__(self):
        super().__init__()
        D = BEV["d_bins"]
        C = LSS["feature_channels"]

        # EfficientNet-B0 — pretrained on ImageNet, remove classifier head
        backbone      = create_model(LSS["backbone"], pretrained=True,
                                     features_only=False)
        # strip classifier + pooling, keep feature extractor only
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # reduce EfficientNet output (1280 ch) → (D + C) channels
        self.reduce = nn.Sequential(
            nn.Conv2d(LSS["backbone_out_ch"], LSS["reduce_channels"], 1, bias=False),
            nn.BatchNorm2d(LSS["reduce_channels"]),
            nn.ReLU(inplace=True),
            nn.Conv2d(LSS["reduce_channels"], D + C, 1),
        )

        # register depth bins as buffer so they move with the model to MPS/CPU
        self.register_buffer("depth_bins", get_depth_bins())

    def forward(self, images):
        """
        images : (B, 3, H, W) normalised RGB

        Returns
        -------
        frustum_feats : (B, D, C, H', W')
            Feature volume weighted by depth probability at each pixel.
        depth_dist    : (B, D, H', W')
            Softmax depth distribution (sums to 1 along D dim).
        """
        B = images.shape[0]
        D = BEV["d_bins"]
        C = LSS["feature_channels"]

        # EfficientNet features: (B, 1280, H/32, W/32)
        feats = self.backbone(images)

        # project to (B, D+C, H', W')
        dc = self.reduce(feats)

        depth_logits = dc[:, :D]         # (B, D, H', W')
        context      = dc[:, D:]         # (B, C, H', W')

        # depth distribution — softmax over depth dimension
        depth_dist = depth_logits.softmax(dim=1)   # (B, D, H', W')

        # outer product: (B,D,1,H',W') × (B,1,C,H',W') → (B,D,C,H',W')
        frustum_feats = depth_dist.unsqueeze(2) * context.unsqueeze(1)

        return frustum_feats, depth_dist