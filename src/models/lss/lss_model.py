import torch
import torch.nn as nn
from models.lss.lift  import LiftModule
from models.lss.splat import SplatModule
from models.lss.shoot import ShootModule


class LSSModel(nn.Module):
    """
    Full Lift-Splat-Shoot model.

    End-to-end differentiable pipeline:
        image → Lift → frustum features
              → Splat → BEV feature grid
              → Shoot → BEV semantic logits

    Supervised entirely in BEV space using nuScenes map + box GT.
    Never sees front-view pixel labels — learns depth implicitly
    through BEV supervision gradients flowing back through Splat.

    Input:
        images    : (B, 3, H, W) — normalised front camera
        K         : (B, 3, 3)   — camera intrinsics
        T_cam2ego : (B, 4, 4)   — sensor→ego rigid transform

    Output:
        logits    : (B, num_classes, bev_size, bev_size) — raw logits
        depth_dist: (B, D, H', W')  — depth probability per pixel
    """

    def __init__(self):
        super().__init__()
        self.lift  = LiftModule()
        self.splat = SplatModule()
        self.shoot = ShootModule()

    def forward(self, images, K, T_cam2ego):
        frustum_feats, depth_dist = self.lift(images)
        bev_feats                 = self.splat(frustum_feats, K, T_cam2ego)
        logits                    = self.shoot(bev_feats)
        return logits, depth_dist

    def get_bev_features(self, images, K, T_cam2ego):
        """
        Expose intermediate BEV feature map (before Shoot head).
        Useful for visualisation and ablation studies.
        """
        frustum_feats, depth_dist = self.lift(images)
        bev_feats                 = self.splat(frustum_feats, K, T_cam2ego)
        return bev_feats, depth_dist