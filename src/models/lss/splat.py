import torch
import torch.nn as nn
from utils.config import LSS, BEV
from models.lss.depth_bins import get_depth_bins


class SplatModule(nn.Module):
    """
    SPLAT: 3D frustum features → 2D BEV feature grid.

    For each pixel (u,v) in the feature map and each depth bin d:
      1. Unproject pixel + depth → 3D point in camera frame
         x_cam = (u - cx) / fx * d
         y_cam = (v - cy) / fy * d
         z_cam = d
      2. Transform camera-frame point → ego frame using T_cam2ego
      3. Find which BEV grid cell (ix, iy) it falls into
      4. Accumulate (scatter-add) feature vector into that BEV cell

    Multiple frustum voxels can land in the same BEV cell — they add up.
    Cells that receive no voxels stay zero.

    This "pillar pooling" operation collapses the Z dimension, giving
    a top-down feature map that preserves spatial layout.
    """

    def __init__(self):
        super().__init__()
        self.D        = BEV["d_bins"]
        self.C        = LSS["feature_channels"]
        self.bev_size = BEV["size"]
        self.bev_res  = BEV["resolution"]
        self.register_buffer("depth_bins", get_depth_bins())

    def _make_frustum_grid(self, H_feat, W_feat, K, device):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # EfficientNet stride=32 — feature (h,w) maps to image pixel centre
        xs = torch.arange(W_feat, device=device).float() * 32 + 15.5  # (W_feat,)
        ys = torch.arange(H_feat, device=device).float() * 32 + 15.5  # (H_feat,)

        # 2D meshgrid: both (H_feat, W_feat)
        ys_2d, xs_2d = torch.meshgrid(ys, xs, indexing='ij')

        # per-pixel direction vectors in camera frame
        x_dir = (xs_2d - cx) / fx   # (H, W)
        y_dir = (ys_2d - cy) / fy   # (H, W)

        # scale by each depth bin → (D, H, W)
        D_bins = self.depth_bins                              # (D,)
        x = D_bins.view(self.D, 1, 1) * x_dir.unsqueeze(0)  # (D, H, W)
        y = D_bins.view(self.D, 1, 1) * y_dir.unsqueeze(0)  # (D, H, W)
        z = D_bins.view(self.D, 1, 1).expand(self.D, H_feat, W_feat)

        return torch.stack([x, y, z], dim=-1)   # (D, H, W, 3)

    def forward(self, frustum_feats, K, T_cam2ego):
        """
        frustum_feats : (B, D, C, H', W')
        K             : (B, 3, 3) camera intrinsics
        T_cam2ego     : (B, 4, 4) sensor-to-ego rigid transform

        Returns
        -------
        bev_feats : (B, C, bev_size, bev_size)
        """
        B, D, C, H, W = frustum_feats.shape
        sz  = self.bev_size
        res = self.bev_res
        dev = frustum_feats.device

        bev_list = []
        for b in range(B):
            # 1. frustum voxel positions in camera frame: (D, H, W, 3)
            coords_cam = self._make_frustum_grid(H, W, K[b], dev)
            N = D * H * W
            coords_flat = coords_cam.view(N, 3)   # (N, 3)

            # 2. camera → ego frame: R @ p + t
            R = T_cam2ego[b, :3, :3]   # (3,3)
            t = T_cam2ego[b, :3, 3]    # (3,)
            coords_ego = (R @ coords_flat.T).T + t   # (N, 3)

            # 3. ego (x,y) → BEV pixel index
            # x = lateral (+right), y = forward (+up in BEV image)
            cx_bev = sz / 2.0
            ix = (coords_ego[:, 0] / res + cx_bev).long()   # col
            iy = (cx_bev - coords_ego[:, 1] / res).long()   # row (flip y)

            valid = (ix >= 0) & (ix < sz) & (iy >= 0) & (iy < sz)

            # 4. scatter-add features into BEV grid
            bev_grid = torch.zeros(C, sz, sz, device=dev, dtype=frustum_feats.dtype)

            # flatten feats: (D, C, H, W) → (N, C)
            feats_flat = frustum_feats[b].permute(0, 2, 3, 1).reshape(N, C)

            ix_v    = ix[valid]             # (M,)
            iy_v    = iy[valid]             # (M,)
            feats_v = feats_flat[valid]     # (M, C)

            # linear index into flattened (sz×sz) grid
            flat_idx = iy_v * sz + ix_v    # (M,)

            bev_flat = bev_grid.view(C, -1)   # (C, sz*sz)
            bev_flat.scatter_add_(
                1,
                flat_idx.unsqueeze(0).expand(C, -1),
                feats_v.T,
            )

            bev_list.append(bev_grid)

        return torch.stack(bev_list, dim=0)   # (B, C, sz, sz)