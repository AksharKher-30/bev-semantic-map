import torch
import torch.nn as nn
from utils.config import LSS, CLASSES


def _conv_bn_relu(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ShootModule(nn.Module):
    """
    SHOOT: BEV feature map → per-class semantic logits.

    Lightweight ResNet-style encoder-decoder operating entirely in BEV space.
    Takes (B, C, H, W) BEV features from Splat and outputs
    (B, num_classes, H, W) raw logits (no sigmoid — applied in loss).

    Architecture:
      Block 1: 2× conv3×3, maintain resolution
      Block 2: 2× conv3×3, stride=2 (downsample)
      Block 3: ConvTranspose2×2, stride=2 (upsample back)
      Head:    Conv1×1 → num_classes
    """

    def __init__(self):
        super().__init__()
        C      = LSS["feature_channels"]          # 64 — input channels from Splat
        ch     = LSS["bev_encoder_ch"]            # [128, 256]
        n_cls  = CLASSES["num_classes"]           # 3

        self.block1 = nn.Sequential(
            _conv_bn_relu(C,     ch[0]),
            _conv_bn_relu(ch[0], ch[0]),
        )

        self.block2 = nn.Sequential(
            _conv_bn_relu(ch[0], ch[1], stride=2),
            _conv_bn_relu(ch[1], ch[1]),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(ch[0], n_cls, kernel_size=1)

    def forward(self, bev_feats):
        """
        bev_feats : (B, C, bev_size, bev_size)
        Returns   : (B, num_classes, bev_size, bev_size) raw logits
        """
        x = self.block1(bev_feats)
        x = self.upsample(self.block2(x))
        return self.head(x)