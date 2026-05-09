import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from utils.config import SEGFORMER


def build_segformer(pretrained=True):
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER["pretrained_ckpt"]
        )
    else:
        from transformers import SegformerConfig
        cfg   = SegformerConfig.from_pretrained(SEGFORMER["pretrained_ckpt"])
        model = SegformerForSemanticSegmentation(cfg)

    # swap 19-class Cityscapes head → 5-class BEV head
    in_ch = model.decode_head.classifier.in_channels
    model.decode_head.classifier = nn.Conv2d(in_ch, SEGFORMER["num_classes"], kernel_size=1)

    return model


def load_checkpoint(path, device):
    model = build_segformer(pretrained=False)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)

def build_segformer_zero_shot():
    """
    Load pretrained 19-class Cityscapes SegFormer-b0.
    No head replacement — uses original Cityscapes weights as-is.
    Zero-shot transfer: remap predictions at inference time.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER["pretrained_ckpt"]
    )
    return model