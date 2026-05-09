from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from utils.config import SEGFORMER, TRAIN


def build_segformer_optimizer(model, steps_per_epoch):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from utils.config import SEGFORMER, TRAIN

    # backbone gets 10x lower LR — preserves Cityscapes pretrained features
    backbone_params = list(model.segformer.parameters())
    head_params     = list(model.decode_head.parameters())

    optimizer = AdamW([
        {"params": backbone_params, "lr": SEGFORMER["lr"] * 0.1},
        {"params": head_params,     "lr": SEGFORMER["lr"]},
    ], weight_decay=TRAIN["segformer"]["weight_decay"])

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=SEGFORMER["epochs"],
        eta_min=1e-7,
    )
    return optimizer, scheduler


def build_lss_optimizer(model):
    cfg = TRAIN["lss"]
    param_groups = [
        {"params": model.lift.backbone.parameters(), "lr": cfg["backbone_lr"]},
        {"params": model.lift.reduce_conv.parameters(), "lr": cfg["head_lr"]},
        {"params": model.splat.parameters(), "lr": cfg["head_lr"]},
        {"params": model.shoot.parameters(), "lr": cfg["head_lr"]},
    ]
    optimizer = AdamW(param_groups, weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=cfg["eta_min"],
    )
    return optimizer, scheduler