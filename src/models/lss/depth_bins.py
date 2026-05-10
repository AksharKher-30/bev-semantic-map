import torch
import numpy as np
from utils.config import LSS, BEV


def get_depth_bins(device=None):
    bins = torch.linspace(BEV["d_min"], BEV["d_max"], BEV["d_bins"])
    if device is not None:
        bins = bins.to(device)
    return bins


def get_depth_bin_size():
    return (BEV["d_max"] - BEV["d_min"]) / (BEV["d_bins"] - 1)


def bin_depths(depth_map, mode="linear"):
    d_min, d_max = BEV["d_min"], BEV["d_max"]
    D = BEV["d_bins"]
    depth_map = depth_map.clamp(d_min, d_max)
    return ((depth_map - d_min) / (d_max - d_min) * (D - 1)).long()