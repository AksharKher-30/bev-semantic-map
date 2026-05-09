import torch
import torch.nn.functional as F
import numpy as np
import cv2
from utils.config import SEGFORMER
from utils.device import get_device


def preprocess(image_bgr, img_h=None, img_w=None):
    h = img_h or SEGFORMER["img_h"]
    w = img_w or SEGFORMER["img_w"]
    rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h))
    tensor  = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)   # (1, 3, H, W)


def run_inference(model, image_bgr, device=None):
    """
    image_bgr : numpy (H, W, 3) - raw OpenCV image
    returns   : numpy (H, W)   - class label per pixel (0-4)
    """
    if device is None:
        device = get_device(verbose=False)

    model.eval()
    x = preprocess(image_bgr).to(device)

    with torch.no_grad():
        logits = model(pixel_values=x).logits   # (1, C, H/4, W/4)

    # upsample back to model input size
    upsampled = F.interpolate(
        logits,
        size=(SEGFORMER["img_h"], SEGFORMER["img_w"]),
        mode="bilinear",
        align_corners=False,
    )
    mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask   # (img_h, img_w)


def run_inference_tensor(model, image_tensor, device=None):
    """
    image_tensor : (3, H, W) float32 tensor already normalised
    returns      : (H, W) uint8 numpy mask
    """
    if device is None:
        device = get_device(verbose=False)

    model.eval()
    x = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(pixel_values=x).logits

    upsampled = F.interpolate(
        logits,
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

def run_inference_19class(model, image_bgr, device=None):
    """
    Run zero-shot inference returning full 19-class Cityscapes mask.
    Use this for rich scene visualization.

    Returns (H, W) uint8 with values 0-18 (Cityscapes class indices).
    """
    if device is None:
        device = get_device(verbose=False)

    model.eval()
    x = preprocess(image_bgr).to(device)

    with torch.no_grad():
        logits = model(pixel_values=x).logits

    upsampled = F.interpolate(
        logits,
        size=(SEGFORMER["img_h"], SEGFORMER["img_w"]),
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def run_inference_5class(model, image_bgr, device=None):
    """
    Zero-shot inference remapped to 5 classes (road, vehicle, ped, sky, bg).
    Use for front-view segmentation visualization overlay.
    """
    from data.class_mappings import remap_19_to_5
    mask_19 = run_inference_19class(model, image_bgr, device)
    return remap_19_to_5(mask_19)


def run_inference_3class(model, image_bgr, device=None):
    """
    Zero-shot inference remapped to 3 classes (road, vehicle, pedestrian).
    Use as input to IPM warp and BEV evaluation.
    255 = ignore (sky, building, vegetation etc.)
    """
    from data.class_mappings import remap_19_to_3
    mask_19 = run_inference_19class(model, image_bgr, device)
    return remap_19_to_3(mask_19)