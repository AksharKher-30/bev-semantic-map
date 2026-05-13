import random
import torch
import torchvision.transforms.functional as TF

def augment_lss(image, bev_gt):
    """
    Geometrically safe augmentations for LSS.
    ALL colour transforms — never geometric (would break K matrix).
    """
    # horizontal flip — safe (symmetric lateral scene)
    if random.random() > 0.5:
        image  = TF.hflip(image)
        bev_gt = torch.flip(bev_gt, dims=[-1])

    # stronger brightness for night scene generalisation
    image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
    image = TF.adjust_contrast(image,   random.uniform(0.7, 1.3))
    image = TF.adjust_saturation(image, random.uniform(0.6, 1.4))

    # random grayscale — forces model to rely on shape not colour
    if random.random() < 0.15:
        # Added .clone() here! TF.rgb_to_grayscale with 3 channels uses .expand()
        # which creates overlapping memory. Cloning makes it contiguous and safe for in-place writes.
        image = TF.rgb_to_grayscale(image, num_output_channels=3).clone()

    # gaussian noise — small regularisation
    if random.random() < 0.3:
        noise = torch.randn_like(image) * 0.02
        image = (image + noise).clamp(0.0, 1.0)

    # random erasing (cutout) — occlusion robustness
    # erase a random rectangle from the image only (not BEV GT)
    if random.random() < 0.3:
        h, w   = image.shape[-2], image.shape[-1]
        eh     = random.randint(h // 8, h // 4)
        ew     = random.randint(w // 8, w // 4)
        y0     = random.randint(0, h - eh)
        x0     = random.randint(0, w - ew)
        image[:, y0:y0+eh, x0:x0+ew] = 0.0   # black patch

    return image, bev_gt