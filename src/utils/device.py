# ============================================================
# src/utils/device.py
# ============================================================
# Centralised device selection for Apple M3.
# All training / inference code calls get_device() - never
# hardcodes torch.device() themselves.
#
# To switch to CPU-only (e.g. for debugging):
#   set FORCE_CPU=1 in your shell before running any script.
#   export FORCE_CPU=1
# ============================================================

import os
import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    Return the best available device in priority order:
        1. MPS  (Apple Silicon GPU - M1/M2/M3)
        2. CUDA (NVIDIA GPU - not present on M3 but kept for portability)
        3. CPU  (fallback)

    Parameters
    ----------
    verbose : bool
        Print the selected device on first call.

    Returns
    -------
    torch.device
    """
    force_cpu = os.environ.get("FORCE_CPU", "0") == "1"

    if force_cpu:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"[device] Using: {device}")
        if device.type == "mps":
            print("[device] Apple MPS backend active - M3 GPU will be used.")
        elif device.type == "cpu":
            print("[device] WARNING: Running on CPU - training will be slow.")

    return device


def move_batch(batch: dict, device: torch.device) -> dict:
    """
    Move every tensor in a DataLoader batch dict to `device`.
    Non-tensor values (strings, ints) are left untouched.

    Usage
    -----
        batch = move_batch(batch, get_device())
    """
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }