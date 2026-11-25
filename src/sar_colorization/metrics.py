"""Reference-based metrics, matching the notebook's evaluation (cell 33).

PSNR and SSIM are only meaningful against a *pixel-aligned* optical reference of
the same scene (the paired dataset provides one). They must never be computed
against an arbitrary uploaded image, which has no ground truth.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .losses import ssim as _ssim


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """PSNR in dB for tensors in [0, max_val]; identical to the notebook's definition."""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return 100.0
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Mean SSIM (11x11 Gaussian window, sigma 1.5) for tensors in [0, 1]."""
    return float(_ssim(img1, img2).item())


def mae(img1: torch.Tensor, img2: torch.Tensor) -> float:
    return float(F.l1_loss(img1, img2).item())
