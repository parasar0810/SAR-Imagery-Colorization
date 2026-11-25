"""Image preprocessing that mirrors the training transforms exactly.

The notebook's ``SAROpticalDataset`` does, for the SAR half of each pair:
    RGB -> bicubic resize to IMG_SIZE -> ToTensor -> Normalize(0.5, 0.5) -> mean over channels
which yields a 1x256x256 tensor in [-1, 1]. Inference must apply the same steps,
otherwise the generator sees a different input distribution than it was trained on.
"""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

IMG_SIZE = 256  # CFG["IMG_SIZE"] in the notebook

# Formats PIL decodes that make sense for single-band or RGB rasters.
SUPPORTED_FORMATS = {"PNG", "JPEG", "TIFF", "WEBP", "BMP"}


class InvalidImageError(ValueError):
    """Raised when an upload cannot be decoded or violates a limit."""


def decode_image(data: bytes, *, max_pixels: int) -> Image.Image:
    """Decode untrusted bytes into a PIL image, enforcing format and pixel limits.

    The header is inspected before full decoding so oversized images are rejected
    without allocating their pixel buffers (decompression-bomb protection).
    """
    if not data:
        raise InvalidImageError("Empty file.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        w, h = probe.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("File is not a readable image.") from exc
    if fmt not in SUPPORTED_FORMATS:
        raise InvalidImageError(f"Unsupported format {fmt!r}. Use PNG, JPEG, TIFF, WEBP or BMP.")
    if w * h > max_pixels:
        raise InvalidImageError(f"Image is {w}x{h}; the limit is {max_pixels:,} pixels.")
    if min(w, h) < 32:
        raise InvalidImageError("Image is smaller than 32 px on one side.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("Image data is corrupted or truncated.") from exc
    return img


def sar_to_tensor(img: Image.Image, img_size: int = IMG_SIZE) -> torch.Tensor:
    """Convert a SAR image to the generator's input: shape (1, 1, H, W), values in [-1, 1]."""
    rgb = img.convert("RGB").resize((img_size, img_size), Image.Resampling.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0  # ToTensor scaling
    arr = (arr - 0.5) / 0.5  # Normalize([0.5]*3, [0.5]*3)
    gray = arr.mean(axis=2, keepdims=True)  # .mean(dim=0, keepdim=True) over channels
    return torch.from_numpy(gray.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """Convert a generator output (1, 3, H, W) in [-1, 1] to an 8-bit RGB image."""
    if t.dim() == 4:
        t = t[0]
    arr = ((t.detach().float().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def preprocessed_preview(img: Image.Image, img_size: int = IMG_SIZE) -> Image.Image:
    """The exact grayscale image the model sees, for display in the UI."""
    t = sar_to_tensor(img, img_size)[0, 0]
    arr = ((t.numpy() + 1) / 2 * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")
