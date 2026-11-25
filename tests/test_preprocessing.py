"""Preprocessing must reproduce the notebook's training transforms exactly."""
import io

import numpy as np
import pytest
import torch
from conftest import png_bytes
from PIL import Image
from torchvision import transforms

from sar_colorization.preprocessing import (
    InvalidImageError,
    decode_image,
    preprocessed_preview,
    sar_to_tensor,
    tensor_to_image,
)


def notebook_transform(img: Image.Image) -> torch.Tensor:
    """The SAR branch of SAROpticalDataset.__getitem__, written with torchvision as in the notebook."""
    resize = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC)
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    return norm(transforms.ToTensor()(resize(img.convert("RGB")))).mean(dim=0, keepdim=True)


def test_matches_notebook_transform_exactly():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (317, 411, 3), dtype=np.uint8))
    ours = sar_to_tensor(img)[0]
    ref = notebook_transform(img)
    assert ours.shape == ref.shape == (1, 256, 256)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_output_shape_and_range():
    t = sar_to_tensor(Image.new("L", (640, 480), 200))
    assert t.shape == (1, 1, 256, 256)
    assert t.dtype == torch.float32
    assert -1.0 <= t.min() and t.max() <= 1.0


def test_normalisation_endpoints():
    assert torch.allclose(sar_to_tensor(Image.new("L", (64, 64), 0)), torch.tensor(-1.0))
    assert torch.allclose(sar_to_tensor(Image.new("L", (64, 64), 255)), torch.tensor(1.0))


def test_tensor_to_image_round_trip():
    t = torch.linspace(-1, 1, 3 * 8 * 8).reshape(1, 3, 8, 8)
    img = tensor_to_image(t)
    assert img.mode == "RGB" and img.size == (8, 8)
    arr = np.asarray(img)
    assert arr.min() == 0 and arr.max() == 255


def test_tensor_to_image_clamps_out_of_range():
    img = tensor_to_image(torch.full((1, 3, 4, 4), 5.0))
    assert np.asarray(img).max() == 255


def test_preview_is_what_model_sees():
    prev = preprocessed_preview(Image.new("RGB", (100, 50), (10, 20, 30)))
    assert prev.size == (256, 256) and prev.mode == "L"


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "TIFF", "WEBP", "BMP"])
def test_decode_supported_formats(fmt):
    img = decode_image(png_bytes(fmt=fmt, mode="RGB"), max_pixels=10_000_000)
    assert img.size == (300, 200)


def test_decode_rejects_empty():
    with pytest.raises(InvalidImageError, match="Empty"):
        decode_image(b"", max_pixels=10)


def test_decode_rejects_non_image():
    with pytest.raises(InvalidImageError, match="not a readable image"):
        decode_image(b"%PDF-1.4 not an image", max_pixels=10_000_000)


def test_decode_rejects_unsupported_format():
    with pytest.raises(InvalidImageError, match="Unsupported"):
        decode_image(png_bytes(fmt="GIF", mode="L"), max_pixels=10_000_000)


def test_decode_rejects_too_many_pixels_before_decoding():
    with pytest.raises(InvalidImageError, match="limit"):
        decode_image(png_bytes(size=(2000, 2000)), max_pixels=1_000_000)


def test_decode_rejects_tiny_images():
    with pytest.raises(InvalidImageError, match="smaller than 32"):
        decode_image(png_bytes(size=(16, 300)), max_pixels=10_000_000)


def test_decode_rejects_truncated_file():
    data = png_bytes(size=(400, 400), mode="RGB", color=(1, 2, 3))
    # Keep the header but drop most pixel data.
    buf = io.BytesIO()
    Image.effect_noise((400, 400), 50).save(buf, format="PNG")
    truncated = buf.getvalue()[: len(buf.getvalue()) // 3]
    assert len(data) > 0
    with pytest.raises(InvalidImageError):
        decode_image(truncated, max_pixels=10_000_000)
