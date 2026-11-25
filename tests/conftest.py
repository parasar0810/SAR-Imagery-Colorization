import io
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def png_bytes(size=(300, 200), mode="L", fmt="PNG", color=128) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def random_generator_ckpt(tmp_path):
    """A generator checkpoint with RANDOM (untrained) weights.

    Used to test loading, shapes and API plumbing only. Outputs from it are noise
    and say nothing about model quality.
    """
    from sar_colorization.models import AttentionUNetGenerator

    torch.manual_seed(0)
    G = AttentionUNetGenerator(1, 3, 64)
    path = tmp_path / "random_generator.pt"
    torch.save({"epoch": 0, "G": G.state_dict(), "cfg": {"IMG_SIZE": 256, "NGF": 64}}, path)
    return path
