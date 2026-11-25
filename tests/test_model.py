"""Model and checkpoint tests. All use RANDOM weights: they check structure, not quality."""
import pickle
import subprocess
import sys

import pytest
import torch
from conftest import ROOT
from PIL import Image

from sar_colorization.inference import (
    Colorizer,
    ModelUnavailableError,
    load_generator_state,
)
from sar_colorization.models import AttentionUNetGenerator, MultiscaleDiscriminator


def test_generator_shapes_and_tanh_range():
    G = AttentionUNetGenerator(1, 3, 64).eval()
    with torch.no_grad():
        y = G(torch.randn(2, 1, 256, 256))
    assert y.shape == (2, 3, 256, 256)
    assert y.min() >= -1 and y.max() <= 1


def test_generator_parameter_count_is_stable():
    # Guards against accidental edits to the verbatim architecture (breaks checkpoint loading).
    assert sum(p.numel() for p in AttentionUNetGenerator(1, 3, 64).parameters()) == 42_497_640


def test_discriminator_returns_three_scales_with_features():
    out = MultiscaleDiscriminator(4, 64, 3, 3)(torch.randn(1, 4, 256, 256))
    assert len(out) == 3
    assert all(len(feats) == 5 for feats in out)
    assert out[0][-1].shape[-1] > out[1][-1].shape[-1] > out[2][-1].shape[-1]


def test_load_training_checkpoint_format(random_generator_ckpt):
    state, meta = load_generator_state(random_generator_ckpt)
    assert meta["format"] == "training-checkpoint" and meta["img_size"] == 256
    AttentionUNetGenerator(1, 3, 64).load_state_dict(state, strict=True)


def test_load_bare_state_dict(tmp_path):
    path = tmp_path / "g.pt"
    torch.save(AttentionUNetGenerator(1, 3, 64).state_dict(), path)
    _, meta = load_generator_state(path)
    assert meta["format"] == "generator-state-dict"


def test_rejects_checkpoint_without_generator(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"something": [1, 2, 3]}, path)
    with pytest.raises(ModelUnavailableError):
        load_generator_state(path)


def test_refuses_pickled_code(tmp_path):
    """weights_only=True must refuse arbitrary objects, so a hostile checkpoint can't run code."""
    path = tmp_path / "evil.pt"
    torch.save({"G": Image.new("L", (2, 2))}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_generator_state(path)


def test_colorizer_reports_missing_weights(tmp_path):
    c = Colorizer(tmp_path / "missing.pt", device="cpu")
    c.load()
    assert not c.ready and "not installed" in c.error
    with pytest.raises(ModelUnavailableError):
        c.colorize(Image.new("L", (64, 64)))


def test_colorizer_end_to_end_with_random_weights(random_generator_ckpt):
    c = Colorizer(random_generator_ckpt, device="cpu")
    c.load()
    assert c.ready and c.meta["parameters"] == 42_497_640
    r = c.colorize(Image.new("L", (500, 300), 90))
    assert r.image.size == (256, 256) and r.image.mode == "RGB"
    assert r.inference_ms > 0 and r.input_size == (500, 300)


def test_export_generator_fp16_loads(random_generator_ckpt, tmp_path):
    out = tmp_path / "gen16.pt"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "export_generator.py"), str(random_generator_ckpt), str(out), "--fp16"],
                   check=True, capture_output=True)
    c = Colorizer(out, device="cpu")
    c.load()
    assert c.ready
    assert out.stat().st_size < random_generator_ckpt.stat().st_size * 0.6
