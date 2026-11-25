"""API tests. 'with_model' tests use RANDOM weights and only verify request handling."""
import base64
import io

import pytest
from conftest import png_bytes
from fastapi.testclient import TestClient
from PIL import Image

import api.main as main
from sar_colorization.inference import Colorizer


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "colorizer", Colorizer(tmp_path / "none.pt", device="cpu"))
    main._hits.clear()
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def client_with_model(monkeypatch, random_generator_ckpt):
    monkeypatch.setattr(main, "colorizer", Colorizer(random_generator_ckpt, device="cpu"))
    main._hits.clear()
    with TestClient(main.app) as c:
        yield c


def upload(c, data, name="sar.png", ctype="image/png"):
    return c.post("/api/colorize", files={"file": (name, data, ctype)})


def test_health_reports_model_state(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["model_ready"] is False


def test_model_info_explains_unavailability(client):
    j = client.get("/api/model").json()
    assert j["ready"] is False and "not installed" in j["reason"]
    assert j["input"]["size"] == [256, 256] and "not measured colour" in j["output"]["meaning"]


def test_colorize_without_weights_is_503(client):
    r = upload(client, png_bytes())
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_invalid_image_is_400_even_without_model(client):
    r = upload(client, b"not an image")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_image"


def test_unsupported_format_is_415(client):
    assert upload(client, png_bytes(fmt="GIF"), "a.gif", "image/gif").status_code == 415


def test_oversized_upload_is_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    assert upload(client, png_bytes(size=(600, 600), mode="RGB", color=(9, 9, 9)) + b"\0" * 2000).status_code == 413


def test_too_many_pixels_is_400(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_PIXELS", 10_000)
    r = upload(client, png_bytes(size=(200, 200)))
    assert r.status_code == 400 and "limit" in r.json()["error"]["message"]


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 2)
    codes = [upload(client, b"x").status_code for _ in range(3)]
    assert codes[:2] == [400, 400] and codes[2] == 429


def test_spoofed_forwarded_for_is_ignored_by_default(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 1)
    a = client.post("/api/colorize", files={"file": ("a", b"x")}, headers={"x-forwarded-for": "1.1.1.1"})
    b = client.post("/api/colorize", files={"file": ("a", b"x")}, headers={"x-forwarded-for": "2.2.2.2"})
    assert a.status_code == 400 and b.status_code == 429


def test_colorize_with_model_returns_png(client_with_model):
    r = upload(client_with_model, png_bytes(size=(512, 384)))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["inference_ms"] > 0 and j["input"] == {"width": 512, "height": 384, "format": "PNG", "mode": "L"}
    assert "not ground truth" in j["label"]
    for key in ("output", "preprocessed"):
        img = Image.open(io.BytesIO(base64.b64decode(j[key].split(",", 1)[1])))
        assert img.size == (256, 256)


def test_samples_listing_and_assets(client):
    items = client.get("/api/samples").json()
    assert len(items) >= 2 and {"sar", "generated", "reference"} <= set(items[0]["assets"])
    assert "precomputed" in items[0]["provenance"]
    r = client.get(f"/api/samples/{items[0]['id']}/sar")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"


@pytest.mark.parametrize("path", ["/api/samples/upstream-01/..%2f..%2fapi%2fmain.py", "/api/samples/nope/sar", "/api/samples/upstream-01/manifest"])
def test_sample_paths_are_restricted(client, path):
    assert client.get(path).status_code == 404


def test_evaluation_is_honest_when_not_run(client):
    j = client.get("/api/evaluation").json()
    if not j["available"]:
        assert "not reproduced" in j["upstream_reported"]["source"]


def test_security_headers(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"


def test_preprocess_works_without_model(client):
    r = client.post("/api/preprocess", files={"file": ("a.png", png_bytes(size=(640, 480)), "image/png")})
    assert r.status_code == 200
    j = r.json()
    img = Image.open(io.BytesIO(base64.b64decode(j["preprocessed"].split(",", 1)[1])))
    assert img.size == (256, 256) and img.mode == "L" and j["input"]["width"] == 640
