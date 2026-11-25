"""Checkpoint loading and single-image inference."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .models import AttentionUNetGenerator
from .preprocessing import IMG_SIZE, sar_to_tensor, tensor_to_image

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when no usable generator checkpoint is installed."""


def pick_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cuda":
        raise RuntimeError("CUDA requested but not available.")
    return torch.device("cpu")


def load_generator_state(path: Path) -> tuple[dict, dict]:
    """Load generator weights with ``weights_only=True`` (no arbitrary code execution).

    Accepts either a full training checkpoint written by the notebook
    (``{"G": ..., "D": ..., "epoch": ..., ...}``) or a bare generator ``state_dict``
    produced by ``scripts/export_generator.py``. Returns (state_dict, metadata).
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "G" in obj and isinstance(obj["G"], dict):
        meta = {"epoch": obj.get("epoch"), "format": "training-checkpoint"}
        cfg = obj.get("cfg")
        if isinstance(cfg, dict):
            meta["img_size"] = cfg.get("IMG_SIZE")
            meta["ngf"] = cfg.get("NGF")
        return obj["G"], meta
    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj, {"format": "generator-state-dict"}
    raise ModelUnavailableError("Checkpoint does not contain generator weights.")


@dataclass
class Result:
    image: Image.Image
    inference_ms: float
    input_size: tuple[int, int]
    model_size: int


class Colorizer:
    """Owns one generator instance. Thread-safe: inference calls are serialised."""

    def __init__(self, weights: Path | None, device: str = "auto", ngf: int = 64):
        self.device = pick_device(device)
        self.weights = Path(weights) if weights else None
        self.ngf = ngf
        self._model: AttentionUNetGenerator | None = None
        self._lock = threading.Lock()
        self.meta: dict = {}
        self.error: str | None = None

    def load(self) -> None:
        if not self.weights or not self.weights.is_file():
            self.error = "Model weights are not installed on this server."
            log.warning("generator weights missing: %s", self.weights)
            return
        try:
            state, meta = load_generator_state(self.weights)
            model = AttentionUNetGenerator(in_channels=1, out_channels=3, ngf=meta.get("ngf") or self.ngf)
            model.load_state_dict(state, strict=True)
            model.to(self.device).eval()
            self._model = model
            digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()
            self.meta = {**meta, "sha256": digest[:16], "parameters": sum(p.numel() for p in model.parameters())}
            self.error = None
            log.info("generator loaded from %s (%s)", self.weights.name, self.meta)
        except Exception as exc:  # report, don't crash the service
            self._model = None
            self.error = f"Model weights could not be loaded: {exc.__class__.__name__}"
            log.exception("failed to load generator")

    @property
    def ready(self) -> bool:
        return self._model is not None

    @torch.inference_mode()
    def colorize(self, img: Image.Image) -> Result:
        if self._model is None:
            raise ModelUnavailableError(self.error or "Model not loaded.")
        x = sar_to_tensor(img, IMG_SIZE).to(self.device)
        with self._lock:
            t0 = time.perf_counter()
            y = self._model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
        return Result(image=tensor_to_image(y), inference_ms=round(ms, 1), input_size=img.size, model_size=IMG_SIZE)
