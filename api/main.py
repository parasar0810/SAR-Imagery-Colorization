"""FastAPI inference service for the SAR colorization generator.

Design notes
- The model is loaded once at startup; requests never train or reload it.
- Uploads are processed entirely in memory (no temp files), decoded with
  format and pixel limits before full decoding.
- One inference runs at a time per process (semaphore) with a timeout, and each
  client IP is rate limited, so a public demo cannot be used to exhaust the CPU.
- Errors use one JSON shape: {"error": {"code": ..., "message": ...}}.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sar_colorization import __version__  # noqa: E402
from sar_colorization.inference import Colorizer, ModelUnavailableError  # noqa: E402
from sar_colorization.preprocessing import (  # noqa: E402
    IMG_SIZE,
    InvalidImageError,
    decode_image,
    preprocessed_preview,
)

log = logging.getLogger("sar.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format='{"t":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}')

WEIGHTS = Path(os.getenv("SAR_WEIGHTS", ROOT / "weights" / "generator.pt"))
MAX_UPLOAD_BYTES = int(float(os.getenv("SAR_MAX_UPLOAD_MB", "8")) * 1024 * 1024)
MAX_PIXELS = int(os.getenv("SAR_MAX_PIXELS", str(4096 * 4096)))
INFER_TIMEOUT_S = float(os.getenv("SAR_INFER_TIMEOUT_S", "30"))
RATE_LIMIT = int(os.getenv("SAR_RATE_LIMIT_PER_MIN", "20"))
CONCURRENCY = int(os.getenv("SAR_CONCURRENCY", "1"))
CORS_ORIGINS = [o for o in os.getenv("SAR_CORS_ORIGINS", "").split(",") if o]
TRUST_PROXY = os.getenv("SAR_TRUST_PROXY", "0") == "1"  # only behind a proxy that sets X-Forwarded-For
SAMPLES_DIR = ROOT / "samples"
EVAL_FILE = ROOT / "evaluation" / "evaluation.json"
FRONTEND_DIR = ROOT / "frontend" / "dist"

colorizer = Colorizer(WEIGHTS, device=os.getenv("SAR_DEVICE", "auto"))
_sem = asyncio.Semaphore(CONCURRENCY)
_hits: dict[str, deque] = defaultdict(deque)


@asynccontextmanager
async def lifespan(_: FastAPI):
    colorizer.load()
    yield


app = FastAPI(title="SAR Colorization API", version=__version__, lifespan=lifespan,
              docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
if CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    return resp


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(HTTPException)
async def http_exc(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for") if TRUST_PROXY else None
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def check_rate(request: Request) -> None:
    now, q = time.monotonic(), _hits[client_ip(request)]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, {"code": "rate_limited", "message": f"Limit is {RATE_LIMIT} requests per minute."})
    q.append(now)


def png_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": colorizer.ready, "version": __version__}


@app.get("/api/model")
def model_info():
    return {
        "ready": colorizer.ready,
        "reason": None if colorizer.ready else colorizer.error,
        "architecture": "Attention U-Net generator (self-attention at 16x16, attention-gated skips)",
        "input": {"channels": 1, "size": [IMG_SIZE, IMG_SIZE], "range": [-1, 1], "preprocessing": "RGB → bicubic resize → [0,1] → (x−0.5)/0.5 → channel mean"},
        "output": {"channels": 3, "size": [IMG_SIZE, IMG_SIZE], "activation": "tanh", "meaning": "model-predicted optical-style RGB, not measured colour"},
        "device": str(colorizer.device),
        "checkpoint": colorizer.meta or None,
        "limits": {"max_upload_mb": MAX_UPLOAD_BYTES / 1024 / 1024, "max_pixels": MAX_PIXELS, "timeout_s": INFER_TIMEOUT_S,
                   "rate_limit_per_min": RATE_LIMIT, "concurrency": CONCURRENCY, "formats": ["PNG", "JPEG", "TIFF", "WEBP", "BMP"]},
        "attribution": "Model design and training code: Fahad Ur Rehman (MIT). Inference service: Priyanshu Parasar.",
    }


@app.post("/api/colorize")
async def colorize(request: Request, file: Annotated[UploadFile, File()]):
    check_rate(request)
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, {"code": "too_large", "message": f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."})
    try:
        img = decode_image(data, max_pixels=MAX_PIXELS)
    except InvalidImageError as exc:
        raise HTTPException(415 if "Unsupported" in str(exc) else 400, {"code": "invalid_image", "message": str(exc)}) from None
    if not colorizer.ready:
        raise HTTPException(503, {"code": "model_unavailable", "message": colorizer.error or "Model not loaded."})
    try:
        async with _sem:
            result = await asyncio.wait_for(run_in_threadpool(colorizer.colorize, img), timeout=INFER_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(504, {"code": "timeout", "message": f"Inference exceeded {INFER_TIMEOUT_S:.0f} s."}) from None
    except ModelUnavailableError as exc:
        raise HTTPException(503, {"code": "model_unavailable", "message": str(exc)}) from None
    log.info("colorize ok %sx%s %.1fms", *img.size, result.inference_ms)
    return {
        "output": png_data_url(result.image),
        "preprocessed": png_data_url(preprocessed_preview(img)),
        "inference_ms": result.inference_ms,
        "input": {"width": img.size[0], "height": img.size[1], "format": img.format, "mode": img.mode},
        "model_input_size": IMG_SIZE,
        "label": "Model prediction — synthetic optical-style colours, not ground truth.",
    }


def load_manifest() -> list[dict]:
    path = SAMPLES_DIR / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


@app.get("/api/samples")
def samples():
    return [{k: v for k, v in s.items() if k != "files"} | {"assets": sorted(s.get("files", {}))} for s in load_manifest()]


@app.get("/api/samples/{sample_id}/{kind}")
def sample_asset(sample_id: str, kind: str):
    for s in load_manifest():
        if s["id"] == sample_id and kind in s.get("files", {}):
            path = (SAMPLES_DIR / s["files"][kind]).resolve()
            if SAMPLES_DIR.resolve() in path.parents and path.is_file():  # no path traversal
                return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    raise HTTPException(404, {"code": "not_found", "message": "Unknown sample or asset."})


@app.get("/api/evaluation")
def evaluation():
    if EVAL_FILE.is_file():
        data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
        data.pop("per_image", None)
        return {"available": True, "source": "measured in this repository (scripts/evaluate.py)", **data}
    return {
        "available": False,
        "message": "No evaluation has been run in this repository yet: it needs a trained checkpoint and the dataset.",
        "upstream_reported": {"psnr_db": "17.669 ± 2.929", "ssim": "0.3393 ± 0.1155", "n_images": 319,
                              "source": "Original author's README (fahad-mughal-rehman/sar-colorization-gan); not reproduced here."},
    }


if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, {"code": "not_found", "message": "Unknown endpoint."})
        candidate = (FRONTEND_DIR / full_path).resolve()
        if full_path and FRONTEND_DIR.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIR / "index.html")
