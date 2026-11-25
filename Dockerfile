# --- 1. Build the web app ---------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- 2. CPU inference runtime -----------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SAR_WEIGHTS=/app/weights/generator.pt PORT=7860
WORKDIR /app

# CPU-only PyTorch keeps the image ~4x smaller than the CUDA build.
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY api/ api/
COPY scripts/fetch_weights.py scripts/
COPY assets/demo/ assets/demo/
COPY --from=web /web/dist frontend/dist

RUN useradd --create-home --uid 1000 app && mkdir -p /app/weights && chown -R app /app
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health',timeout=4)"
# Optional checksum-verified weight download, then serve. One worker: inference is CPU-bound
# and serialised by a semaphore, so more workers would only multiply memory use.
CMD ["sh", "-c", "python scripts/fetch_weights.py && exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]
