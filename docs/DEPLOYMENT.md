# Deployment

The service is one container: FastAPI serves both the API (`/api/*`) and the built web app. Inference is CPU-only.

## Resource needs (measured / estimated)
| | |
|---|---|
| CPU | 1–2 vCPU. Measured per 256² tile: ≈139 ms median warm natively; 146–297 ms inside the container (Docker Desktop VM). Expect slower on shared vCPUs |
| RAM | ≈400 MiB for the running container with the model loaded (measured via `docker stats`); ≈550 MB natively on Windows. Allow 1 GB for headroom |
| Disk | Image 1.61 GB (measured; CPU-only torch) + ≈85–170 MB weights |
| GPU | Not needed |
| Concurrency | 1 inference at a time per process by design (`SAR_CONCURRENCY`) |

## Configuration
| Variable | Default | Meaning |
|---|---|---|
| `SAR_WEIGHTS` | `/app/weights/generator.pt` | Generator checkpoint path |
| `SAR_WEIGHTS_URL` / `SAR_WEIGHTS_SHA256` | — | Optional https download at start, checksum mandatory |
| `SAR_MAX_UPLOAD_MB` | 8 | Upload limit |
| `SAR_MAX_PIXELS` | 16777216 | Pixel limit, checked before full decode |
| `SAR_RATE_LIMIT_PER_MIN` | 20 | Per-client requests on POST endpoints |
| `SAR_INFER_TIMEOUT_S` | 30 | Per-inference timeout |
| `SAR_TRUST_PROXY` | 0 | Set to 1 only behind a proxy that sets `X-Forwarded-For`; otherwise all clients share one rate-limit bucket |
| `SAR_CORS_ORIGINS` | (none) | Comma-separated origins, only if the UI is hosted elsewhere |

## Options considered
| Option | Fit | Notes |
|---|---|---|
| Hugging Face Spaces (Docker SDK) | Good for a public CPU demo | The container already listens on 7860. Check the current free-tier CPU/RAM and sleep policy before relying on it. Weights can live in a model repo and be fetched with `SAR_WEIGHTS_URL`. |
| Existing AegisOps EC2 (c7i-flex.large, 4 GiB) | Possible but not preferred | It already runs 11 containers (~1.2–1.6 GB working set); ~0.6–1 GB more would fit on paper, but CPU-bound inference would compete with AegisOps’s live demo. Needs owner approval and a rollback plan. |
| Render / Railway / Fly.io | Possible | Paid tiers needed for ≥1 GB RAM on most; verify pricing first. |

No paid resource is created without the owner’s approval.

## Verified locally (2026-11-25)
`docker build` succeeds; the container reports healthy, serves the UI, `/api/preprocess` works, `/api/colorize` returns 503 without weights and 200 with a mounted checkpoint (tested with random weights), and the process runs as non-root user `app`.

## Steps (any Docker host)
```bash
docker build -t sar-colorization .
docker run -d -p 7860:7860 -e SAR_TRUST_PROXY=1 \
  -e SAR_WEIGHTS_URL=https://.../generator.pt -e SAR_WEIGHTS_SHA256=<sha256> sar-colorization
curl -s localhost:7860/api/health
```
Without weights, the container still starts and serves the workbench with live inference marked offline.

## Rollback
Redeploy the previous image tag. The service is stateless: no database, and no uploads are stored.
