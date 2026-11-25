"""Download generator weights at container start, verifying a SHA-256 checksum.

Reads SAR_WEIGHTS_URL and SAR_WEIGHTS_SHA256 from the environment. Does nothing if the
weights file already exists or no URL is configured (the service then starts in
"model unavailable" mode). A checksum is mandatory when a URL is set: unverified
checkpoints are never loaded.
"""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

dest = Path(os.getenv("SAR_WEIGHTS", "weights/generator.pt"))
url = os.getenv("SAR_WEIGHTS_URL", "").strip()
expected = os.getenv("SAR_WEIGHTS_SHA256", "").strip().lower()

if dest.is_file() or not url:
    sys.exit(0)
if not url.startswith("https://"):
    sys.exit("SAR_WEIGHTS_URL must use https://")
if len(expected) != 64:
    sys.exit("SAR_WEIGHTS_SHA256 (64 hex chars) is required when SAR_WEIGHTS_URL is set")

dest.parent.mkdir(parents=True, exist_ok=True)
tmp = dest.with_suffix(".part")
h = hashlib.sha256()
with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310 (https enforced above)
    while chunk := r.read(1 << 20):
        h.update(chunk)
        f.write(chunk)
if h.hexdigest() != expected:
    tmp.unlink(missing_ok=True)
    sys.exit("checksum mismatch; refusing to install downloaded weights")
tmp.replace(dest)
print(f"installed {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
