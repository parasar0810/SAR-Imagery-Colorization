"""Strip a training checkpoint down to the generator weights for deployment.

A full checkpoint also holds the discriminator and both optimiser states (~4x larger).
The service only needs the generator. Optionally store it in float16 to halve the size;
weights are cast back to float32 at load time.

    python scripts/export_generator.py checkpoints/last.pt weights/generator.pt [--fp16]
"""
import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sar_colorization.inference import load_generator_state  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    state, meta = load_generator_state(a.src)
    if a.fp16:
        state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    os.makedirs(os.path.dirname(a.dst) or ".", exist_ok=True)
    torch.save(state, a.dst)
    digest = hashlib.sha256(open(a.dst, "rb").read()).hexdigest()
    print(f"wrote {a.dst} ({os.path.getsize(a.dst) / 1e6:.1f} MB, sha256 {digest}) from {meta}")


if __name__ == "__main__":
    main()
