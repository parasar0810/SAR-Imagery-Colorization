"""Evaluate a generator checkpoint on the held-out split — script version of notebook cell 33.

Writes ``evaluation.json`` with per-image PSNR/SSIM/MAE, summary statistics, CPU/GPU latency
and the exact configuration, so every number shown in the web app is traceable.

    python scripts/evaluate.py --data-dir /path/to/Paired_SAR_Optical_images \
        --weights checkpoints/last.pt --out evaluation/
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils as vutils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sar_colorization.data import SAROpticalDataset  # noqa: E402
from sar_colorization.inference import load_generator_state  # noqa: E402
from sar_colorization.metrics import mae, psnr, ssim  # noqa: E402
from sar_colorization.models import AttentionUNetGenerator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="evaluation")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grids", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, meta = load_generator_state(args.weights)
    G = AttentionUNetGenerator(1, 3, 64)
    G.load_state_dict(state, strict=True)
    G.to(device).eval()

    ds = SAROpticalDataset(args.data_dir, 256, "val", args.val_ratio, args.seed)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    rows, latencies, grids = [], [], 0
    with torch.no_grad():
        for b, (sar, opt) in enumerate(loader):
            sar, opt = sar.to(device), opt.to(device)
            t0 = time.perf_counter()
            fake = G(sar)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000 / sar.size(0))
            f01, o01 = ((fake + 1) / 2).clamp(0, 1), ((opt + 1) / 2).clamp(0, 1)
            for i in range(f01.size(0)):
                rows.append({"file": ds.files[b * 8 + i], "psnr": psnr(f01[i:i + 1], o01[i:i + 1]),
                             "ssim": ssim(f01[i:i + 1], o01[i:i + 1]), "mae": mae(f01[i:i + 1], o01[i:i + 1])})
            if grids < args.grids:
                g = torch.cat([(sar.repeat(1, 3, 1, 1) + 1) / 2, f01, o01])
                vutils.save_image(vutils.make_grid(g, nrow=sar.size(0)), os.path.join(args.out, f"comparison_{b:03d}.png"))
                grids += 1

    def stats(key: str) -> dict:
        v = np.array([r[key] for r in rows])
        return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}

    report = {
        "checkpoint": {"path": os.path.basename(args.weights), **meta},
        "split": {"val_ratio": args.val_ratio, "seed": args.seed, "n_images": len(rows), "method": "seeded file shuffle (tile-level)"},
        "metrics": {"psnr_db": stats("psnr"), "ssim": stats("ssim"), "mae": stats("mae")},
        "latency_ms_per_image": {"mean": float(np.mean(latencies)), "device": str(device), "batch_size": 8},
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "torch": torch.__version__},
        "per_image": rows,
    }
    with open(os.path.join(args.out, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("split", "metrics", "latency_ms_per_image")}, indent=2))


if __name__ == "__main__":
    main()
