"""Train the Attention GAN — a script version of the original notebook's training cells.

The training step, losses, optimisers, LR schedule, AMP and checkpoint format are
the original author's (notebook cells 18–24); this file only adds a CLI, seeding,
CPU fallback and logging. Checkpoints are compatible with the notebook.

Example (Kaggle GPU):
    python scripts/train.py --data-dir /kaggle/input/.../Paired_SAR_Optical_images \
        --out /kaggle/working --epochs 60
Quick CPU smoke run (a handful of steps, NOT a real training run):
    python scripts/train.py --data-dir ./data --out ./runs/smoke --epochs 1 --max-steps 5 --batch-size 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils as vutils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sar_colorization.data import SAROpticalDataset  # noqa: E402
from sar_colorization.losses import (  # noqa: E402
    VGGPerceptualLoss,
    feature_matching_loss,
    gan_loss,
    ssim_loss,
)
from sar_colorization.models import AttentionUNetGenerator, MultiscaleDiscriminator  # noqa: E402

log = logging.getLogger("train")

DEFAULTS = dict(IMG_SIZE=256, BATCH_SIZE=8, NUM_WORKERS=2, EPOCHS=60, LR=2e-4, BETA1=0.5, BETA2=0.999,
                VAL_RATIO=0.02, SPLIT_SEED=42, SAVE_EVERY=5, LAMBDA_GAN=1.0, LAMBDA_FM=10.0, LAMBDA_L1=50.0,
                LAMBDA_VGG=10.0, LAMBDA_SSIM=10.0, NGF=64, NDF=64, NUM_SCALES=3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=DEFAULTS["EPOCHS"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["BATCH_SIZE"])
    ap.add_argument("--workers", type=int, default=DEFAULTS["NUM_WORKERS"])
    ap.add_argument("--max-steps", type=int, default=0, help="Stop each epoch after N steps (smoke tests only)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = {**DEFAULTS, "EPOCHS": args.epochs, "BATCH_SIZE": args.batch_size, "NUM_WORKERS": args.workers,
           "DATA_DIR": args.data_dir, "OUTPUT_DIR": args.out, "RESUME": not args.no_resume}
    ckpt_dir, sample_dir = os.path.join(args.out, "checkpoints"), os.path.join(args.out, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.manual_seed(cfg["SPLIT_SEED"])
    random.seed(cfg["SPLIT_SEED"])
    np.random.seed(cfg["SPLIT_SEED"])
    log.info("device=%s amp=%s", device, use_amp)

    train_ds = SAROpticalDataset(cfg["DATA_DIR"], cfg["IMG_SIZE"], "train", cfg["VAL_RATIO"], cfg["SPLIT_SEED"])
    val_ds = SAROpticalDataset(cfg["DATA_DIR"], cfg["IMG_SIZE"], "val", cfg["VAL_RATIO"], cfg["SPLIT_SEED"])
    train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True, num_workers=cfg["NUM_WORKERS"],
                              pin_memory=use_amp, drop_last=True)
    fixed_sar, fixed_opt = next(iter(DataLoader(val_ds, batch_size=8, shuffle=False)))
    fixed_sar, fixed_opt = fixed_sar.to(device), fixed_opt.to(device)

    G = AttentionUNetGenerator(1, 3, cfg["NGF"]).to(device)
    D = MultiscaleDiscriminator(4, cfg["NDF"], num_scales=cfg["NUM_SCALES"]).to(device)
    vgg_loss_fn = VGGPerceptualLoss().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["LR"], betas=(cfg["BETA1"], cfg["BETA2"]))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["LR"], betas=(cfg["BETA1"], cfg["BETA2"]))

    def lr_lambda(epoch: int) -> float:
        decay_start = cfg["EPOCHS"] // 2
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / float(cfg["EPOCHS"] - decay_start + 1))

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)
    scaler_G = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_D = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext

    history: dict[str, list[float]] = {"G": [], "D": [], "L1": [], "SSIM": []}
    start_epoch = 0
    last = os.path.join(ckpt_dir, "last.pt")
    if cfg["RESUME"] and os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=True)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
        start_epoch, history = ck["epoch"] + 1, ck.get("history", history)
        log.info("resumed from epoch %d", start_epoch)

    def save(epoch: int, tag: str) -> None:
        torch.save({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(), "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(), "history": history, "cfg": cfg}, os.path.join(ckpt_dir, f"{tag}.pt"))

    def sample_grid(epoch: int) -> None:
        G.eval()
        with torch.no_grad():
            fake = G(fixed_sar)
        G.train()
        grid = (torch.cat([fixed_sar.repeat(1, 3, 1, 1), fake, fixed_opt]) + 1) / 2
        vutils.save_image(vutils.make_grid(grid, nrow=fixed_sar.size(0)), os.path.join(sample_dir, f"epoch_{epoch:03d}.png"))

    for epoch in range(start_epoch, cfg["EPOCHS"]):
        t0, running, n = time.time(), {"G": 0.0, "D": 0.0, "L1": 0.0, "SSIM": 0.0}, 0
        for step, (sar, opt) in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            sar, opt = sar.to(device, non_blocking=True), opt.to(device, non_blocking=True)
            real_ab = torch.cat([sar, opt], dim=1)

            opt_D.zero_grad(set_to_none=True)
            with autocast():
                fake_opt = G(sar)
                pred_real, pred_fake = D(real_ab), D(torch.cat([sar, fake_opt.detach()], dim=1))
                loss_D = sum(0.5 * (gan_loss(pr[-1], True) + gan_loss(pf[-1], False)) for pr, pf in zip(pred_real, pred_fake)) / len(pred_real)
            scaler_D.scale(loss_D).backward(); scaler_D.step(opt_D); scaler_D.update()

            opt_G.zero_grad(set_to_none=True)
            with autocast():
                pred_fake_g = D(torch.cat([sar, fake_opt], dim=1))
                with torch.no_grad():
                    pred_real_g = D(real_ab)
                loss_gan = sum(gan_loss(pf[-1], True) for pf in pred_fake_g) / len(pred_fake_g)
                loss_fm = feature_matching_loss(pred_real_g, pred_fake_g)
                loss_l1 = F.l1_loss(fake_opt, opt)
                loss_vgg = vgg_loss_fn(fake_opt, opt)
                loss_ss = ssim_loss(fake_opt, opt)
                loss_G = (cfg["LAMBDA_GAN"] * loss_gan + cfg["LAMBDA_FM"] * loss_fm + cfg["LAMBDA_L1"] * loss_l1
                          + cfg["LAMBDA_VGG"] * loss_vgg + cfg["LAMBDA_SSIM"] * loss_ss)
            scaler_G.scale(loss_G).backward(); scaler_G.step(opt_G); scaler_G.update()

            running["G"] += loss_G.item(); running["D"] += loss_D.item()
            running["L1"] += loss_l1.item(); running["SSIM"] += 1 - loss_ss.item(); n += 1

        sched_G.step(); sched_D.step()
        for k in running:
            history[k].append(running[k] / max(n, 1))
        log.info("epoch %d/%d G=%.4f D=%.4f L1=%.4f SSIM=%.4f (%.1fs, %d steps)", epoch + 1, cfg["EPOCHS"],
                 *(history[k][-1] for k in ("G", "D", "L1", "SSIM")), time.time() - t0, n)
        if (epoch + 1) % cfg["SAVE_EVERY"] == 0 or epoch == cfg["EPOCHS"] - 1:
            sample_grid(epoch + 1)
            save(epoch, "last")
            save(epoch, f"epoch_{epoch + 1:03d}")

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f)
    log.info("done; checkpoints in %s", ckpt_dir)


if __name__ == "__main__":
    main()
