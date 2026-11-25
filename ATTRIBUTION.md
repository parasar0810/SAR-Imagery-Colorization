# Attribution

This repository is a **derivative of [fahad-mughal-rehman/sar-colorization-gan](https://github.com/fahad-mughal-rehman/sar-colorization-gan)**
(MIT License), imported with its full commit history on 2026-11-25.

## Original work — Fahad Ur Rehman (Fahad Mughal)
Everything in the upstream repository at commit `2180fbe`, including:
- the model design and its PyTorch implementation: Attention U-Net generator, multi-scale PatchGAN discriminator, and the LSGAN + feature-matching + L1 + VGG19 + SSIM loss;
- the training and evaluation notebook `notebooks/sar_colorization_attention_gan.ipynb`;
- `scripts/crop_results.py`;
- the result figure `assets/results/output.png`, and the reported metrics (PSNR 17.669 ± 2.929 dB, SSIM 0.3393 ± 0.1155 on 319 images), produced with the original author's checkpoint, which is not published.

The upstream commits are kept unchanged, with their original authors, in this repository's history.

## Derivative work — Priyanshu Parasar
Commits after the import, by Priyanshu Parasar (with AI pair-programming assistance), add:
- an importable Python package that reuses the original model code;
- an inference pipeline with safe checkpoint loading;
- a FastAPI inference service with upload, size, pixel and concurrency limits;
- the web application;
- automated tests;
- documentation (model card, dataset notes, evaluation methodology).

These additions do not change the model architecture or the training objective.

## Dataset
[Paired SAR-Optical Dataset (16K Images)](https://www.kaggle.com/datasets/anjaliikakde/paired-sar-optical-dataset-16k-images) on Kaggle, published by its own author under its own licence.
It is **not** redistributed here. See `docs/DATASET.md`.
