# Model card — SAR → optical-style Attention GAN

| | |
|---|---|
| **Model** | Conditional GAN: Attention U-Net generator + multi-scale PatchGAN discriminator |
| **Authors** | Model and training: Fahad Ur Rehman (MIT). Inference packaging: Priyanshu Parasar. |
| **Framework** | PyTorch ≥ 2.0 |
| **Input** | 1 × 256 × 256 single-channel SAR intensity, scaled to [−1, 1] (see preprocessing) |
| **Output** | 3 × 256 × 256 RGB in [−1, 1] (tanh), shown as 8-bit RGB |
| **Parameters** | Generator 42,497,640 · discriminators 8,297,283 (training only) |
| **Checkpoint** | Not published (upstream did not release one). ≈170 MB fp32 / ≈85 MB fp16 generator-only once trained. |

## Intended use
- Visual aid that makes SAR tiles easier for non-specialists to read.
- Teaching and demonstration of image-to-image translation.

## Out-of-scope / inappropriate use
- Any quantitative use of colour: land-cover maps, NDVI-like vegetation analysis, water-quality or burn-severity estimation, change detection.
- Disaster or damage assessment, military or legal decisions. The model can produce plausible-looking but wrong content.
- Presenting outputs as real optical imagery. The UI and API label every output as a model prediction.

## Training data
[Paired SAR-Optical Dataset (16K Images)](https://www.kaggle.com/datasets/anjaliikakde/paired-sar-optical-dataset-16k-images); see [DATASET.md](DATASET.md). 98% train / 2% held out, by a seeded (42) shuffle of tile files.

## Preprocessing (identical at training and inference)
RGB → bicubic resize to 256×256 → `ToTensor` ([0,1]) → `Normalize(0.5, 0.5)` ([−1,1]) → mean over the three channels → 1×256×256.
A unit test checks that the inference implementation matches the notebook’s torchvision pipeline to within 1e-6.

## Evaluation
| Source | PSNR | SSIM | n |
|---|---|---|---|
| Original author (README) | 17.669 ± 2.929 dB | 0.3393 ± 0.1155 | 319 |
| This repository | not measured (no checkpoint) | — | — |

**How to interpret the numbers.** Both metrics compare the prediction with the pixel-aligned optical tile of the same scene. Absolute values are low because much optical information (paint colour, crop type, snow, small roads) is not present in C-band backscatter at this resolution. PSNR also favours blurry averages over sharp, plausible detail.

**Threats to validity**
1. **Tile-level random split.** If the source scenes were tiled with overlap or adjacency, near-duplicate content can appear in both train and test, inflating scores. A scene- or region-level split is needed for a strict generalisation estimate.
2. **Single dataset**, unknown geographic and seasonal coverage. There is no out-of-distribution test.
3. **No training history in the repository.** Curves cannot be shown or checked.

## Known failure modes (visible in the upstream figure)
- Field boundaries and fine linear features (roads) are lost. Outputs are smoother than real optical imagery.
- Snow and bright surfaces are not reproduced in mountain scenes; relief shading is.
- Colours drift towards the dataset’s dominant palette (greens and browns).

## Compute
- Training: designed for a Kaggle P100/T4 with AMP. 60 epochs; the upstream notebook resumes across ~9–12 h sessions.
- Inference: CPU is sufficient. Measured on a 12-thread laptop CPU: ≈139 ms median per 256² tile warm (≈194 ms first call), ≈330 ms through the API, ≈550 MB resident memory.
