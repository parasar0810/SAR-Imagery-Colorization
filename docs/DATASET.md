# Dataset

**Name:** Paired SAR-Optical Dataset (16K Images)
**Where:** https://www.kaggle.com/datasets/anjaliikakde/paired-sar-optical-dataset-16k-images
**Format:** one PNG per pair, with the SAR tile on the **left** half and the co-registered optical tile on the **right** half (`SAROpticalDataset` crops at `width // 2`).

## Not redistributed
The images are **not** included in this repository. Download them from Kaggle under the publisher’s terms (Kaggle account required). The licence and the original upstream source (for example, whether the tiles derive from Sentinel-1/Sentinel-2 pairs such as SEN1-2) are stated by the dataset publisher and have **not been independently verified here**. Check them before any redistribution or commercial use.

## Split used by the model
- `VAL_RATIO = 0.02`, `SPLIT_SEED = 42`: file names are sorted, shuffled with `random.Random(42)`, and the first 2% are held out (319 images in the upstream run).
- The same split serves validation during training and the final test. There is no separate untouched test set.
- The split is **tile-level**. Spatially adjacent or overlapping tiles from one scene may fall on both sides, so treat reported metrics as optimistic until a scene-level split is evaluated.

## Bundled demo images
`assets/demo/` holds six 256×256 panels cropped from the upstream author’s result figure (`assets/results/output.png`, MIT): two scenes, each as SAR input, generated output and optical reference. They are for demonstration only. The figure was rendered by matplotlib, so the panels are resampled and must not be used to compute metrics.
