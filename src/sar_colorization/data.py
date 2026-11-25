"""Paired SAR/optical dataset, copied verbatim from the original notebook (cell 6)
by Fahad Ur Rehman (Fahad Mughal), MIT License.

Each PNG holds the SAR tile on the left half and the optical tile on the right half.
The split is a seeded shuffle of file names (``seed=42``, 2% held out). Note: this is a
random split over tiles, not over scenes; see docs/DATASET.md for the leakage caveat.
"""
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SAROpticalDataset(Dataset):
    def __init__(self, data_dir, img_size=256, split="train", val_ratio=0.02, seed=42):
        self.data_dir = data_dir
        self.img_size = img_size

        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".png")])
        rng = random.Random(seed)
        rng.shuffle(all_files)

        n_val = int(len(all_files) * val_ratio)
        if split == "train":
            self.files = all_files[n_val:]
        else:
            self.files = all_files[:n_val]

        print(f"[{split}] {len(self.files)} images")

        self.resize = transforms.Resize((img_size, img_size),
                                         interpolation=transforms.InterpolationMode.BICUBIC)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        half_w = w // 2
        sar_img = img.crop((0, 0, half_w, h))
        opt_img = img.crop((half_w, 0, w, h))

        sar_img = self.resize(sar_img)
        opt_img = self.resize(opt_img)

        sar_tensor = self.normalize(self.to_tensor(sar_img)).mean(dim=0, keepdim=True)
        opt_tensor = self.normalize(self.to_tensor(opt_img))

        return sar_tensor, opt_tensor
