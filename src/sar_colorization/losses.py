"""Training losses, copied verbatim from the original notebook (cell 16) by
Fahad Ur Rehman (Fahad Mughal), MIT License. Used by ``train.py``; the
``ssim`` function is also reused for evaluation metrics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def feature_matching_loss(real_feats_list, fake_feats_list):
    loss = 0.0
    for real_feats, fake_feats in zip(real_feats_list, fake_feats_list):
        for rf, ff in zip(real_feats[:-1], fake_feats[:-1]):
            loss = loss + F.l1_loss(ff, rf.detach())
    return loss / len(real_feats_list)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features
            self.available = True
        except Exception as e:
            print("WARNING: could not download VGG19 weights (no internet on this Kaggle session?). "
                  "Perceptual loss will be disabled automatically.", e)
            self.available = False
            return

        layer_ids = [2, 7, 12, 21, 30]
        self.slices = nn.ModuleList()
        prev = 0
        for idx in layer_ids:
            self.slices.append(nn.Sequential(*[vgg[j] for j in range(prev, idx)]))
            prev = idx
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.weights = [1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5]

    def forward(self, x, y):
        if not self.available:
            return torch.tensor(0.0, device=x.device)
        x, y = (x + 1) / 2, (y + 1) / 2
        x, y = (x - self.mean) / self.std, (y - self.mean) / self.std
        loss = 0.0
        fx, fy = x, y
        for i, s in enumerate(self.slices):
            fx, fy = s(fx), s(fy)
            loss = loss + self.weights[i] * F.l1_loss(fx, fy.detach())
        return loss


def gaussian_window(window_size, sigma, channel):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = g.mm(g.t()).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11):
    channel = img1.size(1)
    window = gaussian_window(window_size, 1.5, channel).to(img1.device).type_as(img1)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim_loss(img1, img2):
    img1, img2 = (img1 + 1) / 2, (img2 + 1) / 2
    return 1 - ssim(img1, img2)
