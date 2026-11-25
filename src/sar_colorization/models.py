"""Model definitions for SAR -> optical translation.

The classes below are copied verbatim from the original notebook
``notebooks/sar_colorization_attention_gan.ipynb`` by Fahad Ur Rehman
(Fahad Mughal), MIT License — cells 10, 12 and 14. They are reproduced
unchanged so that checkpoints trained with the notebook load directly.
Do not edit the layer definitions: any change breaks checkpoint compatibility.
"""
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, max(in_dim // 8, 1), 1)
        self.key = nn.Conv2d(in_dim, max(in_dim // 8, 1), 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = self.softmax(torch.bmm(q, k))
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.InstanceNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True), nn.InstanceNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.InstanceNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

def down_block(in_c, out_c, normalize=True, dropout=0.0):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not normalize)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    if dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def up_block(in_c, out_c, dropout=0.0):
    layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
              nn.InstanceNorm2d(out_c), nn.ReLU(inplace=True)]
    if dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class AttentionUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, ngf=64):
        super().__init__()
        self.down1 = down_block(in_channels, ngf, normalize=False)   # 256 -> 128
        self.down2 = down_block(ngf, ngf * 2)                         # 128 -> 64
        self.down3 = down_block(ngf * 2, ngf * 4)                     # 64  -> 32
        self.down4 = down_block(ngf * 4, ngf * 8)                     # 32  -> 16
        self.self_attn = SelfAttention(ngf * 8)                       # at 16x16
        self.down5 = down_block(ngf * 8, ngf * 8)                     # 16  -> 8
        self.down6 = down_block(ngf * 8, ngf * 8)                     # 8   -> 4
        self.down7 = down_block(ngf * 8, ngf * 8, normalize=False)    # 4   -> 2

        self.up1 = up_block(ngf * 8, ngf * 8, dropout=0.5)            # 2 -> 4
        self.up2 = up_block(ngf * 16, ngf * 8, dropout=0.5)           # 4 -> 8
        self.up3 = up_block(ngf * 16, ngf * 8, dropout=0.5)           # 8 -> 16
        self.ag3 = AttentionGate(ngf * 8, ngf * 8, ngf * 4)
        self.up4 = up_block(ngf * 16, ngf * 4)                        # 16 -> 32
        self.ag4 = AttentionGate(ngf * 4, ngf * 4, ngf * 2)
        self.up5 = up_block(ngf * 8, ngf * 2)                         # 32 -> 64
        self.ag5 = AttentionGate(ngf * 2, ngf * 2, ngf)
        self.up6 = up_block(ngf * 4, ngf)                             # 64 -> 128
        self.ag6 = AttentionGate(ngf, ngf, ngf // 2)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1),       # 128 -> 256
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d4 = self.self_attn(d4)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7)
        u1 = torch.cat([u1, d6], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d5], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, self.ag3(g=u3, x=d4)], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, self.ag4(g=u4, x=d3)], dim=1)
        u5 = self.up5(u4)
        u5 = torch.cat([u5, self.ag5(g=u5, x=d2)], dim=1)
        u6 = self.up6(u5)
        u6 = torch.cat([u6, self.ag6(g=u6, x=d1)], dim=1)
        return self.final(u6)

class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=4, ndf=64, n_layers=3):
        super().__init__()
        blocks = [[nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            blocks.append([nn.Conv2d(nf_prev, nf, 4, 2, 1), nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)])
        nf_prev, nf = nf, min(nf * 2, 512)
        blocks.append([nn.Conv2d(nf_prev, nf, 4, 1, 1), nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)])
        blocks.append([nn.Conv2d(nf, 1, 4, 1, 1)])
        self.blocks = nn.ModuleList([nn.Sequential(*b) for b in blocks])

    def forward(self, x):
        feats = [x]
        for block in self.blocks:
            feats.append(block(feats[-1]))
        return feats[1:]  # intermediate feats + final patch map (last element)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, in_channels=4, ndf=64, n_layers=3, num_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [NLayerDiscriminator(in_channels, ndf, n_layers) for _ in range(num_scales)]
        )
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        results = []
        inp = x
        for i, D in enumerate(self.discriminators):
            results.append(D(inp))
            if i != len(self.discriminators) - 1:
                inp = self.downsample(inp)
        return results
