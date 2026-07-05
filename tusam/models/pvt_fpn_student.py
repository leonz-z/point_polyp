from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tusam.models.pvt_v2_local import pvt_v2_b2_local


class ConvGNReLU(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.GroupNorm(8, c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PVTv2FPNStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = pvt_v2_b2_local()
        ch = [64, 128, 320, 512]

        self.lats = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in ch])
        self.smooth = nn.ModuleList([ConvGNReLU(128, 128) for _ in ch])

        self.head = nn.Sequential(ConvGNReLU(128 * 4, 128), nn.Conv2d(128, 1, 1))
        self.aux1 = nn.Conv2d(128, 1, 1)
        self.aux2 = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        feats = self.backbone(x)
        p = [lat(f) for lat, f in zip(self.lats, feats)]

        for i in range(2, -1, -1):
            p[i] = p[i] + F.interpolate(p[i + 1], size=p[i].shape[-2:], mode="bilinear", align_corners=False)

        p = [s(v) for s, v in zip(self.smooth, p)]
        h, w = x.shape[-2:]
        up = [F.interpolate(v, size=(h, w), mode="bilinear", align_corners=False) for v in p]

        main = self.head(torch.cat(up, dim=1))
        aux1 = F.interpolate(self.aux1(p[1]), size=(h, w), mode="bilinear", align_corners=False)
        aux2 = F.interpolate(self.aux2(p[2]), size=(h, w), mode="bilinear", align_corners=False)
        return main, aux1, aux2
