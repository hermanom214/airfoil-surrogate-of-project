from __future__ import annotations

import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 5, out_channels: int = 3):
        super().__init__()

        self.enc1 = self.block(in_channels, 32)
        self.enc2 = self.block(32, 64)
        self.enc3 = self.block(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self.block(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.block(64, 32)

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    @staticmethod
    def block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass (U-Net)
#
# Input:
# x: (B, 5, H, W)
#
# Encoder (reduces resolution, increases channels – feature extraction)
# --------------------------------------------------------------------
# e1 = enc1(x)
# → (B, 5, H, W) → (B, 32, H, W)
#   2× Conv + ReLU, keeps resolution, increases channels
#
# e2 = enc2(pool(e1))
# → pool: (B, 32, H, W) → (B, 32, H/2, W/2)
# → enc2: (B, 32, H/2, W/2) → (B, 64, H/2, W/2)
#   downsampling + more feature channels
#
# e3 = enc3(pool(e2))
# → pool: (B, 64, H/2, W/2) → (B, 64, H/4, W/4)
# → enc3: (B, 64, H/4, W/4) → (B, 128, H/4, W/4)
#
# Bottleneck (smallest resolution, highest channel count – global context)
# -----------------------------------------------------------------------
# b = bottleneck(pool(e3))
# → pool: (B, 128, H/4, W/4) → (B, 128, H/8, W/8)
# → bottleneck: (B, 128, H/8, W/8) → (B, 256, H/8, W/8)
#
# Decoder (increases resolution, combines with encoder features)
# --------------------------------------------------------------
# d3 = up3(b)
# → (B, 256, H/8, W/8) → (B, 128, H/4, W/4)
#   upsampling (ConvTranspose)
#
# d3 = cat(d3, e3)
# → (B, 128, H/4, W/4) + (B, 128, H/4, W/4)
# → (B, 256, H/4, W/4)
#   skip connection (restores spatial detail)
#
# d3 = dec3(d3)
# → (B, 256, H/4, W/4) → (B, 128, H/4, W/4)
#
# d2 = up2(d3)
# → (B, 128, H/4, W/4) → (B, 64, H/2, W/2)
#
# d2 = cat(d2, e2)
# → (B, 64, H/2, W/2) + (B, 64, H/2, W/2)
# → (B, 128, H/2, W/2)
#
# d2 = dec2(d2)
# → (B, 128, H/2, W/2) → (B, 64, H/2, W/2)
#
# d1 = up1(d2)
# → (B, 64, H/2, W/2) → (B, 32, H, W)
#
# d1 = cat(d1, e1)
# → (B, 32, H, W) + (B, 32, H, W)
# → (B, 64, H, W)
#
# d1 = dec1(d1)
# → (B, 64, H, W) → (B, 32, H, W)
#
# Output
# ------
# out = self.out(d1)
# → (B, 32, H, W) → (B, 3, H, W)
#   1×1 convolution – maps feature maps to physical quantities (e.g. Ux, Uy, p)
#
# Summary:
# encoder → compresses spatial resolution, learns context
# bottleneck → global representation
# decoder → restores resolution + recovers details via skip connections

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)