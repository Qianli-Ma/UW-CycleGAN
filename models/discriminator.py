from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


def sn_conv(in_channels: int, out_channels: int, stride: int) -> nn.Conv2d:
    return spectral_norm(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=stride,
            padding=1,
        )
    )


class UWCycleGANDiscriminator(nn.Module):
    """70x70 receptive-field PatchGAN with spectral normalisation."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            sn_conv(3, c, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(c, c * 2, stride=2),
            nn.InstanceNorm2d(c * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(c * 2, c * 4, stride=2),
            nn.InstanceNorm2d(c * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(c * 4, c * 8, stride=1),
            nn.InstanceNorm2d(c * 8, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(c * 8, 1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
