from __future__ import annotations

import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int = 64, growth_rate: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseNetBlock(nn.Module):
    """Dense block from UW-CycleGAN Fig. 4b: 256 -> 128, then 8 x 16 growth."""

    def __init__(self, channels: int = 256, transition_channels: int = 128, layers: int = 8) -> None:
        super().__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(channels, transition_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        dense_layers: list[nn.Module] = []
        in_channels = transition_channels
        for _ in range(layers):
            dense_layers.append(DenseLayer(in_channels))
            in_channels += 16
        self.layers = nn.ModuleList(dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [self.transition(x)]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class UWCycleGANGenerator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(c, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(c * 2, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(c * 4, affine=False),
            nn.ReLU(inplace=True),
            DenseNetBlock(c * 4),
            DenseNetBlock(c * 4),
            DenseNetBlock(c * 4),
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(c * 2, affine=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(c, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
