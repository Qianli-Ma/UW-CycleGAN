from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


class VGG19Conv44(nn.Module):
    """Frozen VGG19 feature extractor through conv4_4, torchvision feature index 26."""

    def __init__(self) -> None:
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1
        model = vgg19(weights=weights)
        self.features = nn.Sequential(*list(model.features.children())[:27]).eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) * 0.5
        x = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)
        return self.features(x)
