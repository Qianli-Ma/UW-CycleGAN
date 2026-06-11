from __future__ import annotations

import torch
from torch.nn import functional as F


def gaussian_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd.")
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def gaussian_blur(x: torch.Tensor, kernel_size: int = 21, sigma: float = 3.0) -> torch.Tensor:
    kernel = gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel.expand(x.shape[1], 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, kernel, groups=x.shape[1])
