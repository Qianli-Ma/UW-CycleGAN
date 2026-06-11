from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def content_loss(
    extractor: nn.Module,
    fake_y: torch.Tensor,
    x: torch.Tensor,
    fake_x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return F.l1_loss(extractor(fake_y), extractor(x)) + F.l1_loss(extractor(fake_x), extractor(y))
