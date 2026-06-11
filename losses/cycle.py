from __future__ import annotations

import torch
from torch.nn import functional as F


def cycle_loss(rec_x: torch.Tensor, x: torch.Tensor, rec_y: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(rec_x, x) + F.l1_loss(rec_y, y)
