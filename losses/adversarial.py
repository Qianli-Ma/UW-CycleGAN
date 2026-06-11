from __future__ import annotations

import torch
from torch.nn import functional as F


def generator_adv_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_fake, torch.ones_like(pred_fake))


def discriminator_adv_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    real_loss = F.mse_loss(pred_real, torch.ones_like(pred_real))
    fake_loss = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
    return real_loss + fake_loss


def discriminator_badv_loss(
    pred_real: torch.Tensor,
    pred_blur: torch.Tensor,
    pred_fake: torch.Tensor,
) -> torch.Tensor:
    real_loss = F.mse_loss(pred_real, torch.ones_like(pred_real))
    blur_loss = F.mse_loss(pred_blur, torch.zeros_like(pred_blur))
    fake_loss = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
    return real_loss + blur_loss + fake_loss
