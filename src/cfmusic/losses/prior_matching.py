"""Classwise standard-Gaussian moment matching."""

from __future__ import annotations

import torch
from torch import Tensor


def classwise_prior_matching(
    projected_noise: Tensor, style_id: Tensor, *, min_samples_per_class: int = 2
) -> Tensor:
    losses: list[Tensor] = []
    for style in torch.unique(style_id):
        values = projected_noise[style_id == style]
        if values.shape[0] < min_samples_per_class:
            continue
        mean = values.mean(0)
        variance = values.var(0, unbiased=False)
        losses.append(mean.square().mean() + (variance - 1).square().mean())
    return torch.stack(losses).mean() if losses else projected_noise.sum() * 0.0
