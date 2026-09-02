"""Projected one-dimensional Wasserstein prior diagnostic."""

from __future__ import annotations

import torch
from torch import Tensor


def sliced_wasserstein_standard_normal(
    samples: Tensor, *, num_projections: int = 64, seed: int = 0
) -> Tensor:
    generator = torch.Generator(device=samples.device).manual_seed(seed)
    directions = torch.randn(
        samples.shape[1], num_projections, generator=generator, device=samples.device
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    projected = (samples @ directions).sort(dim=0).values
    reference = (
        torch.randn(
            projected.shape, generator=generator, device=samples.device, dtype=samples.dtype
        )
        .sort(dim=0)
        .values
    )
    return (projected - reference).square().mean().sqrt()
