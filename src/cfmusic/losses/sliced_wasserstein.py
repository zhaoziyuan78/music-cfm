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


def sliced_wasserstein_distance(
    left: Tensor, right: Tensor, *, num_projections: int = 32, seed: int = 0
) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("Empirical SWD inputs must have identical rank-2 shapes")
    generator = torch.Generator(device=left.device).manual_seed(seed)
    directions = torch.randn(
        left.shape[1], num_projections, generator=generator, device=left.device
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    projected_left = (left.float() @ directions).sort(dim=0).values
    projected_right = (right.float() @ directions).sort(dim=0).values
    return (projected_left - projected_right).square().mean().sqrt()


def class_conditional_sliced_wasserstein(
    generated: Tensor,
    factual: Tensor,
    labels: Tensor,
    *,
    num_projections: int = 32,
    seed: int = 0,
) -> Tensor:
    losses = [
        sliced_wasserstein_distance(
            generated[labels == label],
            factual[labels == label],
            num_projections=num_projections,
            seed=seed + int(label),
        )
        for label in torch.unique(labels)
        if int((labels == label).sum()) >= 2
    ]
    return torch.stack(losses).mean() if losses else generated.sum() * 0.0
