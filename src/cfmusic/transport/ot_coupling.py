"""Within-style minibatch OT between Gaussian noise and factual latents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

_PROJECTION_CACHE: dict[tuple[int, int, str], Tensor] = {}


def _fixed_projection(dimension: int, projection_dim: int, device: torch.device) -> Tensor:
    key = (dimension, projection_dim, str(device))
    projection = _PROJECTION_CACHE.get(key)
    if projection is None:
        generator = torch.Generator(device=device).manual_seed(2026)
        projection = (
            torch.randn(dimension, projection_dim, generator=generator, device=device)
            / projection_dim**0.5
        )
        _PROJECTION_CACHE[key] = projection
    return projection


@dataclass(frozen=True)
class OTCouplingResult:
    noise: Tensor
    fallback_groups: int
    total_groups: int

    @property
    def fallback_ratio(self) -> float:
        return self.fallback_groups / max(1, self.total_groups)


def couple_noise_to_data(
    noise: Tensor,
    latent: Tensor,
    style_id: Tensor,
    *,
    solver: str = "hungarian",
    cost_projection_dim: int = 128,
    regularization: float = 0.05,
) -> OTCouplingResult:
    """Reorder only Gaussian samples; no factual music sample is paired to another music sample."""
    if noise.shape != latent.shape:
        raise ValueError("noise and latent must have identical shapes")
    output = noise.clone()
    fallback = 0
    groups = torch.unique(style_id)
    for style in groups:
        indices = torch.nonzero(style_id == style, as_tuple=False).flatten()
        if indices.numel() < 2:
            fallback += 1
            continue
        flat_noise = noise.index_select(0, indices).detach().float().flatten(1)
        flat_data = latent.index_select(0, indices).detach().float().flatten(1)
        dimension = flat_data.shape[1]
        if cost_projection_dim < dimension:
            projection = _fixed_projection(dimension, cost_projection_dim, flat_data.device)
            flat_noise = flat_noise @ projection
            flat_data = flat_data @ projection
        cost = torch.cdist(flat_noise, flat_data).square()
        if solver == "hungarian":
            row, column = linear_sum_assignment(cost.cpu().numpy())
            permutation = np.empty(len(row), dtype=np.int64)
            permutation[column] = row
            chosen = indices[torch.as_tensor(permutation, device=indices.device)]
        elif solver == "sinkhorn":
            kernel = torch.exp(-cost / max(regularization, 1e-6))
            for _ in range(50):
                kernel = kernel / kernel.sum(1, keepdim=True).clamp_min(1e-12)
                kernel = kernel / kernel.sum(0, keepdim=True).clamp_min(1e-12)
            # Independent argmax can select one Gaussian row more than once.  A
            # maximum-weight bipartite rounding preserves the required permutation.
            row, column = linear_sum_assignment(-kernel.detach().cpu().numpy())
            permutation = np.empty(len(row), dtype=np.int64)
            permutation[column] = row
            chosen = indices[torch.as_tensor(permutation, device=indices.device)]
        else:
            raise ValueError(f"Unsupported OT solver: {solver}")
        output.index_copy_(0, indices, noise.index_select(0, chosen))
    return OTCouplingResult(output, fallback, len(groups))
