"""Fixed abduction-noise projections and post-hoc MLP probes."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FixedNoiseProjector(nn.Module):
    projection: Tensor

    def __init__(self, input_dim: int, projection_dim: int = 128, seed: int = 2026) -> None:
        super().__init__()
        if projection_dim > input_dim:
            raise ValueError("projection_dim cannot exceed flattened latent dimension")
        generator = torch.Generator().manual_seed(seed)
        gaussian = torch.randn(input_dim, projection_dim, generator=generator)
        orthogonal, _ = torch.linalg.qr(gaussian, mode="reduced")
        self.register_buffer("projection", orthogonal.T.contiguous())

    def forward(self, noise: Tensor) -> Tensor:
        return noise.flatten(1) @ self.projection.T


def noise_summaries(noise: Tensor) -> Tensor:
    mean = noise.mean(dim=-1).mean(dim=-1, keepdim=True)
    std = noise.std(dim=-1, unbiased=False).mean(dim=-1, keepdim=True)
    if noise.shape[1] > 1:
        left = noise[:, :-1].flatten(1)
        right = noise[:, 1:].flatten(1)
        left = left - left.mean(1, keepdim=True)
        right = right - right.mean(1, keepdim=True)
        lag = (left * right).mean(1, keepdim=True) / (
            left.std(1, keepdim=True, unbiased=False) * right.std(1, keepdim=True, unbiased=False)
        ).clamp_min(1e-6)
    else:
        lag = torch.zeros_like(mean)
    norm = noise.flatten(1).norm(dim=1, keepdim=True)
    return torch.cat([mean, std, lag, norm], dim=-1)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)
