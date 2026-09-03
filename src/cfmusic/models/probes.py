"""Dynamic abduction-noise views and post-hoc MLP probes."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DynamicNoiseProjector(nn.Module):
    """Resampled, deterministic multi-view features for exogeneity training.

    Projection matrices are generated on demand rather than checkpointed.  All
    DDP ranks use the same global-step seed, while validation uses a disjoint
    seed stream.  Rademacher directions preserve unit Gaussian scale and avoid
    the fixed projector's large permanent null space.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        *,
        num_views: int = 3,
        seed: int = 2026,
        refresh_interval: int = 1,
        block_tokens: int = 8,
        block_channels: int = 32,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or projection_dim <= 0 or projection_dim > input_dim:
            raise ValueError("Invalid dynamic projection dimensions")
        if num_views < 2:
            raise ValueError("Dynamic exogeneity projection requires at least two views")
        if refresh_interval <= 0 or block_tokens <= 0 or block_channels <= 0:
            raise ValueError("Projection refresh and block dimensions must be positive")
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.num_views = num_views
        self.seed = seed
        self.refresh_interval = refresh_interval
        self.block_tokens = block_tokens
        self.block_channels = block_channels

    def _seed(self, step: int, view: int, *, validation: bool) -> int:
        refresh = step // self.refresh_interval
        stream = 1_000_000_007 if validation else 0
        return self.seed + stream + refresh * 10_007 + view * 1_009

    def projection_matrix(
        self,
        step: int,
        view: int,
        *,
        device: torch.device,
        validation: bool = False,
    ) -> Tensor:
        if view < 0 or view >= self.num_views:
            raise ValueError(f"Projection view must be in [0, {self.num_views})")
        generator = torch.Generator(device=device).manual_seed(
            self._seed(step, view, validation=validation)
        )
        # Generate directly as float: bool/int temporaries otherwise increase peak memory.
        directions = torch.empty(
            self.input_dim, self.projection_dim, device=device, dtype=torch.float32
        )
        directions.bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        return directions.mul_(self.input_dim**-0.5)

    def _random_block(self, noise: Tensor, step: int, *, validation: bool) -> Tensor:
        if noise.ndim != 3:
            raise ValueError("Dynamic noise views require [batch, tokens, channels]")
        generator = torch.Generator(device=noise.device).manual_seed(
            self._seed(step, self.num_views, validation=validation)
        )
        token_count = min(self.block_tokens, noise.shape[1])
        channel_count = min(self.block_channels, noise.shape[2])
        tokens = torch.randperm(noise.shape[1], generator=generator, device=noise.device)[
            :token_count
        ]
        channels = torch.randperm(noise.shape[2], generator=generator, device=noise.device)[
            :channel_count
        ]
        return noise.index_select(1, tokens).index_select(2, channels).flatten(1).float()

    def forward(self, noise: Tensor, *, step: int, validation: bool = False) -> tuple[Tensor, ...]:
        flattened = noise.flatten(1).float()
        if flattened.shape[1] != self.input_dim:
            raise ValueError(
                f"Noise has flattened dimension {flattened.shape[1]}, expected {self.input_dim}"
            )
        projections = tuple(
            flattened
            @ self.projection_matrix(step, view, device=noise.device, validation=validation)
            for view in range(self.num_views)
        )
        # Per-token summaries expose structure that a random flattened view can miss.
        token_mean = noise.float().mean(dim=-1) * noise.shape[-1] ** 0.5
        token_std = noise.float().std(dim=-1, unbiased=False)
        summaries = torch.cat([token_mean, token_std], dim=1)
        return (*projections, summaries, self._random_block(noise, step, validation=validation))


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
