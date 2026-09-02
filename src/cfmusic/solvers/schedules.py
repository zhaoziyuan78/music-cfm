"""Time and diffusion schedule utilities."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sample_flow_time(batch_size: int, device: torch.device, method: str = "uniform") -> Tensor:
    if method == "uniform":
        return torch.rand(batch_size, device=device)
    if method == "logit_normal":
        return torch.randn(batch_size, device=device).sigmoid()
    raise ValueError(f"Unknown flow time sampling method: {method}")


def cosine_alpha_cumprod(timesteps: int, s: float = 0.008) -> Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    values = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi / 2).square()
    values = values / values[0]
    return values[1:].float().clamp(1e-6, 1.0)
