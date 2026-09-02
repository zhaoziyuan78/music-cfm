"""AdamW and gradient diagnostics."""

from __future__ import annotations

import torch
from torch import nn


def create_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    use_fused = any(parameter.is_cuda for parameter in model.parameters())
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        fused=use_fused,
    )


def parameter_norm(model: nn.Module) -> float:
    total = torch.zeros(())
    for parameter in model.parameters():
        total = total + parameter.detach().float().square().sum().cpu()
    return float(total.sqrt())
