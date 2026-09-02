"""Warmup-cosine learning-rate scheduling."""

from __future__ import annotations

import math

import torch


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, max_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
