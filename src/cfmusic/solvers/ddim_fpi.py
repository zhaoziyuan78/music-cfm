"""Fixed-point DDIM inversion update."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FixedPointResult:
    state: Tensor
    residuals: tuple[float, ...]
    nfe: int


def fixed_point_inversion_step(
    lower_state: Tensor,
    initial_upper_state: Tensor,
    predict_epsilon_upper: Callable[[Tensor], Tensor],
    *,
    alpha_lower: Tensor | float,
    alpha_upper: Tensor | float,
    iterations: int = 3,
    tolerance: float = 1e-5,
    stop_on_convergence: bool = True,
) -> FixedPointResult:
    lower_alpha = torch.as_tensor(alpha_lower, device=lower_state.device, dtype=lower_state.dtype)
    upper_alpha = torch.as_tensor(alpha_upper, device=lower_state.device, dtype=lower_state.dtype)
    state = initial_upper_state
    residuals: list[float] = []
    for _ in range(iterations):
        epsilon = predict_epsilon_upper(state)
        clean = (lower_state - (1 - lower_alpha).sqrt() * epsilon) / lower_alpha.sqrt().clamp_min(
            1e-8
        )
        updated = upper_alpha.sqrt() * clean + (1 - upper_alpha).sqrt() * epsilon
        residual = float((updated - state).detach().float().square().mean().sqrt())
        residuals.append(residual)
        state = updated
        if stop_on_convergence and residual <= tolerance:
            break
    return FixedPointResult(state, tuple(residuals), len(residuals))
