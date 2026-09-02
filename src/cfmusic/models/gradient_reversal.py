"""Gradient-reversal operation for adversarial exogeneity ablations."""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.autograd import Function


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx: Any, value: Tensor, weight: float) -> Tensor:
        ctx.weight = weight
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        return -float(ctx.weight) * gradient, None


def gradient_reverse(value: Tensor, weight: float = 1.0) -> Tensor:
    return _GradientReversal.apply(value, weight)
