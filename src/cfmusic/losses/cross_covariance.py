"""Cross-covariance penalty used only by optional split ablations."""

from __future__ import annotations

from torch import Tensor


def cross_covariance_loss(left: Tensor, right: Tensor) -> Tensor:
    left_flat = left.flatten(1) - left.flatten(1).mean(0)
    right_flat = right.flatten(1) - right.flatten(1).mean(0)
    covariance = left_flat.T @ right_flat / max(1, left.shape[0] - 1)
    return covariance.square().mean()
