"""Adversarial style-prediction loss for the optional GRL ablation."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn

from cfmusic.models.gradient_reversal import gradient_reverse


def adversarial_style_loss(
    projected_noise: Tensor, style_id: Tensor, probe: nn.Module, *, reversal_weight: float = 1.0
) -> Tensor:
    return functional.cross_entropy(
        probe(gradient_reverse(projected_noise, reversal_weight)), style_id
    )
