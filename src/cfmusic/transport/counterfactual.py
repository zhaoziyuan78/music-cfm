"""Counterfactual transport output and negative controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cfmusic.conditioning.schema import ConditionBatch


@dataclass
class CounterfactualOutput:
    source_latent: Tensor
    abducted_noise: Tensor
    reconstructed_source_latent: Tensor
    counterfactual_latent: Tensor
    source_condition: ConditionBatch
    target_condition: ConditionBatch
    inverse_nfe: int
    forward_nfe: int


def resampled_noise_like(latent: Tensor) -> Tensor:
    """Negative control that intentionally discards source-specific exogenous noise."""
    return torch.randn_like(latent)
