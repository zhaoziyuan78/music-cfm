"""Counterfactual algebra metrics in latent space."""

from __future__ import annotations

import torch
from torch import Tensor


def latent_errors(reference: Tensor, prediction: Tensor) -> dict[str, float]:
    return {
        "mse": float((reference - prediction).square().mean()),
        "mae": float((reference - prediction).abs().mean()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference.flatten(1), prediction.flatten(1)
            ).mean()
        ),
    }


def exogenous_consistency(source_noise: Tensor, counterfactual_noise: Tensor) -> float:
    return float((source_noise - counterfactual_noise).square().mean())
