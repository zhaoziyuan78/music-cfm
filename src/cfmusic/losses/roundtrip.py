"""Latent round-trip reconstruction loss."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor


def roundtrip_loss(reconstructed: Tensor, original: Tensor, cosine_weight: float = 0.1) -> Tensor:
    absolute = (reconstructed - original).abs().mean()
    cosine = 1 - functional.cosine_similarity(reconstructed.flatten(1), original.flatten(1)).mean()
    return absolute + cosine_weight * cosine
