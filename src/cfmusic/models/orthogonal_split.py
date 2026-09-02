"""Invertible orthogonal conserved/editable latent rotation ablation."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class OrthogonalLatentSplit(nn.Module):
    def __init__(self, latent_dim: int, editable_fraction: float = 0.5) -> None:
        super().__init__()
        if not 0 < editable_fraction < 1:
            raise ValueError("editable_fraction must lie strictly between zero and one")
        editable_dim = round(latent_dim * editable_fraction)
        self.editable_dim = min(latent_dim - 1, max(1, editable_dim))
        self.conserved_dim = latent_dim - self.editable_dim
        rotation = nn.Linear(latent_dim, latent_dim, bias=False)
        with torch.no_grad():
            rotation.weight.copy_(torch.eye(latent_dim))
        self.rotation = torch.nn.utils.parametrizations.orthogonal(rotation)

    def split(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        rotated = self.rotation(latent)
        return rotated[..., : self.conserved_dim], rotated[..., self.conserved_dim :]

    def merge(self, conserved: Tensor, editable: Tensor) -> Tensor:
        rotated = torch.cat([conserved, editable], dim=-1)
        weight = cast(Tensor, self.rotation.weight)
        return rotated @ weight

    def orthogonality_error(self) -> Tensor:
        weight = cast(Tensor, self.rotation.weight)
        identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
        return (weight.T @ weight - identity).abs().max()
