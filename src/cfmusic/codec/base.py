"""Latent codec contracts and diagonal Gaussian posterior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


@dataclass
class DiagonalGaussian:
    mean: Tensor
    logvar: Tensor

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(
            self.mean.shape,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        return self.mean + torch.exp(0.5 * self.logvar) * noise

    def mode(self) -> Tensor:
        return self.mean


class LatentCodec(Protocol):
    def encode_distribution(self, tokens: Tensor, attention_mask: Tensor) -> DiagonalGaussian: ...

    def encode_mean(self, tokens: Tensor, attention_mask: Tensor) -> Tensor: ...

    def decode_teacher_forced(self, tokens: Tensor, latent: Tensor) -> Tensor: ...

    def generate(
        self,
        latent: Tensor,
        *,
        strategy: str,
        temperature: float,
        top_p: float,
        max_length: int,
        max_bars: int | None = None,
        min_bars: int | None = None,
        use_cache: bool = True,
        show_progress: bool = True,
        progress_description: str = "Decode tokens",
    ) -> Tensor: ...
