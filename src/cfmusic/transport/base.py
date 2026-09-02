"""Conditional transport interface."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.transport.counterfactual import CounterfactualOutput


class ConditionalTransport(Protocol):
    def training_loss(
        self,
        latent: Tensor,
        condition: ConditionBatch,
        *,
        negative_condition: ConditionBatch | None = None,
        condition_contrast_weight: float = 0.0,
        condition_contrast_margin: float = 0.0,
        condition_contrast_samples: int | None = None,
        sample_weight: Tensor | None = None,
    ) -> dict[str, Tensor]: ...

    def abduct(
        self, latent: Tensor, condition: ConditionBatch, *, num_steps: int, track_grad: bool = False
    ) -> Tensor: ...

    def predict(
        self, noise: Tensor, condition: ConditionBatch, *, num_steps: int, track_grad: bool = False
    ) -> Tensor: ...

    def counterfactual(
        self,
        latent: Tensor,
        source_condition: ConditionBatch,
        target_condition: ConditionBatch,
        *,
        num_steps: int,
    ) -> CounterfactualOutput: ...
