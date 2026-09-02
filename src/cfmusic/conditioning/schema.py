"""Typed conditional labels passed to shared transport models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConditionBatch:
    dataset_id: Tensor
    task_id: Tensor
    style_id: Tensor
    genre_id: Tensor | None = None
    emotion_id: Tensor | None = None

    def to(self, device: str | torch.device | Tensor) -> ConditionBatch:
        target = device.device if isinstance(device, Tensor) else device
        return ConditionBatch(
            self.dataset_id.to(target),
            self.task_id.to(target),
            self.style_id.to(target),
            self.genre_id.to(target) if self.genre_id is not None else None,
            self.emotion_id.to(target) if self.emotion_id is not None else None,
        )

    def index_select(self, indices: Tensor) -> ConditionBatch:
        return ConditionBatch(
            self.dataset_id.index_select(0, indices),
            self.task_id.index_select(0, indices),
            self.style_id.index_select(0, indices),
            self.genre_id.index_select(0, indices) if self.genre_id is not None else None,
            self.emotion_id.index_select(0, indices) if self.emotion_id is not None else None,
        )

    @property
    def batch_size(self) -> int:
        return self.style_id.shape[0]
