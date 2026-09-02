"""Training state and exponential moving average."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


@dataclass
class TrainState:
    global_step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0
    world_size: int = 1
    best_validation_loss: float = float("inf")
    best_roundtrip: float = float("inf")


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module, *, steps: int = 1) -> None:
        if steps <= 0:
            raise ValueError("EMA steps must be positive")
        effective_decay = self.decay**steps
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if torch.is_floating_point(shadow):
                shadow.lerp_(value.detach(), 1.0 - effective_decay)
            else:
                shadow.copy_(value)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def to(self, device: torch.device) -> None:
        self.shadow = {name: value.to(device) for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(cast(float, state["decay"]))
        shadow = state["shadow"]
        if not isinstance(shadow, dict):
            raise TypeError("Invalid EMA state")
        self.shadow = shadow
