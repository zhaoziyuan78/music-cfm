"""Typed, versioned conditional labels passed to shared transport models.

There is deliberately only one constructor in this module.  Keeping condition
semantics here prevents training and generation from silently activating a
different set of embedding tables for the same piece of metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

CONDITION_SCHEMA_VERSION = "task-aware-v2"
ConditionTask = Literal["genre", "emotion", "factorial", "style"]
_TASK_IDS: dict[ConditionTask, int] = {
    "genre": 0,
    "emotion": 1,
    "factorial": 2,
    "style": 3,
}


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


def condition_task(task: str, *, factorial: bool = False) -> ConditionTask:
    """Resolve the configured task to one of the schema's explicit modes."""

    normalized = task.strip().lower()
    if factorial:
        return "factorial"
    if normalized not in {"genre", "emotion", "style"}:
        raise ValueError(
            f"Non-factorial condition task must be 'genre', 'emotion', or 'style', got {task!r}"
        )
    return normalized  # type: ignore[return-value]


def _as_long_batch(
    value: Tensor | int | None,
    *,
    name: str,
    device: torch.device,
    batch_size: int | None = None,
) -> Tensor:
    if value is None:
        raise ValueError(f"Condition metadata is missing required field {name!r}")
    tensor = value if isinstance(value, Tensor) else torch.tensor([int(value)])
    tensor = tensor.to(device=device, dtype=torch.long, non_blocking=True).reshape(-1)
    if batch_size is not None and tensor.shape[0] != batch_size:
        if tensor.numel() == 1:
            tensor = tensor.expand(batch_size)
        else:
            raise ValueError(
                f"Condition field {name!r} has batch {tensor.shape[0]}, expected {batch_size}"
            )
    return tensor


def build_condition_batch(
    metadata: Mapping[str, Tensor | str | int],
    device: torch.device,
    *,
    task: str,
    factorial: bool = False,
    active_id: Tensor | int | None = None,
    genre_id: Tensor | int | None = None,
    emotion_id: Tensor | int | None = None,
) -> ConditionBatch:
    """Build the canonical condition for training, generation, and evaluation.

    Non-factorial tasks activate *only* ``style_id``.  Factorial tasks activate
    ``genre_id`` and ``emotion_id`` and use zero as a constant style sentinel.
    Optional overrides are used for counterfactual interventions; inactive
    fields can never leak through from the input metadata.
    """

    mode = condition_task(task, factorial=factorial)
    raw_dataset = metadata.get("dataset_id")
    if not isinstance(raw_dataset, (Tensor, int)):
        raise TypeError("Condition metadata requires tensor or integer dataset_id")
    datasets = _as_long_batch(raw_dataset, name="dataset_id", device=device)
    batch_size = int(datasets.shape[0])
    tasks = torch.full_like(datasets, _TASK_IDS[mode])

    if mode == "factorial":
        raw_genre = genre_id if genre_id is not None else metadata.get("genre_id")
        raw_emotion = emotion_id if emotion_id is not None else metadata.get("emotion_id")
        if not isinstance(raw_genre, (Tensor, int)) or not isinstance(raw_emotion, (Tensor, int)):
            raise ValueError("Factorial conditioning requires both genre_id and emotion_id")
        genres = _as_long_batch(raw_genre, name="genre_id", device=device, batch_size=batch_size)
        emotions = _as_long_batch(
            raw_emotion, name="emotion_id", device=device, batch_size=batch_size
        )
        return ConditionBatch(datasets, tasks, torch.zeros_like(datasets), genres, emotions)

    active_value = active_id
    if active_value is None:
        axis_value = metadata.get(f"{mode}_id")
        active_value = axis_value if isinstance(axis_value, (Tensor, int)) else None
    if active_value is None:
        style_value = metadata.get("style_id")
        active_value = style_value if isinstance(style_value, (Tensor, int)) else None
    styles = _as_long_batch(
        active_value, name=f"{mode}_id/style_id", device=device, batch_size=batch_size
    )
    return ConditionBatch(datasets, tasks, styles, genre_id=None, emotion_id=None)


def condition_schema_provenance(*, task: str, factorial: bool) -> dict[str, str]:
    mode = condition_task(task, factorial=factorial)
    return {
        "condition_schema_version": CONDITION_SCHEMA_VERSION,
        "condition_task": mode,
    }


def validate_condition_checkpoint(
    checkpoint: Mapping[str, object], *, task: str, factorial: bool
) -> None:
    """Reject checkpoints trained with the former contradictory condition schema."""

    provenance = checkpoint.get("provenance")
    recorded_version = checkpoint.get("condition_schema_version")
    recorded_task: object | None = None
    if isinstance(provenance, Mapping):
        recorded_version = recorded_version or provenance.get("condition_schema_version")
        recorded_task = provenance.get("condition_task")
    expected = condition_schema_provenance(task=task, factorial=factorial)
    if recorded_version != CONDITION_SCHEMA_VERSION:
        raise ValueError(
            "Transport checkpoint uses an old or unknown condition schema; retrain Stage 1 "
            "and Stage 2 with task-aware-v2 before generation or resume"
        )
    if recorded_task != expected["condition_task"]:
        raise ValueError(
            f"Transport checkpoint condition task is {recorded_task!r}, but the run requires "
            f"{expected['condition_task']!r}"
        )
