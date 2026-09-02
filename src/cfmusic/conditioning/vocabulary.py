"""Namespaced condition vocabulary persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConditionVocabulary:
    namespace: str
    labels: tuple[str, ...]

    def encode(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError as error:
            raise KeyError(f"Unknown label {label!r} in namespace {self.namespace!r}") from error

    def decode(self, index: int) -> str:
        return self.labels[index]

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"namespace": self.namespace, "labels": self.labels}, indent=2),
            encoding="utf-8",
        )
