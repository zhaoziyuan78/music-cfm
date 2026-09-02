"""Base dataset adapter contract."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from cfmusic.data.schema import RawMidiRecord, ValidationResult


class DatasetAdapter(Protocol):
    def discover(self) -> Iterable[RawMidiRecord]: ...

    def validate_record(self, record: RawMidiRecord) -> ValidationResult: ...

    def style_vocabulary(self) -> list[str]: ...


def iter_dataset_files(root: Path) -> Iterable[Path]:
    """Traverse a dataset tree once, ignoring packaging/version-control artifacts."""

    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = sorted(
            name for name in subdirectories if name not in {".git", "__MACOSX"}
        )
        for filename in sorted(filenames):
            if filename.startswith("._"):
                continue
            yield Path(directory) / filename


def iter_midi_files(root: Path) -> Iterable[Path]:
    return (path for path in iter_dataset_files(root) if path.suffix.lower() in {".mid", ".midi"})


def validate_source(record: RawMidiRecord) -> ValidationResult:
    if not record.source_path.is_file():
        return ValidationResult(False, "missing_source_file")
    if record.source_path.suffix.lower() not in {".mid", ".midi"}:
        return ValidationResult(False, "not_a_midi_file")
    return ValidationResult(True)
