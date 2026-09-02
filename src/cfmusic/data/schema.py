"""Stable schemas used by raw adapters and processed manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LabelValue = str | int | float


@dataclass(frozen=True)
class RawMidiRecord:
    dataset: str
    source_path: Path
    item_id: str
    group_id: str
    labels: dict[str, LabelValue]
    official_split: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str | None = None


@dataclass(frozen=True)
class MidiStatistics:
    exact_file_sha256: str
    canonical_hash: str
    num_notes: int
    num_tracks: int
    duration_seconds: float
    time_signature: str
    tempo_mean: float
    is_drum: bool


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    dataset: str
    dataset_id: int
    source_midi_path: str
    relative_midi_path: str
    group_id: str
    original_split: str | None
    split: str
    style_namespace: str
    style_label: str
    style_id: int
    genre_label: str | None
    genre_id: int | None
    emotion_label: str | None
    emotion_id: int | None
    valence: float | None
    arousal: float | None
    drummer: str | None
    segment_id: str
    segment_index: int
    start_bar: int
    num_bars: int
    canonical_hash: str
    exact_file_sha256: str
    num_notes: int
    num_tracks: int
    duration_seconds: float
    time_signature: str
    tempo_mean: float
    is_drum: bool
    token_count: int
    valid: bool
    invalid_reason: str | None
    raw_token_count: int | None = None
    tokenizer_type: str | None = None
    tokenizer_version: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
