"""Groove MIDI Dataset official-split adapter."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from cfmusic.data.adapters.base import iter_dataset_files, validate_source
from cfmusic.data.schema import RawMidiRecord, ValidationResult


class GrooveAdapter:
    def __init__(self, root: Path, top_k: int = 8, min_train_files: int = 20) -> None:
        self.root = root
        self.top_k = top_k
        self.min_train_files = min_train_files
        self.selected_styles: list[str] = []

    def discover(self) -> Iterable[RawMidiRecord]:
        csv_files = [path for path in iter_dataset_files(self.root) if path.name == "info.csv"]
        if not csv_files:
            return []
        frame = pd.read_csv(csv_files[0])
        frame["primary_style"] = frame["style"].astype(str).str.split("/").str[0]
        counts = Counter(
            frame.loc[frame["split"].astype(str).str.lower() == "train", "primary_style"]
        )
        eligible = [
            (style, count) for style, count in counts.items() if count >= self.min_train_files
        ]
        self.selected_styles = [
            style for style, _ in sorted(eligible, key=lambda x: (-x[1], x[0]))[: self.top_k]
        ]
        records: list[RawMidiRecord] = []
        for row in frame.to_dict(orient="records"):
            style = str(row["primary_style"])
            if style not in self.selected_styles:
                continue
            path = csv_files[0].parent / str(row["midi_filename"])
            records.append(
                RawMidiRecord(
                    "groove",
                    path,
                    str(row["id"]),
                    str(row.get("id", row["midi_filename"])),
                    {"style": style, "bpm": float(row["bpm"])},
                    str(row["split"]).lower(),
                    row,
                )
            )
        return records

    def save_selected_styles(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.selected_styles, indent=2), encoding="utf-8")

    def validate_record(self, record: RawMidiRecord) -> ValidationResult:
        return validate_source(record)

    def style_vocabulary(self) -> list[str]:
        return list(self.selected_styles)
