"""VGMIDI labelled-phrase CSV adapter."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from cfmusic.data.adapters.base import iter_dataset_files, validate_source
from cfmusic.data.schema import RawMidiRecord, ValidationResult
from cfmusic.download.extraction import safe_extract_zip


def quadrant(valence: int, arousal: int) -> str:
    mapping = {(1, 1): "Q1", (-1, 1): "Q2", (-1, -1): "Q3", (1, -1): "Q4"}
    try:
        return mapping[(valence, arousal)]
    except KeyError as error:
        raise ValueError(f"Invalid VGMIDI valence/arousal pair: {(valence, arousal)}") from error


class VGMIDIAdapter:
    def __init__(self, root: Path) -> None:
        self.root = root

    def discover(self) -> Iterable[RawMidiRecord]:
        phrase_archive = self.root / "labelled" / "phrases.zip"
        phrase_dir = phrase_archive.parent / "phrases"
        if phrase_archive.is_file() and (
            not phrase_dir.is_dir() or not any(phrase_dir.glob("*.mid"))
        ):
            safe_extract_zip(phrase_archive, phrase_dir, member_prefix="phrases")
        files = list(iter_dataset_files(self.root))
        csv_files = [path for path in files if path.name == "vgmidi_labelled.csv"]
        if not csv_files:
            return []
        frame = pd.read_csv(csv_files[0])
        required = {"id", "series", "game", "piece", "midi", "valence", "arousal"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"VGMIDI CSV missing columns: {sorted(missing)}")
        midi_by_name = {
            path.name: path
            for path in files
            if path.suffix.lower() in {".mid", ".midi"} and "phrases" in path.parts
        }
        records: list[RawMidiRecord] = []
        for row in frame.to_dict(orient="records"):
            relative_midi = Path(str(row["midi"]))
            candidate = self.root / relative_midi
            if not candidate.is_file():
                candidate = midi_by_name.get(relative_midi.name, candidate)
            valence, arousal = int(row["valence"]), int(row["arousal"])
            style = quadrant(valence, arousal)
            group = "::".join(str(row[key]) for key in ("series", "game", "piece"))
            records.append(
                RawMidiRecord(
                    "vgmidi",
                    candidate,
                    relative_midi.stem,
                    group,
                    {"emotion": style, "style": style, "valence": valence, "arousal": arousal},
                    None,
                    row,
                )
            )
        return records

    def validate_record(self, record: RawMidiRecord) -> ValidationResult:
        return validate_source(record)

    def style_vocabulary(self) -> list[str]:
        return ["Q1", "Q2", "Q3", "Q4"]
