"""EMOPIA annotations adapter with song-grouped records."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from cfmusic.data.adapters.base import iter_dataset_files, validate_source
from cfmusic.data.schema import RawMidiRecord, ValidationResult

EMOTION_MAP = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
_EMOPIA_ID = re.compile(r"^Q(?P<quadrant>[1-4])_(?P<song>.+)_(?P<clip>\d+)$", re.IGNORECASE)


def _quadrant(value: object) -> str:
    text = str(value).strip().upper()
    if text.startswith("Q"):
        text = text[1:]
    try:
        return EMOTION_MAP[int(float(text))]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid EMOPIA 4Q label: {value!r}") from error


def _song_id(item_id: str) -> str:
    match = _EMOPIA_ID.fullmatch(item_id)
    return match.group("song") if match else item_id.rsplit("_", 1)[0]


class EMOPIAAdapter:
    def __init__(self, root: Path) -> None:
        self.root = root

    def discover(self) -> Iterable[RawMidiRecord]:
        files = list(iter_dataset_files(self.root))
        by_stem = {path.stem: path for path in files if path.suffix.lower() in {".mid", ".midi"}}
        label_files = [path for path in files if path.name.casefold() == "label.csv"]
        output: list[RawMidiRecord] = []
        if label_files:
            label_path = min(label_files, key=lambda path: len(path.parts))
            frame = pd.read_csv(label_path)
            required = {"ID", "4Q"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"EMOPIA label.csv missing columns: {sorted(missing)}")
            for annotation in frame.to_dict(orient="records"):
                item_id = str(annotation["ID"])
                path = by_stem.get(item_id, label_path.parent / "midis" / f"{item_id}.mid")
                emotion = _quadrant(annotation["4Q"])
                output.append(
                    RawMidiRecord(
                        "emopia",
                        path,
                        item_id,
                        _song_id(item_id),
                        {"emotion": emotion, "style": emotion},
                        None,
                        dict(annotation),
                    )
                )
            return output

        # Older EMOPIA mirrors may omit label.csv; the official MIDI names still
        # encode both the quadrant and source-song identifier.
        for stem, path in sorted(by_stem.items()):
            match = _EMOPIA_ID.fullmatch(stem)
            if match is None:
                continue
            emotion = EMOTION_MAP[int(match.group("quadrant"))]
            output.append(
                RawMidiRecord(
                    "emopia",
                    path,
                    stem,
                    match.group("song"),
                    {"emotion": emotion, "style": emotion},
                    None,
                )
            )
        return output

    def validate_record(self, record: RawMidiRecord) -> ValidationResult:
        return validate_source(record)

    def style_vocabulary(self) -> list[str]:
        return ["Q1", "Q2", "Q3", "Q4"]
