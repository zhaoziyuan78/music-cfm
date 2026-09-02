"""XMIDI filename-driven metadata adapter."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from cfmusic.data.adapters.base import iter_midi_files, validate_source
from cfmusic.data.schema import RawMidiRecord, ValidationResult
from cfmusic.progress import track

XMIDI_PATTERN = re.compile(
    r"^XMIDI_(?P<emotion>.+?)_(?P<genre>.+?)_(?P<id>[A-Za-z0-9]{8})\.(?:mid|midi)$",
    re.IGNORECASE,
)


class XMIDIAdapter:
    def __init__(self, root: Path, task: str = "genre") -> None:
        self.root = root
        self.task = task
        self._records: list[RawMidiRecord] | None = None

    def discover(self) -> Iterable[RawMidiRecord]:
        records: list[RawMidiRecord] = []
        for path in track(
            iter_midi_files(self.root),
            description="Discover XMIDI files",
            unit="file",
        ):
            match = XMIDI_PATTERN.match(path.name)
            if match is None:
                continue
            values = match.groupdict()
            records.append(
                RawMidiRecord(
                    dataset="xmidi",
                    source_path=path,
                    item_id=values["id"],
                    group_id=values["id"],
                    labels={
                        "genre": values["genre"],
                        "emotion": values["emotion"],
                        "style": values["genre"] if self.task == "factorial" else values[self.task],
                    },
                    official_split=None,
                )
            )
        genres = sorted({str(row.labels["genre"]) for row in records})
        emotions = sorted({str(row.labels["emotion"]) for row in records})
        if records and (len(genres) != 6 or len(emotions) != 11):
            raise ValueError(
                "XMIDI label scan failed: expected 6 genres and 11 emotions; "
                f"found genres={genres}, emotions={emotions}"
            )
        self._records = records
        return records

    def validate_record(self, record: RawMidiRecord) -> ValidationResult:
        return validate_source(record)

    def style_vocabulary(self) -> list[str]:
        records = self._records if self._records is not None else list(self.discover())
        return sorted({str(row.labels["style"]) for row in records})
