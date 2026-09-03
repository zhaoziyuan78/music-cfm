"""Unpaired training datasets backed by processed manifests."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import miditoolkit
import pandas as pd
import torch
from torch.utils.data import Dataset

from cfmusic.data.midi_io import load_midi
from cfmusic.tokenization.factory import MidiTokenizer


class MidiTokenDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    """Each item contains one factual MIDI segment and observed labels only."""

    forbidden_keys = frozenset(
        {"target", "target_tokens", "target_midi", "paired", "reference_target"}
    )

    def __init__(
        self,
        manifest: Path | pd.DataFrame,
        tokenizer: MidiTokenizer,
        *,
        split: str = "train",
        midi_cache_size: int = 0,
    ) -> None:
        if midi_cache_size < 0:
            raise ValueError("midi_cache_size cannot be negative")
        frame = pd.read_parquet(manifest) if isinstance(manifest, Path) else manifest.copy()
        tokenizer_type = getattr(tokenizer.vocabulary, "scheme", "bar_event")
        if tokenizer_type == "beat" and not {
            "tokenizer_type",
            "tokenizer_version",
        }.issubset(frame.columns):
            raise ValueError(
                "This manifest predates BEAT tokenization; rerun cfmusic.commands.prepare"
            )
        if "tokenizer_type" in frame:
            manifest_types = set(frame["tokenizer_type"].dropna().astype(str))
            if manifest_types and manifest_types != {tokenizer_type}:
                raise ValueError(
                    f"Manifest tokenizer {sorted(manifest_types)} does not match {tokenizer_type!r}"
                )
        expected_version = getattr(tokenizer.config, "implementation_version", None)
        if expected_version is not None and "tokenizer_version" in frame:
            manifest_versions = set(frame["tokenizer_version"].dropna().astype(str))
            if manifest_versions and manifest_versions != {expected_version}:
                raise ValueError(
                    f"Manifest tokenizer version {sorted(manifest_versions)} does not match "
                    f"{expected_version!r}"
                )
        self.frame = frame.loc[(frame["split"] == split) & frame["valid"]].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.midi_cache_size = midi_cache_size
        self._midi_cache: OrderedDict[Path, miditoolkit.MidiFile] = OrderedDict()
        self._verified_sources: set[Path] = set()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.frame.iloc[index]
        midi_path = Path(str(row["source_midi_path"]))
        midi = self._midi_cache.get(midi_path)
        if midi is None:
            try:
                midi = load_midi(midi_path)
            except Exception as error:
                raise RuntimeError(
                    f"Cannot load source MIDI {midi_path} for segment "
                    f"{row.get('segment_id', index)!r}: {error}"
                ) from error
            if midi_path not in self._verified_sources and pd.notna(row.get("num_notes")):
                expected_notes = int(row["num_notes"])
                actual_notes = sum(len(instrument.notes) for instrument in midi.instruments)
                if actual_notes != expected_notes:
                    raise RuntimeError(
                        f"Source MIDI changed after manifest creation: {midi_path} has "
                        f"{actual_notes} notes, expected {expected_notes}. Restore the raw "
                        "dataset or rerun preprocessing before training/caching."
                    )
                self._verified_sources.add(midi_path)
            if self.midi_cache_size:
                self._midi_cache[midi_path] = midi
                self._midi_cache.move_to_end(midi_path)
                while len(self._midi_cache) > self.midi_cache_size:
                    self._midi_cache.popitem(last=False)
        elif self.midi_cache_size:
            self._midi_cache.move_to_end(midi_path)
        tokens = self.tokenizer.encode(
            midi, start_bar=int(row["start_bar"]), num_bars=int(row["num_bars"])
        )
        item: dict[str, torch.Tensor | str | int] = {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "style_id": int(row["style_id"]),
            "dataset_id": int(row["dataset_id"]),
            "sample_id": str(row["sample_id"]),
            "segment_id": str(row["segment_id"]),
            "num_bars": int(row["num_bars"]),
        }
        for key in ("genre_id", "emotion_id"):
            if key in row and pd.notna(row[key]):
                item[key] = int(row[key])
        if self.forbidden_keys.intersection(item):
            raise RuntimeError("Training item violates the no-pair contract")
        return item
