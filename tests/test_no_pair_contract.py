from pathlib import Path

import pandas as pd
import pytest

import cfmusic.data.datasets as datasets_module
from cfmusic.data.datasets import MidiTokenDataset
from cfmusic.tokenization.bar_event import BarEventTokenizer


def test_training_items_have_no_pair_fields(tiny_midi_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "source_midi_path": str(tiny_midi_path),
                "split": "train",
                "valid": True,
                "start_bar": 0,
                "num_bars": 2,
                "style_id": 0,
                "dataset_id": 0,
                "sample_id": "one",
                "segment_id": "one:0",
            }
        ]
    )
    item = MidiTokenDataset(frame, BarEventTokenizer())[0]
    assert MidiTokenDataset.forbidden_keys.isdisjoint(item)
    assert "tokens" in item and "style_id" in item


def test_midi_dataset_lru_avoids_reparsing_adjacent_segments(
    tiny_midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(
        [
            {
                "source_midi_path": str(tiny_midi_path),
                "split": "train",
                "valid": True,
                "start_bar": index,
                "num_bars": 1,
                "style_id": 0,
                "dataset_id": 0,
                "sample_id": f"sample-{index}",
                "segment_id": f"sample-{index}:0",
            }
            for index in range(2)
        ]
    )
    original = datasets_module.load_midi
    calls = 0

    def counted_load(path: Path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(datasets_module, "load_midi", counted_load)
    dataset = MidiTokenDataset(frame, BarEventTokenizer(), midi_cache_size=1)
    dataset[0]
    dataset[1]
    assert calls == 1
