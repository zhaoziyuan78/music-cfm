import pandas as pd

from cfmusic.commands.relabel_xmidi_segments_clamp2 import _representative_segments


def test_representative_relabel_uses_middle_existing_cached_segment_per_song() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "a", "a", "b", "b"],
            "segment_id": ["a:0", "a:1", "a:2", "b:0", "b:1"],
            "split": ["train"] * 3 + ["test"] * 2,
            "segment_index": [0, 1, 2, 0, 1],
            "start_bar": [0, 4, 8, 0, 4],
            "num_bars": [8] * 5,
            "source_midi_path": ["a.mid"] * 3 + ["b.mid"] * 2,
            "genre_id": [0, 0, 0, 1, 1],
            "genre_label": ["classical"] * 3 + ["rock"] * 2,
        }
    )
    index = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b"],
            "segment_id": ["a:0", "a:2", "b:0", "b:1"],
            "split": ["train", "train", "test", "test"],
            "shard": ["0.pt"] * 4,
            "offset": range(4),
            "style_id": [0, 0, 1, 1],
            "genre_id": [0, 0, 1, 1],
            "dataset_id": [0] * 4,
        }
    )

    selected = _representative_segments(manifest, index)

    assert selected["segment_id"].tolist() == ["a:0", "b:0"]
    assert selected["sample_id"].is_unique
