from __future__ import annotations

import hashlib
from pathlib import Path

from cfmusic.commands.prepare import _audit_and_segment
from cfmusic.data.manifests import read_manifest, write_manifest_batches
from cfmusic.data.midi_io import load_and_audit_midi
from cfmusic.data.schema import ManifestRow, RawMidiRecord
from cfmusic.data.segmentation import enumerate_segments, segment_note_counts
from cfmusic.tokenization.bar_event import BarEventTokenizer
from cfmusic.tokenization.beat import BeatTokenizer


def test_single_read_audit_and_fast_segment_counts(tiny_midi_path: Path, monkeypatch) -> None:
    original_read_bytes = Path.read_bytes
    reads = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal reads
        reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    midi, statistics = load_and_audit_midi(tiny_midi_path)
    assert reads == 1
    assert (
        statistics.exact_file_sha256
        == hashlib.sha256(original_read_bytes(tiny_midi_path)).hexdigest()
    )

    segments = enumerate_segments(midi, num_bars=1, hop_bars=1, drop_incomplete_final_segment=True)
    counts = segment_note_counts(midi, segments)
    tokenizer = BarEventTokenizer()
    assert counts == [4, 4]
    assert [
        tokenizer.encoded_length(num_bars=segment.num_bars, num_notes=count)
        for segment, count in zip(segments, counts, strict=True)
    ] == [
        len(tokenizer.encode(midi, start_bar=segment.start_bar, num_bars=segment.num_bars))
        for segment in segments
    ]


def test_preprocessing_worker_returns_manifest_statistics(tiny_midi_path: Path) -> None:
    record = RawMidiRecord("test", tiny_midi_path, "sample", "group", {"style": "style"}, None)
    prepared = _audit_and_segment(
        record,
        allowed_time_signatures=("4/4",),
        num_bars=1,
        hop_bars=1,
        drop_incomplete_final_segment=True,
        minimum_notes=1,
        maximum_notes=100,
        max_sequence_length=2048,
    )
    assert prepared.invalid_reason is None
    assert prepared.statistics is not None
    assert len(prepared.segments) == 2
    assert [segment.token_count for segment in prepared.segments] == [25, 25]

    beat_prepared = _audit_and_segment(
        record,
        allowed_time_signatures=("4/4",),
        num_bars=1,
        hop_bars=1,
        drop_incomplete_final_segment=True,
        minimum_notes=1,
        maximum_notes=100,
        max_sequence_length=2560,
        tokenizer_type="beat",
    )
    midi, _ = load_and_audit_midi(tiny_midi_path)
    tokenizer = BeatTokenizer()
    assert [segment.raw_token_count for segment in beat_prepared.segments] == [
        len(tokenizer.encode_untruncated(midi, start_bar=index, num_bars=1)) for index in range(2)
    ]


def test_manifest_rows_are_streamed_in_batches(tmp_path: Path) -> None:
    def rows():
        for index in range(3):
            yield ManifestRow(
                sample_id=f"sample-{index}",
                dataset="test",
                dataset_id=0,
                source_midi_path="/tmp/source.mid",
                relative_midi_path="source.mid",
                group_id=f"group-{index}",
                original_split=None,
                split="train",
                style_namespace="test.style",
                style_label="style",
                style_id=0,
                genre_label=None,
                genre_id=None,
                emotion_label=None,
                emotion_id=None,
                valence=None,
                arousal=None,
                drummer=None,
                segment_id=f"segment-{index}",
                segment_index=index,
                start_bar=index,
                num_bars=1,
                canonical_hash="a" * 64,
                exact_file_sha256="b" * 64,
                num_notes=4,
                num_tracks=1,
                duration_seconds=2.0,
                time_signature="4/4",
                tempo_mean=120.0,
                is_drum=False,
                token_count=25,
                valid=True,
                invalid_reason=None,
                raw_token_count=25,
                tokenizer_type="bar_event",
                tokenizer_version="legacy-bar-event-v1",
            ).to_dict()

    path = tmp_path / "manifest.parquet"
    assert write_manifest_batches(rows(), path, batch_size=1) == 3
    frame = read_manifest(path)
    assert frame["sample_id"].tolist() == ["sample-0", "sample-1", "sample-2"]
    assert {"raw_token_count", "tokenizer_type", "tokenizer_version"}.issubset(frame.columns)
    assert frame["raw_token_count"].tolist() == [25, 25, 25]
    assert frame["tokenizer_type"].tolist() == ["bar_event"] * 3
    assert frame["tokenizer_version"].tolist() == ["legacy-bar-event-v1"] * 3
