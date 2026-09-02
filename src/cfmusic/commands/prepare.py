"""Audit, group-split, segment, tokenize, and index one raw MIDI dataset."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from cfmusic.config import CONFIG_DIR, prepare_config
from cfmusic.data.adapters.base import DatasetAdapter
from cfmusic.data.adapters.emopia import EMOPIAAdapter
from cfmusic.data.adapters.groove import GrooveAdapter
from cfmusic.data.adapters.vgmidi import VGMIDIAdapter
from cfmusic.data.adapters.xmidi import XMIDIAdapter
from cfmusic.data.deduplicate import duplicate_clusters
from cfmusic.data.manifests import write_manifest_batches
from cfmusic.data.midi_io import load_and_audit_midi
from cfmusic.data.schema import MidiStatistics, RawMidiRecord
from cfmusic.data.segmentation import enumerate_segments, segment_note_counts
from cfmusic.data.splitting import assert_disjoint_groups, grouped_stratified_split
from cfmusic.progress import track
from cfmusic.tokenization.beat import BeatTokenizer, BeatTokenizerConfig


def _adapter(name: str, root: Path, cfg: DictConfig) -> DatasetAdapter:
    if name == "xmidi":
        return XMIDIAdapter(root, str(cfg.task))
    if name == "emopia":
        return EMOPIAAdapter(root)
    if name == "vgmidi":
        return VGMIDIAdapter(root)
    if name == "groove":
        selection = cfg.style_selection
        return GrooveAdapter(root, int(selection.top_k), int(selection.min_train_files))
    raise ValueError(f"Preparation requires a single concrete dataset, got {name!r}")


def _raw_root(raw_dir: Path, name: str) -> Path:
    base = raw_dir / name
    for child in (base / "extracted", base / "repository"):
        if child.exists():
            return child
    return base


@dataclass(frozen=True, slots=True)
class _PreparedSegment:
    index: int
    start_bar: int
    num_bars: int
    token_count: int
    raw_token_count: int


@dataclass(frozen=True, slots=True)
class _PreparedMidi:
    statistics: MidiStatistics | None
    segments: tuple[_PreparedSegment, ...]
    invalid_reason: str | None = None


def _audit_and_segment(
    record: RawMidiRecord,
    *,
    allowed_time_signatures: tuple[str, ...],
    num_bars: int,
    hop_bars: int,
    drop_incomplete_final_segment: bool,
    minimum_notes: int,
    maximum_notes: int,
    max_sequence_length: int,
    tokenizer_type: str = "bar_event",
    tokenizer_version: str = "cfmusic-beat-v1",
    tokenizer_steps_per_beat: int = 4,
    tokenizer_preserve_velocity: bool = True,
) -> _PreparedMidi:
    """Parse one MIDI once and return all lightweight preprocessing results."""

    if record.source_path.suffix.lower() not in {".mid", ".midi"}:
        return _PreparedMidi(None, (), "not_a_midi_file")
    try:
        midi, statistics = load_and_audit_midi(record.source_path)
        prepared: list[_PreparedSegment] = []
        if statistics.time_signature in allowed_time_signatures:
            segments = enumerate_segments(
                midi,
                num_bars=num_bars,
                hop_bars=hop_bars,
                drop_incomplete_final_segment=drop_incomplete_final_segment,
            )
            beat_lengths: list[int] | None = None
            if tokenizer_type == "beat":
                beat_tokenizer = BeatTokenizer(
                    BeatTokenizerConfig(
                        implementation_version=tokenizer_version,
                        steps_per_beat=tokenizer_steps_per_beat,
                        max_sequence_length=max_sequence_length,
                        preserve_velocity=tokenizer_preserve_velocity,
                    )
                )
                beat_lengths = beat_tokenizer.encoded_segment_lengths(
                    midi,
                    [(segment.start_bar, segment.num_bars) for segment in segments],
                )
            for segment, note_count in zip(
                segments, segment_note_counts(midi, segments), strict=True
            ):
                if minimum_notes <= note_count <= maximum_notes:
                    raw_token_count = (
                        beat_lengths[segment.index]
                        if beat_lengths is not None
                        else 2 + 3 * segment.num_bars + 5 * note_count
                    )
                    prepared.append(
                        _PreparedSegment(
                            segment.index,
                            segment.start_bar,
                            segment.num_bars,
                            min(raw_token_count, max_sequence_length),
                            raw_token_count,
                        )
                    )
        return _PreparedMidi(statistics, tuple(prepared))
    except FileNotFoundError:
        return _PreparedMidi(None, (), "missing_source_file")
    except (OSError, ValueError, EOFError, KeyError, IndexError) as error:
        return _PreparedMidi(None, (), f"{type(error).__name__}: {error}")


def _available_preprocessing_workers(configured: int) -> int:
    if configured < 0:
        raise ValueError("preprocessing.workers must be non-negative")
    if configured:
        return configured
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(8, available))


def prepare_dataset(cfg: DictConfig) -> Path:
    paths = prepare_config(cfg)
    name = str(cfg.data.name)
    root = _raw_root(paths["raw_dir"], name)
    adapter = _adapter(name, root, cfg.data)
    records = list(adapter.discover())
    if not records:
        raise FileNotFoundError(f"No raw {name} MIDI records discovered under {root}")
    processed_dir = paths["processed_dir"] / name
    processed_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(adapter, GrooveAdapter):
        adapter.save_selected_styles(processed_dir / "selected_styles.json")
    segment_cfg = cfg.data.segmentation
    preprocessing_cfg = cfg.get("preprocessing", {})
    workers = _available_preprocessing_workers(int(preprocessing_cfg.get("workers", 0)))
    worker_chunksize = max(1, int(preprocessing_cfg.get("worker_chunksize", 32)))
    worker = partial(
        _audit_and_segment,
        allowed_time_signatures=tuple(str(value) for value in segment_cfg.allowed_time_signatures),
        num_bars=int(segment_cfg.num_bars),
        hop_bars=int(segment_cfg.hop_bars),
        drop_incomplete_final_segment=bool(segment_cfg.drop_incomplete_final_segment),
        minimum_notes=int(segment_cfg.minimum_notes),
        maximum_notes=int(segment_cfg.maximum_notes),
        max_sequence_length=int(cfg.tokenizer.max_sequence_length),
        tokenizer_type=str(cfg.tokenizer.get("type", "bar_event")),
        tokenizer_version=str(cfg.tokenizer.get("implementation_version", "cfmusic-beat-v1")),
        tokenizer_steps_per_beat=int(cfg.tokenizer.get("steps_per_beat", 4)),
        tokenizer_preserve_velocity=bool(cfg.tokenizer.get("preserve_velocity", True)),
    )
    valid_records: list[RawMidiRecord] = []
    statistics: dict[str, MidiStatistics] = {}
    prepared_segments: dict[str, tuple[_PreparedSegment, ...]] = {}
    invalid: list[dict[str, object]] = []
    prepared_segment_count = 0
    executor: ProcessPoolExecutor | None
    prepared_results: Iterator[_PreparedMidi]
    if workers == 1:
        prepared_results = map(worker, records)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        prepared_results = executor.map(worker, records, chunksize=worker_chunksize)
    audit_progress = track(
        zip(records, prepared_results, strict=True),
        description=f"Audit/segment {name} ({workers} workers)",
        total=len(records),
        unit="file",
    )
    try:
        for record, prepared in audit_progress:
            if prepared.statistics is None:
                invalid.append(
                    {
                        "source_midi_path": str(record.source_path),
                        "invalid_reason": prepared.invalid_reason,
                    }
                )
            else:
                statistics[record.item_id] = prepared.statistics
                prepared_segments[record.item_id] = prepared.segments
                prepared_segment_count += len(prepared.segments)
                valid_records.append(record)
            audit_progress.set_postfix(
                valid=len(valid_records),
                invalid=len(invalid),
                segments=prepared_segment_count,
                refresh=False,
            )
    finally:
        if executor is not None:
            executor.shutdown()
    invalid_frame = pd.DataFrame(invalid, columns=["source_midi_path", "invalid_reason"])
    invalid_frame.to_parquet(processed_dir / "invalid_files.parquet", index=False)
    if not valid_records:
        raise RuntimeError(f"All {name} files failed MIDI validation")
    cluster_map = duplicate_clusters(
        (record.item_id, statistics[record.item_id].canonical_hash) for record in valid_records
    )
    group_ids = [
        cluster_map[record.item_id]
        if cluster_map[record.item_id].startswith("duplicate:")
        else record.group_id
        for record in valid_records
    ]
    labels = [str(record.labels["style"]) for record in valid_records]
    if all(record.official_split is not None for record in valid_records):
        split_map = {"valid": "validation", "val": "validation", "dev": "validation"}
        splits = [
            split_map.get(str(record.official_split), str(record.official_split))
            for record in valid_records
        ]
    else:
        split_cfg = cfg.data.split
        splits = grouped_stratified_split(
            group_ids,
            labels,
            fractions=(float(split_cfg.train), float(split_cfg.validation), float(split_cfg.test)),
            seed=int(split_cfg.seed),
        )
    assert_disjoint_groups(group_ids, splits)
    vocabulary = adapter.style_vocabulary()
    style_to_id = {label: index for index, label in enumerate(vocabulary)}
    genre_vocabulary = sorted(
        {str(record.labels["genre"]) for record in valid_records if "genre" in record.labels}
    )
    emotion_vocabulary = sorted(
        {str(record.labels["emotion"]) for record in valid_records if "emotion" in record.labels}
    )
    genre_to_id = {label: index for index, label in enumerate(genre_vocabulary)}
    emotion_to_id = {label: index for index, label in enumerate(emotion_vocabulary)}
    label_counts: Counter[tuple[str, str]] = Counter()
    split_counts: Counter[str] = Counter()
    segment_count = 0

    def manifest_rows() -> Iterator[dict[str, object]]:
        nonlocal segment_count
        segment_progress = track(
            zip(valid_records, group_ids, splits, strict=True),
            description=f"Build {name} manifest",
            total=len(valid_records),
            unit="file",
        )
        for record, group_id, split in segment_progress:
            stats = statistics[record.item_id]
            source_path = (
                record.source_path
                if record.source_path.is_absolute()
                else record.source_path.absolute()
            )
            style = str(record.labels["style"])
            genre_label = str(record.labels["genre"]) if "genre" in record.labels else None
            emotion_label = str(record.labels["emotion"]) if "emotion" in record.labels else None
            common_row: dict[str, object] = {
                "sample_id": record.item_id,
                "dataset": name,
                "dataset_id": 0,
                "source_midi_path": str(source_path),
                "relative_midi_path": str(record.source_path.relative_to(root)),
                "group_id": group_id,
                "original_split": record.official_split,
                "split": split,
                "style_namespace": str(cfg.data.style_namespace),
                "style_label": style,
                "style_id": style_to_id[style],
                "genre_label": genre_label,
                "genre_id": genre_to_id.get(genre_label) if genre_label is not None else None,
                "emotion_label": emotion_label,
                "emotion_id": emotion_to_id.get(emotion_label)
                if emotion_label is not None
                else None,
                "valence": float(record.labels["valence"]) if "valence" in record.labels else None,
                "arousal": float(record.labels["arousal"]) if "arousal" in record.labels else None,
                "drummer": str(record.metadata["drummer"])
                if "drummer" in record.metadata
                else None,
                "canonical_hash": stats.canonical_hash,
                "exact_file_sha256": stats.exact_file_sha256,
                "num_notes": stats.num_notes,
                "num_tracks": stats.num_tracks,
                "duration_seconds": stats.duration_seconds,
                "time_signature": stats.time_signature,
                "tempo_mean": stats.tempo_mean,
                "is_drum": stats.is_drum,
                "valid": True,
                "invalid_reason": None,
            }
            for segment in prepared_segments.pop(record.item_id, ()):
                segment_id = f"{record.item_id}:bar{segment.start_bar}:n{segment.num_bars}"
                segment_count += 1
                label_counts[(split, style)] += 1
                split_counts[split] += 1
                yield {
                    **common_row,
                    "segment_id": segment_id,
                    "segment_index": segment.index,
                    "start_bar": segment.start_bar,
                    "num_bars": segment.num_bars,
                    "token_count": segment.token_count,
                    "raw_token_count": segment.raw_token_count,
                    "tokenizer_type": str(cfg.tokenizer.get("type", "bar_event")),
                    "tokenizer_version": str(
                        cfg.tokenizer.get("implementation_version", "legacy-bar-event-v1")
                    ),
                }
            segment_progress.set_postfix(segments=segment_count, refresh=False)

    manifest = processed_dir / "manifest.parquet"
    try:
        write_manifest_batches(manifest_rows(), manifest)
    except ValueError as error:
        if str(error) == "Cannot write an empty manifest":
            raise RuntimeError(f"No {name} segments passed segmentation constraints") from error
        raise
    pd.DataFrame(
        [
            {"split": split, "style_label": style, "count": count}
            for (split, style), count in sorted(label_counts.items())
        ]
    ).to_csv(processed_dir / "label_counts.csv", index=False)
    card = {
        "dataset": name,
        "raw_records": len(records),
        "valid_files": len(valid_records),
        "invalid_files": len(invalid),
        "segments": segment_count,
        "style_vocabulary": vocabulary,
        "genre_vocabulary": genre_vocabulary,
        "emotion_vocabulary": emotion_vocabulary,
        "splits": split_counts,
        "preprocessing_workers": workers,
        "single_pass_midi_io": True,
        "tokenizer_type": str(cfg.tokenizer.get("type", "bar_event")),
        "tokenizer_version": str(
            cfg.tokenizer.get("implementation_version", "legacy-bar-event-v1")
        ),
        "tokenizer_source": cfg.tokenizer.get("source"),
        "tokenizer_source_commit": cfg.tokenizer.get("source_commit"),
        "tokenizer_vocabulary_size": 593
        if str(cfg.tokenizer.get("type", "bar_event")) == "beat"
        else None,
        "deduplication_hashes": ["exact_file_sha256", "canonical_event_sha256"],
        "no_pair_training_contract": True,
    }
    (processed_dir / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    report = (
        f"# {name} audit\n\nValid MIDI files: {len(valid_records)}\n\n"
        f"Invalid MIDI files: {len(invalid)}\n\nSegments: {segment_count}\n"
    )
    (processed_dir / "audit_report.md").write_text(report, encoding="utf-8")
    return manifest


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Prepared manifest: {prepare_dataset(cfg)}")


if __name__ == "__main__":
    main()
