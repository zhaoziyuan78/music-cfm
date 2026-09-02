"""Parquet manifest persistence and stable hashing."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from itertools import islice
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("dataset", pa.string()),
        ("dataset_id", pa.int64()),
        ("source_midi_path", pa.string()),
        ("relative_midi_path", pa.string()),
        ("group_id", pa.string()),
        ("original_split", pa.string()),
        ("split", pa.string()),
        ("style_namespace", pa.string()),
        ("style_label", pa.string()),
        ("style_id", pa.int64()),
        ("genre_label", pa.string()),
        ("genre_id", pa.int64()),
        ("emotion_label", pa.string()),
        ("emotion_id", pa.int64()),
        ("valence", pa.float64()),
        ("arousal", pa.float64()),
        ("drummer", pa.string()),
        ("segment_id", pa.string()),
        ("segment_index", pa.int64()),
        ("start_bar", pa.int64()),
        ("num_bars", pa.int64()),
        ("canonical_hash", pa.string()),
        ("exact_file_sha256", pa.string()),
        ("num_notes", pa.int64()),
        ("num_tracks", pa.int64()),
        ("duration_seconds", pa.float64()),
        ("time_signature", pa.string()),
        ("tempo_mean", pa.float64()),
        ("is_drum", pa.bool_()),
        ("token_count", pa.int64()),
        ("raw_token_count", pa.int64()),
        ("tokenizer_type", pa.string()),
        ("tokenizer_version", pa.string()),
        ("valid", pa.bool_()),
        ("invalid_reason", pa.string()),
    ]
)


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    write_manifest_batches(rows, path)


def write_manifest_batches(
    rows: Iterable[Mapping[str, object]], path: Path, *, batch_size: int = 50_000
) -> int:
    """Stream manifest rows to Parquet without retaining a second full table in memory."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    iterator = iter(rows)
    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        while batch := list(islice(iterator, batch_size)):
            table = pa.Table.from_pylist([dict(row) for row in batch], schema=MANIFEST_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(temporary, MANIFEST_SCHEMA, compression="snappy")
            writer.write_table(table)
            count += len(batch)
        if writer is None:
            raise ValueError("Cannot write an empty manifest")
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    temporary.replace(path)
    return count


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
