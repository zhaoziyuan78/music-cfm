"""Versioned sharded latent cache writer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from cfmusic.latent.normalization import (
    NORMALIZATION_SCHEMA_VERSION,
    StreamingLatentStatistics,
    save_statistics,
)
from cfmusic.progress import progress_bar


def stable_json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_latent_cache(
    samples: Iterable[dict[str, object]],
    destination: Path,
    *,
    samples_per_shard: int = 4096,
    dtype: torch.dtype = torch.float16,
    metadata: dict[str, str],
    verify_after_write: bool = True,
    finalize: bool = True,
) -> Path:
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    buffer: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    statistics = StreamingLatentStatistics()
    shard_number = 0
    progress = progress_bar(description="Write latent cache", total=None, unit="sample", position=1)

    def flush() -> None:
        nonlocal shard_number
        if not buffer:
            return
        shard_name = f"shard-{shard_number:05d}.pt"
        latents = torch.stack(
            [item["latent"] for item in buffer if isinstance(item["latent"], Tensor)]
        ).to(dtype)
        payload = {
            "latent": latents,
            "sample_id": [str(item["sample_id"]) for item in buffer],
            "style_id": torch.tensor([int(str(item["style_id"])) for item in buffer]),
            "dataset_id": torch.tensor([int(str(item["dataset_id"])) for item in buffer]),
            "split": [str(item["split"]) for item in buffer],
            "metadata": metadata,
        }
        path = destination / shard_name
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        if verify_after_write:
            restored = torch.load(path, map_location="cpu", weights_only=False)
            if restored["latent"].shape != latents.shape or restored["metadata"] != metadata:
                raise OSError(f"Latent shard verification failed: {path}")
        train_offsets = [
            offset for offset, item in enumerate(buffer) if str(item["split"]) == "train"
        ]
        if train_offsets:
            statistics.update(latents[torch.tensor(train_offsets, dtype=torch.long)])
        for offset, item in enumerate(buffer):
            index_rows.append(
                {
                    "sample_id": str(item["sample_id"]),
                    "segment_id": str(item.get("segment_id", item["sample_id"])),
                    "shard": shard_name,
                    "offset": offset,
                    "style_id": int(str(item["style_id"])),
                    "dataset_id": int(str(item["dataset_id"])),
                    "genre_id": item.get("genre_id"),
                    "emotion_id": item.get("emotion_id"),
                    "split": str(item["split"]),
                }
            )
        buffer.clear()
        shard_number += 1

    try:
        for sample in samples:
            if not isinstance(sample.get("latent"), Tensor):
                raise TypeError("Every cache sample requires a Tensor latent")
            buffer.append(sample)
            progress.update(1)
            progress.set_postfix(shards=shard_number, refresh=False)
            if len(buffer) >= samples_per_shard:
                flush()
        flush()
    finally:
        progress.close()
    if not index_rows:
        raise ValueError("Cannot write an empty latent cache")
    index_path = destination / "index.parquet"
    pd.DataFrame(index_rows).to_parquet(index_path, index=False)
    if finalize:
        normalization_hash = save_statistics(statistics.finalize(), destination)
        _write_cache_metadata(destination, metadata, normalization_hash)
    else:
        torch.save(statistics.state_dict(), destination / "partial_latent_stats.pt")
    return index_path


def _write_cache_metadata(
    destination: Path, metadata: dict[str, str], normalization_hash: str
) -> None:
    cache_metadata = dict(metadata)
    cache_metadata["normalization_hash"] = normalization_hash
    cache_metadata["normalization_schema_version"] = NORMALIZATION_SCHEMA_VERSION
    (destination / "cache_metadata.json").write_text(
        json.dumps(cache_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def finalize_latent_cache_partitions(
    destination: Path,
    partition_names: Iterable[str],
    *,
    metadata: dict[str, str],
) -> Path:
    """Merge independently written rank partitions and compute streaming train statistics."""

    names = tuple(partition_names)
    frames: list[pd.DataFrame] = []
    for partition_name in names:
        partition = destination / partition_name
        frame = pd.read_parquet(partition / "index.parquet")
        frame["shard"] = f"{partition_name}/" + frame["shard"].astype(str)
        frames.append(frame)
    if not frames:
        raise ValueError("Cannot finalize an empty set of latent cache partitions")
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("Cannot finalize an empty latent cache")
    index_path = destination / "index.parquet"
    combined.to_parquet(index_path, index=False)

    statistics = StreamingLatentStatistics()
    for partition_name in names:
        partial_path = destination / partition_name / "partial_latent_stats.pt"
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if not isinstance(partial, dict):
            raise TypeError(f"Invalid partial latent statistics: {partial_path}")
        statistics.merge_state_dict(partial)
        partial_path.unlink()
    normalization_hash = save_statistics(statistics.finalize(), destination)
    _write_cache_metadata(destination, metadata, normalization_hash)
    return index_path
