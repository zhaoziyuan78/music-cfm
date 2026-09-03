"""Encode deterministic posterior means into versioned latent shards."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cfmusic.codec.checkpoint import checkpoint_hash
from cfmusic.commands.train_codec import codec_from_config
from cfmusic.config import CONFIG_DIR, prepare_config
from cfmusic.data.collate import collate_token_batch
from cfmusic.data.datasets import MidiTokenDataset
from cfmusic.data.manifests import manifest_hash
from cfmusic.data.samplers import GroupedLengthBatchSampler
from cfmusic.distributed import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
)
from cfmusic.latent.cache import (
    finalize_latent_cache_partitions,
    stable_json_hash,
    write_latent_cache,
)
from cfmusic.memory import autocast_context, peak_memory_gib, reset_peak_memory
from cfmusic.progress import progress_bar, track
from cfmusic.tokenization.factory import tokenizer_from_config


def _partition_bounds(length: int, rank: int, world_size: int) -> tuple[int, int]:
    base, remainder = divmod(length, world_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


def _weighted_partition_bounds(frame: pd.DataFrame, rank: int, world_size: int) -> tuple[int, int]:
    """Split contiguous rows by estimated attention cost instead of row count.

    Codec attention cost grows approximately quadratically with token length. XMIDI is
    ordered enough that equal-sized contiguous partitions can otherwise leave one rank
    processing long sequences for more than ten minutes after its peers have finished.
    """

    if "raw_token_count" not in frame or frame.empty:
        return _partition_bounds(len(frame), rank, world_size)
    lengths = (
        pd.to_numeric(frame["raw_token_count"], errors="coerce")
        .fillna(1)
        .clip(lower=1)
        .to_numpy(dtype=np.float64)
    )
    prefix = np.empty(len(lengths) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(np.square(lengths), out=prefix[1:])
    total = float(prefix[-1])

    def boundary(partition: int) -> int:
        if partition <= 0:
            return 0
        if partition >= world_size:
            return len(frame)
        target = total * partition / world_size
        upper = min(int(np.searchsorted(prefix, target, side="left")), len(frame))
        lower = max(0, upper - 1)
        chosen = lower if target - prefix[lower] <= prefix[upper] - target else upper
        if len(frame) >= world_size:
            chosen = min(max(chosen, partition), len(frame) - (world_size - partition))
        return chosen

    return boundary(rank), boundary(rank + 1)


def _broadcast_build_id(
    context: DistributedContext,
    *,
    staging: Path,
    resume: bool,
) -> str:
    build_id: object = None
    if context.is_main:
        if resume:
            try:
                candidate = (staging / ".build_id").read_text(encoding="utf-8").strip()
            except (FileNotFoundError, OSError):
                candidate = ""
            if len(candidate) == 32 and all(
                character in "0123456789abcdef" for character in candidate
            ):
                build_id = candidate
        if build_id is None:
            build_id = uuid.uuid4().hex
    if context.world_size > 1:
        values = [build_id]
        dist.broadcast_object_list(values, src=0, device=context.device)
        build_id = values[0]
    if not isinstance(build_id, str):
        raise TypeError("Invalid latent-cache build ID broadcast")
    return build_id


def _write_marker(path: Path, build_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(build_id, encoding="utf-8")
    temporary.replace(path)


def _write_heartbeat(
    path: Path,
    build_id: str,
    *,
    rank: int,
    phase: str,
    samples: int,
) -> None:
    payload = {
        "build_id": build_id,
        "rank": rank,
        "phase": phase,
        "samples": samples,
        "updated_at": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_failure_marker(path: Path, build_id: str, *, rank: int, error: BaseException) -> None:
    payload = {
        "build_id": build_id,
        "rank": rank,
        "error": f"{type(error).__name__}: {error}"[:2000],
        "updated_at": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_status(path: Path, build_id: str) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("build_id") != build_id:
        return None
    return payload


def _partition_is_reusable(
    partition: Path,
    build_id: str,
    metadata: dict[str, str],
) -> bool:
    """Validate a completed rank partition before reusing an interrupted build."""

    try:
        if (partition / ".complete").read_text(encoding="utf-8") != build_id:
            return False
        if _read_status(partition / ".failed", build_id) is not None:
            return False
        index = pd.read_parquet(partition / "index.parquet", columns=["shard"])
        if index.empty or not (partition / "partial_latent_stats.pt").is_file():
            return False
        shard_names = set(index["shard"].astype(str))
        if not shard_names or any(
            not (partition / shard_name).is_file() for shard_name in shard_names
        ):
            return False
        payload = torch.load(
            partition / min(shard_names),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        return isinstance(payload, Mapping) and payload.get("metadata") == metadata
    except Exception:  # Corrupt or partial artifacts must be rebuilt.
        return False


def _wait_for_markers(
    markers: list[Path],
    build_id: str,
    *,
    description: str,
    poll_seconds: float = 1.0,
    timeout_seconds: float = 7200.0,
    monitor_partitions: bool = False,
    stall_timeout_seconds: float = 300.0,
) -> None:
    """Synchronize long NFS work without holding an NCCL collective open."""

    pending = set(markers)
    started = time.monotonic()
    progress = progress_bar(description=description, total=len(pending), unit="rank")
    try:
        while pending:
            completed: list[Path] = []
            for marker in pending:
                try:
                    matches = marker.read_text(encoding="utf-8") == build_id
                except (FileNotFoundError, OSError):
                    matches = False
                if matches:
                    completed.append(marker)
            for marker in completed:
                pending.remove(marker)
            progress.update(len(completed))
            if pending:
                if monitor_partitions:
                    now = time.time()
                    for marker in sorted(pending):
                        failure = _read_status(marker.parent / ".failed", build_id)
                        if failure is not None:
                            raise RuntimeError(
                                f"Latent-cache rank {failure.get('rank', '?')} failed: "
                                f"{failure.get('error', 'unknown error')}"
                            )
                        heartbeat = _read_status(marker.parent / ".heartbeat", build_id)
                        if heartbeat is None:
                            age = time.monotonic() - started
                            detail = "no heartbeat"
                        else:
                            updated_at = heartbeat.get("updated_at")
                            age = (
                                now - float(updated_at)
                                if isinstance(updated_at, int | float)
                                else time.monotonic() - started
                            )
                            detail = (
                                f"phase={heartbeat.get('phase', '?')}, "
                                f"samples={heartbeat.get('samples', '?')}"
                            )
                        if age >= stall_timeout_seconds:
                            raise RuntimeError(
                                f"Latent-cache rank {marker.parent.name} made no progress for "
                                f"{age:.0f}s ({detail}); aborting all ranks instead of waiting "
                                "indefinitely"
                            )
                if time.monotonic() - started >= timeout_seconds:
                    missing = ", ".join(str(path) for path in sorted(pending))
                    raise TimeoutError(f"{description} timed out waiting for: {missing}")
                time.sleep(poll_seconds)
    finally:
        progress.close()


def _broadcast_metadata(
    checkpoint_path: Path,
    manifest_path: Path,
    tokenizer_hash: str,
    extra_metadata: dict[str, str],
    context: DistributedContext,
) -> dict[str, str]:
    metadata: dict[str, str] | None = None
    if context.is_main:
        print("Hashing codec checkpoint once on rank 0...")
        metadata = {
            "codec_checkpoint_hash": checkpoint_hash(checkpoint_path),
            "tokenizer_hash": tokenizer_hash,
            "dataset_manifest_hash": manifest_hash(manifest_path),
            **extra_metadata,
        }
    if context.world_size > 1:
        values: list[object] = [metadata]
        dist.broadcast_object_list(values, src=0, device=context.device)
        received = values[0]
        if not isinstance(received, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in received.items()
        ):
            raise TypeError("Invalid latent-cache metadata broadcast")
        metadata = received
    if metadata is None:
        raise RuntimeError("Latent-cache metadata was not constructed")
    return metadata


def _checkpoint_codec_config(checkpoint: Mapping[str, object]) -> DictConfig:
    payload = checkpoint.get("config")
    if not isinstance(payload, Mapping):
        raise TypeError("Codec checkpoint is missing its embedded configuration")
    codec_cfg = OmegaConf.create(dict(payload))
    if not isinstance(codec_cfg, DictConfig):
        raise TypeError("Codec checkpoint configuration must be a mapping")
    if not isinstance(codec_cfg.get("tokenizer"), DictConfig):
        raise ValueError(
            "Codec checkpoint predates embedded tokenizer metadata; it cannot build a safe cache"
        )
    return codec_cfg


def _checkpoint_model_state(
    checkpoint: Mapping[str, object], weights: str
) -> Mapping[str, torch.Tensor]:
    if weights == "raw":
        state = checkpoint.get("model")
    elif weights == "ema":
        ema = checkpoint.get("ema_model")
        state = ema.get("shadow") if isinstance(ema, Mapping) else None
    else:
        raise ValueError("latent_cache.codec_weights must be 'raw' or 'ema'")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()
    ):
        raise TypeError(f"Codec checkpoint has no valid {weights} model state")
    return state


def _filter_overlength(
    frame: pd.DataFrame, *, max_sequence_length: int, enabled: bool
) -> tuple[pd.DataFrame, int]:
    if not enabled or "raw_token_count" not in frame:
        return frame, 0
    overlength = frame["raw_token_count"].astype(int) > max_sequence_length
    return frame.loc[~overlength].reset_index(drop=True), int(overlength.sum())


def _publish_cache(staging: Path, destination: Path) -> None:
    """Atomically expose a complete cache while keeping the old cache on failure."""

    previous = destination.with_name(f".{destination.name}.previous")
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        destination.replace(previous)
    try:
        staging.replace(destination)
    except BaseException:
        if previous.exists() and not destination.exists():
            previous.replace(destination)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def _cache(cfg: DictConfig, context: DistributedContext) -> None:
    if cfg.codec_checkpoint is None:
        raise ValueError("codec_checkpoint must point to a trained codec checkpoint")
    paths = prepare_config(cfg)
    checkpoint_path = Path(str(cfg.codec_checkpoint)).resolve()
    checkpoint_value = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(checkpoint_value, Mapping):
        raise TypeError("Codec checkpoint must contain a mapping")
    checkpoint = checkpoint_value
    codec_cfg = _checkpoint_codec_config(checkpoint)
    tokenizer_cfg = codec_cfg.tokenizer
    tokenizer = tokenizer_from_config(
        tokenizer_cfg, max_sequence_length=int(codec_cfg.max_sequence_length)
    )
    tokenizer_digest = hashlib.sha256(
        json.dumps(asdict(tokenizer.config), sort_keys=True).encode()
    ).hexdigest()
    provenance = checkpoint.get("provenance")
    checkpoint_tokenizer_hash = (
        provenance.get("tokenizer_hash") if isinstance(provenance, Mapping) else None
    )
    if checkpoint_tokenizer_hash != tokenizer_digest:
        raise ValueError(
            "Codec checkpoint tokenizer provenance does not match its embedded tokenizer config"
        )
    weights = str(cfg.latent_cache.get("codec_weights", "raw"))
    model = codec_from_config(codec_cfg, tokenizer)
    model.load_state_dict(_checkpoint_model_state(checkpoint, weights))
    del checkpoint, checkpoint_value
    device = context.device
    model.to(device).eval()
    reset_peak_memory(device)

    data_name = str(cfg.data.name)
    manifest_path = paths["processed_dir"] / data_name / "manifest.parquet"
    frame = pd.read_parquet(manifest_path)
    cache_columns = {
        "sample_id",
        "dataset_id",
        "source_midi_path",
        "split",
        "style_id",
        "genre_id",
        "emotion_id",
        "segment_id",
        "start_bar",
        "num_bars",
        "raw_token_count",
        "tokenizer_type",
        "tokenizer_version",
        "valid",
    }
    frame = frame[[column for column in frame.columns if column in cache_columns]].copy()
    frame, overlength_count = _filter_overlength(
        frame,
        max_sequence_length=int(codec_cfg.max_sequence_length),
        enabled=bool(codec_cfg.training.get("drop_overlength", True)),
    )
    if context.is_main and overlength_count:
        print(
            f"Dropping {overlength_count} overlength segments from latent caching to match "
            "codec training"
        )
    dtype = getattr(torch, str(cfg.latent_cache.dtype))
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Invalid latent cache dtype: {cfg.latent_cache.dtype}")
    codec_payload = OmegaConf.to_container(codec_cfg, resolve=True)
    metadata = _broadcast_metadata(
        checkpoint_path,
        manifest_path,
        tokenizer_digest,
        {
            "codec_config_hash": stable_json_hash(codec_payload),
            "codec_weights": weights,
            "latent_tokens": str(int(codec_cfg.latent_tokens)),
            "latent_dim": str(int(codec_cfg.latent_dim)),
            "latent_dtype": str(dtype).removeprefix("torch."),
            "max_sequence_length": str(int(codec_cfg.max_sequence_length)),
            "overlength_policy": "drop" if overlength_count else "none_needed",
        },
        context,
    )

    destination = paths["latent_dir"] / data_name
    staging = destination.with_name(f".{destination.name}.building")
    resume_cache = bool(cfg.latent_cache.get("resume", True))
    build_id = _broadcast_build_id(context, staging=staging, resume=resume_cache)
    synchronization_timeout = float(cfg.latent_cache.get("synchronization_timeout_seconds", 7200))
    if synchronization_timeout <= 0:
        raise ValueError("latent_cache.synchronization_timeout_seconds must be positive")
    build_marker = staging / ".build_id"
    if context.is_main:
        try:
            existing_build_id = build_marker.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            existing_build_id = ""
        preserve_staging = resume_cache and existing_build_id == build_id
        if staging.exists() and not preserve_staging:
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        _write_marker(build_marker, build_id)
    _wait_for_markers(
        [build_marker],
        build_id,
        description="Initialize latent-cache staging",
        timeout_seconds=synchronization_timeout,
    )
    partition_name = f"rank-{context.rank:05d}"
    partition = staging / partition_name
    partition_names = [f"rank-{rank:05d}" for rank in range(context.world_size)]

    configured_workers = int(cfg.latent_cache.get("dataloader_workers", 16))
    workers = max(1, configured_workers // context.world_size) if configured_workers > 0 else 0
    batch_size = int(cfg.latent_cache.get("batch_size_per_gpu", codec_cfg.inference.batch_size))
    if batch_size <= 0:
        raise ValueError("latent_cache.batch_size_per_gpu must be positive")
    midi_cache_size = int(cfg.latent_cache.get("midi_cache_size", 8))
    prefetch_factor = int(cfg.latent_cache.get("prefetch_factor", 4))
    dataloader_timeout = float(cfg.latent_cache.get("dataloader_timeout_seconds", 180))
    stall_timeout = float(cfg.latent_cache.get("rank_stall_timeout_seconds", 300))
    if dataloader_timeout <= 0 or stall_timeout <= 0:
        raise ValueError("Latent-cache DataLoader and rank-stall timeouts must be positive")
    length_bucketing = bool(cfg.latent_cache.get("length_bucketing", True))
    if context.is_main:
        total = int(frame.loc[frame["valid"]].shape[0])
        print(
            f"Latent caching plan: {total} samples, world_size={context.world_size}, "
            f"batch_size_per_gpu={batch_size}, workers_per_rank={workers}, "
            f"dtype={dtype}, source_grouped_length_bucketing={length_bucketing}"
        )

    collate = partial(collate_token_batch, pad_id=tokenizer.vocabulary.pad_id)
    reuse_partition = resume_cache and _partition_is_reusable(partition, build_id, metadata)
    if reuse_partition:
        print(f"Rank {context.rank}: reusing completed latent-cache partition {partition}")
    else:
        if partition.exists():
            shutil.rmtree(partition)
        partition.mkdir(parents=True)
    heartbeat_path = partition / ".heartbeat"

    def encoded_samples() -> Iterator[dict[str, object]]:
        total_encoded = 0
        with torch.inference_mode():
            for split in ("train", "validation", "test"):
                split_frame = frame.loc[
                    (frame["split"] == split) & frame["valid"]
                ].reset_index(drop=True)
                start, stop = _weighted_partition_bounds(
                    split_frame, context.rank, context.world_size
                )
                local_frame = split_frame.iloc[start:stop].reset_index(drop=True)
                del split_frame
                dataset = MidiTokenDataset(
                    local_frame,
                    tokenizer,
                    split=split,
                    midi_cache_size=midi_cache_size,
                )
                del local_frame
                if length_bucketing and len(dataset) and "raw_token_count" in dataset.frame:
                    local_lengths = dataset.frame["raw_token_count"].astype(int).tolist()
                    group_column = (
                        "source_midi_path"
                        if "source_midi_path" in dataset.frame
                        else "sample_id"
                    )
                    batch_sampler = GroupedLengthBatchSampler(
                        local_lengths,
                        group_ids=dataset.frame[group_column].astype(str).tolist(),
                        batch_size=batch_size,
                    )
                    if context.is_main:
                        print(
                            f"Cache batches {data_name}/{split}: {len(dataset)} samples, "
                            f"{len(batch_sampler)} batches, estimated attention efficiency="
                            f"{batch_sampler.estimated_attention_efficiency:.1%}"
                        )
                    loader = DataLoader(
                        dataset,
                        batch_sampler=batch_sampler,
                        collate_fn=collate,
                        num_workers=workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=False,
                        prefetch_factor=prefetch_factor if workers else None,
                        multiprocessing_context="spawn" if workers else None,
                        timeout=dataloader_timeout if workers else 0,
                    )
                else:
                    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate,
                        num_workers=workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=False,
                        prefetch_factor=prefetch_factor if workers else None,
                        multiprocessing_context="spawn" if workers else None,
                        timeout=dataloader_timeout if workers else 0,
                    )
                encoded = 0
                _write_heartbeat(
                    heartbeat_path,
                    build_id,
                    rank=context.rank,
                    phase=f"{split}:loading",
                    samples=total_encoded,
                )
                heartbeat_updated = time.monotonic()
                batch_progress = track(
                    loader,
                    description=f"Encode {data_name}/{split} rank {context.rank}",
                    total=len(loader),
                    unit="batch",
                )
                for batch in batch_progress:
                    batch_started = time.perf_counter()
                    tokens = batch["tokens"]
                    mask = batch["attention_mask"]
                    if not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor):
                        raise TypeError("Invalid token batch")
                    device_tokens = tokens.to(device, non_blocking=True)
                    device_mask = mask.to(device, non_blocking=True)
                    with autocast_context(device, str(codec_cfg.inference.precision)):
                        device_latent = model.encode_mean(device_tokens, device_mask)
                    # This is the intentional CUDA synchronization point. A traceback
                    # ending here after Ctrl-C means the process was interrupted while
                    # the current encoder batch was completing, not that .cpu() deadlocked.
                    latent = device_latent.to(dtype=dtype).cpu()
                    encoded += latent.shape[0]
                    total_encoded += latent.shape[0]
                    elapsed = max(time.perf_counter() - batch_started, 1e-9)
                    padding_efficiency = float(mask.sum()) / max(1, mask.numel())
                    valid_lengths = mask.sum(dim=1, dtype=torch.float32)
                    attention_efficiency = float(valid_lengths.square().sum()) / max(
                        1, mask.shape[0] * mask.shape[1] ** 2
                    )
                    batch_progress.set_postfix(
                        samples=encoded,
                        seq=tokens.shape[1],
                        pad=f"{padding_efficiency:.0%}",
                        attn=f"{attention_efficiency:.0%}",
                        batch_s=f"{elapsed:.2f}",
                        sample_s=f"{latent.shape[0] / elapsed:.0f}",
                        refresh=False,
                    )
                    if time.monotonic() - heartbeat_updated >= 30:
                        _write_heartbeat(
                            heartbeat_path,
                            build_id,
                            rank=context.rank,
                            phase=f"{split}:encoding",
                            samples=total_encoded,
                        )
                        heartbeat_updated = time.monotonic()
                    sample_ids = batch["sample_id"]
                    segment_ids = batch["segment_id"]
                    style_ids = batch["style_id"]
                    dataset_ids = batch["dataset_id"]
                    genre_ids = batch.get("genre_id")
                    emotion_ids = batch.get("emotion_id")
                    for index in range(latent.shape[0]):
                        yield {
                            "sample_id": sample_ids[index],
                            "segment_id": segment_ids[index],
                            "latent": latent[index],
                            "style_id": int(style_ids[index]),
                            "dataset_id": int(dataset_ids[index]),
                            "split": split,
                            "genre_id": int(genre_ids[index])
                            if isinstance(genre_ids, torch.Tensor)
                            else None,
                            "emotion_id": int(emotion_ids[index])
                            if isinstance(emotion_ids, torch.Tensor)
                            else None,
                        }
                batch_progress.close()
                _write_heartbeat(
                    heartbeat_path,
                    build_id,
                    rank=context.rank,
                    phase=f"{split}:complete",
                    samples=total_encoded,
                )

    if not reuse_partition:
        _write_heartbeat(
            heartbeat_path,
            build_id,
            rank=context.rank,
            phase="initialized",
            samples=0,
        )
        try:
            write_latent_cache(
                encoded_samples(),
                partition,
                samples_per_shard=int(cfg.latent_cache.samples_per_shard),
                dtype=dtype,
                metadata=metadata,
                verify_after_write=bool(cfg.latent_cache.verify_after_write),
                finalize=False,
            )
        except BaseException as error:
            _write_failure_marker(partition / ".failed", build_id, rank=context.rank, error=error)
            raise
        _write_marker(partition / ".complete", build_id)
    _wait_for_markers(
        [staging / name / ".complete" for name in partition_names],
        build_id,
        description="Wait for latent-cache partitions",
        timeout_seconds=synchronization_timeout,
        monitor_partitions=True,
        stall_timeout_seconds=stall_timeout,
    )
    index: Path | None = None
    if context.is_main:
        print("Merging rank indexes and computing streaming train statistics...")
        index = finalize_latent_cache_partitions(
            staging,
            partition_names,
            metadata=metadata,
        )
        _publish_cache(staging, destination)
        index = destination / index.relative_to(staging)
        _write_marker(destination / ".publish_complete", build_id)
        print(f"Latent cache index: {index}; peak GPU memory: {peak_memory_gib(device):.2f} GiB")
    _wait_for_markers(
        [destination / ".publish_complete"],
        build_id,
        description="Publish latent cache",
        timeout_seconds=synchronization_timeout,
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    context = initialize_distributed()
    try:
        _cache(cfg, context)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
