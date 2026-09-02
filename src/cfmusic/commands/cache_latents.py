"""Encode deterministic posterior means into versioned latent shards."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from cfmusic.codec.checkpoint import checkpoint_hash
from cfmusic.commands.train_codec import codec_from_config
from cfmusic.config import CONFIG_DIR, prepare_config
from cfmusic.data.collate import collate_token_batch
from cfmusic.data.datasets import MidiTokenDataset
from cfmusic.data.manifests import manifest_hash
from cfmusic.distributed import (
    DistributedContext,
    cleanup_distributed,
    distributed_barrier,
    initialize_distributed,
)
from cfmusic.latent.cache import (
    finalize_latent_cache_partitions,
    stable_json_hash,
    write_latent_cache,
)
from cfmusic.memory import autocast_context, peak_memory_gib, reset_peak_memory
from cfmusic.progress import track
from cfmusic.tokenization.factory import tokenizer_from_config


def _partition_bounds(length: int, rank: int, world_size: int) -> tuple[int, int]:
    base, remainder = divmod(length, world_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


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
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
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
    if context.is_main:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
    distributed_barrier(context)
    partition_name = f"rank-{context.rank:05d}"
    partition = staging / partition_name

    configured_workers = int(cfg.latent_cache.get("dataloader_workers", 16))
    workers = max(1, configured_workers // context.world_size) if configured_workers > 0 else 0
    batch_size = int(cfg.latent_cache.get("batch_size_per_gpu", codec_cfg.inference.batch_size))
    if batch_size <= 0:
        raise ValueError("latent_cache.batch_size_per_gpu must be positive")
    midi_cache_size = int(cfg.latent_cache.get("midi_cache_size", 8))
    prefetch_factor = int(cfg.latent_cache.get("prefetch_factor", 4))
    if context.is_main:
        total = int(frame.loc[frame["valid"]].shape[0])
        print(
            f"Latent caching plan: {total} samples, world_size={context.world_size}, "
            f"batch_size_per_gpu={batch_size}, workers_per_rank={workers}, "
            f"dtype={dtype}"
        )

    collate = partial(collate_token_batch, pad_id=tokenizer.vocabulary.pad_id)

    def encoded_samples() -> Iterator[dict[str, object]]:
        with torch.inference_mode():
            for split in ("train", "validation", "test"):
                dataset = MidiTokenDataset(
                    frame,
                    tokenizer,
                    split=split,
                    midi_cache_size=midi_cache_size,
                )
                start, stop = _partition_bounds(len(dataset), context.rank, context.world_size)
                local_dataset = Subset(dataset, range(start, stop))
                loader = DataLoader(
                    local_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate,
                    num_workers=workers,
                    pin_memory=device.type == "cuda",
                    persistent_workers=bool(workers),
                    prefetch_factor=prefetch_factor if workers else None,
                )
                encoded = 0
                batch_progress = track(
                    loader,
                    description=f"Encode {data_name}/{split} rank {context.rank}",
                    total=len(loader),
                    unit="batch",
                )
                for batch in batch_progress:
                    tokens = batch["tokens"]
                    mask = batch["attention_mask"]
                    if not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor):
                        raise TypeError("Invalid token batch")
                    with autocast_context(device, str(codec_cfg.inference.precision)):
                        latent = (
                            model.encode_mean(
                                tokens.to(device, non_blocking=True),
                                mask.to(device, non_blocking=True),
                            )
                            .to(dtype=dtype)
                            .cpu()
                        )
                    encoded += latent.shape[0]
                    batch_progress.set_postfix(samples=encoded, refresh=False)
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

    write_latent_cache(
        encoded_samples(),
        partition,
        samples_per_shard=int(cfg.latent_cache.samples_per_shard),
        dtype=dtype,
        metadata=metadata,
        verify_after_write=bool(cfg.latent_cache.verify_after_write),
        finalize=False,
    )
    distributed_barrier(context)
    index: Path | None = None
    if context.is_main:
        print("Merging rank indexes and computing streaming train statistics...")
        index = finalize_latent_cache_partitions(
            staging,
            [f"rank-{rank:05d}" for rank in range(context.world_size)],
            metadata=metadata,
        )
        _publish_cache(staging, destination)
        index = destination / index.relative_to(staging)
        print(f"Latent cache index: {index}; peak GPU memory: {peak_memory_gib(device):.2f} GiB")
    distributed_barrier(context)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    context = initialize_distributed()
    try:
        _cache(cfg, context)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
