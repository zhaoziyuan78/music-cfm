"""Fail-fast compatibility checks for cached codec latents and transports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from omegaconf import DictConfig

from cfmusic.latent.dataset import LatentDataset


def _metadata_int(value: object) -> int:
    if not isinstance(value, (int, str)):
        raise TypeError("Latent shape metadata must be an integer string")
    return int(value)


def load_cache_metadata(root: Path) -> dict[str, object]:
    payload = json.loads((root / "cache_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid latent cache metadata in {root}")
    return {str(key): value for key, value in payload.items()}


def serialize_cache_metadata(metadata: Sequence[Mapping[str, object]]) -> str:
    """Serialize one or more cache identities into transport provenance."""

    return json.dumps(
        [{str(key): value for key, value in item.items()} for item in metadata],
        sort_keys=True,
        separators=(",", ":"),
    )


def expected_transport_input_dim(transport_cfg: DictConfig) -> int:
    split = transport_cfg.get("split")
    if split is not None and bool(split.get("enabled", False)):
        return int(split.original_latent_dim)
    return int(transport_cfg.model.latent_dim)


def validate_latent_dataset(
    dataset: LatentDataset,
    *,
    codec_cfg: DictConfig,
    transport_cfg: DictConfig | None = None,
    dataset_name: str,
) -> tuple[int, int]:
    """Ensure cache metadata, tensors, codec, and transport agree exactly."""

    expected_shape = (int(codec_cfg.latent_tokens), int(codec_cfg.latent_dim))
    recorded_shape: tuple[int, int]
    try:
        recorded_shape = (
            _metadata_int(dataset.metadata["latent_tokens"]),
            _metadata_int(dataset.metadata["latent_dim"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"The {dataset_name} latent cache predates shape metadata; rebuild it from the "
            "current codec checkpoint"
        ) from error
    if recorded_shape != expected_shape:
        raise ValueError(
            f"The {dataset_name} latent cache records shape {recorded_shape}, but the configured "
            f"codec requires {expected_shape}; rebuild cache_latents"
        )
    if len(dataset) == 0:
        raise RuntimeError(f"The {dataset_name} latent cache contains no samples for this split")
    latent = dataset[0].get("latent")
    if not isinstance(latent, torch.Tensor) or latent.ndim != 2:
        raise TypeError(f"The {dataset_name} latent cache must contain rank-2 sample tensors")
    actual_shape = (int(latent.shape[0]), int(latent.shape[1]))
    if actual_shape != recorded_shape:
        raise ValueError(
            f"The {dataset_name} latent tensor shape {actual_shape} disagrees with cache metadata "
            f"{recorded_shape}; delete and rebuild the cache"
        )
    if transport_cfg is not None:
        transport_dim = expected_transport_input_dim(transport_cfg)
        if transport_dim != actual_shape[1]:
            raise ValueError(
                f"Transport input dimension {transport_dim} is incompatible with the "
                f"{dataset_name} codec latent dimension {actual_shape[1]}"
            )
    return actual_shape


def validate_transport_cache_provenance(
    checkpoint: Mapping[str, object], metadata: Sequence[Mapping[str, object]]
) -> None:
    """Reject a transport checkpoint trained from any other latent cache."""

    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Transport checkpoint has no provenance; retrain it on the current cache")
    recorded = provenance.get("latent_cache_metadata_json")
    if not isinstance(recorded, str):
        raise ValueError(
            "Transport checkpoint predates strict latent-cache provenance; retrain it on the "
            "current latent cache"
        )
    expected = serialize_cache_metadata(metadata)
    if recorded != expected:
        raise ValueError(
            "Transport checkpoint was trained from a different latent cache; checkpoint, codec, "
            "manifest, tokenizer, or normalization statistics do not match"
        )
