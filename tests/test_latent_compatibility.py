from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from cfmusic.latent.cache import write_latent_cache
from cfmusic.latent.compatibility import (
    serialize_cache_metadata,
    validate_latent_dataset,
    validate_transport_cache_provenance,
)
from cfmusic.latent.dataset import LatentDataset


def _cache(root: Path, *, latent_tokens: int = 2, latent_dim: int = 4) -> LatentDataset:
    metadata = {
        "codec_checkpoint_hash": "checkpoint",
        "tokenizer_hash": "tokenizer",
        "dataset_manifest_hash": "manifest",
        "latent_tokens": str(latent_tokens),
        "latent_dim": str(latent_dim),
    }
    write_latent_cache(
        [
            {
                "sample_id": "sample",
                "latent": torch.randn(latent_tokens, latent_dim),
                "style_id": 0,
                "dataset_id": 0,
                "split": "train",
            }
        ],
        root,
        metadata=metadata,
        verify_after_write=False,
    )
    return LatentDataset(root)


def test_current_codec_transport_and_cache_shapes_must_match(tmp_path: Path) -> None:
    dataset = _cache(tmp_path)
    codec = OmegaConf.create({"latent_tokens": 2, "latent_dim": 4})
    transport = OmegaConf.create({"model": {"latent_dim": 4}})

    assert validate_latent_dataset(
        dataset,
        codec_cfg=codec,
        transport_cfg=transport,
        dataset_name="xmidi",
    ) == (2, 4)

    transport.model.latent_dim = 8
    with pytest.raises(ValueError, match="Transport input dimension"):
        validate_latent_dataset(
            dataset,
            codec_cfg=codec,
            transport_cfg=transport,
            dataset_name="xmidi",
        )


def test_transport_checkpoint_is_bound_to_exact_cache_metadata(tmp_path: Path) -> None:
    dataset = _cache(tmp_path)
    checkpoint = {
        "provenance": {
            "latent_cache_metadata_json": serialize_cache_metadata([dataset.metadata])
        }
    }
    validate_transport_cache_provenance(checkpoint, [dataset.metadata])

    changed = dict(dataset.metadata)
    changed["codec_checkpoint_hash"] = "other"
    with pytest.raises(ValueError, match="different latent cache"):
        validate_transport_cache_provenance(checkpoint, [changed])
