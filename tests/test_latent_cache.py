from pathlib import Path

import pandas as pd
import pytest
import torch

from cfmusic.latent.cache import finalize_latent_cache_partitions, write_latent_cache
from cfmusic.latent.dataset import LatentDataset
from cfmusic.latent.normalization import compute_train_statistics, load_statistics


def test_versioned_latent_cache(tmp_path: Path) -> None:
    torch.manual_seed(5)
    samples = [
        {
            "sample_id": f"s{index}",
            "latent": torch.randn(2, 4),
            "style_id": index % 2,
            "dataset_id": 0,
            "split": "train" if index < 4 else "test",
        }
        for index in range(6)
    ]
    write_latent_cache(
        samples,
        tmp_path,
        samples_per_shard=2,
        metadata={
            "codec_checkpoint_hash": "abc",
            "tokenizer_hash": "def",
            "dataset_manifest_hash": "ghi",
        },
    )
    dataset = LatentDataset(tmp_path, expected_metadata={"codec_checkpoint_hash": "abc"})
    assert len(dataset) == 4
    assert dataset[0]["latent"].shape == (2, 4)
    with pytest.raises(ValueError, match="provenance"):
        LatentDataset(tmp_path, expected_metadata={"codec_checkpoint_hash": "wrong"})
    expected = compute_train_statistics(
        torch.stack([sample["latent"] for sample in samples[:4]]).half().float()
    )
    actual = load_statistics(tmp_path)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.std, expected.std)
    assert actual.count == 4


def test_distributed_partitions_finalize_into_one_cache(tmp_path: Path) -> None:
    metadata = {
        "codec_checkpoint_hash": "abc",
        "tokenizer_hash": "def",
        "dataset_manifest_hash": "ghi",
    }
    all_samples: list[dict[str, object]] = []
    for rank in range(2):
        samples = [
            {
                "sample_id": f"r{rank}s{index}",
                "segment_id": f"r{rank}s{index}:0",
                "latent": torch.full((2, 3), rank * 4 + index, dtype=torch.float32),
                "style_id": index % 2,
                "dataset_id": 0,
                "split": "train" if index < 3 else "test",
            }
            for index in range(4)
        ]
        all_samples.extend(samples)
        write_latent_cache(
            samples,
            tmp_path / f"rank-{rank:05d}",
            samples_per_shard=2,
            dtype=torch.float16,
            metadata=metadata,
            verify_after_write=False,
            finalize=False,
        )

    index_path = finalize_latent_cache_partitions(
        tmp_path, ["rank-00000", "rank-00001"], metadata=metadata
    )
    assert not list(tmp_path.glob("rank-*/partial_latent_stats.pt"))
    frame = pd.read_parquet(index_path)
    assert len(frame) == 8
    assert set(frame["shard"].str.split("/").str[0]) == {"rank-00000", "rank-00001"}
    dataset = LatentDataset(tmp_path, normalize=False)
    assert len(dataset) == 6
    expected = compute_train_statistics(
        torch.stack([sample["latent"] for sample in all_samples if sample["split"] == "train"])
        .half()
        .float()
    )
    actual = load_statistics(tmp_path)
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.std, expected.std)
    assert actual.count == 6
