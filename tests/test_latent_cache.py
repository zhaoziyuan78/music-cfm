import json
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
    assert actual.mean.shape == (2, 4)
    assert actual.count == 4


def test_latent_label_overlay_reuses_tensors_and_changes_condition_ids(tmp_path: Path) -> None:
    samples = [
        {
            "sample_id": f"s{index}",
            "latent": torch.full((2, 4), float(index)),
            "style_id": index % 2,
            "dataset_id": 0,
            "genre_id": index % 2,
            "split": "train",
        }
        for index in range(4)
    ]
    write_latent_cache(
        samples,
        tmp_path,
        samples_per_shard=2,
        metadata={
            "codec_checkpoint_hash": "abc",
            "tokenizer_hash": "def",
            "dataset_manifest_hash": "base-manifest",
        },
    )
    overlay_path = tmp_path / "index_clamp2_prompt.parquet"
    overlay = pd.read_parquet(tmp_path / "index.parquet")
    overlay["style_id"] = [1, 1, 0, 0]
    overlay["genre_id"] = overlay["style_id"]
    overlay.to_parquet(overlay_path, index=False)
    overlay_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "cfmusic.latent-label-overlay.v1",
                "label_source": "clamp2_nearest_genre_prompt",
                "label_assignment_hash": "assignment",
                "base_dataset_manifest_hash": "base-manifest",
                "rows": 4,
            }
        ),
        encoding="utf-8",
    )

    original = LatentDataset(tmp_path, normalize=False)
    relabeled = LatentDataset(tmp_path, index_path=overlay_path, normalize=False)

    assert original[0]["style_id"] == 0
    assert relabeled[0]["style_id"] == 1
    torch.testing.assert_close(original[0]["latent"], relabeled[0]["latent"])
    assert relabeled.metadata["label_assignment_hash"] == "assignment"


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


def test_obsolete_latent_normalization_is_rejected(tmp_path: Path) -> None:
    torch.save(torch.zeros(4), tmp_path / "latent_mean.pt")
    torch.save(torch.ones(4), tmp_path / "latent_std.pt")
    (tmp_path / "latent_stats.json").write_text(
        json.dumps({"train_samples": 2, "latent_dim": 4}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="obsolete normalization"):
        load_statistics(tmp_path)
