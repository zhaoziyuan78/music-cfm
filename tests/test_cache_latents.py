import json
import time
from pathlib import Path

import pandas as pd
import pytest
import torch

from cfmusic.commands.cache_latents import (
    _filter_overlength,
    _partition_bounds,
    _partition_is_reusable,
    _wait_for_markers,
    _weighted_partition_bounds,
    _write_failure_marker,
    _write_marker,
)
from cfmusic.latent.cache import write_latent_cache


def test_cache_partitions_cover_samples_once_without_padding() -> None:
    bounds = [_partition_bounds(10, rank, 4) for rank in range(4)]
    indices = [index for start, stop in bounds for index in range(start, stop)]

    assert bounds == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert indices == list(range(10))


def test_cache_drops_the_same_overlength_rows_as_codec_training() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["short", "limit", "long"],
            "raw_token_count": [100, 2560, 2561],
        }
    )

    filtered, dropped = _filter_overlength(frame, max_sequence_length=2560, enabled=True)

    assert filtered["sample_id"].tolist() == ["short", "limit"]
    assert dropped == 1


def test_cache_partitions_balance_estimated_attention_cost() -> None:
    frame = pd.DataFrame({"raw_token_count": [1, 1, 4, 4]})

    bounds = [_weighted_partition_bounds(frame, rank, 2) for rank in range(2)]
    indices = [index for start, stop in bounds for index in range(start, stop)]

    assert bounds == [(0, 3), (3, 4)]
    assert indices == list(range(len(frame)))


def test_cache_filesystem_markers_are_scoped_to_one_build(tmp_path: Path) -> None:
    markers = [tmp_path / f"rank-{rank}" / ".complete" for rank in range(2)]
    _write_marker(markers[0], "old-build")
    _write_marker(markers[0], "current-build")
    _write_marker(markers[1], "current-build")

    _wait_for_markers(
        markers,
        "current-build",
        description="test cache markers",
        poll_seconds=0,
    )

    assert [marker.read_text(encoding="utf-8") for marker in markers] == [
        "current-build",
        "current-build",
    ]


def test_cache_wait_propagates_a_rank_failure(tmp_path: Path) -> None:
    marker = tmp_path / "rank-00001" / ".complete"
    _write_failure_marker(
        marker.parent / ".failed",
        "current-build",
        rank=1,
        error=RuntimeError("worker exited"),
    )

    with pytest.raises(RuntimeError, match=r"rank 1 failed.*worker exited"):
        _wait_for_markers(
            [marker],
            "current-build",
            description="test failed rank",
            poll_seconds=0,
            monitor_partitions=True,
        )


def test_cache_wait_rejects_a_stale_rank_heartbeat(tmp_path: Path) -> None:
    marker = tmp_path / "rank-00001" / ".complete"
    marker.parent.mkdir(parents=True)
    (marker.parent / ".heartbeat").write_text(
        json.dumps(
            {
                "build_id": "current-build",
                "rank": 1,
                "phase": "validation:encoding",
                "samples": 42,
                "updated_at": time.time() - 20,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"rank-00001 made no progress.*samples=42"):
        _wait_for_markers(
            [marker],
            "current-build",
            description="test stale rank",
            poll_seconds=0,
            monitor_partitions=True,
            stall_timeout_seconds=10,
        )


def test_cache_reuses_only_complete_matching_partitions(tmp_path: Path) -> None:
    partition = tmp_path / "rank-00000"
    metadata = {"codec_checkpoint_hash": "current"}
    write_latent_cache(
        [
            {
                "sample_id": "sample",
                "segment_id": "sample:0",
                "latent": torch.zeros(2, 3),
                "style_id": 0,
                "dataset_id": 0,
                "split": "train",
            }
        ],
        partition,
        metadata=metadata,
        verify_after_write=False,
        finalize=False,
    )
    _write_marker(partition / ".complete", "a" * 32)

    assert _partition_is_reusable(partition, "a" * 32, metadata)
    assert not _partition_is_reusable(
        partition, "a" * 32, {"codec_checkpoint_hash": "different"}
    )
    assert not _partition_is_reusable(partition, "b" * 32, metadata)
