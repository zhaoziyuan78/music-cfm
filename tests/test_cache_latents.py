import pandas as pd

from cfmusic.commands.cache_latents import _filter_overlength, _partition_bounds


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

    filtered, dropped = _filter_overlength(
        frame, max_sequence_length=2560, enabled=True
    )

    assert filtered["sample_id"].tolist() == ["short", "limit"]
    assert dropped == 1
