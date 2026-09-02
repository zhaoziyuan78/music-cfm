from cfmusic.data.samplers import (
    BalancedStyleBatchSampler,
    DatasetTemperatureLengthBatchSampler,
    DistributedBatchSampler,
    LengthBucketBatchSampler,
    ShardBatchSampler,
)


def test_length_bucket_sampler_covers_samples_and_limits_padding() -> None:
    lengths = [100, 1, 99, 2, 98, 3, 97, 4]
    sampler = LengthBucketBatchSampler(lengths, batch_size=2, seed=7)
    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    assert all(
        max(lengths[index] for index in batch) - min(lengths[index] for index in batch) <= 1
        for batch in batches
    )


def test_length_bucket_sampler_epoch_changes_batch_order() -> None:
    sampler = LengthBucketBatchSampler(list(range(32)), batch_size=4, seed=3)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert {frozenset(batch) for batch in first} == {frozenset(batch) for batch in second}
    assert first != second


def test_distributed_batch_sampler_evenly_shards_and_pads_batches() -> None:
    ranks = [
        list(
            DistributedBatchSampler(
                LengthBucketBatchSampler(list(range(10)), batch_size=2, seed=5),
                rank=rank,
                world_size=4,
            )
        )
        for rank in range(4)
    ]

    assert {len(batches) for batches in ranks} == {2}
    flattened = [index for batches in ranks for batch in batches for index in batch]
    assert set(flattened) == set(range(10))
    assert len(flattened) == 16  # three padded whole batches keep collective counts equal


def test_distributed_batch_sampler_forwards_epoch() -> None:
    base = LengthBucketBatchSampler(list(range(16)), batch_size=2, seed=2)
    sampler = DistributedBatchSampler(base, rank=0, world_size=2)
    sampler.set_epoch(9)
    assert base.epoch == 9


def test_distributed_batch_sampler_aligns_rank_costs() -> None:
    lengths = list(range(1, 65))
    ranks = [
        list(
            DistributedBatchSampler(
                LengthBucketBatchSampler(lengths, batch_size=4, seed=3),
                rank=rank,
                world_size=4,
                batch_costs=lengths,
                seed=3,
            )
        )
        for rank in range(4)
    ]

    for step_batches in zip(*ranks, strict=True):
        maxima = [max(lengths[index] for index in batch) for batch in step_batches]
        assert max(maxima) - min(maxima) <= 12


def test_distributed_batch_sampler_repeats_one_batch_for_every_rank() -> None:
    ranks = [list(DistributedBatchSampler([[0, 1]], rank=rank, world_size=4)) for rank in range(4)]
    assert ranks == [[[0, 1]], [[0, 1]], [[0, 1]], [[0, 1]]]


def test_dataset_temperature_sampler_increases_small_dataset_exposure() -> None:
    dataset_ids = [0] * 900 + [1] * 100
    sampler = DatasetTemperatureLengthBatchSampler(
        list(range(1000)),
        dataset_ids=dataset_ids,
        batch_size=20,
        sampling_exponent=0.5,
        seed=4,
    )
    batches = list(sampler)
    selected = [index for batch in batches for index in batch]
    counts = {label: sum(dataset_ids[index] == label for index in selected) for label in (0, 1)}

    assert len(selected) == 1000
    assert counts == {0: 750, 1: 250}
    assert all(len(set(batch)) == len(batch) for batch in batches)


def test_dataset_temperature_sampler_is_epoch_deterministic() -> None:
    arguments = dict(
        lengths=list(range(12)),
        dataset_ids=[0] * 8 + [1] * 4,
        batch_size=3,
        sampling_exponent=0.5,
        seed=9,
    )
    first = DatasetTemperatureLengthBatchSampler(**arguments)
    second = DatasetTemperatureLengthBatchSampler(**arguments)
    assert list(first) == list(second)

    first.set_epoch(2)
    assert list(first) != list(second)


def test_shard_sampler_keeps_batches_local_and_balances_ranks() -> None:
    shard_ids = ["a"] * 9 + ["b"] * 7 + ["c"] * 5 + ["d"] * 3
    ranks = [
        list(
            ShardBatchSampler(
                shard_ids,
                batch_size=2,
                rank=rank,
                world_size=2,
                seed=7,
            )
        )
        for rank in range(2)
    ]

    assert len(ranks[0]) == len(ranks[1])
    assert all(len({shard_ids[index] for index in batch}) == 1 for rank in ranks for batch in rank)
    assert set(index for rank in ranks for batch in rank for index in batch) == set(
        range(len(shard_ids))
    )


def test_balanced_style_sampler_selects_one_shard_per_class() -> None:
    labels = [0] * 8 + [1] * 8
    groups = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
    sampler = BalancedStyleBatchSampler(
        labels,
        classes_per_batch=2,
        samples_per_class=3,
        group_ids=groups,
        seed=11,
    )

    for batch in sampler:
        for label in {labels[index] for index in batch}:
            selected_groups = {groups[index] for index in batch if labels[index] == label}
            assert len(selected_groups) == 1
