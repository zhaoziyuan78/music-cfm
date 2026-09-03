"""Dataset-balanced and style-balanced batch samplers."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Protocol

from torch.utils.data import Sampler


class BatchSamplerProtocol(Protocol):
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[list[int]]: ...


class BalancedStyleBatchSampler(Sampler[list[int]]):
    """Sample ``style -> unique song -> one segment`` balanced batches."""

    def __init__(
        self,
        labels: Sequence[int],
        *,
        classes_per_batch: int = 4,
        samples_per_class: int = 16,
        replacement_for_small_classes: bool = True,
        group_ids: Sequence[str] | None = None,
        seed: int = 0,
    ) -> None:
        if classes_per_batch <= 0 or samples_per_class <= 0:
            raise ValueError("Batch dimensions must be positive")
        by_class: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            by_class[int(label)].append(index)
        if not by_class:
            raise ValueError("BalancedStyleBatchSampler requires at least one sample")
        self.by_class = dict(by_class)
        if group_ids is not None and len(group_ids) != len(labels):
            raise ValueError("group_ids and labels must have the same length")
        by_class_group: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        if group_ids is not None:
            for index, (label, group_id) in enumerate(zip(labels, group_ids, strict=True)):
                by_class_group[int(label)][str(group_id)].append(index)
        self.by_class_group = {label: dict(groups) for label, groups in by_class_group.items()}
        self.group_names = {label: sorted(groups) for label, groups in self.by_class_group.items()}
        self.classes_per_batch = min(classes_per_batch, len(by_class))
        self.samples_per_class = samples_per_class
        self.replacement = replacement_for_small_classes
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return max(
            1,
            len([index for values in self.by_class.values() for index in values])
            // (self.classes_per_batch * self.samples_per_class),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        classes = sorted(self.by_class)
        for batch_index in range(len(self)):
            offset = batch_index * self.classes_per_batch
            rotated = classes[offset % len(classes) :] + classes[: offset % len(classes)]
            chosen_classes = rotated[: self.classes_per_batch]
            batch: list[int] = []
            for label in chosen_classes:
                candidates = self.by_class[label]
                if self.by_class_group:
                    groups = self.by_class_group[label]
                    group_names = self.group_names[label]
                    if len(group_names) < self.samples_per_class:
                        raise ValueError(
                            f"Class {label} has only {len(group_names)} unique sample_id groups, "
                            f"fewer than samples_per_class={self.samples_per_class}; reducing the "
                            "batch is required because a song may occur at most once"
                        )
                    chosen_groups = rng.sample(group_names, self.samples_per_class)
                    batch.extend(rng.choice(groups[name]) for name in chosen_groups)
                elif len(candidates) >= self.samples_per_class:
                    batch.extend(rng.sample(candidates, self.samples_per_class))
                elif self.replacement:
                    batch.extend(rng.choices(candidates, k=self.samples_per_class))
                else:
                    batch.extend(candidates)
            rng.shuffle(batch)
            yield batch


class ShardBatchSampler(Sampler[list[int]]):
    """Balance shard-local work over ranks, optionally sampling one segment per song.

    Random sample-level shuffling is pathological for large tensor shards: a one-shard
    dataset cache otherwise reloads a roughly 128 MiB file for nearly every sample.  This
    sampler shuffles shards and samples *within* a shard, retaining stochastic training
    while each rank reads every assigned shard only once per epoch.
    """

    def __init__(
        self,
        shard_ids: Sequence[str],
        *,
        batch_size: int,
        sample_ids: Sequence[str] | None = None,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not shard_ids:
            raise ValueError("ShardBatchSampler requires at least one sample")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("Invalid distributed rank/world_size")
        if sample_ids is not None and len(sample_ids) != len(shard_ids):
            raise ValueError("sample_ids and shard_ids must have the same length")
        self._song_segments: dict[int, list[int]] = {}
        by_shard: dict[str, list[int]] = defaultdict(list)
        if sample_ids is None:
            for index, shard_id in enumerate(shard_ids):
                by_shard[str(shard_id)].append(index)
        else:
            by_song: dict[str, list[int]] = defaultdict(list)
            for index, sample_id in enumerate(sample_ids):
                by_song[str(sample_id)].append(index)
            for indices in by_song.values():
                shards = [str(shard_ids[index]) for index in indices]
                primary = max(sorted(set(shards)), key=shards.count)
                local_indices = [index for index in indices if str(shard_ids[index]) == primary]
                representative = local_indices[0]
                self._song_segments[representative] = local_indices
                by_shard[primary].append(representative)
        self.by_shard = dict(by_shard)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self._rank_groups = self._assign_groups()
        self._length = max(self._batch_count(groups) for groups in self._rank_groups)
        if self._song_segments:
            # Keep the optimizer budget comparable to segment-level training while
            # giving every song equal exposure and changing its selected window on
            # repeated song passes.
            self._length = max(
                self._length,
                math.ceil(len(shard_ids) / (self.batch_size * self.world_size)),
            )

    def _count(self, sample_count: int) -> int:
        full, remainder = divmod(sample_count, self.batch_size)
        return full + int(bool(remainder) and not self.drop_last)

    def _batch_count(self, groups: dict[str, list[int]]) -> int:
        return sum(self._count(len(indices)) for indices in groups.values())

    def _assign_groups(self) -> list[dict[str, list[int]]]:
        assignments: list[dict[str, list[int]]] = [dict() for _ in range(self.world_size)]
        loads = [0] * self.world_size
        ordered = sorted(
            self.by_shard.items(), key=lambda item: (-self._count(len(item[1])), item[0])
        )
        if len(ordered) >= self.world_size:
            for shard_id, indices in ordered:
                target = min(range(self.world_size), key=lambda rank: (loads[rank], rank))
                assignments[target][shard_id] = indices
                loads[target] += self._count(len(indices))
            return assignments

        # Very small/legacy caches can contain fewer shards than GPU ranks. Split their
        # already-local batches across ranks; all ranks still receive equal DDP steps.
        batch_units: list[tuple[str, list[int]]] = []
        for shard_id, indices in ordered:
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batch_units.append((shard_id, batch))
        for shard_id, batch in batch_units:
            target = min(range(self.world_size), key=lambda rank: (loads[rank], rank))
            assignments[target].setdefault(shard_id, []).extend(batch)
            loads[target] += 1
        for target, groups in enumerate(assignments):
            if groups:
                continue
            shard_id, batch = batch_units[target % len(batch_units)]
            groups[shard_id] = list(batch)
            loads[target] = 1
        return assignments

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = self._rank_groups[self.rank]
        if self._song_segments:
            produced = 0
            while produced < self._length:
                shard_names = list(groups)
                rng.shuffle(shard_names)
                representatives: list[int] = []
                for shard_name in shard_names:
                    shard_representatives = list(groups[shard_name])
                    rng.shuffle(shard_representatives)
                    representatives.extend(shard_representatives)
                if not representatives:
                    raise RuntimeError("ShardBatchSampler rank has no song representatives")
                if self.drop_last and len(representatives) < self.batch_size:
                    raise RuntimeError(
                        "drop_last=True cannot produce a song-unique batch on this rank"
                    )
                for start in range(0, len(representatives), self.batch_size):
                    if produced >= self._length:
                        break
                    batch_representatives = representatives[start : start + self.batch_size]
                    if len(batch_representatives) < self.batch_size and self.drop_last:
                        continue
                    batch = [
                        rng.choice(self._song_segments[index]) for index in batch_representatives
                    ]
                    produced += 1
                    yield batch
            return
        shard_names = list(groups)
        rng.shuffle(shard_names)
        batches: list[list[int]] = []
        for shard_name in shard_names:
            indices = [
                rng.choice(self._song_segments.get(index, [index])) for index in groups[shard_name]
            ]
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if not batches:
            raise RuntimeError("ShardBatchSampler produced no batches")
        original = list(batches)
        while len(batches) < self._length:
            batches.append(list(original[(len(batches) - len(original)) % len(original)]))
        yield from batches


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle batches of similarly sized sequences to minimize padding."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not lengths:
            raise ValueError("LengthBucketBatchSampler requires at least one sample")
        self.sorted_indices = sorted(range(len(lengths)), key=lambda index: int(lengths[index]))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        full, remainder = divmod(len(self.sorted_indices), self.batch_size)
        return full + int(bool(remainder) and not self.drop_last)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        starts = list(range(0, len(self.sorted_indices), self.batch_size))
        if self.drop_last and starts and starts[-1] + self.batch_size > len(self.sorted_indices):
            starts.pop()
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(starts)
        for start in starts:
            batch = self.sorted_indices[start : start + self.batch_size]
            rng.shuffle(batch)
            yield batch
        self.epoch += 1


class GroupedLengthBatchSampler(Sampler[list[int]]):
    """Pack source-local groups into similarly sized inference batches.

    A global length sort minimizes padding but scatters the windows from each source
    MIDI over many workers and batches.  Keeping every (small) source group together
    lets :class:`MidiTokenDataset` parse that MIDI once while sorting groups by their
    maximum length still avoids padding every batch to an unrelated long sequence.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        group_ids: Sequence[str],
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(lengths) != len(group_ids) or not lengths:
            raise ValueError("lengths and group_ids must have the same non-zero length")
        normalized_lengths = [max(1, int(length)) for length in lengths]
        by_group: dict[str, list[int]] = defaultdict(list)
        for index, group_id in enumerate(group_ids):
            by_group[str(group_id)].append(index)

        # A source normally contributes only a few dozen windows.  Chunk unusually
        # large sources so the batch-size bound remains strict.
        chunks: list[tuple[int, str, int, list[int]]] = []
        for group_id, indices in by_group.items():
            for chunk_number, start in enumerate(range(0, len(indices), batch_size)):
                chunk = indices[start : start + batch_size]
                maximum = max(normalized_lengths[index] for index in chunk)
                chunks.append((maximum, group_id, chunk_number, chunk))
        chunks.sort(key=lambda item: (item[0], item[1], item[2]))

        batches: list[list[int]] = []
        current: list[int] = []
        for _, _, _, chunk in chunks:
            if current and len(current) + len(chunk) > batch_size:
                batches.append(current)
                current = []
            current.extend(chunk)
        if current:
            batches.append(current)
        self.batches = batches
        self.lengths = normalized_lengths

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        # Cache construction is deterministic inference, so no epoch shuffle is
        # needed.  Group contents stay contiguous for the per-worker MIDI LRU cache.
        for batch in self.batches:
            yield list(batch)

    @property
    def estimated_attention_efficiency(self) -> float:
        """Useful token-square work divided by padded token-square work."""

        useful = sum(length * length for length in self.lengths)
        padded = sum(
            len(batch) * max(self.lengths[index] for index in batch) ** 2
            for batch in self.batches
        )
        return useful / padded


class DatasetTemperatureLengthBatchSampler(Sampler[list[int]]):
    """Balance dataset exposure while retaining length-efficient batches.

    Dataset sampling probabilities are proportional to ``count ** sampling_exponent``.
    An exponent of one preserves the natural mixture, zero gives equal exposure, and
    the default square-root mixture increases small-dataset coverage without repeating
    every small corpus hundreds of times per epoch.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        dataset_ids: Sequence[int],
        batch_size: int,
        sampling_exponent: float = 0.5,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(lengths) != len(dataset_ids) or not lengths:
            raise ValueError("lengths and dataset_ids must have the same non-zero length")
        if not 0.0 <= sampling_exponent <= 1.0:
            raise ValueError("sampling_exponent must be in [0, 1]")
        by_dataset: dict[int, list[int]] = defaultdict(list)
        for index, dataset_id in enumerate(dataset_ids):
            by_dataset[int(dataset_id)].append(index)
        self.lengths = [int(length) for length in lengths]
        self.by_dataset = dict(by_dataset)
        self.total_samples = len(lengths)
        self.batch_size = batch_size
        self.sampling_exponent = sampling_exponent
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        full, remainder = divmod(self.total_samples, self.batch_size)
        return full + int(bool(remainder) and not self.drop_last)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _quotas(self) -> dict[int, int]:
        labels = sorted(self.by_dataset)
        weights = {label: len(self.by_dataset[label]) ** self.sampling_exponent for label in labels}
        denominator = sum(weights.values())
        raw = {label: self.total_samples * weights[label] / denominator for label in labels}
        quotas = {label: math.floor(raw[label]) for label in labels}
        remainder = self.total_samples - sum(quotas.values())
        ranked = sorted(
            labels, key=lambda label: (raw[label] - quotas[label], -label), reverse=True
        )
        for label in ranked[:remainder]:
            quotas[label] += 1
        return quotas

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        selected: list[int] = []
        for label, quota in self._quotas().items():
            candidates = self.by_dataset[label]
            while quota > 0:
                cycle = list(candidates)
                rng.shuffle(cycle)
                take = min(quota, len(cycle))
                selected.extend(cycle[:take])
                quota -= take
        selected.sort(key=self.lengths.__getitem__)
        # Repeated small-dataset samples would be adjacent after a pure length sort.
        # Round-robin within a local mega-bucket keeps the length range narrow while
        # spreading repeated indices across different optimizer batches.
        batches: list[list[int]] = []
        mega_bucket_size = self.batch_size * 64
        for start in range(0, len(selected), mega_bucket_size):
            mega_bucket = selected[start : start + mega_bucket_size]
            batch_count = math.ceil(len(mega_bucket) / self.batch_size)
            local_batches: list[list[int]] = [[] for _ in range(batch_count)]
            for offset, index in enumerate(mega_bucket):
                local_batches[offset % batch_count].append(index)
            batches.extend(local_batches)
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        rng.shuffle(batches)
        for batch in batches:
            rng.shuffle(batch)
            yield batch
        self.epoch += 1


class DistributedBatchSampler(Sampler[list[int]]):
    """Shard an existing batch sampler evenly across synchronous workers.

    A few leading batches are repeated when necessary so every rank performs the
    same number of optimizer collectives.  This matches ``DistributedSampler``'s
    non-dropping behavior while retaining length/style-balanced whole batches.
    """

    def __init__(
        self,
        batch_sampler: BatchSamplerProtocol,
        *,
        rank: int,
        world_size: int,
        batch_costs: Sequence[int] | None = None,
        seed: int = 0,
    ) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.batch_sampler = batch_sampler
        self.rank = rank
        self.world_size = world_size
        self.batch_costs = batch_costs
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        total = len(self.batch_sampler)
        return (total + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        setter = getattr(self.batch_sampler, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        total = len(self.batch_sampler)
        target_total = len(self) * self.world_size
        padding = target_total - total
        if self.batch_costs is not None:
            batch_costs = self.batch_costs
            batches = [list(batch) for batch in self.batch_sampler]
            if not batches:
                raise RuntimeError("DistributedBatchSampler produced no batches")
            batches.extend(list(batches[index % len(batches)]) for index in range(padding))
            batches.sort(key=lambda batch: max(batch_costs[index] for index in batch))
            rank_groups = [
                batches[start : start + self.world_size]
                for start in range(0, len(batches), self.world_size)
            ]
            random.Random(self.seed + self.epoch).shuffle(rank_groups)
            for group in rank_groups:
                yield group[self.rank]
            return
        leading: list[list[int]] = []
        for index, batch in enumerate(self.batch_sampler):
            if index < padding:
                leading.append(batch)
            if index % self.world_size == self.rank:
                yield batch
        for offset in range(padding):
            batch = leading[offset % len(leading)]
            if (total + offset) % self.world_size == self.rank:
                yield list(batch)
