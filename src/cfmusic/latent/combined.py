"""Joint 4Q latent dataset with explicit dataset identities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from cfmusic.latent.dataset import LatentDataset


class CombinedLatentDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    def __init__(self, roots: list[Path], *, split: str = "train") -> None:
        self.datasets = [LatentDataset(root, split=split) for root in roots]
        self.offsets: list[tuple[int, int, int]] = []
        self.shard_ids: list[str] = []
        start = 0
        for dataset_id, dataset in enumerate(self.datasets):
            self.offsets.append((start, start + len(dataset), dataset_id))
            self.shard_ids.extend(
                f"dataset-{dataset_id:05d}/{shard_id}" for shard_id in dataset.shard_ids
            )
            start += len(dataset)
        self.length = start

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        for start, end, dataset_id in self.offsets:
            if start <= index < end:
                item = dict(self.datasets[dataset_id][index - start])
                item["dataset_id"] = dataset_id
                return item
        raise IndexError(index)
