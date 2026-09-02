"""Lazy latent shard dataset with provenance validation."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from cfmusic.latent.normalization import LatentStatistics, load_statistics


class LatentDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    def __init__(
        self,
        root: Path,
        *,
        split: str = "train",
        expected_metadata: dict[str, str] | None = None,
        normalize: bool = True,
        shard_cache_size: int = 1,
        mmap_shards: bool = True,
    ) -> None:
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive")
        self.root = root
        metadata = json.loads((root / "cache_metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise TypeError("Latent cache metadata must be a mapping")
        self.metadata: dict[str, object] = {
            str(key): value for key, value in metadata.items()
        }
        if expected_metadata is not None:
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"Latent cache provenance mismatch: {mismatches}")
        frame = pd.read_parquet(root / "index.parquet")
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        self.normalize = normalize
        self.statistics: LatentStatistics = load_statistics(root)
        self.shard_cache_size = shard_cache_size
        self.mmap_shards = mmap_shards
        self._shard_cache: OrderedDict[str, dict[str, object]] = OrderedDict()
        self.refresh_index()

    def refresh_index(self) -> None:
        """Refresh compact column arrays after an intentional frame mutation."""

        self.shard_ids = self.frame["shard"].astype(str).tolist()
        self._offsets = self.frame["offset"].astype(int).tolist()
        self._style_ids = self.frame["style_id"].astype(int).tolist()
        self._dataset_ids = self.frame["dataset_id"].astype(int).tolist()
        self._sample_ids = self.frame["sample_id"].astype(str).tolist()
        self._genre_ids = self.frame["genre_id"].tolist() if "genre_id" in self.frame else None
        self._emotion_ids = (
            self.frame["emotion_id"].tolist() if "emotion_id" in self.frame else None
        )

    def _load_shard(self, shard_name: str) -> dict[str, object]:
        shard = self._shard_cache.get(shard_name)
        if shard is not None:
            self._shard_cache.move_to_end(shard_name)
            return shard
        restored = torch.load(
            self.root / shard_name,
            map_location="cpu",
            weights_only=False,
            mmap=self.mmap_shards,
        )
        if not isinstance(restored, dict):
            raise TypeError("Latent shard contains invalid data")
        self._shard_cache[shard_name] = restored
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return restored

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        shard = self._load_shard(self.shard_ids[index])
        latents = shard["latent"]
        if not isinstance(latents, torch.Tensor):
            raise TypeError("Latent shard contains invalid data")
        latent = latents[self._offsets[index]].float()
        if self.normalize:
            latent = self.statistics.normalize(latent)
        item: dict[str, torch.Tensor | str | int] = {
            "latent": latent,
            "style_id": self._style_ids[index],
            "dataset_id": self._dataset_ids[index],
            "sample_id": self._sample_ids[index],
        }
        if self._genre_ids is not None and pd.notna(self._genre_ids[index]):
            item["genre_id"] = int(self._genre_ids[index])
        if self._emotion_ids is not None and pd.notna(self._emotion_ids[index]):
            item["emotion_id"] = int(self._emotion_ids[index])
        return item
