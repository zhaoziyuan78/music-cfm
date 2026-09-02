"""Train-only latent feature normalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class LatentStatistics:
    mean: Tensor
    std: Tensor
    count: int

    def normalize(self, latent: Tensor, eps: float = 1e-6) -> Tensor:
        return (latent - self.mean.to(latent)) / (self.std.to(latent) + eps)

    def denormalize(self, latent: Tensor) -> Tensor:
        return latent * self.std.to(latent) + self.mean.to(latent)


class StreamingLatentStatistics:
    """Numerically stable feature statistics without retaining every latent."""

    def __init__(self) -> None:
        self.sample_count = 0
        self.vector_count = 0
        self.feature_sum: Tensor | None = None
        self.feature_square_sum: Tensor | None = None

    def update(self, latents: Tensor) -> None:
        if latents.ndim != 3 or latents.shape[0] == 0:
            raise ValueError("Expected nonempty [samples, latent_tokens, latent_dim] tensor")
        values = latents.detach().cpu().float().reshape(-1, latents.shape[-1])
        feature_sum = values.sum(dim=0, dtype=torch.float64)
        feature_square_sum = values.square().sum(dim=0, dtype=torch.float64)
        if self.feature_sum is None:
            self.feature_sum = feature_sum
            self.feature_square_sum = feature_square_sum
        else:
            self.feature_sum += feature_sum
            if self.feature_square_sum is None:
                raise RuntimeError("Streaming statistics state is inconsistent")
            self.feature_square_sum += feature_square_sum
        self.sample_count += latents.shape[0]
        self.vector_count += values.shape[0]

    def state_dict(self) -> dict[str, Tensor | int]:
        if self.feature_sum is None or self.feature_square_sum is None:
            raise ValueError("Cannot save empty streaming statistics")
        return {
            "sample_count": self.sample_count,
            "vector_count": self.vector_count,
            "feature_sum": self.feature_sum,
            "feature_square_sum": self.feature_square_sum,
        }

    def merge_state_dict(self, state: dict[str, object]) -> None:
        feature_sum = state.get("feature_sum")
        feature_square_sum = state.get("feature_square_sum")
        sample_count = state.get("sample_count")
        vector_count = state.get("vector_count")
        if (
            not isinstance(feature_sum, Tensor)
            or not isinstance(feature_square_sum, Tensor)
            or not isinstance(sample_count, int)
            or not isinstance(vector_count, int)
        ):
            raise TypeError("Invalid streaming statistics state")
        if self.feature_sum is None:
            self.feature_sum = feature_sum.double().clone()
            self.feature_square_sum = feature_square_sum.double().clone()
        else:
            if self.feature_square_sum is None:
                raise RuntimeError("Streaming statistics state is inconsistent")
            self.feature_sum += feature_sum.double()
            self.feature_square_sum += feature_square_sum.double()
        self.sample_count += sample_count
        self.vector_count += vector_count

    def finalize(self) -> LatentStatistics:
        if (
            self.sample_count == 0
            or self.vector_count == 0
            or self.feature_sum is None
            or self.feature_square_sum is None
        ):
            raise ValueError("Cannot normalize without train split latents")
        mean = self.feature_sum / self.vector_count
        variance = self.feature_square_sum / self.vector_count - mean.square()
        return LatentStatistics(
            mean.float(),
            variance.clamp_min(1e-12).sqrt().float().clamp_min(1e-6),
            self.sample_count,
        )


def compute_train_statistics(latents: Tensor) -> LatentStatistics:
    if latents.ndim != 3 or latents.shape[0] == 0:
        raise ValueError("Expected nonempty [samples, latent_tokens, latent_dim] tensor")
    flattened = latents.float().reshape(-1, latents.shape[-1])
    return LatentStatistics(
        flattened.mean(0), flattened.std(0, unbiased=False).clamp_min(1e-6), latents.shape[0]
    )


def save_statistics(stats: LatentStatistics, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(stats.mean, directory / "latent_mean.pt")
    torch.save(stats.std, directory / "latent_std.pt")
    digest = hashlib.sha256(stats.mean.numpy().tobytes() + stats.std.numpy().tobytes()).hexdigest()
    payload = {
        "train_samples": stats.count,
        "latent_dim": stats.mean.numel(),
        "normalization_hash": digest,
    }
    (directory / "latent_stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return digest


def load_statistics(directory: Path) -> LatentStatistics:
    payload = json.loads((directory / "latent_stats.json").read_text(encoding="utf-8"))
    return LatentStatistics(
        torch.load(directory / "latent_mean.pt", weights_only=True),
        torch.load(directory / "latent_std.pt", weights_only=True),
        int(payload["train_samples"]),
    )
