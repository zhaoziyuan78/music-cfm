"""Train-only latent feature normalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

NORMALIZATION_SCHEMA_VERSION = "per-token-v2"


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
    """Per-token statistics without retaining every latent.

    A VAE latent token is produced by a distinct learned query, so token
    positions are not exchangeable.  Statistics therefore retain ``[T, D]``
    instead of flattening tokens into a shared ``[D]`` distribution.
    """

    def __init__(self) -> None:
        self.sample_count = 0
        self.vector_count = 0
        self.feature_sum: Tensor | None = None
        self.feature_square_sum: Tensor | None = None

    def update(self, latents: Tensor) -> None:
        if latents.ndim != 3 or latents.shape[0] == 0:
            raise ValueError("Expected nonempty [samples, latent_tokens, latent_dim] tensor")
        values = latents.detach().cpu().float()
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
        mean = self.feature_sum / self.sample_count
        variance = self.feature_square_sum / self.sample_count - mean.square()
        return LatentStatistics(
            mean.float(),
            variance.clamp_min(1e-12).sqrt().float().clamp_min(1e-6),
            self.sample_count,
        )


def compute_train_statistics(latents: Tensor) -> LatentStatistics:
    if latents.ndim != 3 or latents.shape[0] == 0:
        raise ValueError("Expected nonempty [samples, latent_tokens, latent_dim] tensor")
    values = latents.float()
    return LatentStatistics(
        values.mean(0), values.std(0, unbiased=False).clamp_min(1e-6), latents.shape[0]
    )


def save_statistics(stats: LatentStatistics, directory: Path) -> str:
    mean = stats.mean.detach().cpu().contiguous()
    std = stats.std.detach().cpu().contiguous()
    if mean.ndim != 2 or mean.shape != std.shape:
        raise ValueError("Per-token latent statistics must have matching [tokens, dim] shapes")
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(mean, directory / "latent_mean.pt")
    torch.save(std, directory / "latent_std.pt")
    digest = hashlib.sha256(
        NORMALIZATION_SCHEMA_VERSION.encode() + mean.numpy().tobytes() + std.numpy().tobytes()
    ).hexdigest()
    payload = {
        "train_samples": stats.count,
        "latent_tokens": mean.shape[0],
        "latent_dim": mean.shape[1],
        "normalization_shape": list(mean.shape),
        "normalization_schema_version": NORMALIZATION_SCHEMA_VERSION,
        "normalization_hash": digest,
    }
    (directory / "latent_stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return digest


def load_statistics(directory: Path) -> LatentStatistics:
    payload = json.loads((directory / "latent_stats.json").read_text(encoding="utf-8"))
    if payload.get("normalization_schema_version") != NORMALIZATION_SCHEMA_VERSION:
        raise ValueError(
            f"Latent cache uses obsolete normalization; rebuild it with "
            f"{NORMALIZATION_SCHEMA_VERSION}"
        )
    mean = torch.load(directory / "latent_mean.pt", weights_only=True)
    std = torch.load(directory / "latent_std.pt", weights_only=True)
    expected_shape = tuple(int(value) for value in payload.get("normalization_shape", ()))
    if mean.ndim != 2 or mean.shape != std.shape or tuple(mean.shape) != expected_shape:
        raise ValueError("Latent normalization tensors do not match their recorded per-token shape")
    return LatentStatistics(
        mean,
        std,
        int(payload["train_samples"]),
    )
