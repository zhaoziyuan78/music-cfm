"""Distributional two-sample losses used by endpoint and exogeneity checks."""

from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor


def maximum_mean_discrepancy(left: Tensor, right: Tensor) -> Tensor:
    """Biased RBF MMD with a detached median bandwidth."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("MMD inputs must be rank-2 tensors with matching feature dimensions")
    joined = torch.cat([left.float(), right.float()])
    distances = torch.cdist(joined, joined).square()
    positive = distances.detach()[distances.detach() > 0]
    bandwidth = (positive.median() if positive.numel() else distances.new_tensor(1.0)).clamp_min(
        1e-6
    )
    kernel = torch.exp(-distances / (2 * bandwidth))
    count = left.shape[0]
    return (
        kernel[:count, :count].mean()
        + kernel[count:, count:].mean()
        - 2 * kernel[:count, count:].mean()
    ).clamp_min(0)


def cross_class_mmd(features: Tensor, labels: Tensor) -> Tensor:
    """Mean pairwise MMD between every observed class distribution."""

    losses = [
        maximum_mean_discrepancy(features[labels == left], features[labels == right])
        for left, right in combinations(torch.unique(labels).tolist(), 2)
        if int((labels == left).sum()) >= 2 and int((labels == right).sum()) >= 2
    ]
    return torch.stack(losses).mean() if losses else features.sum() * 0.0


def class_conditional_mmd(generated: Tensor, factual: Tensor, labels: Tensor) -> Tensor:
    losses = [
        maximum_mean_discrepancy(generated[labels == label], factual[labels == label])
        for label in torch.unique(labels)
        if int((labels == label).sum()) >= 2
    ]
    return torch.stack(losses).mean() if losses else generated.sum() * 0.0
