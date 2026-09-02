"""Differentiable normalized HSIC with detached median bandwidth."""

from __future__ import annotations

import torch
from torch import Tensor


def _center(kernel: Tensor) -> Tensor:
    return kernel - kernel.mean(0, keepdim=True) - kernel.mean(1, keepdim=True) + kernel.mean()


def rbf_kernel(features: Tensor, minimum_bandwidth: float = 1e-3) -> Tensor:
    distances = torch.cdist(features.float(), features.float()).square()
    positive = distances.detach()[distances.detach() > 0]
    median = positive.median() if positive.numel() else distances.new_tensor(1.0)
    bandwidth_squared = median.clamp_min(minimum_bandwidth**2)
    return torch.exp(-distances / (2 * bandwidth_squared))


def normalized_hsic(
    features: Tensor,
    labels: Tensor,
    *,
    unbiased: bool = False,
    minimum_bandwidth: float = 1e-3,
) -> Tensor:
    batch_size = features.shape[0]
    if batch_size < (4 if unbiased else 2):
        return features.sum() * 0.0
    feature_kernel = rbf_kernel(features, minimum_bandwidth)
    label_kernel = labels[:, None].eq(labels[None, :]).to(feature_kernel.dtype)
    if unbiased:
        feature_kernel = feature_kernel.clone().fill_diagonal_(0)
        label_kernel = label_kernel.clone().fill_diagonal_(0)
        n = float(batch_size)
        term1 = (feature_kernel * label_kernel).sum()
        term2 = feature_kernel.sum() * label_kernel.sum() / ((n - 1) * (n - 2))
        term3 = 2 * (feature_kernel.sum(0) * label_kernel.sum(0)).sum() / (n - 2)
        return ((term1 + term2 - term3) / (n * (n - 3))).clamp_min(0)
    centered_features = _center(feature_kernel)
    centered_labels = _center(label_kernel)
    numerator = (centered_features * centered_labels).sum()
    denominator = (
        centered_features.square().sum().sqrt() * centered_labels.square().sum().sqrt()
    ).clamp_min(1e-12)
    return (numerator / denominator).clamp_min(0)
