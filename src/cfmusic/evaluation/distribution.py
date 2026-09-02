"""Distribution distances for unpaired generated samples."""

from __future__ import annotations

import torch
from torch import Tensor


def rbf_mmd(left: Tensor, right: Tensor) -> Tensor:
    combined = torch.cat([left.flatten(1), right.flatten(1)])
    distance = torch.cdist(combined, combined).square()
    positive = distance[distance > 0]
    bandwidth = (
        positive.median().detach().clamp_min(1e-6) if positive.numel() else distance.new_tensor(1)
    )
    kernel = torch.exp(-distance / (2 * bandwidth))
    n = left.shape[0]
    return kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()
