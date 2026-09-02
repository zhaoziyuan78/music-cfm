"""Cross-seed counterfactual spread."""

from __future__ import annotations

import itertools

from torch import Tensor


def counterfactual_spread(samples: list[Tensor]) -> float:
    if len(samples) < 2:
        return 0.0
    distances = [
        float((left - right).square().mean().sqrt())
        for left, right in itertools.combinations(samples, 2)
    ]
    return sum(distances) / len(distances)
