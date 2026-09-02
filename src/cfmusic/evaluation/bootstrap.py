"""Seeded nonparametric bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np

from cfmusic.progress import track


def bootstrap_interval(
    values: np.ndarray, *, confidence: float = 0.95, samples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array")
    generator = np.random.default_rng(seed)
    estimates = np.array(
        [
            generator.choice(values, len(values), replace=True).mean()
            for _ in track(
                range(samples),
                description="Bootstrap confidence interval",
                total=samples,
                unit="sample",
            )
        ]
    )
    tail = (1 - confidence) / 2
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail))
