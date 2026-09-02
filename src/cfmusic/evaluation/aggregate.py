"""Macro aggregation helpers."""

from __future__ import annotations

import pandas as pd


def macro_transition_average(frame: pd.DataFrame, metric: str) -> float:
    if metric not in frame:
        raise KeyError(metric)
    return float(frame.groupby(["source_style", "target_style"])[metric].mean().mean())
