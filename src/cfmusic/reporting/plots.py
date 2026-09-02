"""Core research plots with deterministic rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def scatter_plot(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    label: str,
    output: Path,
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    for _, row in frame.iterrows():
        axis.scatter(row[x], row[y])
        axis.annotate(str(row[label]), (row[x], row[y]), fontsize=8)
    axis.set(xlabel=x, ylabel=y, title=title)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
