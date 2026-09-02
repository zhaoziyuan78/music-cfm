"""CSV, Markdown, and LaTeX result table writers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_result_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(frame.to_markdown(index=False), encoding="utf-8")
    stem.with_suffix(".tex").write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")
