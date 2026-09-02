"""Standalone LaTeX table document generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_latex_document(frame: pd.DataFrame, path: Path, caption: str) -> None:
    table = frame.to_latex(index=False, escape=True, caption=caption)
    path.write_text(
        "\\documentclass{article}\n\\usepackage{booktabs}\n\\begin{document}\n"
        + table
        + "\n\\end{document}\n",
        encoding="utf-8",
    )
