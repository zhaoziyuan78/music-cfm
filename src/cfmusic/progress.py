"""Consistent progress reporting for long-running command-line workloads."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

_FALSE_VALUES = {"0", "false", "no", "off"}


def progress_enabled() -> bool:
    """Return whether this process should render progress bars.

    Progress is enabled by default, including in redirected Slurm logs.  Set
    ``CFMUSIC_PROGRESS=0`` to silence it.  Under ``torchrun`` only rank zero
    renders bars so workers do not duplicate output.
    """

    configured = os.environ.get("CFMUSIC_PROGRESS", "1").strip().lower()
    try:
        rank = int(os.environ.get("RANK", "0"))
    except ValueError:
        rank = 0
    return configured not in _FALSE_VALUES and rank == 0


def track(
    iterable: Iterable[T],
    *,
    description: str,
    total: int | None = None,
    unit: str = "item",
    leave: bool = True,
    position: int | None = None,
) -> tqdm[T]:
    """Wrap an iterable in the project's standard progress display."""

    return tqdm(
        iterable,
        desc=description,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.1,
        leave=leave,
        position=position,
        disable=not progress_enabled(),
    )


def progress_bar(
    *,
    description: str,
    total: int | None,
    initial: int = 0,
    unit: str = "item",
    leave: bool = True,
    position: int | None = None,
) -> tqdm[object]:
    """Create a manually updated progress bar with the standard settings."""

    return tqdm(
        desc=description,
        total=total,
        initial=initial,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.1,
        leave=leave,
        position=position,
        disable=not progress_enabled(),
    )
