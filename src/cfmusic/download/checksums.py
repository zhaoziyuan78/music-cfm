"""Streaming checksum helpers."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path

from cfmusic.progress import progress_bar


def sha256_file(path: Path, chunk_size: int = 1024 * 1024, *, show_progress: bool = False) -> str:
    digest = hashlib.sha256()
    progress_context = (
        progress_bar(
            description=f"SHA-256 {path.name}",
            total=path.stat().st_size,
            unit="B",
            position=1,
        )
        if show_progress
        else nullcontext(None)
    )
    with (
        path.open("rb") as stream,
        progress_context as progress_value,
    ):
        progress = progress_value
        if progress is not None:
            progress.unit_scale = True
            progress.unit_divisor = 1024
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
            if progress is not None:
                progress.update(len(chunk))
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> str:
    actual = sha256_file(path, show_progress=True)
    if expected is not None and actual.lower() != expected.lower():
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return actual
