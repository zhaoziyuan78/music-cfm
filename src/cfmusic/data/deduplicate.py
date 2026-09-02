"""Exact canonical duplicate grouping."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def duplicate_clusters(hashes: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Map sample IDs to stable cluster IDs from canonical hashes."""
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id, digest in hashes:
        groups[digest].append(sample_id)
    return {
        sample_id: f"duplicate:{digest}" if len(samples) > 1 else sample_id
        for digest, samples in groups.items()
        for sample_id in samples
    }
