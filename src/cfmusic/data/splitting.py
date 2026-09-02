"""Deterministic grouped stratified splitting without item-level leakage."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence


def grouped_stratified_split(
    group_ids: Sequence[str],
    labels: Sequence[str],
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> list[str]:
    """Assign whole groups while greedily matching split sizes and label ratios."""
    if len(group_ids) != len(labels):
        raise ValueError("group_ids and labels must have the same length")
    if not group_ids:
        return []
    if abs(sum(fractions) - 1.0) > 1e-6 or any(value < 0 for value in fractions):
        raise ValueError(f"Invalid split fractions: {fractions}")
    names = ("train", "validation", "test")
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        group_indices[group].append(index)
    rng = random.Random(seed)
    groups = list(group_indices)
    rng.shuffle(groups)
    groups.sort(key=lambda group: -len(group_indices[group]))
    target_sizes = [len(group_ids) * fraction for fraction in fractions]
    label_totals = Counter(labels)
    target_labels = [
        {label: count * fraction for label, count in label_totals.items()} for fraction in fractions
    ]
    split_sizes = [0, 0, 0]
    split_labels: list[Counter[str]] = [Counter(), Counter(), Counter()]
    assignments: dict[str, str] = {}
    for group in groups:
        indices = group_indices[group]
        group_counts = Counter(labels[index] for index in indices)
        scores: list[float] = []
        for split_index in range(3):
            size_error = abs(
                split_sizes[split_index] + len(indices) - target_sizes[split_index]
            ) / max(1.0, target_sizes[split_index])
            label_error = sum(
                abs(split_labels[split_index][label] + count - target_labels[split_index][label])
                / max(1.0, target_labels[split_index][label])
                for label, count in group_counts.items()
            )
            overfill = max(0.0, split_sizes[split_index] + len(indices) - target_sizes[split_index])
            scores.append(
                size_error + label_error + 2.0 * overfill / max(1.0, target_sizes[split_index])
            )
        chosen = min(range(3), key=lambda index: (scores[index], split_sizes[index], index))
        assignments[group] = names[chosen]
        split_sizes[chosen] += len(indices)
        split_labels[chosen].update(group_counts)
    return [assignments[group] for group in group_ids]


def assert_disjoint_groups(group_ids: Sequence[str], splits: Sequence[str]) -> None:
    owners: dict[str, str] = {}
    for group, split in zip(group_ids, splits, strict=True):
        previous = owners.setdefault(group, split)
        if previous != split:
            raise ValueError(f"Group {group!r} appears in both {previous!r} and {split!r}")
