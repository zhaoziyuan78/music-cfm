"""Padding collator for variable-length event sequences."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_token_batch(
    items: list[dict[str, torch.Tensor | str | int]], pad_id: int = 0
) -> dict[str, torch.Tensor | list[str]]:
    tokens = pad_sequence(
        [item["tokens"] for item in items if isinstance(item["tokens"], torch.Tensor)],
        batch_first=True,
        padding_value=pad_id,
    )
    batch: dict[str, torch.Tensor | list[str]] = {
        "tokens": tokens,
        "attention_mask": tokens.ne(pad_id),
        "style_id": torch.tensor([int(item["style_id"]) for item in items]),
        "dataset_id": torch.tensor([int(item["dataset_id"]) for item in items]),
        "sample_id": [str(item["sample_id"]) for item in items],
        "segment_id": [str(item["segment_id"]) for item in items],
    }
    for key in ("genre_id", "emotion_id"):
        if all(key in item for item in items):
            batch[key] = torch.tensor([int(item[key]) for item in items])
    if all("num_bars" in item for item in items):
        batch["num_bars"] = torch.tensor([int(item["num_bars"]) for item in items])
    return batch
