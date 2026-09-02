"""Grammar-constrained autoregressive decoding."""

from __future__ import annotations

from collections.abc import Callable

import torch

from cfmusic.progress import track
from cfmusic.tokenization.grammar import EventGrammar


def grammar_greedy_decode(
    next_logits: Callable[[torch.Tensor], torch.Tensor],
    grammar: EventGrammar,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    tokens = torch.full((batch_size, 1), grammar.vocabulary.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for _ in track(
        range(max_length - 1),
        description="Grammar-constrained decoding",
        total=max_length - 1,
        unit="token",
        leave=False,
    ):
        logits = next_logits(tokens)[:, -1]
        allowed = grammar.mask(tokens[:, -1])
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        chosen = logits.argmax(dim=-1)
        chosen = torch.where(finished, grammar.vocabulary.pad_id, chosen)
        tokens = torch.cat([tokens, chosen[:, None]], dim=1)
        finished |= chosen == grammar.vocabulary.eos_id
        if bool(finished.all()):
            break
    return tokens
