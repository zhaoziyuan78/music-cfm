"""Codec reconstruction helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from cfmusic.codec.base import LatentCodec


@torch.inference_mode()
def reconstruct_greedy(
    codec: LatentCodec, tokens: Tensor, attention_mask: Tensor, *, max_length: int
) -> Tensor:
    latent = codec.encode_mean(tokens, attention_mask)
    return codec.generate(
        latent, strategy="greedy", temperature=1.0, top_p=1.0, max_length=max_length
    )
