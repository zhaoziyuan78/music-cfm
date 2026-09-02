"""Reusable Transformer masks and latent-query pooling."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)


class LatentQueryPool(nn.Module):
    def __init__(self, d_model: int, latent_tokens: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(latent_tokens, d_model) * 0.02)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, states: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        query = self.queries.unsqueeze(0).expand(states.shape[0], -1, -1)
        pooled, _ = self.attention(
            query, states, states, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.norm(query + pooled)
