"""Shared conditional DiT vector field over complete VAE latents."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from cfmusic.conditioning.embeddings import AdditiveConditionEmbedding
from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.models.dit_blocks import AdaLNBlock


def sinusoidal_time_embedding(time: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=time.device) / max(1, half - 1)
    )
    phases = time[:, None].float() * frequencies[None] * 1000.0
    embedding = torch.cat([phases.sin(), phases.cos()], dim=-1)
    return torch.nn.functional.pad(embedding, (0, dimension - embedding.shape[-1]))


class ConditionalVectorField(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
        condition_embedding: AdditiveConditionEmbedding,
        zero_init_output: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.SiLU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.condition_embedding = condition_embedding
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList(
            [AdaLNBlock(hidden_dim, heads, mlp_ratio, dropout) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.output = nn.Linear(hidden_dim, latent_dim)
        if zero_init_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, state: Tensor, time: Tensor, condition: ConditionBatch) -> Tensor:
        if time.ndim == 0:
            time = time.expand(state.shape[0])
        conditioning = self.condition_embedding(condition) + self.time_mlp(
            sinusoidal_time_embedding(time, self.input.out_features)
        )
        hidden = self.input(state)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, conditioning, use_reentrant=False)
            else:
                hidden = block(hidden, conditioning)
        return self.output(self.output_norm(hidden))

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled
