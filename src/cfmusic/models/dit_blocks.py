"""AdaLN latent Transformer blocks."""

from __future__ import annotations

from torch import Tensor, nn


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))

    def forward(self, state: Tensor, conditioning: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(conditioning).chunk(6, -1)
        modulated = self.norm1(state) * (1 + scale1[:, None]) + shift1[:, None]
        attended, _ = self.attention(modulated, modulated, modulated, need_weights=False)
        state = state + gate1[:, None] * attended
        modulated = self.norm2(state) * (1 + scale2[:, None]) + shift2[:, None]
        return state + gate2[:, None] * self.mlp(modulated)
