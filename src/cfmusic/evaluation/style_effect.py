"""Independent token and symbolic-descriptor style evaluators."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class TokenStyleClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        *,
        d_model: int = 384,
        layers: int = 6,
        heads: int = 6,
        dropout: float = 0.1,
        max_length: int = 2048,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position = nn.Parameter(torch.randn(max_length, d_model) * 0.01)
        self.gradient_checkpointing = gradient_checkpointing
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.output = nn.Linear(d_model, num_classes)

    def forward(self, tokens: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.embedding(tokens) + self.position[: tokens.shape[1]]
        padding_mask = ~attention_mask.bool()
        if self.gradient_checkpointing and self.training:
            for layer in self.encoder.layers:
                hidden = checkpoint(
                    layer,
                    hidden,
                    src_key_padding_mask=padding_mask,
                    use_reentrant=False,
                )
            if self.encoder.norm is not None:
                hidden = self.encoder.norm(hidden)
        else:
            hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        scores = hidden @ self.query
        scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        pooled = (scores.softmax(-1).unsqueeze(-1) * hidden).sum(1)
        return self.output(pooled)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(1)
    prediction = probabilities.argmax(1)
    error = 0.0
    for lower, upper in zip(
        np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:], strict=True
    ):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.mean() * abs(
                (prediction[selected] == labels[selected]).mean() - confidence[selected].mean()
            )
    return float(error)
