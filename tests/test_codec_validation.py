import math

import torch

from cfmusic.codec.transformer_vae import TransformerVAE
from cfmusic.training.codec_trainer import evaluate_codec_batches


def test_codec_validation_logs_reconstruction_and_latent_reliance() -> None:
    model = TransformerVAE(
        vocab_size=24,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        ff_multiplier=2,
        dropout=0.0,
        latent_tokens=2,
        latent_dim=4,
        max_sequence_length=16,
    ).train()
    tokens = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 9, 2], [1, 4, 8, 6, 2], [1, 7, 5, 9, 2]])
    metrics = evaluate_codec_batches(
        model,
        [{"tokens": tokens, "attention_mask": tokens.ne(0)}],
        device=torch.device("cpu"),
        precision="fp32",
    )

    assert model.training
    assert metrics["evaluated_samples"] == 4
    assert metrics["evaluated_tokens"] == 16
    assert 0.0 <= metrics["teacher_forced_token_accuracy"] <= 1.0
    assert all(math.isfinite(value) for value in metrics.values())
