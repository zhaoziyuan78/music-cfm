from pathlib import Path

import torch

from cfmusic.codec.losses import vae_loss
from cfmusic.codec.transformer_vae import TransformerVAE
from cfmusic.conditioning.embeddings import AdditiveConditionEmbedding
from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.data.midi_io import load_midi
from cfmusic.models.latent_vector_field import ConditionalVectorField
from cfmusic.tokenization.bar_event import BarEventTokenizer
from cfmusic.transport.conditional_flow import ConditionalFlow


def test_two_step_codec_and_cfm_smoke(tiny_midi_path: Path) -> None:
    tokenizer = BarEventTokenizer()
    tokens_list = tokenizer.encode(load_midi(tiny_midi_path), num_bars=2)
    tokens = torch.tensor([tokens_list, tokens_list])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    codec = TransformerVAE(
        vocab_size=len(tokenizer.vocabulary),
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        ff_multiplier=2,
        dropout=0,
        latent_tokens=2,
        latent_dim=8,
        max_sequence_length=128,
        vocabulary=tokenizer.vocabulary,
    )
    codec_optimizer = torch.optim.Adam(codec.parameters(), lr=1e-3)
    for _ in range(2):
        logits, posterior = codec(tokens, mask)
        losses = vae_loss(
            logits, tokens[:, 1:], posterior, pad_id=0, beta=0.001, free_bits_per_dim=0
        )
        codec_optimizer.zero_grad()
        losses["loss"].backward()
        codec_optimizer.step()
    latent = codec.encode_mean(tokens, mask).detach()
    embedding = AdditiveConditionEmbedding(
        num_datasets=1, num_tasks=1, num_styles=2, num_genres=1, num_emotions=1, embedding_dim=32
    )
    field = ConditionalVectorField(
        latent_dim=8,
        hidden_dim=32,
        layers=1,
        heads=4,
        mlp_ratio=2,
        dropout=0,
        condition_embedding=embedding,
        zero_init_output=False,
        gradient_checkpointing=True,
    )
    transport = ConditionalFlow(field, solver_method="heun")
    condition = ConditionBatch(
        torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), torch.tensor([0, 1])
    )
    optimizer = torch.optim.Adam(transport.parameters(), lr=1e-3)
    for _ in range(2):
        loss = transport.training_loss(latent, condition)["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    output = transport.counterfactual(
        latent,
        condition,
        ConditionBatch(condition.dataset_id, condition.task_id, condition.style_id.flip(0)),
        num_steps=2,
    )
    decoded_tokens = codec.generate(
        output.counterfactual_latent, strategy="greedy", temperature=1, top_p=1, max_length=24
    )
    midi = tokenizer.decode(decoded_tokens[0].tolist())
    assert decoded_tokens.shape[0] == 2
    assert output.abducted_noise.shape == latent.shape
    assert midi.ticks_per_beat == 480
