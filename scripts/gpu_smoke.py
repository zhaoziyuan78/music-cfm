"""Run a compact bf16 VAE/CFM/DDIM smoke test on one CUDA device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import miditoolkit
import torch

from cfmusic.codec.losses import vae_loss
from cfmusic.codec.transformer_vae import TransformerVAE
from cfmusic.conditioning.embeddings import AdditiveConditionEmbedding
from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.models.latent_denoiser import ConditionalLatentDenoiser
from cfmusic.models.latent_vector_field import ConditionalVectorField
from cfmusic.tokenization.bar_event import BarEventTokenizer
from cfmusic.transport.conditional_ddim import ConditionalDDIM
from cfmusic.transport.conditional_flow import ConditionalFlow


def make_tokens(tokenizer: BarEventTokenizer, batch_size: int) -> torch.Tensor:
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    midi.tempo_changes = [miditoolkit.TempoChange(120, 0)]
    midi.time_signature_changes = [miditoolkit.TimeSignature(4, 4, 0)]
    instrument = miditoolkit.Instrument(program=0)
    for bar in range(2):
        for step, pitch in enumerate((60, 64, 67, 72)):
            start = bar * 1920 + step * 480
            instrument.notes.append(miditoolkit.Note(72, pitch, start, start + 360))
    midi.instruments = [instrument]
    values = tokenizer.encode(midi, num_bars=2)
    return torch.tensor([values for _ in range(batch_size)], dtype=torch.long)


def embedding() -> AdditiveConditionEmbedding:
    return AdditiveConditionEmbedding(
        num_datasets=2,
        num_tasks=2,
        num_styles=4,
        num_genres=6,
        num_emotions=11,
        embedding_dim=64,
    )


def run(output: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script on a GPU node")
    device = torch.device("cuda:0")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    tokenizer = BarEventTokenizer()
    tokens = make_tokens(tokenizer, 4).to(device)
    mask = tokens.ne(tokenizer.vocabulary.pad_id)
    codec = TransformerVAE(
        vocab_size=len(tokenizer.vocabulary),
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        num_heads=4,
        ff_multiplier=2,
        dropout=0,
        latent_tokens=4,
        latent_dim=16,
        max_sequence_length=128,
        vocabulary=tokenizer.vocabulary,
    ).to(device)
    codec_optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, posterior = codec(tokens, mask)
        codec_losses = vae_loss(
            logits,
            tokens[:, 1:],
            posterior,
            pad_id=tokenizer.vocabulary.pad_id,
            beta=0.001,
            free_bits_per_dim=0,
        )
    codec_losses["loss"].backward()
    codec_optimizer.step()
    latent = codec.encode_mean(tokens, mask).detach().float()
    zeros = torch.zeros(tokens.shape[0], dtype=torch.long, device=device)
    source = ConditionBatch(zeros, zeros, torch.tensor([0, 1, 2, 3], device=device))
    target = ConditionBatch(zeros, zeros, torch.tensor([1, 2, 3, 0], device=device))

    vector_field = ConditionalVectorField(
        latent_dim=16,
        hidden_dim=64,
        layers=2,
        heads=4,
        mlp_ratio=2,
        dropout=0,
        condition_embedding=embedding(),
        zero_init_output=False,
    ).to(device)
    flow = ConditionalFlow(vector_field, solver_method="heun")
    flow_optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        flow_loss = flow.training_loss(latent, source)["loss"]
    flow_loss.backward()
    flow_optimizer.step()
    flow_output = flow.counterfactual(latent, source, target, num_steps=4)

    denoiser = ConditionalLatentDenoiser(
        latent_dim=16,
        hidden_dim=64,
        layers=2,
        heads=4,
        mlp_ratio=2,
        dropout=0,
        condition_embedding=embedding(),
        zero_init_output=False,
    ).to(device)
    ddim = ConditionalDDIM(
        denoiser,
        train_timesteps=20,
        inversion_method="fixed_point",
        fpi_iterations=2,
    ).to(device)
    ddim_optimizer = torch.optim.AdamW(ddim.parameters(), lr=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ddim_loss = ddim.training_loss(latent, source)["loss"]
    ddim_loss.backward()
    ddim_optimizer.step()
    ddim_output = ddim.counterfactual(latent, source, target, num_steps=4)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "codec": codec.state_dict(),
            "flow": flow.state_dict(),
            "ddim": ddim.state_dict(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        temporary,
    )
    temporary.replace(output)
    return {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "codec_loss": float(codec_losses["loss"].detach()),
        "cfm_loss": float(flow_loss.detach()),
        "cfm_roundtrip_mse": float(
            (flow_output.reconstructed_source_latent - latent).square().mean()
        ),
        "ddim_loss": float(ddim_loss.detach()),
        "ddim_roundtrip_mse": float(
            (ddim_output.reconstructed_source_latent - latent).square().mean()
        ),
        "reserved_memory_mb": torch.cuda.memory_reserved() / 1024**2,
        "checkpoint": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output), indent=2))


if __name__ == "__main__":
    main()
