"""Measure full-size training and inference peaks against one 40 GiB A100."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cfmusic.codec.losses import vae_loss
from cfmusic.codec.transformer_vae import TransformerVAE
from cfmusic.commands.train_codec import codec_from_config
from cfmusic.conditioning.embeddings import AdditiveConditionEmbedding
from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.config import CONFIG_DIR
from cfmusic.evaluation.style_effect import TokenStyleClassifier
from cfmusic.memory import (
    autocast_context,
    peak_memory_gib,
    reset_peak_memory,
    sdpa_kernel_context,
    total_memory_gib,
)
from cfmusic.models.latent_denoiser import ConditionalLatentDenoiser
from cfmusic.models.latent_vector_field import ConditionalVectorField
from cfmusic.tokenization.beat import BeatTokenizer, BeatTokenizerConfig
from cfmusic.tokenization.factory import tokenizer_from_config
from cfmusic.training.state import ExponentialMovingAverage
from cfmusic.transport.conditional_ddim import ConditionalDDIM
from cfmusic.transport.conditional_flow import ConditionalFlow

Case = Callable[[torch.device], dict[str, float | int | str]]
CODEC_BATCH = 32
CODEC_TOKENS = 2560
CODEC_PROFILE = "transformer_vae"
CODEC_STEPS = 1
CODEC_GRADIENT_CHECKPOINTING = True
CACHE_BATCH = 384
LATENT_TOKENS = 64
LATENT_DIM = 512


def _codec(device: torch.device, *, training: bool) -> TransformerVAE:
    codec_config = OmegaConf.load(Path(CONFIG_DIR) / "codec" / f"{CODEC_PROFILE}.yaml")
    tokenizer_config = OmegaConf.load(Path(CONFIG_DIR) / "tokenizer" / "beat.yaml")
    tokenizer = tokenizer_from_config(tokenizer_config, max_sequence_length=CODEC_TOKENS)
    codec_config.max_sequence_length = CODEC_TOKENS
    codec_config.training.gradient_checkpointing = CODEC_GRADIENT_CHECKPOINTING
    model = codec_from_config(codec_config, tokenizer).to(device)
    return model.train(training)


def _embedding(hidden_dim: int = 512) -> AdditiveConditionEmbedding:
    return AdditiveConditionEmbedding(
        num_datasets=4,
        num_tasks=8,
        num_styles=32,
        num_genres=6,
        num_emotions=11,
        embedding_dim=hidden_dim,
    )


def _condition(batch_size: int, device: torch.device) -> ConditionBatch:
    zeros = torch.zeros(batch_size, dtype=torch.long, device=device)
    styles = torch.arange(batch_size, device=device) % 4
    return ConditionBatch(zeros, zeros, styles)


def _flow(device: torch.device) -> ConditionalFlow:
    field = ConditionalVectorField(
        latent_dim=LATENT_DIM,
        hidden_dim=512,
        layers=8,
        heads=8,
        mlp_ratio=4,
        dropout=0,
        condition_embedding=_embedding(),
        zero_init_output=False,
        gradient_checkpointing=False,
    ).to(device)
    return ConditionalFlow(field, solver_method="heun")


def codec_train(device: torch.device) -> dict[str, float | int | str]:
    tokenizer = BeatTokenizer(BeatTokenizerConfig(max_sequence_length=CODEC_TOKENS))
    model = _codec(device, training=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ema = ExponentialMovingAverage(model)
    tokens = torch.randint(
        0, len(tokenizer.vocabulary), (CODEC_BATCH, CODEC_TOKENS), dtype=torch.long, device=device
    )
    tokens[:, 0] = tokenizer.vocabulary.bos_id
    step_times: list[float] = []
    losses: dict[str, torch.Tensor] = {}
    for _ in range(CODEC_STEPS):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with autocast_context(device, "bf16"):
            logits, posterior = model(tokens, torch.ones_like(tokens, dtype=torch.bool))
            losses = vae_loss(
                logits,
                tokens[:, 1:],
                posterior,
                pad_id=tokenizer.vocabulary.pad_id,
                beta=0.0001,
                free_bits_per_dim=0.05,
            )
        losses["loss"].backward()
        optimizer.step()
        ema.update(model)
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - started)
    measured_times = step_times[1:] or step_times
    return {
        "loss": float(losses["loss"].detach()),
        "micro_batch": CODEC_BATCH,
        "tokens": CODEC_TOKENS,
        "steps": CODEC_STEPS,
        "steady_step_seconds": statistics.mean(measured_times),
        "steady_samples_per_second": CODEC_BATCH / statistics.mean(measured_times),
        "gradient_checkpointing": str(CODEC_GRADIENT_CHECKPOINTING).lower(),
    }


def codec_encode(device: torch.device) -> dict[str, float | int | str]:
    tokenizer = BeatTokenizer(BeatTokenizerConfig(max_sequence_length=CODEC_TOKENS))
    model = _codec(device, training=False)
    tokens = torch.randint(
        0,
        len(tokenizer.vocabulary),
        (CACHE_BATCH, CODEC_TOKENS),
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode(), autocast_context(device, "bf16"):
        latent = model.encode_mean(tokens, torch.ones_like(tokens, dtype=torch.bool))
    return {"latent_elements": latent.numel(), "batch": CACHE_BATCH, "tokens": CODEC_TOKENS}


def codec_decode(device: torch.device) -> dict[str, float | int | str]:
    tokenizer = BeatTokenizer(BeatTokenizerConfig(max_sequence_length=CODEC_TOKENS))
    model = _codec(device, training=False)
    tokens = torch.randint(
        0,
        len(tokenizer.vocabulary),
        (1, CODEC_TOKENS - 1),
        dtype=torch.long,
        device=device,
    )
    latent = torch.randn(1, model.latent_tokens, model.latent_dim, device=device)
    with torch.inference_mode(), autocast_context(device, "bf16"):
        logits = model.decode_teacher_forced(tokens, latent)
    return {"logit_elements": logits.numel(), "batch": 1, "tokens": CODEC_TOKENS - 1}


def evaluator_train(device: torch.device) -> dict[str, float | int | str]:
    tokenizer = BeatTokenizer()
    model = TokenStyleClassifier(
        len(tokenizer.vocabulary),
        11,
        d_model=384,
        layers=6,
        heads=6,
        dropout=0.1,
        max_length=2048,
        gradient_checkpointing=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    tokens = torch.randint(
        1, len(tokenizer.vocabulary), (32, 2048), dtype=torch.long, device=device
    )
    with autocast_context(device, "bf16"):
        logits = model(tokens, torch.ones_like(tokens, dtype=torch.bool))
        loss = torch.nn.functional.cross_entropy(
            logits, torch.arange(32, dtype=torch.long, device=device) % 11
        )
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach()), "micro_batch": 32, "tokens": 2048}


def transport_train(device: torch.device) -> dict[str, float | int | str]:
    model = _flow(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ema = ExponentialMovingAverage(model)
    latent = torch.randn(512, LATENT_TOKENS, LATENT_DIM, device=device)
    with sdpa_kernel_context(device, "math"):
        with autocast_context(device, "bf16"):
            loss = model.training_loss(latent, _condition(512, device))["loss"]
        loss.backward()
    optimizer.step()
    ema.update(model)
    return {
        "loss": float(loss.detach()),
        "micro_batch": 512,
        "latent_tokens": LATENT_TOKENS,
        "latent_dim": LATENT_DIM,
    }


def transport_inference(device: torch.device) -> dict[str, float | int | str]:
    model = _flow(device).eval()
    latent = torch.randn(1, LATENT_TOKENS, LATENT_DIM, device=device)
    source = _condition(1, device)
    target = ConditionBatch(source.dataset_id, source.task_id, source.style_id + 1)
    with torch.inference_mode():
        output = model.counterfactual(latent, source, target, num_steps=32)
    return {"output_elements": output.counterfactual_latent.numel(), "solver_steps": 32}


def cfm_abduction_train(device: torch.device) -> dict[str, float | int | str]:
    model = _flow(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    latent = torch.randn(64, LATENT_TOKENS, LATENT_DIM, device=device)
    condition = _condition(64, device)
    with sdpa_kernel_context(device, "math"):
        with autocast_context(device, "bf16"):
            base = model.training_loss(latent, condition)["loss"]
            noise = model.abduct(latent, condition, num_steps=4, track_grad=True)
            reconstruction = model.predict(noise, condition, num_steps=4, track_grad=True)
            loss = base + 0.1 * torch.nn.functional.mse_loss(reconstruction, latent)
        loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach()), "micro_batch": 64, "solver_steps": 4}


def ddim_abduction_train(device: torch.device) -> dict[str, float | int | str]:
    denoiser = ConditionalLatentDenoiser(
        latent_dim=LATENT_DIM,
        hidden_dim=512,
        layers=8,
        heads=8,
        mlp_ratio=4,
        dropout=0,
        condition_embedding=_embedding(),
        zero_init_output=False,
        gradient_checkpointing=False,
    ).to(device)
    model = ConditionalDDIM(
        denoiser,
        train_timesteps=1000,
        inversion_method="fixed_point",
        fpi_iterations=3,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    latent = torch.randn(64, LATENT_TOKENS, LATENT_DIM, device=device)
    condition = _condition(64, device)
    with sdpa_kernel_context(device, "math"):
        with autocast_context(device, "bf16"):
            base = model.training_loss(latent, condition)["loss"]
            noise = model.abduct(latent, condition, num_steps=4, track_grad=True)
            reconstruction = model.predict(noise, condition, num_steps=4, track_grad=True)
            loss = base + 0.1 * torch.nn.functional.mse_loss(reconstruction, latent)
        loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach()), "micro_batch": 64, "solver_steps": 4}


CASES: dict[str, Case] = {
    "codec_train": codec_train,
    "codec_encode": codec_encode,
    "codec_decode": codec_decode,
    "evaluator_train": evaluator_train,
    "transport_train": transport_train,
    "transport_inference": transport_inference,
    "cfm_abduction_train": cfm_abduction_train,
    "ddim_abduction_train": ddim_abduction_train,
}


def run_case(
    name: str, function: Case, device: torch.device, limit_gib: float
) -> dict[str, object]:
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak_memory(device)
    started = time.perf_counter()
    try:
        details = function(device)
        torch.cuda.synchronize(device)
        peak = peak_memory_gib(device)
        if peak > limit_gib:
            raise RuntimeError(f"{name} used {peak:.2f} GiB, above {limit_gib:.2f} GiB")
        return {
            "status": "ok",
            "peak_memory_gib": peak,
            "wall_time_seconds": time.perf_counter() - started,
            **details,
        }
    except torch.OutOfMemoryError as error:
        peak = peak_memory_gib(device)
        raise RuntimeError(f"{name} OOM at {peak:.2f} GiB") from error


def main() -> None:
    global CACHE_BATCH, CODEC_BATCH, CODEC_GRADIENT_CHECKPOINTING
    global CODEC_PROFILE, CODEC_STEPS, CODEC_TOKENS
    global LATENT_DIM, LATENT_TOKENS

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="all", help="Comma-separated case names or 'all'")
    parser.add_argument("--limit-gib", type=float, default=32.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--codec-batch", type=int, default=32)
    parser.add_argument("--codec-tokens", type=int, default=2560)
    parser.add_argument("--codec-steps", type=int, default=1)
    parser.add_argument("--cache-batch", type=int, default=384)
    parser.add_argument(
        "--codec-profile",
        choices=("transformer_vae", "drum_transformer_vae"),
        default="transformer_vae",
    )
    parser.add_argument(
        "--codec-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    arguments = parser.parse_args()
    if (
        arguments.codec_batch <= 0
        or arguments.codec_tokens <= 1
        or arguments.codec_steps <= 0
        or arguments.cache_batch <= 0
    ):
        raise ValueError("Codec and cache batch/token/step arguments must be positive")
    CODEC_BATCH = arguments.codec_batch
    CODEC_TOKENS = arguments.codec_tokens
    CODEC_PROFILE = arguments.codec_profile
    CODEC_STEPS = arguments.codec_steps
    CODEC_GRADIENT_CHECKPOINTING = arguments.codec_gradient_checkpointing
    CACHE_BATCH = arguments.cache_batch
    codec_config = OmegaConf.load(Path(CONFIG_DIR) / "codec" / f"{CODEC_PROFILE}.yaml")
    LATENT_TOKENS = int(codec_config.latent_tokens)
    LATENT_DIM = int(codec_config.latent_dim)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this script on a GPU node")
    device = torch.device("cuda:0")
    selected = list(CASES) if arguments.cases == "all" else arguments.cases.split(",")
    unknown = sorted(set(selected) - set(CASES))
    if unknown:
        raise ValueError(f"Unknown memory smoke cases: {unknown}")
    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "total_memory_gib": total_memory_gib(device),
        "limit_gib": arguments.limit_gib,
        "cases": {},
    }
    case_results = results["cases"]
    if not isinstance(case_results, dict):
        raise TypeError("Invalid results container")
    for name in selected:
        case_results[name] = run_case(name, CASES[name], device, arguments.limit_gib)
        print(json.dumps({name: case_results[name]}, indent=2), flush=True)
    serialized = json.dumps(results, indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
