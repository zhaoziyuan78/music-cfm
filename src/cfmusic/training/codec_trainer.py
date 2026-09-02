"""Native PyTorch VAE training loop."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor

from cfmusic.codec.losses import kl_beta, vae_loss
from cfmusic.codec.transformer_vae import TransformerVAE
from cfmusic.distributed import (
    DistributedContext,
    decorrelate_worker_rng,
    distributed_barrier,
    distributed_max_int,
    distributed_model,
    maybe_no_sync,
    set_data_epoch,
)
from cfmusic.evaluation.reconstruction import (
    aligned_token_accuracy,
    multiset_f1,
    symbolic_note_events,
    trim_token_sequence,
)
from cfmusic.logging import MetricLogger
from cfmusic.memory import autocast_context, peak_memory_gib, reset_peak_memory
from cfmusic.progress import progress_bar, track
from cfmusic.training.checkpointing import load_checkpoint, save_rolling_checkpoint
from cfmusic.training.state import ExponentialMovingAverage, TrainState


def evaluate_codec_batches(
    model: TransformerVAE,
    batches: Iterable[Mapping[str, Tensor | list[str]]],
    *,
    device: torch.device,
    precision: str,
    generation_samples: int = 0,
    generation_max_length: int | None = None,
) -> dict[str, float]:
    """Measure validation reconstruction and whether the decoder uses its latent."""

    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_shuffled_nll = 0.0
    total_correct = 0
    total_tokens = 0
    posterior_means: list[Tensor] = []
    generation_references: list[Tensor] = []
    generation_latents: list[Tensor] = []
    generation_bars: list[int] = []
    progress = track(
        batches,
        description="Validate codec",
        total=len(batches) if isinstance(batches, Sized) else None,
        unit="batch",
        leave=False,
    )
    try:
        with torch.inference_mode():
            for batch in progress:
                tokens_value = batch["tokens"]
                mask_value = batch["attention_mask"]
                if not isinstance(tokens_value, Tensor) or not isinstance(mask_value, Tensor):
                    raise TypeError("Codec validation requires tensor tokens and attention_mask")
                tokens = tokens_value.to(device, non_blocking=True)
                attention_mask = mask_value.to(device, non_blocking=True)
                with autocast_context(device, precision):
                    posterior = model.encode_distribution(tokens, attention_mask)
                    logits = model.decode_teacher_forced(tokens[:, :-1], posterior.mean)
                    shuffled_logits = model.decode_teacher_forced(
                        tokens[:, :-1], posterior.mean.roll(1, dims=0)
                    )
                targets = tokens[:, 1:]
                valid = targets.ne(model.pad_id)
                total_nll += float(
                    functional.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        ignore_index=model.pad_id,
                        reduction="sum",
                    )
                )
                total_shuffled_nll += float(
                    functional.cross_entropy(
                        shuffled_logits.float().reshape(-1, shuffled_logits.shape[-1]),
                        targets.reshape(-1),
                        ignore_index=model.pad_id,
                        reduction="sum",
                    )
                )
                total_correct += int((logits.argmax(-1).eq(targets) & valid).sum())
                total_tokens += int(valid.sum())
                posterior_means.append(posterior.mean.float().cpu())
                remaining = generation_samples - len(generation_references)
                if remaining > 0:
                    selected = min(remaining, tokens.shape[0])
                    generation_references.extend(tokens[:selected].detach().cpu())
                    generation_latents.extend(posterior.mean[:selected].detach().float().cpu())
                    bars_value = batch.get("num_bars")
                    if isinstance(bars_value, Tensor):
                        generation_bars.extend(
                            int(value) for value in bars_value[:selected].tolist()
                        )
                progress.set_postfix(tokens=total_tokens, refresh=False)
    finally:
        model.train(was_training)
    means = torch.cat(posterior_means)
    token_ce = total_nll / max(1, total_tokens)
    shuffled_ce = total_shuffled_nll / max(1, total_tokens)
    metrics = {
        "token_ce": token_ce,
        "teacher_forced_token_accuracy": total_correct / max(1, total_tokens),
        "shuffled_latent_token_ce": shuffled_ce,
        "shuffled_latent_ce_increase": shuffled_ce - token_ce,
        "posterior_mean_variance": float(means.var(dim=0, unbiased=False).mean()),
        "active_latent_dimensions": float(means.var(dim=0, unbiased=False).gt(1e-2).sum()),
        "evaluated_samples": float(means.shape[0]),
        "evaluated_tokens": float(total_tokens),
    }
    if generation_references:
        model.eval()
        max_length = generation_max_length or model.max_sequence_length
        generated_by_index: list[Tensor | None] = [None] * len(generation_references)
        groups: dict[int | None, list[int]] = defaultdict(list)
        for index in range(len(generation_references)):
            bars = generation_bars[index] if index < len(generation_bars) else None
            groups[bars].append(index)
        try:
            with torch.inference_mode(), autocast_context(device, precision):
                for bars, indices in groups.items():
                    generated_group = model.generate(
                        torch.stack([generation_latents[index] for index in indices]).to(device),
                        strategy="greedy",
                        max_length=min(max_length, model.max_sequence_length),
                        min_bars=bars,
                        max_bars=bars,
                        show_progress=True,
                        progress_description="Validate autoregressive reconstruction",
                    ).cpu()
                    for index, generated in zip(indices, generated_group, strict=True):
                        generated_by_index[index] = generated
        finally:
            model.train(was_training)
        token_accuracies: list[float] = []
        note_f1s: list[float] = []
        exact_sequences = 0
        if model.grammar is None:
            raise RuntimeError("Autoregressive codec validation requires an event grammar")
        for reference_tensor, generated_tensor in zip(
            generation_references, generated_by_index, strict=True
        ):
            if generated_tensor is None:
                raise RuntimeError("Missing an autoregressive validation result")
            reference = trim_token_sequence(
                reference_tensor.tolist(), eos_id=model.eos_id, pad_id=model.pad_id
            )
            prediction = trim_token_sequence(
                generated_tensor.tolist(), eos_id=model.eos_id, pad_id=model.pad_id
            )
            token_accuracies.append(aligned_token_accuracy(reference, prediction))
            note_f1s.append(
                multiset_f1(
                    symbolic_note_events(model.grammar.vocabulary, reference),
                    symbolic_note_events(model.grammar.vocabulary, prediction),
                )
            )
            exact_sequences += int(reference == prediction)
        metrics.update(
            {
                "autoregressive_samples": float(len(generation_references)),
                "autoregressive_token_accuracy": sum(token_accuracies) / len(token_accuracies),
                "autoregressive_note_event_f1": sum(note_f1s) / len(note_f1s),
                "autoregressive_exact_sequence_rate": exact_sequences / len(generation_references),
            }
        )
    return metrics


def train_codec_steps(
    model: TransformerVAE,
    batches: Iterable[Mapping[str, Tensor | list[str]]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    max_steps: int,
    gradient_accumulation: int,
    gradient_clip_norm: float,
    precision: str,
    warmup_steps: int,
    beta_max: float,
    free_bits_per_dim: float,
    checkpoint_dir: Path,
    checkpoint_interval: int,
    config: Mapping[str, object],
    provenance: Mapping[str, str],
    ema_decay: float | None = 0.9999,
    start_state: TrainState | None = None,
    resume_from: Path | None = None,
    distributed: DistributedContext | None = None,
    validation_batches: Iterable[Mapping[str, Tensor | list[str]]] | None = None,
    validation_interval: int | None = None,
    validation_generation_samples: int = 0,
    validation_generation_max_length: int | None = None,
) -> TrainState:
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    context = distributed or DistributedContext(0, 0, 1, device)
    state = start_state or TrainState()
    model.train()
    ema = ExponentialMovingAverage(model, ema_decay) if ema_decay is not None else None
    use_amp = precision in {"bf16", "fp16"} and device.type == "cuda"
    scaler = torch.GradScaler("cuda", enabled=use_amp and precision == "fp16")
    if resume_from is not None:
        state = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
        )
    append_metrics = resume_from is not None or state.global_step > 0
    logger = MetricLogger(checkpoint_dir, append=append_metrics) if context.is_main else None
    validation_logger = (
        MetricLogger(checkpoint_dir / "validation", append=append_metrics, curve_interval=1)
        if context.is_main and validation_batches is not None
        else None
    )
    if state.world_size != context.world_size:
        if context.is_main and state.global_step:
            print(
                f"Resume world size changed from {state.world_size} to {context.world_size}; "
                "resetting the within-epoch data cursor"
            )
        state.batch_in_epoch = 0
    state.world_size = context.world_size
    training_model = distributed_model(model, context)
    decorrelate_worker_rng(context)
    optimizer.zero_grad(set_to_none=True)
    reset_peak_memory(device)
    started = time.perf_counter()
    progress = progress_bar(
        description="Train codec",
        total=max_steps,
        initial=state.global_step,
        unit="step",
    )
    while state.global_step < max_steps:
        set_data_epoch(batches, state.epoch)
        saw_batch = False
        completed_epoch = True
        batch_count = len(batches) if isinstance(batches, Sized) else None
        for batch_index, batch in enumerate(batches):
            saw_batch = True
            if batch_index < state.batch_in_epoch:
                continue
            tokens_value = batch["tokens"]
            mask_value = batch["attention_mask"]
            if not isinstance(tokens_value, Tensor) or not isinstance(mask_value, Tensor):
                raise TypeError("Codec batches require tensor tokens and attention_mask")
            tokens = tokens_value.to(device, non_blocking=True)
            attention_mask = mask_value.to(device, non_blocking=True)
            maximum_sequence_length = distributed_max_int(tokens.shape[1], context)
            model_limit = getattr(model, "max_sequence_length", None)
            if isinstance(model_limit, int) and maximum_sequence_length > model_limit:
                raise ValueError(
                    "A codec batch exceeds the model positional embedding limit on at least "
                    f"one rank: {maximum_sequence_length} > {model_limit}. Ensure the tokenizer "
                    "is capped to codec.max_sequence_length."
                )
            last_batch = batch_count is not None and batch_index + 1 == batch_count
            synchronize = (batch_index + 1) % gradient_accumulation == 0 or last_batch
            try:
                with maybe_no_sync(training_model, synchronize=synchronize):
                    with autocast_context(device, precision):
                        logits, posterior = training_model(tokens, attention_mask)
                        beta = kl_beta(
                            state.global_step, warmup_steps=warmup_steps, beta_max=beta_max
                        )
                        losses = vae_loss(
                            logits,
                            tokens[:, 1:],
                            posterior,
                            pad_id=model.pad_id,
                            beta=beta,
                            free_bits_per_dim=free_bits_per_dim,
                        )
                        loss = losses["loss"] / gradient_accumulation
                    scaler.scale(loss).backward()
            except torch.OutOfMemoryError as error:
                raise RuntimeError(
                    "Codec CUDA OOM at "
                    f"micro_batch={tokens.shape[0]}, sequence_length={tokens.shape[1]}. "
                    "Reduce codec.training.batch_size and increase gradient_accumulation "
                    "by the same factor to preserve the effective batch."
                ) from error
            if not synchronize:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            if not torch.isfinite(gradient_norm):
                if context.is_main:
                    torch.save(batch, checkpoint_dir / "offending_batch.pt")
                raise FloatingPointError(f"Non-finite codec gradient at step {state.global_step}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema:
                ema.update(model)
            state.global_step += 1
            state.batch_in_epoch = batch_index + 1
            elapsed = max(time.perf_counter() - started, 1e-6)
            if logger is not None:
                logger.log(
                    {
                        "step": state.global_step,
                        "loss": float(losses["loss"].detach()),
                        "token_ce": float(losses["token_ce"].detach()),
                        "kl": float(losses["raw_kl"].detach()),
                        "active_units": float(losses["active_units"].detach()),
                        "gradient_norm": float(gradient_norm),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "steps_per_second": max(1, progress.n + 1) / elapsed,
                        "peak_gpu_memory_gib": peak_memory_gib(device),
                        "sequence_length": maximum_sequence_length,
                    }
                )
            progress.update(1)
            progress.set_postfix(
                epoch=state.epoch,
                loss=f"{float(losses['loss'].detach()):.4f}",
                ce=f"{float(losses['token_ce'].detach()):.4f}",
                kl=f"{float(losses['raw_kl'].detach()):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                seq=maximum_sequence_length,
                gpu=f"{peak_memory_gib(device):.1f}GiB" if device.type == "cuda" else "cpu",
                refresh=False,
            )
            should_validate = validation_interval is not None and (
                state.global_step % validation_interval == 0 or state.global_step == max_steps
            )
            if should_validate:
                if context.is_main:
                    if validation_batches is None or validation_logger is None:
                        raise RuntimeError("Codec validation interval requires validation batches")
                    validation_metrics = evaluate_codec_batches(
                        model,
                        validation_batches,
                        device=device,
                        precision=precision,
                        generation_samples=validation_generation_samples,
                        generation_max_length=validation_generation_max_length,
                    )
                    state.best_validation_loss = min(
                        state.best_validation_loss, validation_metrics["token_ce"]
                    )
                    autoregressive_f1 = validation_metrics.get("autoregressive_note_event_f1")
                    if autoregressive_f1 is not None:
                        state.best_roundtrip = min(state.best_roundtrip, 1.0 - autoregressive_f1)
                    validation_logger.log({"step": state.global_step, **validation_metrics})
                    print(
                        "Codec validation: "
                        f"step={state.global_step}, ce={validation_metrics['token_ce']:.4f}, "
                        "accuracy="
                        f"{validation_metrics['teacher_forced_token_accuracy']:.4f}, "
                        "autoregressive_note_f1="
                        f"{validation_metrics.get('autoregressive_note_event_f1', float('nan')):.4f}, "
                        "shuffled_latent_ce_increase="
                        f"{validation_metrics['shuffled_latent_ce_increase']:.4f}"
                    )
                distributed_barrier(context)
            if state.global_step % checkpoint_interval == 0 or state.global_step == max_steps:
                if context.is_main:
                    save_rolling_checkpoint(
                        checkpoint_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        state=state,
                        ema=ema,
                        config=config,
                        provenance=provenance,
                    )
                distributed_barrier(context)
            if state.global_step >= max_steps:
                completed_epoch = False
                break
        if not saw_batch:
            raise RuntimeError("Codec DataLoader is empty")
        if completed_epoch:
            state.epoch += 1
            state.batch_in_epoch = 0
    progress.close()
    if logger is not None:
        logger.close()
    if validation_logger is not None:
        validation_logger.close()
    return state
