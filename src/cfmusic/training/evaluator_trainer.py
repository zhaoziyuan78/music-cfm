"""Training loop for token Transformer style evaluators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor, nn

from cfmusic.distributed import (
    DistributedContext,
    decorrelate_worker_rng,
    distributed_barrier,
    distributed_model,
    maybe_no_sync,
    set_data_epoch,
)
from cfmusic.logging import MetricLogger
from cfmusic.memory import autocast_context, peak_memory_gib, reset_peak_memory
from cfmusic.progress import progress_bar
from cfmusic.training.checkpointing import load_checkpoint, save_rolling_checkpoint
from cfmusic.training.optim import create_adamw
from cfmusic.training.state import TrainState


def train_token_evaluator(
    model: nn.Module,
    batches: Iterable[Mapping[str, Tensor | list[str]]],
    *,
    device: torch.device,
    max_steps: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_dir: Path,
    checkpoint_interval: int,
    config: Mapping[str, object],
    provenance: Mapping[str, str],
    gradient_accumulation: int = 1,
    precision: str = "fp32",
    log_interval: int = 10,
    resume_from: Path | None = None,
    distributed: DistributedContext | None = None,
) -> TrainState:
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    context = distributed or DistributedContext(0, 0, 1, device)
    optimizer = create_adamw(model, learning_rate=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    use_fp16_scaler = precision == "fp16" and device.type == "cuda"
    scaler = torch.GradScaler("cuda", enabled=use_fp16_scaler)
    state = TrainState()
    if resume_from is not None:
        state = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
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
    logger = (
        MetricLogger(checkpoint_dir, append=resume_from is not None) if context.is_main else None
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    reset_peak_memory(device)
    progress = progress_bar(
        description="Train evaluator",
        total=max_steps,
        initial=state.global_step,
        unit="step",
    )
    report_started = perf_counter()
    report_samples = 0
    report_steps = 0
    while state.global_step < max_steps:
        set_data_epoch(batches, state.epoch)
        saw_batch = False
        completed_epoch = True
        batch_count = len(batches) if isinstance(batches, Sized) else None
        for batch_index, batch in enumerate(batches):
            saw_batch = True
            if batch_index < state.batch_in_epoch:
                continue
            tokens, mask, labels = batch["tokens"], batch["attention_mask"], batch["style_id"]
            if (
                not isinstance(tokens, Tensor)
                or not isinstance(mask, Tensor)
                or not isinstance(labels, Tensor)
            ):
                raise TypeError("Invalid evaluator batch")
            maximum_sequence_length = tokens.shape[1]
            report_samples += tokens.shape[0]
            position = getattr(model, "position", None)
            if isinstance(position, Tensor) and maximum_sequence_length > position.shape[0]:
                raise ValueError(
                    "An evaluator batch exceeds the positional embedding limit on at least "
                    f"one rank: {maximum_sequence_length} > {position.shape[0]}"
                )
            last_batch = batch_count is not None and batch_index + 1 == batch_count
            synchronize = (batch_index + 1) % gradient_accumulation == 0 or last_batch
            with maybe_no_sync(training_model, synchronize=synchronize):
                with autocast_context(device, precision):
                    logits = training_model(
                        tokens.to(device, non_blocking=True),
                        mask.to(device, non_blocking=True),
                    )
                    raw_loss = torch.nn.functional.cross_entropy(
                        logits, labels.to(device, non_blocking=True)
                    )
                    loss = raw_loss / gradient_accumulation
                scaler.scale(loss).backward()
            if not synchronize:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                if context.is_main:
                    torch.save(batch, checkpoint_dir / "offending_batch.pt")
                raise FloatingPointError(
                    f"Non-finite evaluator gradient at step {state.global_step}"
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            state.global_step += 1
            state.batch_in_epoch = batch_index + 1
            report_steps += 1
            progress.update(1)
            report = (
                state.global_step == 1
                or state.global_step % log_interval == 0
                or state.global_step == max_steps
            )
            if report:
                elapsed = max(perf_counter() - report_started, 1e-9)
                metrics = {
                    "step": state.global_step,
                    "loss": float(raw_loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "peak_gpu_memory_gib": peak_memory_gib(device),
                    "sequence_length": maximum_sequence_length,
                    "step_seconds": elapsed / report_steps,
                    "global_samples_per_second": report_samples * context.world_size / elapsed,
                }
                if logger is not None:
                    logger.log(metrics)
                progress.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    rate=f"{metrics['global_samples_per_second']:.0f}sample/s",
                    seq=maximum_sequence_length,
                    gpu=(
                        f"{metrics['peak_gpu_memory_gib']:.1f}GiB"
                        if device.type == "cuda"
                        else "cpu"
                    ),
                    refresh=False,
                )
                report_started = perf_counter()
                report_samples = 0
                report_steps = 0
            if state.global_step % checkpoint_interval == 0 or state.global_step == max_steps:
                if context.is_main:
                    save_rolling_checkpoint(
                        checkpoint_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        state=state,
                        ema=None,
                        config=config,
                        provenance=provenance,
                    )
                distributed_barrier(context)
            if state.global_step >= max_steps:
                completed_epoch = False
                break
        if not saw_batch:
            raise RuntimeError("Evaluator DataLoader is empty")
        if completed_epoch:
            state.epoch += 1
            state.batch_in_epoch = 0
    progress.close()
    if logger is not None:
        logger.close()
    return state
