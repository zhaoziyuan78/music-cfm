"""Stage-2 exogeneity fine-tuning with differentiable abduction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import cast

import torch
from torch import Tensor, nn

from cfmusic.distributed import (
    DistributedContext,
    decorrelate_worker_rng,
    distributed_barrier,
    distributed_model,
    set_data_epoch,
)
from cfmusic.logging import MetricLogger
from cfmusic.losses.hsic import normalized_hsic
from cfmusic.losses.prior_matching import classwise_prior_matching
from cfmusic.losses.roundtrip import roundtrip_loss
from cfmusic.memory import (
    autocast_context,
    peak_memory_gib,
    reset_peak_memory,
    sdpa_kernel_context,
)
from cfmusic.models.probes import FixedNoiseProjector
from cfmusic.progress import progress_bar
from cfmusic.training.checkpointing import load_checkpoint, save_rolling_checkpoint
from cfmusic.training.state import ExponentialMovingAverage, TrainState
from cfmusic.training.transport_trainer import (
    conditions_from_batch,
    contrasting_conditions,
    evaluate_condition_following,
)
from cfmusic.transport.base import ConditionalTransport


def regularization_scale(step: int, *, warmup_steps: int, ramp_steps: int) -> float:
    if step < warmup_steps:
        return 0.0
    return min(1.0, (step - warmup_steps + 1) / max(1, ramp_steps))


class AbductionLossModule(nn.Module):
    """Run the complete Stage-2 objective inside one DDP forward pass."""

    _projector: FixedNoiseProjector

    def __init__(self, transport: nn.Module, projector: FixedNoiseProjector) -> None:
        super().__init__()
        self.transport = transport
        # The projector is already registered below ``transport``.  A raw reference
        # avoids registering the same module twice in the DDP wrapper hierarchy.
        self.__dict__["_projector"] = projector

    def forward(
        self,
        latent: Tensor,
        condition: object,
        *,
        run_abduction: bool,
        inverse_steps: int,
        factorial_conditioning: bool,
        regularization: float,
        hsic_weight: float,
        prior_weight: float,
        roundtrip_weight: float,
        cosine_weight: float,
        negative_condition: object | None,
        condition_contrast_weight: float,
        condition_contrast_margin: float,
        condition_contrast_samples: int | None,
    ) -> dict[str, Tensor]:
        from cfmusic.conditioning.schema import ConditionBatch

        if not isinstance(condition, ConditionBatch):
            raise TypeError("Invalid Stage-2 condition batch")
        if negative_condition is not None and not isinstance(negative_condition, ConditionBatch):
            raise TypeError("Invalid Stage-2 negative condition batch")
        transport_api = cast(ConditionalTransport, self.transport)
        base_losses = transport_api.training_loss(
            latent,
            condition,
            negative_condition=negative_condition,
            condition_contrast_weight=condition_contrast_weight,
            condition_contrast_margin=condition_contrast_margin,
            condition_contrast_samples=condition_contrast_samples,
        )
        total = base_losses["loss"]
        hsic = latent.new_zeros(())
        prior = latent.new_zeros(())
        roundtrip = latent.new_zeros(())
        if run_abduction:
            noise = transport_api.abduct(
                latent, condition, num_steps=inverse_steps, track_grad=True
            )
            reconstruction = transport_api.predict(
                noise, condition, num_steps=inverse_steps, track_grad=True
            )
            projected = self._projector(noise)
            if (
                factorial_conditioning
                and condition.genre_id is not None
                and condition.emotion_id is not None
            ):
                hsic = normalized_hsic(projected, condition.genre_id) + normalized_hsic(
                    projected, condition.emotion_id
                )
                prior = 0.5 * (
                    classwise_prior_matching(projected, condition.genre_id)
                    + classwise_prior_matching(projected, condition.emotion_id)
                )
            else:
                hsic = normalized_hsic(projected, condition.style_id)
                prior = classwise_prior_matching(projected, condition.style_id)
            roundtrip = roundtrip_loss(reconstruction, latent, cosine_weight)
            total = total + regularization * (
                hsic_weight * hsic + prior_weight * prior + roundtrip_weight * roundtrip
            )
        results = {
            "loss": total,
            "base_loss": base_losses["loss"],
            "hsic": hsic,
            "prior_loss": prior,
            "roundtrip_loss": roundtrip,
        }
        for name in (
            "condition_contrast_loss",
            "condition_gap",
            "condition_accuracy",
            "condition_correct_error",
            "condition_wrong_error",
        ):
            results[name] = base_losses.get(name, latent.new_zeros(()))
        return results


def finetune_abduction_steps(
    transport: nn.Module,
    projector: FixedNoiseProjector,
    batches: Iterable[Mapping[str, Tensor | str | int]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    max_steps: int,
    abduction_interval: int,
    inverse_steps: int,
    hsic_weight: float,
    prior_weight: float,
    roundtrip_weight: float,
    cosine_weight: float,
    warmup_steps: int,
    ramp_steps: int,
    gradient_clip_norm: float,
    precision: str,
    sdpa_backend: str = "math",
    checkpoint_dir: Path,
    checkpoint_interval: int,
    config: Mapping[str, object],
    provenance: Mapping[str, str],
    factorial_conditioning: bool = False,
    condition_contrast_weight: float = 0.0,
    condition_contrast_margin: float = 0.0,
    condition_contrast_samples: int | None = None,
    condition_vocabularies: Mapping[str, Sequence[int]] | None = None,
    validation_batch: Mapping[str, Tensor | str | int] | None = None,
    validation_seed: int = 2026,
    ema_update_interval: int = 10,
    log_interval: int = 10,
    resume_from: Path | None = None,
    distributed: DistributedContext | None = None,
) -> TrainState:
    if abduction_interval <= 0:
        raise ValueError("abduction_interval must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if ema_update_interval <= 0 or log_interval <= 0:
        raise ValueError("EMA and log intervals must be positive")
    if condition_contrast_weight < 0 or condition_contrast_margin < 0:
        raise ValueError("Condition contrast weight and margin must be non-negative")
    if condition_contrast_weight > 0 and condition_vocabularies is None:
        raise ValueError("Condition contrast requires observed-label vocabularies")
    context = distributed or DistributedContext(0, 0, 1, device)
    state = TrainState()
    ema = ExponentialMovingAverage(transport)
    use_amp = precision in {"bf16", "fp16"} and device.type == "cuda"
    scaler = torch.GradScaler("cuda", enabled=use_amp and precision == "fp16")
    if resume_from is not None:
        state = load_checkpoint(
            resume_from,
            model=transport,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
        )
    logger = (
        MetricLogger(checkpoint_dir, append=resume_from is not None) if context.is_main else None
    )
    if state.world_size != context.world_size:
        if context.is_main and state.global_step:
            print(
                f"Resume world size changed from {state.world_size} to {context.world_size}; "
                "resetting the within-epoch data cursor"
            )
        state.batch_in_epoch = 0
    state.world_size = context.world_size
    if getattr(transport, "noise_projector", None) is not projector:
        raise ValueError("The Stage-2 projector must be registered on the transport")
    objective = AbductionLossModule(transport, projector)
    training_model = distributed_model(objective, context)
    decorrelate_worker_rng(context)
    transport.train()
    reset_peak_memory(device)
    progress = progress_bar(
        description="Fine-tune abduction",
        total=max_steps,
        initial=state.global_step,
        unit="step",
    )
    report_started = perf_counter()
    report_samples = 0
    report_steps = 0
    report_loss = torch.zeros((), device=device)
    report_base_loss = torch.zeros((), device=device)
    report_regularizers = {
        "hsic": torch.zeros((), device=device),
        "prior_loss": torch.zeros((), device=device),
        "roundtrip_loss": torch.zeros((), device=device),
    }
    report_conditions = {
        "condition_contrast_loss": torch.zeros((), device=device),
        "condition_gap": torch.zeros((), device=device),
        "condition_accuracy": torch.zeros((), device=device),
        "condition_correct_error": torch.zeros((), device=device),
        "condition_wrong_error": torch.zeros((), device=device),
    }
    report_abductions = 0
    while state.global_step < max_steps:
        set_data_epoch(batches, state.epoch)
        saw_batch = False
        completed_epoch = True
        for batch_index, batch in enumerate(batches):
            saw_batch = True
            if batch_index < state.batch_in_epoch:
                continue
            latent_value = batch["latent"]
            if not isinstance(latent_value, Tensor):
                raise TypeError("Abduction batch requires tensor latent")
            latent = latent_value.to(device, non_blocking=True)
            condition = conditions_from_batch(batch, device, factorial=factorial_conditioning)
            negative_condition = (
                contrasting_conditions(
                    condition,
                    condition_vocabularies,
                    factorial=factorial_conditioning,
                )
                if condition_contrast_weight > 0 and condition_vocabularies is not None
                else None
            )
            report_samples += latent.shape[0]
            run_abduction = state.global_step % abduction_interval == 0
            with sdpa_kernel_context(device, sdpa_backend):
                with autocast_context(device, precision):
                    losses = training_model(
                        latent,
                        condition,
                        run_abduction=run_abduction,
                        inverse_steps=inverse_steps,
                        factorial_conditioning=factorial_conditioning,
                        regularization=regularization_scale(
                            state.global_step, warmup_steps=warmup_steps, ramp_steps=ramp_steps
                        ),
                        hsic_weight=hsic_weight,
                        prior_weight=prior_weight,
                        roundtrip_weight=roundtrip_weight,
                        cosine_weight=cosine_weight,
                        negative_condition=negative_condition,
                        condition_contrast_weight=condition_contrast_weight,
                        condition_contrast_margin=condition_contrast_margin,
                        condition_contrast_samples=condition_contrast_samples,
                    )
                    total = losses["loss"]
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                transport.parameters(), gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                if context.is_main:
                    torch.save(batch, checkpoint_dir / "offending_batch.pt")
                raise FloatingPointError(
                    f"Non-finite abduction gradient at step {state.global_step}"
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if (state.global_step + 1) % ema_update_interval == 0:
                ema.update(transport, steps=ema_update_interval)
            state.global_step += 1
            state.batch_in_epoch = batch_index + 1
            report_steps += 1
            report_loss += total.detach()
            report_base_loss += losses["base_loss"].detach()
            for name in report_conditions:
                report_conditions[name] += losses[name].detach()
            if run_abduction:
                report_abductions += 1
                for name in report_regularizers:
                    report_regularizers[name] += losses[name].detach()
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
                    "loss": float(report_loss / report_steps),
                    "base_loss": float(report_base_loss / report_steps),
                    "hsic": float(report_regularizers["hsic"] / max(1, report_abductions)),
                    "prior_loss": float(
                        report_regularizers["prior_loss"] / max(1, report_abductions)
                    ),
                    "roundtrip_loss": float(
                        report_regularizers["roundtrip_loss"] / max(1, report_abductions)
                    ),
                    "condition_contrast_loss": float(
                        report_conditions["condition_contrast_loss"] / report_steps
                    ),
                    "condition_gap": float(report_conditions["condition_gap"] / report_steps),
                    "condition_accuracy": float(
                        report_conditions["condition_accuracy"] / report_steps
                    ),
                    "condition_correct_error": float(
                        report_conditions["condition_correct_error"] / report_steps
                    ),
                    "condition_wrong_error": float(
                        report_conditions["condition_wrong_error"] / report_steps
                    ),
                    "abduction_steps": report_abductions,
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "peak_gpu_memory_gib": peak_memory_gib(device),
                    "step_seconds": elapsed / report_steps,
                    "global_samples_per_second": report_samples * context.world_size / elapsed,
                }
                if (
                    context.is_main
                    and validation_batch is not None
                    and condition_vocabularies is not None
                    and state.global_step % checkpoint_interval == 0
                ):
                    metrics.update(
                        evaluate_condition_following(
                            transport,
                            validation_batch,
                            device=device,
                            precision=precision,
                            sdpa_backend=sdpa_backend,
                            factorial_conditioning=factorial_conditioning,
                            condition_vocabularies=condition_vocabularies,
                            condition_contrast_margin=condition_contrast_margin,
                            condition_contrast_samples=condition_contrast_samples,
                            seed=validation_seed,
                        )
                    )
                if logger is not None:
                    logger.log(metrics)
                progress.set_postfix(
                    epoch=state.epoch,
                    loss=f"{metrics['loss']:.4f}",
                    rate=f"{metrics['global_samples_per_second']:.0f}sample/s",
                    lr=f"{metrics['learning_rate']:.2e}",
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
                report_loss.zero_()
                report_base_loss.zero_()
                for value in report_regularizers.values():
                    value.zero_()
                for value in report_conditions.values():
                    value.zero_()
                report_abductions = 0
            if state.global_step % checkpoint_interval == 0 or state.global_step == max_steps:
                if context.is_main:
                    save_rolling_checkpoint(
                        checkpoint_dir,
                        model=transport,
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
            raise RuntimeError("Abduction DataLoader is empty")
        if completed_epoch:
            state.epoch += 1
            state.batch_in_epoch = 0
    progress.close()
    if logger is not None:
        logger.close()
    return state
