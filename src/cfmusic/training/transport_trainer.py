"""Native AMP transport training loop for CFM and DDIM."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence, Sized
from pathlib import Path
from time import perf_counter
from typing import cast

import torch
from torch import Tensor, nn

from cfmusic.conditioning.schema import ConditionBatch, build_condition_batch
from cfmusic.distributed import (
    DistributedContext,
    decorrelate_worker_rng,
    distributed_barrier,
    distributed_model,
    maybe_no_sync,
    set_data_epoch,
)
from cfmusic.latent.dataset import LatentDataset
from cfmusic.logging import MetricLogger
from cfmusic.losses.mmd import class_conditional_mmd
from cfmusic.losses.sliced_wasserstein import class_conditional_sliced_wasserstein
from cfmusic.memory import (
    autocast_context,
    peak_memory_gib,
    reset_peak_memory,
    sdpa_kernel_context,
)
from cfmusic.progress import progress_bar, track
from cfmusic.training.checkpointing import load_checkpoint, save_rolling_checkpoint
from cfmusic.training.state import ExponentialMovingAverage, TrainState
from cfmusic.transport.base import ConditionalTransport


class TransportLossModule(nn.Module):
    """Expose the transport loss through ``forward`` so native DDP owns backward hooks."""

    def __init__(self, transport: nn.Module) -> None:
        super().__init__()
        self.transport = transport

    def forward(
        self,
        latent: Tensor,
        condition: ConditionBatch,
        negative_condition: ConditionBatch | None,
        sample_weight: Tensor | None,
        *,
        condition_contrast_weight: float,
        condition_contrast_margin: float,
        condition_contrast_samples: int | None,
    ) -> dict[str, Tensor]:
        return cast(ConditionalTransport, self.transport).training_loss(
            latent,
            condition,
            negative_condition=negative_condition,
            condition_contrast_weight=condition_contrast_weight,
            condition_contrast_margin=condition_contrast_margin,
            condition_contrast_samples=condition_contrast_samples,
            sample_weight=sample_weight,
        )


def _different_labels(values: Tensor, vocabulary: Sequence[int]) -> Tensor:
    """Sample a valid label other than each observed label."""

    labels = torch.as_tensor(tuple(vocabulary), dtype=torch.long, device=values.device)
    if labels.numel() < 2:
        raise ValueError("Condition contrast requires at least two observed labels")
    matches = values[:, None] == labels[None]
    if not bool(matches.any(dim=1).all()):
        unknown = torch.unique(values[~matches.any(dim=1)]).tolist()
        raise ValueError(f"Condition vocabulary is missing observed labels: {unknown}")
    positions = matches.to(torch.int64).argmax(dim=1)
    offsets = torch.randint(1, labels.numel(), (values.shape[0],), device=values.device)
    return labels[(positions + offsets) % labels.numel()]


def contrasting_conditions(
    condition: ConditionBatch,
    vocabularies: Mapping[str, Sequence[int]],
    *,
    factorial: bool,
    active_axis: str | None = None,
) -> ConditionBatch:
    """Construct wrong, but in-support, labels for condition-discrimination training."""

    if factorial and condition.genre_id is not None and condition.emotion_id is not None:
        axis = active_axis or "genre"
        if axis not in {"genre", "emotion"}:
            raise ValueError("Factorial wrong condition must select exactly one active axis")
        genres = (
            _different_labels(condition.genre_id, vocabularies["genre_id"])
            if axis == "genre"
            else condition.genre_id
        )
        emotions = (
            _different_labels(condition.emotion_id, vocabularies["emotion_id"])
            if axis == "emotion"
            else condition.emotion_id
        )
        return ConditionBatch(
            condition.dataset_id,
            condition.task_id,
            condition.style_id,
            genres,
            emotions,
        )
    return ConditionBatch(
        condition.dataset_id,
        condition.task_id,
        _different_labels(condition.style_id, vocabularies["style_id"]),
        condition.genre_id,
        condition.emotion_id,
    )


def shifted_conditions(
    condition: ConditionBatch,
    vocabularies: Mapping[str, Sequence[int]],
    *,
    offset: int,
    factorial: bool,
    active_axis: str | None = None,
) -> ConditionBatch:
    """Deterministically shift labels for reproducible held-out comparisons."""

    def shift(values: Tensor, name: str) -> Tensor:
        labels = torch.as_tensor(tuple(vocabularies[name]), device=values.device)
        if labels.numel() < 2:
            raise ValueError("Condition validation requires at least two observed labels")
        matches = values[:, None] == labels[None]
        if not bool(matches.any(dim=1).all()):
            raise ValueError(f"Condition validation vocabulary is missing {name} labels")
        positions = matches.to(torch.int64).argmax(dim=1)
        return labels[(positions + offset) % labels.numel()]

    if factorial and condition.genre_id is not None and condition.emotion_id is not None:
        axis = active_axis or "genre"
        if axis not in {"genre", "emotion"}:
            raise ValueError("Factorial validation must select exactly one active axis")
        return ConditionBatch(
            condition.dataset_id,
            condition.task_id,
            condition.style_id,
            shift(condition.genre_id, "genre_id") if axis == "genre" else condition.genre_id,
            shift(condition.emotion_id, "emotion_id")
            if axis == "emotion"
            else condition.emotion_id,
        )
    return ConditionBatch(
        condition.dataset_id,
        condition.task_id,
        shift(condition.style_id, "style_id"),
        condition.genre_id,
        condition.emotion_id,
    )


def inverse_frequency_weights(labels: Sequence[int], *, exponent: float) -> dict[int, float]:
    """Return weights with mean one under the empirical label distribution."""

    if not 0.0 <= exponent <= 1.0:
        raise ValueError("Class-balance exponent must be in [0, 1]")
    counts = Counter(int(label) for label in labels)
    if not counts:
        raise ValueError("Class balancing requires at least one label")
    raw = {label: count ** (-exponent) for label, count in counts.items()}
    mean = sum(counts[label] * raw[label] for label in counts) / len(labels)
    return {label: weight / mean for label, weight in raw.items()}


def heldout_condition_batch(
    dataset: LatentDataset,
    *,
    samples_per_style: int,
    task: str = "genre",
    factorial: bool = False,
    active_axis: str | None = None,
) -> dict[str, Tensor]:
    """Load a small balanced probe while touching only one shard per style."""

    if samples_per_style <= 0:
        raise ValueError("validation_samples_per_style must be positive")
    requested_column = f"{active_axis or 'genre'}_id" if factorial else f"{task}_id"
    balance_column = requested_column if requested_column in dataset.frame else "style_id"
    if balance_column not in dataset.frame:
        raise ValueError(f"Validation cache is missing condition column {balance_column!r}")
    selected: list[int] = []
    for _style, group in dataset.frame.groupby(balance_column, sort=True):
        shard = str(group["shard"].astype(str).value_counts().index[0])
        local = group.loc[group["shard"].astype(str) == shard]
        if "sample_id" in local:
            local = local.loc[~local["sample_id"].astype(str).duplicated()]
        selected.extend(int(index) for index in local.index[:samples_per_style])
    items = [
        dataset[index]
        for index in track(
            selected,
            description="Load held-out condition probe",
            total=len(selected),
            unit="latent",
            leave=False,
        )
    ]
    latents: list[Tensor] = []
    for item in items:
        latent = item["latent"]
        if not isinstance(latent, Tensor):
            raise TypeError("Held-out condition probe requires tensor latents")
        latents.append(latent)
    batch = {
        "latent": torch.stack(latents),
        "style_id": torch.tensor([int(item["style_id"]) for item in items]),
        "dataset_id": torch.tensor([int(item["dataset_id"]) for item in items]),
    }
    for column in ("genre_id", "emotion_id"):
        if all(column in item for item in items):
            batch[column] = torch.tensor([int(item[column]) for item in items])
    return batch


@torch.no_grad()
def evaluate_condition_following(
    transport: nn.Module,
    batch: Mapping[str, Tensor | str | int],
    *,
    device: torch.device,
    precision: str,
    sdpa_backend: str,
    factorial_conditioning: bool,
    condition_task: str,
    active_axis: str | None,
    condition_vocabularies: Mapping[str, Sequence[int]],
    condition_contrast_margin: float,
    condition_contrast_samples: int | None,
    seed: int,
) -> dict[str, float]:
    """Evaluate all in-support wrong labels on a fixed held-out latent batch."""

    latent_value = batch["latent"]
    if not isinstance(latent_value, Tensor):
        raise TypeError("Condition validation batch requires tensor latent")
    latent = latent_value.to(device, non_blocking=True)
    condition = conditions_from_batch(
        batch, device, task=condition_task, factorial=factorial_conditioning
    )
    if factorial_conditioning:
        axis = active_axis or "genre"
        comparison_count = len(condition_vocabularies[f"{axis}_id"]) - 1
    else:
        comparison_count = len(condition_vocabularies["style_id"]) - 1
    if comparison_count <= 0:
        raise ValueError("Condition validation requires multiple observed labels")

    metric_names = (
        "condition_gap",
        "condition_accuracy",
        "condition_correct_error",
        "condition_wrong_error",
        "condition_contrast_loss",
    )
    totals = {name: 0.0 for name in metric_names}
    was_training = transport.training
    transport.eval()
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        for offset in range(1, comparison_count + 1):
            negative = shifted_conditions(
                condition,
                condition_vocabularies,
                offset=offset,
                factorial=factorial_conditioning,
                active_axis=active_axis,
            )
            # Every wrong-label comparison sees exactly the same random path.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                with (
                    sdpa_kernel_context(device, sdpa_backend),
                    autocast_context(device, precision),
                ):
                    losses = cast(ConditionalTransport, transport).training_loss(
                        latent,
                        condition,
                        negative_condition=negative,
                        condition_contrast_weight=1.0,
                        condition_contrast_margin=condition_contrast_margin,
                        condition_contrast_samples=condition_contrast_samples,
                    )
            for name in metric_names:
                totals[name] += float(losses[name])
    finally:
        transport.train(was_training)
    return {f"validation_{name}": value / comparison_count for name, value in totals.items()}


@torch.no_grad()
def evaluate_endpoint_matching(
    transport: nn.Module,
    batch: Mapping[str, Tensor | str | int],
    *,
    device: torch.device,
    precision: str,
    sdpa_backend: str,
    factorial_conditioning: bool,
    condition_task: str,
    active_axis: str | None,
    num_steps: int,
    seed: int,
) -> dict[str, float]:
    """Compare true conditional endpoints, without constructing invalid flow pairs."""

    latent_value = batch["latent"]
    if not isinstance(latent_value, Tensor):
        raise TypeError("Endpoint validation batch requires tensor latent")
    latent = latent_value.to(device, non_blocking=True)
    condition = conditions_from_batch(
        batch, device, task=condition_task, factorial=factorial_conditioning
    )
    if factorial_conditioning:
        axis = active_axis or "genre"
        label = condition.genre_id if axis == "genre" else condition.emotion_id
        if label is None:
            raise ValueError(f"Factorial endpoint validation has no {axis} labels")
    else:
        label = condition.style_id
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)
    was_training = transport.training
    transport.eval()
    try:
        with sdpa_kernel_context(device, sdpa_backend), autocast_context(device, precision):
            generated = cast(ConditionalTransport, transport).predict(
                noise, condition, num_steps=num_steps
            )
        flat_generated = generated.float().flatten(1)
        flat_factual = latent.float().flatten(1)
        mmd = class_conditional_mmd(flat_generated, flat_factual, label)
        swd = class_conditional_sliced_wasserstein(
            flat_generated, flat_factual, label, num_projections=32, seed=seed + 1
        )
    finally:
        transport.train(was_training)
    return {
        "endpoint_mmd": float(mmd),
        "endpoint_swd": float(swd),
    }


def evaluate_raw_and_ema(
    transport: nn.Module,
    ema: ExponentialMovingAverage | None,
    evaluator: object,
) -> dict[str, float]:
    """Run a zero-argument evaluator for both weight variants."""

    if not callable(evaluator):
        raise TypeError("Validation evaluator must be callable")
    raw = evaluator()
    if not isinstance(raw, Mapping):
        raise TypeError("Validation evaluator must return a mapping")
    metrics = {f"validation_raw_{name}": float(value) for name, value in raw.items()}
    if ema is not None:
        with ema.average_parameters(transport):
            averaged = evaluator()
        if not isinstance(averaged, Mapping):
            raise TypeError("Validation evaluator must return a mapping")
        metrics.update({f"validation_ema_{name}": float(value) for name, value in averaged.items()})
    return metrics


def conditions_from_batch(
    batch: Mapping[str, Tensor | str | int],
    device: torch.device,
    *,
    task: str = "genre",
    factorial: bool = False,
) -> ConditionBatch:
    return build_condition_batch(batch, device, task=task, factorial=factorial)


def train_transport_steps(
    transport: nn.Module,
    batches: Iterable[Mapping[str, Tensor | str | int]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    max_steps: int,
    gradient_accumulation: int,
    gradient_clip_norm: float,
    precision: str,
    sdpa_backend: str = "math",
    checkpoint_dir: Path,
    checkpoint_interval: int,
    config: Mapping[str, object],
    provenance: Mapping[str, str],
    ema_decay: float | None = 0.9999,
    ema_update_interval: int = 10,
    log_interval: int = 10,
    find_unused_parameters: bool = False,
    factorial_conditioning: bool = False,
    condition_task: str = "genre",
    factorial_active_axis: str | None = None,
    condition_contrast_weight: float = 0.0,
    condition_contrast_margin: float = 0.0,
    condition_contrast_samples: int | None = None,
    condition_vocabularies: Mapping[str, Sequence[int]] | None = None,
    style_loss_weights: Mapping[int, float] | None = None,
    validation_batch: Mapping[str, Tensor | str | int] | None = None,
    validation_seed: int = 2026,
    validation_solver_steps: int = 8,
    resume_from: Path | None = None,
    distributed: DistributedContext | None = None,
) -> TrainState:
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
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
    ema = ExponentialMovingAverage(transport, ema_decay) if ema_decay else None
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
    loss_module = TransportLossModule(transport)
    training_model = distributed_model(
        loss_module, context, find_unused_parameters=find_unused_parameters
    )
    style_weight_lookup: Tensor | None = None
    if style_loss_weights:
        maximum_style = max(style_loss_weights)
        style_weight_lookup = torch.ones(maximum_style + 1, device=device)
        for style, weight in style_loss_weights.items():
            style_weight_lookup[int(style)] = float(weight)
    decorrelate_worker_rng(context)
    optimizer.zero_grad(set_to_none=True)
    reset_peak_memory(device)
    transport.train()
    progress = progress_bar(
        description="Train transport",
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
            latent = batch["latent"]
            if not isinstance(latent, Tensor):
                raise TypeError("Transport batch requires tensor latent")
            latent = latent.to(device, non_blocking=True)
            condition = conditions_from_batch(
                batch, device, task=condition_task, factorial=factorial_conditioning
            )
            negative_condition = (
                contrasting_conditions(
                    condition,
                    condition_vocabularies,
                    factorial=factorial_conditioning,
                    active_axis=factorial_active_axis,
                )
                if condition_contrast_weight > 0 and condition_vocabularies is not None
                else None
            )
            sample_weight = (
                style_weight_lookup[condition.style_id]
                if style_weight_lookup is not None and not factorial_conditioning
                else None
            )
            report_samples += latent.shape[0]
            last_batch = batch_count is not None and batch_index + 1 == batch_count
            synchronize = (batch_index + 1) % gradient_accumulation == 0 or last_batch
            with (
                maybe_no_sync(training_model, synchronize=synchronize),
                sdpa_kernel_context(device, sdpa_backend),
            ):
                with autocast_context(device, precision):
                    losses = training_model(
                        latent,
                        condition,
                        negative_condition,
                        sample_weight,
                        condition_contrast_weight=condition_contrast_weight,
                        condition_contrast_margin=condition_contrast_margin,
                        condition_contrast_samples=condition_contrast_samples,
                    )
                    loss = losses["loss"] / gradient_accumulation
                scaler.scale(loss).backward()
            if not synchronize:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                transport.parameters(), gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                if context.is_main:
                    torch.save(batch, checkpoint_dir / "offending_batch.pt")
                raise FloatingPointError(
                    f"Non-finite transport gradient at step {state.global_step}"
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema and (state.global_step + 1) % ema_update_interval == 0:
                ema.update(transport, steps=ema_update_interval)
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
                    "loss": float(losses["loss"].detach()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "peak_gpu_memory_gib": peak_memory_gib(device),
                    "step_seconds": elapsed / report_steps,
                    "global_samples_per_second": report_samples * context.world_size / elapsed,
                }
                metrics.update(
                    {key: float(value.detach()) for key, value in losses.items() if key != "loss"}
                )
                if (
                    context.is_main
                    and validation_batch is not None
                    and condition_vocabularies is not None
                    and state.global_step % checkpoint_interval == 0
                ):
                    metrics.update(
                        evaluate_raw_and_ema(
                            transport,
                            ema,
                            lambda: evaluate_endpoint_matching(
                                transport,
                                validation_batch,
                                device=device,
                                precision=precision,
                                sdpa_backend=sdpa_backend,
                                factorial_conditioning=factorial_conditioning,
                                condition_task=condition_task,
                                active_axis=factorial_active_axis,
                                num_steps=validation_solver_steps,
                                seed=validation_seed,
                            ),
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
            raise RuntimeError("Transport DataLoader is empty")
        if completed_epoch:
            state.epoch += 1
            state.batch_in_epoch = 0
    progress.close()
    if logger is not None:
        logger.close()
    return state
