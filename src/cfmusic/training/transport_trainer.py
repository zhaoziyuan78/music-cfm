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
    all_gather_tensor,
    decorrelate_worker_rng,
    differentiable_all_gather,
    distributed_barrier,
    distributed_model,
    maybe_no_sync,
    set_data_epoch,
)
from cfmusic.latent.dataset import LatentDataset
from cfmusic.logging import MetricLogger
from cfmusic.losses.hsic import normalized_hsic
from cfmusic.losses.mmd import class_conditional_mmd, cross_class_mmd
from cfmusic.losses.roundtrip import roundtrip_loss
from cfmusic.losses.sliced_wasserstein import (
    class_conditional_sliced_wasserstein,
    cross_class_sliced_wasserstein,
    sliced_wasserstein_standard_normal,
)
from cfmusic.memory import (
    autocast_context,
    peak_memory_gib,
    reset_peak_memory,
    sdpa_kernel_context,
)
from cfmusic.models.probes import DynamicNoiseProjector
from cfmusic.progress import progress_bar, track
from cfmusic.training.checkpointing import load_checkpoint, save_rolling_checkpoint
from cfmusic.training.state import ExponentialMovingAverage, TrainState
from cfmusic.transport.base import ConditionalTransport
from cfmusic.transport.conditional_flow import ConditionalFlow


def _active_labels(
    condition: ConditionBatch, *, factorial: bool, active_axis: str | None
) -> Tensor:
    if not factorial:
        return condition.style_id
    axis = active_axis or "genre"
    values = condition.genre_id if axis == "genre" else condition.emotion_id
    if values is None:
        raise ValueError(f"Factorial regularization requires {axis}_id")
    return values


def _balanced_condition_indices(labels: Tensor, samples_per_style: int) -> Tensor:
    if samples_per_style <= 0:
        raise ValueError("samples_per_style must be positive")
    selected = []
    for label in torch.unique(labels, sorted=True):
        candidates = torch.nonzero(labels == label, as_tuple=False).flatten()
        count = min(samples_per_style, len(candidates))
        if count:
            order = torch.randperm(len(candidates), device=labels.device)[:count]
            selected.append(candidates.index_select(0, order))
    if not selected:
        raise ValueError("No samples available for conditional regularization")
    return torch.cat(selected)


def _unguided_predict(
    transport: ConditionalTransport,
    noise: Tensor,
    condition: ConditionBatch,
    *,
    num_steps: int,
    track_grad: bool,
) -> Tensor:
    if isinstance(transport, ConditionalFlow):
        return transport.predict(
            noise,
            condition,
            num_steps=num_steps,
            track_grad=track_grad,
            guidance_scale=1.0,
            source_repulsion_scale=0.0,
        )
    return transport.predict(noise, condition, num_steps=num_steps, track_grad=track_grad)


def _unguided_abduct(
    transport: ConditionalTransport,
    latent: Tensor,
    condition: ConditionBatch,
    *,
    num_steps: int,
    track_grad: bool,
) -> Tensor:
    if isinstance(transport, ConditionalFlow):
        return transport.abduct(
            latent,
            condition,
            num_steps=num_steps,
            track_grad=track_grad,
            guidance_scale=1.0,
        )
    return transport.abduct(latent, condition, num_steps=num_steps, track_grad=track_grad)


class TransportLossModule(nn.Module):
    """Expose the transport loss through ``forward`` so native DDP owns backward hooks."""

    def __init__(
        self, transport: nn.Module, noise_projector: DynamicNoiseProjector | None = None
    ) -> None:
        super().__init__()
        self.transport = transport
        self.noise_projector = noise_projector

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
        roundtrip_weight: float,
        roundtrip_steps: int,
        roundtrip_samples: int | None,
        roundtrip_cosine_weight: float,
        endpoint_weight: float,
        endpoint_steps: int,
        endpoint_samples_per_style: int,
        exogeneity_hsic_weight: float,
        exogeneity_prior_weight: float,
        exogeneity_cross_mmd_weight: float,
        exogeneity_cross_swd_weight: float,
        exogeneity_steps: int,
        exogeneity_samples_per_style: int,
        global_step: int,
        factorial_conditioning: bool,
        factorial_active_axis: str | None,
    ) -> dict[str, Tensor]:
        transport = cast(ConditionalTransport, self.transport)
        losses = transport.training_loss(
            latent,
            condition,
            negative_condition=negative_condition,
            condition_contrast_weight=condition_contrast_weight,
            condition_contrast_margin=condition_contrast_margin,
            condition_contrast_samples=condition_contrast_samples,
            sample_weight=sample_weight,
        )
        consistency = latent.new_zeros(())
        if roundtrip_weight > 0:
            count = latent.shape[0]
            if roundtrip_samples is not None:
                count = min(count, max(1, roundtrip_samples))
            indices = torch.randperm(latent.shape[0], device=latent.device)[:count]
            factual = latent.index_select(0, indices)
            factual_condition = condition.index_select(indices)
            noise = _unguided_abduct(
                transport, factual, factual_condition, num_steps=roundtrip_steps, track_grad=True
            )
            reconstructed = _unguided_predict(
                transport,
                noise,
                factual_condition,
                num_steps=roundtrip_steps,
                track_grad=True,
            )
            consistency = roundtrip_loss(
                reconstructed.float(), factual.float(), cosine_weight=roundtrip_cosine_weight
            )
            losses["loss"] = losses["loss"] + roundtrip_weight * consistency
        losses["roundtrip_loss"] = consistency
        losses["roundtrip_weight"] = latent.new_tensor(roundtrip_weight)

        endpoint_mmd = latent.new_zeros(())
        endpoint_swd = latent.new_zeros(())
        labels = _active_labels(
            condition,
            factorial=factorial_conditioning,
            active_axis=factorial_active_axis,
        )
        if endpoint_weight > 0:
            indices = _balanced_condition_indices(labels, endpoint_samples_per_style)
            factual = latent.index_select(0, indices)
            factual_condition = condition.index_select(indices)
            endpoint_labels = labels.index_select(0, indices)
            generated = _unguided_predict(
                transport,
                torch.randn_like(factual),
                factual_condition,
                num_steps=endpoint_steps,
                track_grad=True,
            )
            flat_generated = generated.float().flatten(1)
            flat_factual = factual.float().flatten(1)
            endpoint_mmd = class_conditional_mmd(flat_generated, flat_factual, endpoint_labels)
            endpoint_swd = class_conditional_sliced_wasserstein(
                flat_generated,
                flat_factual,
                endpoint_labels,
                num_projections=32,
                seed=global_step + 101,
            )
            losses["loss"] = losses["loss"] + endpoint_weight * (endpoint_mmd + endpoint_swd)
        losses["endpoint_mmd"] = endpoint_mmd
        losses["endpoint_swd"] = endpoint_swd
        losses["endpoint_weight"] = latent.new_tensor(endpoint_weight)

        noise_hsic = latent.new_zeros(())
        noise_prior = latent.new_zeros(())
        noise_cross_mmd = latent.new_zeros(())
        noise_cross_swd = latent.new_zeros(())
        exogeneity_active = any(
            weight > 0
            for weight in (
                exogeneity_hsic_weight,
                exogeneity_prior_weight,
                exogeneity_cross_mmd_weight,
                exogeneity_cross_swd_weight,
            )
        )
        if exogeneity_active:
            if self.noise_projector is None:
                raise ValueError("Exogeneity loss requires a dynamic noise projector")
            indices = _balanced_condition_indices(labels, exogeneity_samples_per_style)
            factual = latent.index_select(0, indices)
            factual_condition = condition.index_select(indices)
            noise_labels = labels.index_select(0, indices)
            noise = _unguided_abduct(
                transport,
                factual,
                factual_condition,
                num_steps=exogeneity_steps,
                track_grad=True,
            )
            views = self.noise_projector(noise, step=global_step)
            global_labels = all_gather_tensor(noise_labels)
            global_views = tuple(differentiable_all_gather(view) for view in views)
            noise_hsic = torch.stack(
                [normalized_hsic(view, global_labels) for view in global_views]
            ).mean()
            noise_cross_mmd = torch.stack(
                [cross_class_mmd(view, global_labels) for view in global_views]
            ).mean()
            noise_cross_swd = torch.stack(
                [
                    cross_class_sliced_wasserstein(
                        view,
                        global_labels,
                        num_projections=32,
                        seed=global_step * 131 + view_index,
                    )
                    for view_index, view in enumerate(global_views)
                ]
            ).mean()
            gaussian_views = (
                *global_views[: self.noise_projector.num_views],
                global_views[-1],
            )
            noise_prior = torch.stack(
                [
                    sliced_wasserstein_standard_normal(
                        view,
                        num_projections=32,
                        seed=global_step * 137 + view_index,
                    )
                    for view_index, view in enumerate(gaussian_views)
                ]
            ).mean()
            losses["loss"] = losses["loss"] + (
                exogeneity_hsic_weight * noise_hsic
                + exogeneity_prior_weight * noise_prior
                + exogeneity_cross_mmd_weight * noise_cross_mmd
                + exogeneity_cross_swd_weight * noise_cross_swd
            )
        losses["noise_hsic"] = noise_hsic
        losses["noise_prior_swd"] = noise_prior
        losses["noise_cross_class_mmd"] = noise_cross_mmd
        losses["noise_cross_class_swd"] = noise_cross_swd
        losses["exogeneity_active"] = latent.new_tensor(float(exogeneity_active))
        return losses


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
            condition.condition_mask,
        )
    return ConditionBatch(
        condition.dataset_id,
        condition.task_id,
        _different_labels(condition.style_id, vocabularies["style_id"]),
        condition.genre_id,
        condition.emotion_id,
        condition.condition_mask,
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
            condition.condition_mask,
        )
    return ConditionBatch(
        condition.dataset_id,
        condition.task_id,
        shift(condition.style_id, "style_id"),
        condition.genre_id,
        condition.emotion_id,
        condition.condition_mask,
    )


def roundtrip_schedule_scale(step: int, *, warmup_steps: int, ramp_steps: int) -> float:
    """Delay consistency training until the CFM field has learned a useful path."""

    if warmup_steps < 0 or ramp_steps < 0:
        raise ValueError("Round-trip warmup and ramp steps must be non-negative")
    if step < warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, (step - warmup_steps + 1) / ramp_steps)


def sparse_regularizer_scale(
    step: int,
    *,
    warmup_steps: int,
    ramp_steps: int,
    interval: int,
    offset: int,
) -> float:
    """Return a ramped scale only on one offset of a sparse interval."""

    if interval <= 0 or not 0 <= offset < interval:
        raise ValueError("Sparse regularizer offset must be within its positive interval")
    if step < warmup_steps or (step - warmup_steps) % interval != offset:
        return 0.0
    return roundtrip_schedule_scale(step, warmup_steps=warmup_steps, ramp_steps=ramp_steps)


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
    roundtrip_weight: float = 0.0,
    roundtrip_warmup_steps: int = 0,
    roundtrip_ramp_steps: int = 0,
    roundtrip_interval: int = 1,
    roundtrip_offset: int = 0,
    roundtrip_inverse_steps: int = 2,
    roundtrip_samples_per_batch: int | None = None,
    roundtrip_cosine_weight: float = 0.1,
    endpoint_weight: float = 0.0,
    endpoint_warmup_steps: int = 0,
    endpoint_ramp_steps: int = 0,
    endpoint_interval: int = 1,
    endpoint_offset: int = 0,
    endpoint_solver_steps: int = 4,
    endpoint_samples_per_style: int = 4,
    exogeneity_hsic_weight: float = 0.0,
    exogeneity_prior_weight: float = 0.0,
    exogeneity_cross_mmd_weight: float = 0.0,
    exogeneity_cross_swd_weight: float = 0.0,
    exogeneity_warmup_steps: int = 0,
    exogeneity_ramp_steps: int = 0,
    exogeneity_interval: int = 1,
    exogeneity_offset: int = 0,
    exogeneity_inverse_steps: int = 4,
    exogeneity_samples_per_style: int = 4,
    noise_projector: DynamicNoiseProjector | None = None,
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
    if roundtrip_weight < 0 or roundtrip_cosine_weight < 0:
        raise ValueError("Round-trip weights must be non-negative")
    if roundtrip_interval <= 0 or roundtrip_inverse_steps <= 0:
        raise ValueError("Round-trip interval and inverse steps must be positive")
    if not 0 <= roundtrip_offset < roundtrip_interval:
        raise ValueError("Round-trip offset must be within its interval")
    if roundtrip_samples_per_batch is not None and roundtrip_samples_per_batch <= 0:
        raise ValueError("roundtrip_samples_per_batch must be positive or null")
    if endpoint_weight < 0 or endpoint_solver_steps <= 0 or endpoint_samples_per_style <= 0:
        raise ValueError("Endpoint matching weight/steps/samples are invalid")
    if any(
        value < 0
        for value in (
            exogeneity_hsic_weight,
            exogeneity_prior_weight,
            exogeneity_cross_mmd_weight,
            exogeneity_cross_swd_weight,
        )
    ):
        raise ValueError("Exogeneity weights must be non-negative")
    if exogeneity_inverse_steps <= 0 or exogeneity_samples_per_style <= 0:
        raise ValueError("Exogeneity inverse steps and samples must be positive")
    sparse_regularizer_scale(
        0,
        warmup_steps=endpoint_warmup_steps,
        ramp_steps=endpoint_ramp_steps,
        interval=endpoint_interval,
        offset=endpoint_offset,
    )
    sparse_regularizer_scale(
        0,
        warmup_steps=exogeneity_warmup_steps,
        ramp_steps=exogeneity_ramp_steps,
        interval=exogeneity_interval,
        offset=exogeneity_offset,
    )
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
    loss_module = TransportLossModule(transport, noise_projector)
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
    report_roundtrip = torch.zeros((), device=device)
    report_roundtrip_weight = 0.0
    report_roundtrip_steps = 0
    report_endpoint = {
        "endpoint_mmd": torch.zeros((), device=device),
        "endpoint_swd": torch.zeros((), device=device),
    }
    report_endpoint_weight = 0.0
    report_endpoint_steps = 0
    report_exogeneity = {
        "noise_hsic": torch.zeros((), device=device),
        "noise_prior_swd": torch.zeros((), device=device),
        "noise_cross_class_mmd": torch.zeros((), device=device),
        "noise_cross_class_swd": torch.zeros((), device=device),
    }
    report_exogeneity_steps = 0
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
            batch_sample_weight = batch.get("sample_weight")
            sample_weight = (
                batch_sample_weight.to(device, non_blocking=True).flatten()
                if isinstance(batch_sample_weight, Tensor)
                else None
            )
            if style_weight_lookup is not None and not factorial_conditioning:
                style_weight = style_weight_lookup[condition.style_id]
                sample_weight = (
                    style_weight if sample_weight is None else sample_weight * style_weight
                )
            roundtrip_scale = sparse_regularizer_scale(
                state.global_step,
                warmup_steps=roundtrip_warmup_steps,
                ramp_steps=roundtrip_ramp_steps,
                interval=roundtrip_interval,
                offset=roundtrip_offset,
            )
            run_roundtrip = roundtrip_weight > 0 and roundtrip_scale > 0
            active_roundtrip_weight = roundtrip_weight * roundtrip_scale if run_roundtrip else 0.0
            endpoint_scale = sparse_regularizer_scale(
                state.global_step,
                warmup_steps=endpoint_warmup_steps,
                ramp_steps=endpoint_ramp_steps,
                interval=endpoint_interval,
                offset=endpoint_offset,
            )
            active_endpoint_weight = endpoint_weight * endpoint_scale
            exogeneity_scale = sparse_regularizer_scale(
                state.global_step,
                warmup_steps=exogeneity_warmup_steps,
                ramp_steps=exogeneity_ramp_steps,
                interval=exogeneity_interval,
                offset=exogeneity_offset,
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
                        roundtrip_weight=active_roundtrip_weight,
                        roundtrip_steps=roundtrip_inverse_steps,
                        roundtrip_samples=roundtrip_samples_per_batch,
                        roundtrip_cosine_weight=roundtrip_cosine_weight,
                        endpoint_weight=active_endpoint_weight,
                        endpoint_steps=endpoint_solver_steps,
                        endpoint_samples_per_style=endpoint_samples_per_style,
                        exogeneity_hsic_weight=exogeneity_hsic_weight * exogeneity_scale,
                        exogeneity_prior_weight=exogeneity_prior_weight * exogeneity_scale,
                        exogeneity_cross_mmd_weight=exogeneity_cross_mmd_weight * exogeneity_scale,
                        exogeneity_cross_swd_weight=exogeneity_cross_swd_weight * exogeneity_scale,
                        exogeneity_steps=exogeneity_inverse_steps,
                        exogeneity_samples_per_style=exogeneity_samples_per_style,
                        global_step=state.global_step,
                        factorial_conditioning=factorial_conditioning,
                        factorial_active_axis=factorial_active_axis,
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
            if active_roundtrip_weight > 0:
                report_roundtrip += losses["roundtrip_loss"].detach()
                report_roundtrip_weight += active_roundtrip_weight
                report_roundtrip_steps += 1
            if active_endpoint_weight > 0:
                for name in report_endpoint:
                    report_endpoint[name] += losses[name].detach()
                report_endpoint_weight += active_endpoint_weight
                report_endpoint_steps += 1
            if exogeneity_scale > 0 and any(
                value > 0
                for value in (
                    exogeneity_hsic_weight,
                    exogeneity_prior_weight,
                    exogeneity_cross_mmd_weight,
                    exogeneity_cross_swd_weight,
                )
            ):
                for name in report_exogeneity:
                    report_exogeneity[name] += losses[name].detach()
                report_exogeneity_steps += 1
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
                metrics["roundtrip_loss"] = float(report_roundtrip / max(1, report_roundtrip_steps))
                metrics["roundtrip_weight"] = report_roundtrip_weight / max(
                    1, report_roundtrip_steps
                )
                metrics["roundtrip_active_fraction"] = report_roundtrip_steps / report_steps
                for name, value in report_endpoint.items():
                    metrics[name] = float(value / max(1, report_endpoint_steps))
                metrics["endpoint_weight"] = report_endpoint_weight / max(1, report_endpoint_steps)
                metrics["endpoint_active_fraction"] = report_endpoint_steps / report_steps
                for name, value in report_exogeneity.items():
                    metrics[name] = float(value / max(1, report_exogeneity_steps))
                metrics["exogeneity_active_fraction"] = report_exogeneity_steps / report_steps
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
                report_roundtrip.zero_()
                report_roundtrip_weight = 0.0
                report_roundtrip_steps = 0
                for value in report_endpoint.values():
                    value.zero_()
                report_endpoint_weight = 0.0
                report_endpoint_steps = 0
                for value in report_exogeneity.values():
                    value.zero_()
                report_exogeneity_steps = 0
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
