"""Train shared conditional CFM or DDIM on frozen normalized latents."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from cfmusic.config import CONFIG_DIR, config_mapping, prepare_config
from cfmusic.data.samplers import ShardBatchSampler
from cfmusic.distributed import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
)
from cfmusic.latent.combined import CombinedLatentDataset
from cfmusic.latent.compatibility import (
    serialize_cache_metadata,
    validate_latent_dataset,
    validate_transport_cache_provenance,
)
from cfmusic.latent.dataset import LatentDataset
from cfmusic.paths import save_run_context
from cfmusic.reproducibility import seed_everything
from cfmusic.training.checkpointing import resolve_resume_checkpoint
from cfmusic.training.optim import create_adamw
from cfmusic.training.schedules import warmup_cosine_scheduler
from cfmusic.training.transport_trainer import (
    heldout_condition_batch,
    inverse_frequency_weights,
    train_transport_steps,
)
from cfmusic.transport.factory import create_transport


def _train(cfg: DictConfig, context: DistributedContext) -> None:
    paths = prepare_config(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    if context.is_main:
        save_run_context(cfg, run_dir)
    seed_everything(int(cfg.seed))
    data_name = str(cfg.data.name)
    latent_root = paths["latent_dir"] / data_name
    dataset: Dataset[dict[str, torch.Tensor | str | int]]
    if "datasets" in cfg.data:
        dataset_names = [str(name) for name in cfg.data.datasets]
        combined = CombinedLatentDataset(
            [paths["latent_dir"] / name for name in dataset_names], split="train"
        )
        dataset = combined
        latent_datasets = combined.datasets
    else:
        dataset = LatentDataset(latent_root, split="train")
        dataset_names = [data_name]
        latent_datasets = [dataset]
    for name, latent_dataset in zip(dataset_names, latent_datasets, strict=True):
        validate_latent_dataset(
            latent_dataset,
            codec_cfg=cfg.codec,
            transport_cfg=cfg.transport,
            dataset_name=name,
        )
    cache_metadata = [latent_dataset.metadata for latent_dataset in latent_datasets]
    training = cfg.transport.training
    factorial = bool(cfg.experiment.get("factorial", False))
    label_values: dict[str, list[int]] = {}
    for column in ("genre_id", "emotion_id") if factorial else ("style_id",):
        label_values[column] = sorted(
            {
                int(value)
                for latent_dataset in latent_datasets
                if column in latent_dataset.frame
                for value in latent_dataset.frame[column].dropna().astype(int).tolist()
            }
        )
    all_style_labels = [
        int(value)
        for latent_dataset in latent_datasets
        for value in latent_dataset.frame["style_id"].astype(int).tolist()
    ]
    class_balance_exponent = float(training.get("class_balance_exponent", 0.0))
    style_loss_weights = (
        inverse_frequency_weights(all_style_labels, exponent=class_balance_exponent)
        if class_balance_exponent > 0 and not factorial
        else None
    )
    condition_objective = cfg.transport.get("condition_objective", {})
    condition_contrast_weight = (
        float(condition_objective.get("weight", 0.0))
        if bool(condition_objective.get("enabled", False))
        else 0.0
    )
    validation_batch: dict[str, torch.Tensor] | None = None
    if context.is_main and condition_contrast_weight > 0 and isinstance(dataset, LatentDataset):
        validation_dataset = LatentDataset(
            latent_root,
            split="validation",
            shard_cache_size=max(1, len(label_values.get("style_id", []))),
        )
        validate_latent_dataset(
            validation_dataset,
            codec_cfg=cfg.codec,
            transport_cfg=cfg.transport,
            dataset_name=data_name,
        )
        validation_batch = heldout_condition_batch(
            validation_dataset,
            samples_per_style=int(training.get("validation_samples_per_style", 16)),
        )
    if bool(cfg.experiment.get("shuffled_labels", False)) and isinstance(dataset, LatentDataset):
        generator = torch.Generator().manual_seed(int(cfg.seed))
        original = torch.as_tensor(dataset.frame["style_id"].astype(int).to_numpy())
        shuffled = original[torch.randperm(len(original), generator=generator)]
        dataset.frame["style_id"] = shuffled.numpy()
        dataset.refresh_index()
        import json

        checkpoint_dir = paths["checkpoints_dir"] / str(cfg.experiment.name) / "transport_stage1"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if context.is_main:
            (checkpoint_dir / "label_shuffle.json").write_text(
                json.dumps({"seed": int(cfg.seed), "permutation": shuffled.tolist()}),
                encoding="utf-8",
            )
    shard_ids = getattr(dataset, "shard_ids", None)
    if not isinstance(shard_ids, list):
        raise TypeError("Latent datasets must expose shard_ids for locality-aware training")
    batch_sampler = ShardBatchSampler(
        shard_ids,
        batch_size=int(training.batch_size),
        rank=context.rank,
        world_size=context.world_size,
        seed=int(cfg.seed),
    )
    workers = int(training.get("dataloader_workers", 2))
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=int(training.get("prefetch_factor", 2)) if workers > 0 else None,
    )
    gradient_accumulation = int(training.gradient_accumulation)
    optimizer_steps_per_epoch = (len(loader) + gradient_accumulation - 1) // gradient_accumulation
    max_steps = int(training.max_steps)
    if training.get("max_epochs") is not None:
        epoch_limited_steps = int(training.max_epochs) * optimizer_steps_per_epoch
        max_steps = min(
            max_steps,
            max(int(training.get("min_steps", 1)), epoch_limited_steps),
        )
    device = context.device
    transport = create_transport(cfg.transport).to(device)
    optimizer = create_adamw(
        transport,
        learning_rate=float(training.learning_rate),
        weight_decay=float(training.weight_decay),
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, warmup_steps=min(int(training.warmup_steps), max_steps), max_steps=max_steps
    )
    checkpoint_dir = paths["checkpoints_dir"] / str(cfg.experiment.name) / "transport_stage1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    explicit_resume = Path(str(cfg.resume_from)) if cfg.resume_from is not None else None
    resume_checkpoint = resolve_resume_checkpoint(
        checkpoint_dir,
        resume=bool(cfg.get("resume", False)),
        resume_from=explicit_resume,
        announce=context.is_main,
    )
    if resume_checkpoint is not None:
        resumed = torch.load(resume_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(resumed, dict):
            raise TypeError("Transport checkpoint must contain a mapping")
        validate_transport_cache_provenance(resumed, cache_metadata)
        del resumed
    if context.is_main:
        global_batch = (
            int(training.batch_size) * int(training.gradient_accumulation) * context.world_size
        )
        print(
            f"Transport training plan: {len(dataset)} samples, {len(loader)} "
            f"micro-batches/rank/epoch, world_size={context.world_size}, "
            f"global_effective_batch={global_batch}, optimizer_steps={max_steps} "
            f"(hard_cap={int(training.max_steps)}), "
            f"sdpa_backend={training.get('sdpa_backend', 'math')!s}"
        )
        if style_loss_weights is not None:
            formatted = ", ".join(
                f"{style}:{weight:.3f}" for style, weight in sorted(style_loss_weights.items())
            )
            print(f"Stage-1 class-balanced loss weights (mean=1): {formatted}")
        if condition_contrast_weight > 0:
            print(
                "Stage-1 condition contrast: "
                f"weight={condition_contrast_weight:g}, "
                f"margin={float(condition_objective.get('margin', 0.0)):g}, "
                f"samples/rank={int(condition_objective.get('samples_per_batch', 0))}"
            )
            if validation_batch is not None:
                print(
                    f"Held-out condition probe: {validation_batch['latent'].shape[0]} "
                    f"balanced validation latents, evaluated every {int(training.checkpoint_interval)} steps"
                )
    train_transport_steps(
        transport,
        loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_steps=max_steps,
        gradient_accumulation=gradient_accumulation,
        gradient_clip_norm=float(training.gradient_clip_norm),
        precision=str(training.precision),
        sdpa_backend=str(training.get("sdpa_backend", "math")),
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=int(training.checkpoint_interval),
        config=config_mapping(cfg.transport),
        provenance={"latent_cache_metadata_json": serialize_cache_metadata(cache_metadata)},
        factorial_conditioning=factorial,
        condition_contrast_weight=condition_contrast_weight,
        condition_contrast_margin=float(condition_objective.get("margin", 0.0)),
        condition_contrast_samples=int(condition_objective.get("samples_per_batch", 0)) or None,
        condition_vocabularies=label_values,
        style_loss_weights=style_loss_weights,
        validation_batch=validation_batch,
        validation_seed=int(cfg.seed) + 2026,
        ema_decay=float(training.get("ema_decay", 0.9999)),
        ema_update_interval=int(training.get("ema_update_interval", 10)),
        log_interval=int(training.get("log_interval", 10)),
        find_unused_parameters=bool(cfg.transport.get("independent_per_style", False)),
        resume_from=resume_checkpoint,
        distributed=context,
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    context = initialize_distributed()
    try:
        _train(cfg, context)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
