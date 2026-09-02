"""Fine-tune a Stage-1 transport for distributional exogeneity."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from cfmusic.config import CONFIG_DIR, config_mapping, prepare_config
from cfmusic.data.samplers import BalancedStyleBatchSampler, DistributedBatchSampler
from cfmusic.distributed import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
)
from cfmusic.latent.compatibility import (
    serialize_cache_metadata,
    validate_latent_dataset,
    validate_transport_cache_provenance,
)
from cfmusic.latent.dataset import LatentDataset
from cfmusic.models.latent_vector_field import ConditionalVectorField
from cfmusic.models.probes import FixedNoiseProjector
from cfmusic.paths import save_run_context
from cfmusic.reproducibility import seed_everything
from cfmusic.training.abduction_trainer import finetune_abduction_steps
from cfmusic.training.checkpointing import checkpoint_model_state, resolve_resume_checkpoint
from cfmusic.training.optim import create_adamw
from cfmusic.training.schedules import warmup_cosine_scheduler
from cfmusic.training.transport_trainer import heldout_condition_batch
from cfmusic.transport.factory import create_transport


def _train(cfg: DictConfig, context: DistributedContext) -> None:
    paths = prepare_config(cfg)
    if context.is_main:
        save_run_context(cfg, Path(HydraConfig.get().runtime.output_dir))
    seed_everything(int(cfg.seed))
    checkpoint_dir = paths["checkpoints_dir"] / str(cfg.experiment.name) / "transport_stage2"
    explicit_resume = Path(str(cfg.resume_from)) if cfg.resume_from is not None else None
    resume_checkpoint = resolve_resume_checkpoint(
        checkpoint_dir,
        resume=bool(cfg.get("resume", False)),
        resume_from=explicit_resume,
        announce=context.is_main,
    )
    if cfg.transport_checkpoint is None and resume_checkpoint is None:
        raise ValueError(
            "transport_checkpoint must point to Stage 1 when no Stage-2 checkpoint is resumed"
        )
    latent_root = paths["latent_dir"] / str(cfg.data.name)
    sampler_cfg = cfg.independence.sampler
    training = cfg.independence.training
    dataset = LatentDataset(
        latent_root,
        split="train",
        shard_cache_size=max(1, int(sampler_cfg.classes_per_batch)),
    )
    validate_latent_dataset(
        dataset,
        codec_cfg=cfg.codec,
        transport_cfg=cfg.transport,
        dataset_name=str(cfg.data.name),
    )
    cache_metadata = [dataset.metadata]
    labels = dataset.frame["style_id"].astype(int).tolist()
    factorial = bool(cfg.experiment.get("factorial", False))
    label_values = {
        column: sorted(dataset.frame[column].dropna().astype(int).unique().tolist())
        for column in (("genre_id", "emotion_id") if factorial else ("style_id",))
    }
    validation_batch: dict[str, torch.Tensor] | None = None
    if context.is_main:
        validation_dataset = LatentDataset(
            latent_root,
            split="validation",
            shard_cache_size=max(1, len(label_values.get("style_id", []))),
        )
        validate_latent_dataset(
            validation_dataset,
            codec_cfg=cfg.codec,
            transport_cfg=cfg.transport,
            dataset_name=str(cfg.data.name),
        )
        validation_batch = heldout_condition_batch(
            validation_dataset,
            samples_per_style=int(training.get("validation_samples_per_style", 16)),
        )
    base_sampler = BalancedStyleBatchSampler(
        labels,
        classes_per_batch=int(sampler_cfg.classes_per_batch),
        samples_per_class=int(sampler_cfg.samples_per_class),
        replacement_for_small_classes=bool(sampler_cfg.replacement_for_small_classes),
        group_ids=dataset.shard_ids,
        seed=int(cfg.seed),
    )
    sampler = (
        DistributedBatchSampler(
            base_sampler,
            rank=context.rank,
            world_size=context.world_size,
        )
        if context.world_size > 1
        else base_sampler
    )
    workers = int(training.get("dataloader_workers", 2))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=int(training.get("prefetch_factor", 2)) if workers > 0 else None,
    )
    device = context.device
    transport = create_transport(cfg.transport).to(device)
    checkpoint_path: Path | None = None
    if cfg.transport_checkpoint is not None:
        checkpoint_path = Path(str(cfg.transport_checkpoint)).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(checkpoint, dict):
            raise TypeError("Stage-1 checkpoint must contain a mapping")
        validate_transport_cache_provenance(checkpoint, cache_metadata)
        stage1_weights = str(training.get("stage1_weights", "raw"))
        transport.load_state_dict(checkpoint_model_state(checkpoint, weights=stage1_weights))
        del checkpoint
    if resume_checkpoint is not None:
        resumed = torch.load(resume_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(resumed, dict):
            raise TypeError("Stage-2 checkpoint must contain a mapping")
        validate_transport_cache_provenance(resumed, cache_metadata)
        del resumed
    sample = dataset[0]["latent"]
    if not isinstance(sample, torch.Tensor):
        raise TypeError("Latent dataset returned invalid sample")
    projection_cfg = cfg.independence.noise_projection
    projector = FixedNoiseProjector(
        sample.numel(), int(projection_cfg.dim), int(projection_cfg.seed)
    ).to(device)
    transport.add_module("noise_projector", projector)
    for module in transport.modules():
        if isinstance(module, ConditionalVectorField):
            module.set_gradient_checkpointing(bool(training.gradient_checkpointing))
    optimizer = create_adamw(
        transport,
        learning_rate=float(training.learning_rate),
        weight_decay=float(cfg.transport.training.weight_decay),
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, warmup_steps=0, max_steps=int(training.max_steps)
    )
    condition_objective = cfg.transport.get("condition_objective", {})
    condition_contrast_weight = (
        float(condition_objective.get("weight", 0.0))
        if bool(condition_objective.get("enabled", False))
        else 0.0
    )
    if context.is_main:
        batch_size = int(sampler_cfg.classes_per_batch) * int(sampler_cfg.samples_per_class)
        print(
            f"Stage-2 training plan: {len(dataset)} samples, {len(loader)} "
            f"balanced batches/rank/epoch, world_size={context.world_size}, "
            f"global_batch={batch_size * context.world_size}, "
            f"sdpa_backend={training.get('sdpa_backend', 'math')!s}"
        )
        print(f"Stage-2 initialization weights: {training.get('stage1_weights', 'raw')!s}")
        if condition_contrast_weight > 0:
            contrast_samples = int(
                training.get(
                    "condition_contrast_samples_per_batch",
                    condition_objective.get("samples_per_batch", 0),
                )
            )
            print(
                "Stage-2 condition contrast: "
                f"weight={condition_contrast_weight:g}, "
                f"margin={float(condition_objective.get('margin', 0.0)):g}, "
                f"samples/rank={contrast_samples}"
            )
            if validation_batch is not None:
                print(
                    f"Held-out condition probe: {validation_batch['latent'].shape[0]} "
                    f"balanced validation latents, evaluated every {int(training.checkpoint_interval)} steps"
                )
    finetune_abduction_steps(
        transport,
        projector,
        loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_steps=int(training.max_steps),
        abduction_interval=int(training.abduction_interval),
        inverse_steps=int(training.train_inverse_steps),
        hsic_weight=float(cfg.independence.hsic_weight),
        prior_weight=float(cfg.independence.prior_weight),
        roundtrip_weight=float(cfg.independence.roundtrip.weight),
        cosine_weight=float(cfg.independence.roundtrip.cosine_weight),
        warmup_steps=int(cfg.independence.warmup_steps),
        ramp_steps=int(cfg.independence.ramp_steps),
        gradient_clip_norm=float(training.gradient_clip_norm),
        precision=str(training.precision),
        sdpa_backend=str(training.get("sdpa_backend", "math")),
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=int(training.checkpoint_interval),
        config=config_mapping(cfg),
        provenance={
            "stage1_checkpoint": (
                str(checkpoint_path) if checkpoint_path is not None else "restored_from_stage2"
            ),
            "stage1_weights": str(training.get("stage1_weights", "raw")),
            "latent_cache_metadata_json": serialize_cache_metadata(cache_metadata),
        },
        factorial_conditioning=factorial,
        condition_contrast_weight=condition_contrast_weight,
        condition_contrast_margin=float(condition_objective.get("margin", 0.0)),
        condition_contrast_samples=(
            int(
                training.get(
                    "condition_contrast_samples_per_batch",
                    condition_objective.get("samples_per_batch", 0),
                )
            )
            or None
        ),
        condition_vocabularies=label_values,
        validation_batch=validation_batch,
        validation_seed=int(cfg.seed) + 2026,
        ema_update_interval=int(training.get("ema_update_interval", 10)),
        log_interval=int(training.get("log_interval", 10)),
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
