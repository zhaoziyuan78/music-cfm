"""Train the frozen latent Transformer VAE."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import cast

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cfmusic.codec.transformer_vae import TransformerVAE
from cfmusic.config import CONFIG_DIR, config_mapping, prepare_config
from cfmusic.data.collate import collate_token_batch
from cfmusic.data.datasets import MidiTokenDataset
from cfmusic.data.manifests import manifest_hash
from cfmusic.data.samplers import (
    BatchSamplerProtocol,
    DatasetTemperatureLengthBatchSampler,
    DistributedBatchSampler,
    LengthBucketBatchSampler,
)
from cfmusic.distributed import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
)
from cfmusic.paths import save_run_context
from cfmusic.reproducibility import seed_everything
from cfmusic.tokenization.factory import MidiTokenizer, tokenizer_from_config
from cfmusic.training.checkpointing import resolve_resume_checkpoint
from cfmusic.training.codec_trainer import train_codec_steps
from cfmusic.training.optim import create_adamw
from cfmusic.training.schedules import warmup_cosine_scheduler


def codec_from_config(cfg: DictConfig, tokenizer: MidiTokenizer) -> TransformerVAE:
    if tokenizer.config.max_sequence_length > int(cfg.max_sequence_length):
        raise ValueError(
            "Tokenizer maximum length exceeds the codec positional embedding limit: "
            f"{tokenizer.config.max_sequence_length} > {int(cfg.max_sequence_length)}"
        )
    return TransformerVAE(
        vocab_size=len(tokenizer.vocabulary),
        d_model=int(cfg.d_model),
        encoder_layers=int(cfg.encoder_layers),
        decoder_layers=int(cfg.decoder_layers),
        num_heads=int(cfg.num_heads),
        ff_multiplier=int(cfg.ff_multiplier),
        dropout=float(cfg.dropout),
        latent_tokens=int(cfg.latent_tokens),
        latent_dim=int(cfg.latent_dim),
        max_sequence_length=int(cfg.max_sequence_length),
        pad_id=tokenizer.vocabulary.pad_id,
        bos_id=tokenizer.vocabulary.bos_id,
        eos_id=tokenizer.vocabulary.eos_id,
        unk_id=tokenizer.vocabulary.id("UNK"),
        vocabulary=tokenizer.vocabulary,
        gradient_checkpointing=bool(cfg.training.get("gradient_checkpointing", False)),
        decoder_token_dropout=float(cfg.training.get("decoder_token_dropout", 0.0)),
    )


def _manifest(paths: dict[str, Path], data_cfg: DictConfig) -> tuple[pd.DataFrame, str]:
    names = list(data_cfg.datasets) if "datasets" in data_cfg else [str(data_cfg.name)]
    frames: list[pd.DataFrame] = []
    digests: list[str] = []
    for dataset_id, name_value in enumerate(names):
        name = str(name_value)
        path = paths["processed_dir"] / name / "manifest.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest {path}; run prepare first")
        frame = pd.read_parquet(path)
        frame["dataset_id"] = dataset_id
        frames.append(frame)
        digests.append(manifest_hash(path))
    return pd.concat(frames, ignore_index=True), hashlib.sha256(
        "".join(digests).encode()
    ).hexdigest()


def _validation_subset(frame: pd.DataFrame, *, samples: int, seed: int) -> pd.DataFrame:
    validation = frame.loc[(frame["split"] == "validation") & frame["valid"]]
    if validation.empty:
        raise RuntimeError("Codec training requires at least one valid validation segment")
    if samples <= 0 or len(validation) <= samples:
        return validation
    groups = [group for _, group in validation.groupby("dataset_id", sort=True)]
    quota = max(1, samples // len(groups))
    selected = pd.concat(
        [group.sample(n=min(quota, len(group)), random_state=seed) for group in groups]
    )
    remaining_count = samples - len(selected)
    if remaining_count > 0:
        remaining = validation.drop(index=selected.index)
        selected = pd.concat(
            [
                selected,
                remaining.sample(n=min(remaining_count, len(remaining)), random_state=seed + 1),
            ]
        )
    return selected


def _train(cfg: DictConfig, context: DistributedContext) -> None:
    paths = prepare_config(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    if context.is_main:
        save_run_context(cfg, run_dir)
    training = cfg.codec.training
    seed_everything(int(training.seed))
    tokenizer = tokenizer_from_config(
        cfg.tokenizer, max_sequence_length=int(cfg.codec.max_sequence_length)
    )
    frame, manifest_digest = _manifest(paths, cfg.data)
    if "raw_token_count" in frame and bool(training.get("drop_overlength", True)):
        overlength = frame["raw_token_count"].astype(int) > tokenizer.config.max_sequence_length
        if context.is_main and bool(overlength.any()):
            print(
                f"Dropping {int(overlength.sum())} overlength BEAT segments instead of "
                "training on truncated music"
            )
        frame = frame.loc[~overlength].reset_index(drop=True)
    dataset = MidiTokenDataset(frame, tokenizer, split="train")
    workers = int(training.get("dataloader_workers", 0))
    collate = partial(collate_token_batch, pad_id=tokenizer.vocabulary.pad_id)
    if bool(training.get("length_bucketing", True)):
        length_column = "raw_token_count" if "raw_token_count" in dataset.frame else "token_count"
        raw_lengths = dataset.frame[length_column].astype(int).tolist()
        lengths = [min(length, tokenizer.config.max_sequence_length) for length in raw_lengths]
        truncated = sum(length > tokenizer.config.max_sequence_length for length in raw_lengths)
        if context.is_main and truncated:
            print(
                f"Codec tokenizer will truncate {truncated} training segments to "
                f"{tokenizer.config.max_sequence_length} tokens to match the model limit"
            )
        multidataset = cfg.data.get("multidataset", {})
        base_batch_sampler: BatchSamplerProtocol
        if bool(multidataset.get("balanced_sampling", False)):
            base_batch_sampler = DatasetTemperatureLengthBatchSampler(
                lengths,
                dataset_ids=dataset.frame["dataset_id"].astype(int).tolist(),
                batch_size=int(training.batch_size),
                sampling_exponent=float(multidataset.get("sampling_exponent", 0.5)),
                seed=int(training.seed),
            )
        else:
            base_batch_sampler = LengthBucketBatchSampler(
                lengths,
                batch_size=int(training.batch_size),
                seed=int(training.seed),
            )
        batch_sampler = (
            DistributedBatchSampler(
                base_batch_sampler,
                rank=context.rank,
                world_size=context.world_size,
                batch_costs=lengths,
                seed=int(training.seed),
            )
            if context.world_size > 1
            else base_batch_sampler
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=bool(workers),
            prefetch_factor=2 if workers else None,
        )
    else:
        sampler: DistributedSampler[dict[str, torch.Tensor | str | int]] | None = (
            DistributedSampler(
                dataset,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
                seed=int(training.seed),
            )
            if context.world_size > 1
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=int(training.batch_size),
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=bool(workers),
            prefetch_factor=2 if workers else None,
        )
    gradient_accumulation = int(training.gradient_accumulation)
    optimizer_steps_per_epoch = max(1, len(loader) // gradient_accumulation)
    max_steps = int(training.max_steps)
    if training.get("max_epochs") is not None:
        max_steps = min(max_steps, optimizer_steps_per_epoch * int(training.max_epochs))
    if context.is_main:
        global_batch = int(training.batch_size) * context.world_size * gradient_accumulation
        print(
            f"Codec training plan: {len(dataset)} samples, {len(loader)} micro-batches/rank/epoch, "
            f"{optimizer_steps_per_epoch} optimizer steps/epoch, {max_steps} total steps, "
            f"world_size={context.world_size}, global_effective_batch={global_batch}"
        )
    device = context.device
    model = codec_from_config(cfg.codec, tokenizer).to(device)
    optimizer = create_adamw(
        model,
        learning_rate=float(training.learning_rate),
        weight_decay=float(training.weight_decay),
        betas=(float(training.betas[0]), float(training.betas[1])),
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, warmup_steps=int(training.warmup_steps), max_steps=max_steps
    )
    tokenizer_digest = hashlib.sha256(
        json.dumps(asdict(tokenizer.config), sort_keys=True).encode()
    ).hexdigest()
    checkpoint_dir = (
        paths["checkpoints_dir"] / str(cfg.experiment.name) / "codec" / str(cfg.data.name)
    )
    explicit_resume = Path(str(cfg.resume_from)) if cfg.resume_from is not None else None
    resume_checkpoint = resolve_resume_checkpoint(
        checkpoint_dir,
        resume=bool(cfg.get("resume", False)),
        resume_from=explicit_resume,
        announce=context.is_main,
    )
    validation_loader: Iterable[Mapping[str, torch.Tensor | list[str]]] | None = None
    validation_interval = int(training.get("validation_interval", 0))
    if context.is_main and validation_interval > 0:
        validation_frame = _validation_subset(
            frame,
            samples=int(training.get("validation_samples", 384)),
            seed=int(training.seed),
        )
        validation_dataset = MidiTokenDataset(validation_frame, tokenizer, split="validation")
        raw_validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(cfg.codec.inference.batch_size),
            shuffle=False,
            collate_fn=collate,
            num_workers=min(workers, 4),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=bool(min(workers, 4)),
        )
        validation_loader = cast(
            Iterable[Mapping[str, torch.Tensor | list[str]]], raw_validation_loader
        )
    train_codec_steps(
        model,
        loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_steps=max_steps,
        gradient_accumulation=gradient_accumulation,
        gradient_clip_norm=float(training.gradient_clip_norm),
        precision=str(training.precision),
        warmup_steps=int(cfg.codec.kl.warmup_steps),
        beta_max=float(cfg.codec.kl.beta_max),
        free_bits_per_dim=float(cfg.codec.kl.free_bits_per_dim),
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=int(training.checkpoint_interval),
        config={
            **config_mapping(cfg.codec),
            "tokenizer": config_mapping(cfg.tokenizer),
        },
        provenance={"manifest_hash": manifest_digest, "tokenizer_hash": tokenizer_digest},
        ema_decay=float(training.ema.decay) if training.ema.enabled else None,
        resume_from=resume_checkpoint,
        distributed=context,
        validation_batches=validation_loader,
        validation_interval=validation_interval if validation_interval > 0 else None,
        validation_generation_samples=int(training.get("validation_generation_samples", 0)),
        validation_generation_max_length=int(
            training.get("validation_generation_max_length", cfg.codec.max_sequence_length)
        ),
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
