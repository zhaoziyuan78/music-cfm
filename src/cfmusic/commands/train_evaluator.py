"""Train real-data-only token or descriptor style evaluators."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from cfmusic.config import CONFIG_DIR, config_mapping, prepare_config
from cfmusic.data.collate import collate_token_batch
from cfmusic.data.datasets import MidiTokenDataset
from cfmusic.data.samplers import DistributedBatchSampler, LengthBucketBatchSampler
from cfmusic.distributed import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
)
from cfmusic.evaluation.content_preservation import symbolic_descriptors
from cfmusic.evaluation.style_effect import TokenStyleClassifier, expected_calibration_error
from cfmusic.logging import MetricLogger
from cfmusic.memory import autocast_context
from cfmusic.progress import progress_bar, track
from cfmusic.reproducibility import seed_everything
from cfmusic.tokenization.factory import tokenizer_from_config
from cfmusic.training.checkpointing import resolve_resume_checkpoint
from cfmusic.training.evaluator_trainer import train_token_evaluator


def _train_descriptor_mlp(
    cfg: DictConfig,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
    resume_checkpoint: Path | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    train_x = np.stack(
        [
            symbolic_descriptors(Path(path))
            for path in track(
                train["source_midi_path"],
                description="Extract training descriptors",
                total=len(train),
                unit="file",
            )
        ]
    )
    validation_x = np.stack(
        [
            symbolic_descriptors(Path(path))
            for path in track(
                validation["source_midi_path"],
                description="Extract validation descriptors",
                total=len(validation),
                unit="file",
            )
        ]
    )
    train_y = train["style_id"].to_numpy(dtype=np.int64)
    truth = validation["style_id"].to_numpy(dtype=np.int64)
    step = 0
    best_loss = float("inf")
    stale_steps = 0
    if resume_checkpoint is not None:
        payload = joblib.load(resume_checkpoint)
        if not isinstance(payload, dict):
            raise TypeError("Invalid descriptor MLP checkpoint")
        scaler = payload.get("scaler")
        classifier = payload.get("classifier")
        if not isinstance(scaler, StandardScaler) or not isinstance(classifier, MLPClassifier):
            raise TypeError("Descriptor checkpoint has incompatible estimator state")
        step = int(payload.get("global_step", 0))
        best_loss = float(payload.get("best_loss", float("inf")))
        stale_steps = int(payload.get("stale_steps", 0))
    else:
        scaler = StandardScaler().fit(train_x)
        classifier = MLPClassifier(
            hidden_layer_sizes=tuple(int(value) for value in cfg.evaluator.hidden_layers),
            max_iter=1,
            warm_start=True,
            random_state=int(cfg.evaluator.random_state),
        )
    train_scaled = scaler.transform(train_x)
    classes = np.unique(train_y)
    max_steps = int(cfg.evaluator.max_iter)
    checkpoint_interval = int(cfg.evaluator.get("checkpoint_interval", 25))
    minimum_steps = int(cfg.evaluator.get("min_iter", 20))
    patience = int(cfg.evaluator.get("early_stopping_patience", 10))
    tolerance = float(cfg.evaluator.get("early_stopping_tolerance", 1e-4))
    if checkpoint_interval <= 0:
        raise ValueError("evaluator.checkpoint_interval must be positive")
    progress = progress_bar(
        description="Train descriptor MLP",
        total=max_steps,
        initial=step,
        unit="step",
    )
    checkpoint_path = output_dir / "last.joblib"
    logger = MetricLogger(
        output_dir / "descriptor_mlp_training",
        append=resume_checkpoint is not None,
        curve_interval=5,
    )
    while step < max_steps:
        classifier.partial_fit(train_scaled, train_y, classes=classes)
        step += 1
        loss = float(classifier.loss_)
        if loss < best_loss - tolerance:
            best_loss = loss
            stale_steps = 0
        else:
            stale_steps += 1
        progress.update(1)
        progress.set_postfix(loss=f"{loss:.4f}", stale=stale_steps, refresh=False)
        logger.log(
            {
                "step": step,
                "loss": loss,
                "best_loss": best_loss,
                "stale_steps": stale_steps,
            }
        )
        converged = step >= minimum_steps and stale_steps >= patience
        if step % checkpoint_interval == 0 or step == max_steps or converged:
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            joblib.dump(
                {
                    "scaler": scaler,
                    "classifier": classifier,
                    "global_step": step,
                    "best_loss": best_loss,
                    "stale_steps": stale_steps,
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
        if converged:
            break
    progress.close()
    logger.close()
    model = make_pipeline(scaler, classifier)
    joblib.dump(model, output_dir / "descriptor_mlp.joblib")
    prediction = model.predict(validation_x).astype(np.int64, copy=False)
    probabilities = model.predict_proba(validation_x).astype(np.float64, copy=False)
    return prediction, probabilities, truth


def _train(cfg: DictConfig, context: DistributedContext) -> None:
    paths = prepare_config(cfg)
    seed_everything(int(cfg.seed))
    manifest_path = paths["processed_dir"] / str(cfg.data.name) / "manifest.parquet"
    frame = pd.read_parquet(manifest_path)
    output_dir = paths["checkpoints_dir"] / "evaluators" / str(cfg.data.name) / str(cfg.task)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = frame.loc[(frame["split"] == "train") & frame["valid"]].drop_duplicates("sample_id")
    validation = frame.loc[(frame["split"] == "validation") & frame["valid"]].drop_duplicates(
        "sample_id"
    )
    if validation.empty:
        validation = train.copy()
        validation["split"] = "validation"
    if str(cfg.evaluator.type) == "descriptor_mlp":
        if context.world_size > 1:
            raise ValueError(
                "descriptor_mlp is a CPU scikit-learn baseline; run it once without torchrun"
            )
        explicit_resume = Path(str(cfg.resume_from)) if cfg.resume_from is not None else None
        resume_checkpoint = resolve_resume_checkpoint(
            output_dir,
            resume=bool(cfg.get("resume", False)),
            resume_from=explicit_resume,
            checkpoint_name="last.joblib",
        )
        prediction, probabilities, truth = _train_descriptor_mlp(
            cfg, train, validation, output_dir, resume_checkpoint
        )
    else:
        tokenizer = tokenizer_from_config(cfg.tokenizer)
        # One factual segment per song is sufficient for the evaluator and prevents
        # overlapping windows from multiplying XMIDI training by roughly 20x.
        train_dataset = MidiTokenDataset(train, tokenizer, split="train")
        validation_dataset = MidiTokenDataset(validation, tokenizer, split="validation")
        lengths = train_dataset.frame["token_count"].astype(int).tolist()
        base_batch_sampler = LengthBucketBatchSampler(
            lengths,
            batch_size=int(cfg.evaluator.training.batch_size),
            seed=int(cfg.seed),
        )
        batch_sampler = (
            DistributedBatchSampler(
                base_batch_sampler,
                rank=context.rank,
                world_size=context.world_size,
                batch_costs=lengths,
                seed=int(cfg.seed),
            )
            if context.world_size > 1
            else base_batch_sampler
        )
        collate = partial(collate_token_batch, pad_id=tokenizer.vocabulary.pad_id)
        workers = int(cfg.evaluator.training.get("dataloader_workers", 4))
        loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=context.device.type == "cuda",
            persistent_workers=workers > 0,
            prefetch_factor=int(cfg.evaluator.training.get("prefetch_factor", 2))
            if workers > 0
            else None,
        )
        num_classes = int(frame["style_id"].max()) + 1
        model = TokenStyleClassifier(
            len(tokenizer.vocabulary),
            num_classes,
            d_model=int(cfg.evaluator.d_model),
            layers=int(cfg.evaluator.layers),
            heads=int(cfg.evaluator.heads),
            dropout=float(cfg.evaluator.dropout),
            max_length=int(cfg.tokenizer.max_sequence_length),
            gradient_checkpointing=bool(cfg.evaluator.get("gradient_checkpointing", False)),
        )
        device = context.device
        model.to(device)
        explicit_resume = Path(str(cfg.resume_from)) if cfg.resume_from is not None else None
        resume_checkpoint = resolve_resume_checkpoint(
            output_dir,
            resume=bool(cfg.get("resume", False)),
            resume_from=explicit_resume,
            announce=context.is_main,
        )
        gradient_accumulation = int(cfg.evaluator.training.gradient_accumulation)
        optimizer_steps_per_epoch = (
            len(loader) + gradient_accumulation - 1
        ) // gradient_accumulation
        max_steps = int(cfg.evaluator.training.max_steps)
        if cfg.evaluator.training.get("max_epochs") is not None:
            max_steps = min(
                max_steps,
                max(
                    int(cfg.evaluator.training.get("min_steps", 1)),
                    int(cfg.evaluator.training.max_epochs) * optimizer_steps_per_epoch,
                ),
            )
        if context.is_main:
            global_batch = (
                int(cfg.evaluator.training.batch_size)
                * int(cfg.evaluator.training.gradient_accumulation)
                * context.world_size
            )
            print(
                f"Evaluator training plan: {len(train_dataset)} samples, {len(loader)} "
                f"micro-batches/rank/epoch, world_size={context.world_size}, "
                f"global_effective_batch={global_batch}, optimizer_steps={max_steps} "
                f"(hard_cap={int(cfg.evaluator.training.max_steps)})"
            )
        train_token_evaluator(
            model,
            loader,
            device=device,
            max_steps=max_steps,
            learning_rate=float(cfg.evaluator.training.learning_rate),
            weight_decay=float(cfg.evaluator.training.weight_decay),
            checkpoint_dir=output_dir,
            checkpoint_interval=int(cfg.evaluator.training.checkpoint_interval),
            config=config_mapping(cfg.evaluator),
            provenance={"training_manifest": str(manifest_path)},
            gradient_accumulation=gradient_accumulation,
            precision=str(cfg.evaluator.training.precision),
            log_interval=int(cfg.evaluator.training.get("log_interval", 10)),
            resume_from=resume_checkpoint,
            distributed=context,
        )
        if not context.is_main:
            return
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(cfg.evaluator.inference.batch_size),
            shuffle=False,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=context.device.type == "cuda",
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
        )
        all_probabilities, all_truth = [], []
        model.eval()
        with torch.inference_mode():
            for batch in track(
                validation_loader,
                description="Validate evaluator",
                total=len(validation_loader),
                unit="batch",
            ):
                tokens, mask = batch["tokens"], batch["attention_mask"]
                if not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor):
                    raise TypeError("Invalid validation batch")
                with autocast_context(device, str(cfg.evaluator.inference.precision)):
                    probabilities_batch = model(
                        tokens.to(device, non_blocking=True),
                        mask.to(device, non_blocking=True),
                    ).softmax(-1)
                all_probabilities.append(probabilities_batch.float().cpu().numpy())
                labels = batch["style_id"]
                if not isinstance(labels, torch.Tensor):
                    raise TypeError("Invalid validation labels")
                all_truth.append(labels.numpy())
        probabilities = np.concatenate(all_probabilities)
        truth = np.concatenate(all_truth)
        prediction = probabilities.argmax(1)
        torch.save(
            {
                "model": model.state_dict(),
                "config": OmegaConf.to_container(cfg.evaluator, resolve=True),
            },
            output_dir / "transformer.pt",
        )
    metrics = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(truth, prediction).tolist(),
        "calibration_error": expected_calibration_error(probabilities, truth),
        "trained_on_generated_samples": False,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    context = initialize_distributed()
    try:
        _train(cfg, context)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
