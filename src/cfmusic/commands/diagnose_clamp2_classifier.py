"""Train a held-out XMIDI genre classifier on frozen CLaMP 2 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import Tensor, nn

from cfmusic.progress import progress_bar, track


@dataclass(frozen=True)
class Candidate:
    """One validation-selected classifier configuration."""

    name: str
    hidden_dims: tuple[int, ...]
    class_weight_power: float
    learning_rate: float


CANDIDATES = (
    Candidate("linear_balanced", (), 1.0, 3e-3),
    Candidate("mlp_unweighted", (512, 256), 0.0, 1e-3),
    Candidate("mlp_sqrt_balanced", (512, 256), 0.5, 1e-3),
    Candidate("mlp_balanced", (512, 256), 1.0, 1e-3),
)


class ClampGenreClassifier(nn.Module):
    """Small MLP operating on standardized, frozen CLaMP 2 embeddings."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = (input_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for current, following in pairwise(dimensions):
            layers.extend([nn.Linear(current, following), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dimensions[-1], num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.network(embeddings)


def _parse_args() -> argparse.Namespace:
    project_root = Path(os.environ.get("CFMUSIC_PROJECT_ROOT", Path.cwd()))
    artifacts_root = Path(
        os.environ.get(
            "CFMUSIC_ARTIFACTS_DIR", "/l/users/gus.xia/ziyuan/music-scm/artifacts"
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-artifact-dir",
        type=Path,
        default=artifacts_root / "diagnostics" / "clamp2_xmidi_genre_centroids",
    )
    parser.add_argument(
        "--centroid-report-dir",
        type=Path,
        default=project_root / "reports" / "diagnostics" / "clamp2_xmidi_genre_centroids",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=artifacts_root / "diagnostics" / "clamp2_xmidi_genre_classifier",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=project_root / "reports" / "diagnostics" / "clamp2_xmidi_genre_classifier",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_embedding_index(artifact_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    file_lists = sorted((artifact_dir / "file_lists").glob("rank-*.jsonl"))
    if not file_lists:
        raise FileNotFoundError(f"No CLaMP 2 embedding index under {artifact_dir}")
    for file_list in file_lists:
        rank = int(file_list.stem.split("-")[-1])
        for line in file_list.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            key = str(json.loads(line)["key"])
            path = artifact_dir / "embeddings" / f"rank-{rank}" / f"{key}.npy"
            if key in index:
                raise ValueError(f"Duplicate embedding key: {key}")
            index[key] = path
    return index


def _normalized_embedding(path: Path) -> np.ndarray:
    feature = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Invalid CLaMP 2 embedding: {path}")
    return feature / norm


def _materialize_feature_matrix(
    sources: pd.DataFrame,
    embedding_index: dict[str, Path],
    *,
    path: Path,
) -> np.ndarray:
    metadata_path = path.with_suffix(".json")
    expected = {
        "rows": len(sources),
        "embedding_keys_sha256": hashlib.sha256(
            "\n".join(sources["embedding_key"].astype(str)).encode()
        ).hexdigest(),
    }
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        matrix = np.load(path, mmap_mode="r")
        if metadata != expected or matrix.shape[0] != len(sources):
            raise ValueError(f"Incompatible frozen feature cache: {path}")
        return matrix
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete feature cache: {path}")
    first_key = str(sources.iloc[0]["embedding_key"])
    first = _normalized_embedding(embedding_index[first_key])
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    matrix = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(sources), len(first))
    )
    for row_index, key in enumerate(
        track(
            sources["embedding_key"].astype(str),
            description="Pack frozen XMIDI embeddings",
            total=len(sources),
            unit="song",
        )
    ):
        matrix[row_index] = _normalized_embedding(embedding_index[key])
    matrix.flush()
    del matrix
    temporary.replace(path)
    metadata_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return np.load(path, mmap_mode="r")


def _metrics(truth: np.ndarray, predicted: np.ndarray, labels: list[str]) -> dict[str, Any]:
    matrix = confusion_matrix(truth, predicted, labels=np.arange(len(labels)))
    report = classification_report(
        truth,
        predicted,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    recalls = np.divide(
        np.diag(matrix), matrix.sum(axis=1), out=np.zeros(len(labels)), where=matrix.sum(axis=1) > 0
    )
    return {
        "accuracy": float((truth == predicted).mean()),
        "balanced_accuracy": float(recalls.mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_genre": {label: report[label] for label in labels},
        "confusion_matrix": matrix.tolist(),
    }


def _predict(model: nn.Module, features: Tensor, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[Tensor] = []
    probabilities: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            logits = model(features[start : start + batch_size])
            probability = logits.softmax(dim=-1)
            probabilities.append(probability.float().cpu())
            predictions.append(probability.argmax(dim=-1).cpu())
    return torch.cat(predictions).numpy(), torch.cat(probabilities).numpy()


def _candidate_weights(labels: Tensor, num_classes: int, power: float) -> Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = (counts.mean() / counts).pow(power)
    return weights / weights.mean()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _train_candidate(
    candidate: Candidate,
    *,
    train_x: Tensor,
    train_y: Tensor,
    validation_x: Tensor,
    validation_y: np.ndarray,
    labels: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Tensor], list[dict[str, float]], dict[str, Any], int]:
    model = ClampGenreClassifier(
        train_x.shape[1],
        len(labels),
        hidden_dims=candidate.hidden_dims,
        dropout=args.dropout,
    ).to(train_x.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.learning_rate, weight_decay=args.weight_decay
    )
    class_weights = _candidate_weights(
        train_y, len(labels), candidate.class_weight_power
    ).to(train_x.device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )
    run_dir = args.artifact_dir / "training" / candidate.name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest.pt"
    log_path = run_dir / "metrics.jsonl"
    start_epoch = 0
    stale_epochs = 0
    best_epoch = 0
    best_score = -1.0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    if args.resume and checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location=train_x.device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"])
        stale_epochs = int(payload["stale_epochs"])
        best_epoch = int(payload["best_epoch"])
        best_score = float(payload["best_score"])
        best_state = payload["best_state"]
        history = payload["history"]
    elif checkpoint_path.exists() or log_path.exists():
        raise FileExistsError(
            f"Existing training state requires --resume; refusing to overwrite {run_dir}"
        )
    log_mode = "a" if start_epoch else "x"
    log_stream = log_path.open(log_mode, encoding="utf-8")
    progress = progress_bar(
        description=f"Train {candidate.name}",
        total=args.epochs,
        initial=start_epoch,
        unit="epoch",
    )
    validation_y_tensor = torch.as_tensor(validation_y, device=train_x.device)
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train_x), device=train_x.device)
            loss_sum = 0.0
            examples = 0
            for start in range(0, len(order), args.batch_size):
                indices = order[start : start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=train_x.device.type,
                    dtype=torch.bfloat16,
                    enabled=train_x.device.type == "cuda",
                ):
                    loss = criterion(model(train_x[indices]), train_y[indices])
                loss.backward()
                optimizer.step()
                count = len(indices)
                loss_sum += float(loss.detach()) * count
                examples += count
            predicted, probabilities = _predict(
                model, validation_x, batch_size=args.batch_size
            )
            validation_metrics = _metrics(validation_y, predicted, labels)
            probability_tensor = torch.from_numpy(probabilities).to(train_x.device)
            validation_loss = float(
                nn.functional.nll_loss(
                    probability_tensor.clamp_min(1e-8).log(), validation_y_tensor
                ).cpu()
            )
            row = {
                "epoch": float(epoch),
                "train_loss": loss_sum / examples,
                "validation_loss": validation_loss,
                "validation_accuracy": float(validation_metrics["accuracy"]),
                "validation_balanced_accuracy": float(
                    validation_metrics["balanced_accuracy"]
                ),
                "validation_macro_f1": float(validation_metrics["macro_f1"]),
            }
            history.append(row)
            log_stream.write(json.dumps(row) + "\n")
            log_stream.flush()
            score = row["validation_balanced_accuracy"]
            if score > best_score + 1e-5:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
            _atomic_torch_save(
                {
                    "candidate": asdict(candidate),
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "stale_epochs": stale_epochs,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "best_state": best_state,
                    "history": history,
                },
                checkpoint_path,
            )
            progress.update(1)
            progress.set_postfix(
                val_bal=f"{score:.4f}", best=f"{best_score:.4f}", stale=stale_epochs
            )
            if epoch >= args.min_epochs and stale_epochs >= args.patience:
                break
    finally:
        progress.close()
        log_stream.close()
    if best_state is None:
        raise RuntimeError(f"Candidate {candidate.name} produced no checkpoint")
    model.load_state_dict(best_state)
    predicted, _ = _predict(model, validation_x, batch_size=args.batch_size)
    return best_state, history, _metrics(validation_y, predicted, labels), best_epoch


def _plot_history(histories: dict[str, list[dict[str, float]]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label=name)
        axes[1].plot(
            epochs,
            [row["validation_balanced_accuracy"] for row in history],
            label=name,
        )
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[1].set(
        title="Validation balanced accuracy", xlabel="Epoch", ylabel="Balanced accuracy"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_confusion(matrix: list[list[int]], labels: list[str], path: Path) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    values /= values.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set(xlabel="Classifier prediction", ylabel="XMIDI genre", title="Held-out XMIDI test")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _evaluate_counterfactuals(
    *,
    model: nn.Module,
    feature_mean: Tensor,
    feature_scale: Tensor,
    embedding_index: dict[str, Path],
    centroid_report_dir: Path,
    report_dir: Path,
    labels: list[str],
    batch_size: int,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for csv_path in sorted(centroid_report_dir.glob("*_counterfactual_predictions.csv")):
        name = csv_path.name.removesuffix("_counterfactual_predictions.csv")
        frame = pd.read_csv(csv_path)
        valid = frame[frame["midi_valid"]].copy()
        raw = np.stack(
            [
                _normalized_embedding(embedding_index[str(key)])
                for key in track(
                    valid["embedding_key"],
                    description=f"Load {name} counterfactual embeddings",
                    total=len(valid),
                    unit="midi",
                )
            ]
        )
        features = torch.from_numpy(raw).to(feature_mean.device)
        features = (features - feature_mean) / feature_scale
        predicted, probabilities = _predict(model, features, batch_size=batch_size)
        target = valid["target_style_id"].to_numpy(dtype=np.int64)
        source = valid["source_style_id"].to_numpy(dtype=np.int64)
        rows = np.arange(len(valid))
        valid["classifier_predicted_genre_id"] = predicted
        valid["classifier_predicted_genre"] = [labels[index] for index in predicted]
        valid["classifier_confidence"] = probabilities.max(axis=1)
        valid["classifier_target_probability"] = probabilities[rows, target]
        valid["classifier_source_probability"] = probabilities[rows, source]
        valid["classifier_target_success"] = predicted == target
        valid["classifier_source_retained"] = predicted == source
        for index, label in enumerate(labels):
            valid[f"classifier_probability_{label}"] = probabilities[:, index]
        invalid = frame[~frame["midi_valid"]].copy()
        combined = pd.concat([valid, invalid], ignore_index=True, sort=False)
        combined.to_csv(report_dir / f"{name}_counterfactual_predictions.csv", index=False)
        summaries[name] = {
            "total": len(frame),
            "valid_midi": len(valid),
            "valid_midi_rate": float(len(valid) / len(frame)),
            "target_success_valid_only": float(valid["classifier_target_success"].mean()),
            "target_success_invalid_as_failure": float(
                valid["classifier_target_success"].sum() / len(frame)
            ),
            "source_retention": float(valid["classifier_source_retained"].mean()),
            "mean_target_probability": float(
                valid["classifier_target_probability"].mean()
            ),
            "mean_source_probability": float(
                valid["classifier_source_probability"].mean()
            ),
            "mean_target_minus_source_probability": float(
                (
                    valid["classifier_target_probability"]
                    - valid["classifier_source_probability"]
                ).mean()
            ),
            "success_by_target_genre": {
                str(target_name): float(group["classifier_target_success"].mean())
                for target_name, group in valid.groupby("target_style", sort=True)
            },
        }
    return summaries


def main() -> None:
    args = _parse_args()
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing classifier diagnostic: {args.report_dir}"
        )
    if min(args.epochs, args.min_epochs, args.patience, args.batch_size) <= 0:
        raise ValueError("Epoch, patience, and batch settings must be positive")
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    sources = pd.read_parquet(args.embedding_artifact_dir / "xmidi_unique_sources.parquet")
    if sources["sample_id"].duplicated().any():
        raise ValueError("The classifier requires exactly one embedding per XMIDI song")
    labels = (
        sources[["genre_id", "genre_label"]]
        .drop_duplicates()
        .sort_values("genre_id")["genre_label"]
        .astype(str)
        .tolist()
    )
    if len(labels) != 6 or set(sources["split"]) != {"train", "validation", "test"}:
        raise ValueError("Expected six genres and distinct train/validation/test splits")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    embedding_index = _load_embedding_index(args.embedding_artifact_dir)
    matrix = _materialize_feature_matrix(
        sources,
        embedding_index,
        path=args.artifact_dir / "xmidi_clamp2_embeddings.npy",
    )
    raw_features = torch.from_numpy(np.array(matrix, copy=True)).to(device)
    all_y = torch.as_tensor(
        sources["genre_id"].to_numpy(dtype=np.int64, copy=True), device=device
    )
    masks = {split: sources["split"].eq(split).to_numpy() for split in ("train", "validation", "test")}
    train_indices = torch.from_numpy(np.flatnonzero(masks["train"])).to(device)
    validation_indices = torch.from_numpy(np.flatnonzero(masks["validation"])).to(device)
    test_indices = torch.from_numpy(np.flatnonzero(masks["test"])).to(device)
    feature_mean = raw_features[train_indices].mean(dim=0)
    feature_scale = raw_features[train_indices].std(dim=0).clamp_min(1e-5)
    features = (raw_features - feature_mean) / feature_scale
    del raw_features
    train_x, train_y = features[train_indices], all_y[train_indices]
    validation_x = features[validation_indices]
    validation_y = all_y[validation_indices].cpu().numpy()
    test_x = features[test_indices]
    test_y = all_y[test_indices].cpu().numpy()

    candidate_results: dict[str, dict[str, Any]] = {}
    candidate_states: dict[str, dict[str, Tensor]] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    for candidate in CANDIDATES:
        state, history, metrics, best_epoch = _train_candidate(
            candidate,
            train_x=train_x,
            train_y=train_y,
            validation_x=validation_x,
            validation_y=validation_y,
            labels=labels,
            args=args,
        )
        candidate_states[candidate.name] = state
        histories[candidate.name] = history
        candidate_results[candidate.name] = {
            "configuration": asdict(candidate),
            "best_epoch": best_epoch,
            "validation": metrics,
        }
    selected = max(
        CANDIDATES,
        key=lambda candidate: (
            candidate_results[candidate.name]["validation"]["balanced_accuracy"],
            candidate_results[candidate.name]["validation"]["macro_f1"],
            candidate_results[candidate.name]["validation"]["accuracy"],
        ),
    )
    model = ClampGenreClassifier(
        features.shape[1],
        len(labels),
        hidden_dims=selected.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(candidate_states[selected.name])
    test_prediction, test_probabilities = _predict(model, test_x, batch_size=args.batch_size)
    test_metrics = _metrics(test_y, test_prediction, labels)

    args.report_dir.mkdir(parents=True)
    test_rows = sources[masks["test"]][
        ["sample_id", "source_midi_path", "genre_id", "genre_label"]
    ].copy()
    test_rows["predicted_genre_id"] = test_prediction
    test_rows["predicted_genre"] = [labels[index] for index in test_prediction]
    test_rows["correct"] = test_prediction == test_y
    test_rows["confidence"] = test_probabilities.max(axis=1)
    for index, label in enumerate(labels):
        test_rows[f"probability_{label}"] = test_probabilities[:, index]
    test_rows.to_parquet(args.report_dir / "xmidi_test_predictions.parquet", index=False)
    pd.DataFrame(
        test_metrics["confusion_matrix"], index=labels, columns=labels
    ).to_csv(args.report_dir / "xmidi_test_confusion_matrix.csv")
    _plot_confusion(
        test_metrics["confusion_matrix"],
        labels,
        args.report_dir / "xmidi_test_confusion_matrix.png",
    )
    _plot_history(histories, args.report_dir / "training_curves.png")
    history_rows = [
        {"candidate": name, **row} for name, history in histories.items() for row in history
    ]
    pd.DataFrame(history_rows).to_csv(args.report_dir / "training_history.csv", index=False)

    counterfactuals = _evaluate_counterfactuals(
        model=model,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        embedding_index=embedding_index,
        centroid_report_dir=args.centroid_report_dir,
        report_dir=args.report_dir,
        labels=labels,
        batch_size=args.batch_size,
    )
    checkpoint = {
        "format": "cfmusic.clamp2_genre_classifier.v1",
        "selected_candidate": asdict(selected),
        "model_state": {name: value.cpu() for name, value in model.state_dict().items()},
        "input_dim": features.shape[1],
        "labels": labels,
        "feature_mean": feature_mean.cpu(),
        "feature_scale": feature_scale.cpu(),
        "dropout": args.dropout,
        "seed": args.seed,
        "source_embedding_artifact": str(args.embedding_artifact_dir.resolve()),
        "source_index_sha256": hashlib.sha256(
            (args.embedding_artifact_dir / "xmidi_unique_sources.parquet").read_bytes()
        ).hexdigest(),
    }
    final_checkpoint = args.artifact_dir / "genre_classifier.pt"
    if final_checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite classifier checkpoint: {final_checkpoint}")
    _atomic_torch_save(checkpoint, final_checkpoint)
    summary = {
        "method": "supervised six-class MLP on frozen normalized CLaMP 2 MIDI embeddings",
        "selection_rule": "highest validation balanced accuracy; test used once after selection",
        "device": str(device),
        "seed": args.seed,
        "split_counts": {split: int(mask.sum()) for split, mask in masks.items()},
        "genre_labels": labels,
        "candidates": candidate_results,
        "selected_candidate": selected.name,
        "xmidi_test": test_metrics,
        "counterfactuals": counterfactuals,
        "checkpoint": str(final_checkpoint.resolve()),
    }
    (args.report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    candidate_rows = "\n".join(
        f"| {name} | {values['best_epoch']} | "
        f"{values['validation']['accuracy']:.4f} | "
        f"{values['validation']['balanced_accuracy']:.4f} | "
        f"{values['validation']['macro_f1']:.4f} |"
        for name, values in candidate_results.items()
    )
    counterfactual_rows = "\n".join(
        f"| {name} | {values['valid_midi']}/{values['total']} | "
        f"{values['target_success_valid_only']:.4f} | "
        f"{values['source_retention']:.4f} | "
        f"{values['mean_target_minus_source_probability']:+.4f} |"
        for name, values in counterfactuals.items()
    )
    report = f"""# Supervised CLaMP 2 XMIDI genre-classifier diagnostic

## Protocol

The classifier uses one frozen, normalized CLaMP 2 MIDI embedding per unique XMIDI song. The
official XMIDI split is preserved: {masks['train'].sum():,} train, {masks['validation'].sum():,}
validation, and {masks['test'].sum():,} held-out test songs. Candidate selection uses validation
balanced accuracy only; test labels are evaluated once after selection.

## Validation model selection

| Candidate | Best epoch | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
{candidate_rows}

Selected: **{selected.name}**.

## Held-out XMIDI test

- Accuracy: **{test_metrics['accuracy']:.4f}**
- Balanced accuracy: **{test_metrics['balanced_accuracy']:.4f}**
- Macro F1: **{test_metrics['macro_f1']:.4f}**

## Generated counterfactuals

Success means that the supervised classifier predicts the requested target genre.

| Result set | Valid MIDI | Target success | Source retained | Target-source probability margin |
|---|---:|---:|---:|---:|
{counterfactual_rows}

See `summary.json`, `xmidi_test_predictions.parquet`, and the per-result CSV files for complete
metrics. These outputs and the checkpoint are isolated from all previous diagnostics.
"""
    (args.report_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Classifier diagnostic written to {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
