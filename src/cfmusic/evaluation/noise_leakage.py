"""Post-hoc leakage probes trained separately from transport adversaries."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from cfmusic.logging import MetricLogger
from cfmusic.losses.hsic import normalized_hsic
from cfmusic.losses.sliced_wasserstein import sliced_wasserstein_standard_normal
from cfmusic.progress import track


class TemporalNoiseProbe(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(latent_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, noise: Tensor) -> Tensor:
        return self.network(noise.transpose(1, 2))


def _probe_scores(features: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    models = {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed),
        ),
    }
    results: dict[str, float] = {}
    chance = 1 / len(np.unique(labels))
    for name, model in track(
        models.items(), description="Fit leakage probes", total=len(models), unit="model"
    ):
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)
        accuracy = balanced_accuracy_score(test_y, prediction)
        results[f"{name}_balanced_accuracy"] = float(accuracy)
        results[f"{name}_macro_f1"] = float(f1_score(test_y, prediction, average="macro"))
        results[f"{name}_chance_corrected_leakage"] = float((accuracy - chance) / (1 - chance))
    return results


def leakage_metrics(noise: Tensor, labels: Tensor, *, seed: int = 0) -> dict[str, float]:
    if noise.shape[0] != labels.shape[0]:
        raise ValueError("noise and labels batch dimensions differ")
    flattened = noise.detach().float().flatten(1).cpu().numpy()
    label_array = labels.detach().cpu().numpy()
    results = _probe_scores(flattened, label_array, seed)
    projected = torch.as_tensor(flattened)
    results["hsic"] = float(normalized_hsic(projected, labels.cpu()))
    class_means, class_variances = [], []
    for label in torch.unique(labels):
        values = projected[labels.cpu() == label.cpu()]
        class_means.append(float(values.mean(0).square().mean().sqrt()))
        class_variances.append(float((values.var(0, unbiased=False) - 1).square().mean().sqrt()))
    results["classwise_mean_deviation"] = float(np.mean(class_means))
    results["classwise_variance_deviation"] = float(np.mean(class_variances))
    results["projected_sliced_wasserstein"] = float(sliced_wasserstein_standard_normal(projected))
    pairwise: list[float] = []
    pairs = list(combinations(np.unique(label_array), 2))
    for left, right in track(
        pairs, description="Pairwise leakage tests", total=len(pairs), unit="pair"
    ):
        selected = (label_array == left) | (label_array == right)
        binary = (label_array[selected] == right).astype(int)
        if min(np.bincount(binary)) >= 2:
            pairwise.append(
                _probe_scores(flattened[selected], binary, seed)["logistic_balanced_accuracy"]
            )
    results["pairwise_classifier_two_sample_accuracy"] = (
        float(np.mean(pairwise)) if pairwise else 0.5
    )
    return results


def train_temporal_probe(
    noise: Tensor,
    labels: Tensor,
    *,
    epochs: int = 20,
    seed: int = 0,
    log_dir: Path | None = None,
) -> dict[str, float]:
    torch.manual_seed(seed)
    indices = torch.randperm(noise.shape[0])
    split = max(1, round(noise.shape[0] * 0.7))
    train_indices, test_indices = indices[:split], indices[split:]
    if test_indices.numel() == 0:
        test_indices = train_indices
    model = TemporalNoiseProbe(noise.shape[-1], int(labels.max()) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logger = MetricLogger(log_dir, append=False, curve_interval=5) if log_dir else None
    epoch_progress = track(
        range(epochs), description="Train temporal leakage probe", total=epochs, unit="epoch"
    )
    for epoch in epoch_progress:
        logits = model(noise[train_indices].float())
        loss = torch.nn.functional.cross_entropy(logits, labels[train_indices].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_progress.set_postfix(loss=f"{float(loss.detach()):.4f}", refresh=False)
        if logger is not None:
            logger.log(
                {
                    "step": epoch + 1,
                    "loss": float(loss.detach()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
    if logger is not None:
        logger.close()
    prediction = model(noise[test_indices].float()).argmax(-1).detach().numpy()
    truth = labels[test_indices].numpy()
    return {
        "temporal_cnn_balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "temporal_cnn_macro_f1": float(f1_score(truth, prediction, average="macro")),
    }
