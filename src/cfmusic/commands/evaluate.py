"""Evaluate generated artifacts without paired counterfactual references."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from cfmusic.config import CONFIG_DIR, prepare_config
from cfmusic.data.midi_io import load_midi, validate_midi
from cfmusic.evaluation.content_preservation import descriptor_preservation, symbolic_descriptors
from cfmusic.evaluation.noise_leakage import leakage_metrics, train_temporal_probe
from cfmusic.evaluation.style_effect import TokenStyleClassifier
from cfmusic.memory import autocast_context, peak_memory_gib, reset_peak_memory
from cfmusic.progress import track
from cfmusic.tokenization.factory import tokenizer_from_config


def artifact_midi_validity(source: Path, generated: Path) -> dict[str, object]:
    """Return explicit validity metrics without aborting the evaluation run."""

    source_result = validate_midi(source)
    generated_result = validate_midi(generated)
    metrics: dict[str, object] = {
        "source_midi_valid": float(source_result.valid),
        "generated_midi_valid": float(generated_result.valid),
    }
    if source_result.reason is not None:
        metrics["source_midi_error"] = source_result.reason
    if generated_result.reason is not None:
        metrics["generated_midi_error"] = generated_result.reason
    return metrics


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    paths = prepare_config(cfg)
    data_name = str(cfg.data.name)
    artifact_root = paths["artifacts_dir"] / str(cfg.experiment.name) / data_name
    generation_manifest = artifact_root / "generation_manifest.json"
    if generation_manifest.exists():
        generation_index = json.loads(generation_manifest.read_text(encoding="utf-8"))
        metadata_files = [
            artifact_root / str(relative_path)
            for relative_path in generation_index["metadata_files"]
            if (artifact_root / str(relative_path)).exists()
        ]
    else:
        metadata_files = sorted(artifact_root.rglob("counterfactual_metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No generated artifacts found under {artifact_root}")
    descriptor_path = (
        paths["checkpoints_dir"]
        / "evaluators"
        / data_name
        / str(cfg.task)
        / "descriptor_mlp.joblib"
    )
    descriptor_evaluator = joblib.load(descriptor_path) if descriptor_path.exists() else None
    transformer_path = (
        paths["checkpoints_dir"] / "evaluators" / data_name / str(cfg.task) / "transformer.pt"
    )
    tokenizer = tokenizer_from_config(cfg.tokenizer)
    transformer_evaluator: TokenStyleClassifier | None = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_peak_memory(device)
    if transformer_path.exists():
        transformer_checkpoint = torch.load(
            transformer_path, map_location="cpu", weights_only=False
        )
        evaluator_config = transformer_checkpoint["config"]
        evaluator_state = transformer_checkpoint["model"]
        output_weight = evaluator_state["output.weight"]
        transformer_evaluator = TokenStyleClassifier(
            len(tokenizer.vocabulary),
            int(output_weight.shape[0]),
            d_model=int(evaluator_config["d_model"]),
            layers=int(evaluator_config["layers"]),
            heads=int(evaluator_config["heads"]),
            dropout=float(evaluator_config["dropout"]),
            max_length=int(cfg.tokenizer.max_sequence_length),
            gradient_checkpointing=bool(evaluator_config.get("gradient_checkpointing", False)),
        ).to(device)
        transformer_evaluator.load_state_dict(evaluator_state)
        transformer_evaluator.eval()
    rows: list[dict[str, object]] = []
    noise_by_sample: dict[str, tuple[torch.Tensor, int]] = {}
    invalid_generated_midis = 0
    evaluation_progress = track(
        metadata_files,
        description="Evaluate counterfactuals",
        total=len(metadata_files),
        unit="transition",
    )
    for metadata_path in evaluation_progress:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        directory = metadata_path.parent
        source, generated = directory / "source.mid", directory / "counterfactual.mid"
        metrics = artifact_midi_validity(source, generated)
        source_is_valid = bool(metrics["source_midi_valid"])
        generated_is_valid = bool(metrics["generated_midi_valid"])
        if not generated_is_valid:
            invalid_generated_midis += 1
        if source_is_valid and generated_is_valid:
            metrics.update(descriptor_preservation(source, generated))
        if descriptor_evaluator is not None:
            target_id = int(metadata["target_style_id"])
            if generated_is_valid:
                probabilities = descriptor_evaluator.predict_proba(
                    symbolic_descriptors(load_midi(generated))[None]
                )[0]
                if target_id < len(probabilities):
                    metrics["descriptor_target_style_probability"] = float(
                        probabilities[target_id]
                    )
                    metrics["descriptor_target_style_success"] = float(
                        probabilities.argmax() == target_id
                    )
            else:
                metrics["descriptor_target_style_probability"] = 0.0
                metrics["descriptor_target_style_success"] = 0.0
        if transformer_evaluator is not None and generated_is_valid:
            token_ids = tokenizer.encode(load_midi(generated))
            token_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            with (
                torch.inference_mode(),
                autocast_context(device, str(cfg.evaluator.inference.precision)),
            ):
                transformer_probabilities = (
                    transformer_evaluator(
                        token_tensor, token_tensor.ne(tokenizer.vocabulary.pad_id)
                    )
                    .softmax(-1)[0]
                    .float()
                )
            target_id = int(metadata["target_style_id"])
            if target_id < transformer_probabilities.numel():
                metrics["transformer_target_style_probability"] = float(
                    transformer_probabilities[target_id]
                )
                metrics["transformer_target_style_success"] = float(
                    int(transformer_probabilities.argmax()) == target_id
                )
                if "descriptor_target_style_success" in metrics:
                    metrics["classifier_agreement"] = float(
                        metrics["descriptor_target_style_success"]
                        == metrics["transformer_target_style_success"]
                    )
        elif transformer_evaluator is not None:
            metrics["transformer_target_style_probability"] = 0.0
            metrics["transformer_target_style_success"] = 0.0
        row: dict[str, object] = {**metadata, **metrics}
        rows.append(row)
        sample_id = str(metadata["sample_id"])
        if sample_id not in noise_by_sample:
            noise = torch.load(
                directory / "abducted_noise.pt", map_location="cpu", weights_only=True
            )[0]
            noise_by_sample[sample_id] = (noise, int(metadata["source_style_id"]))
        (directory / "evaluation_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        evaluation_progress.set_postfix(
            completed=len(rows), invalid=invalid_generated_midis, refresh=False
        )
    output_dir = artifact_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_transition_results.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number]).mean().to_dict()
    numeric["invalid_generated_midis"] = invalid_generated_midis
    numeric["peak_gpu_memory_gib"] = peak_memory_gib(device)
    (output_dir / "aggregate.json").write_text(json.dumps(numeric, indent=2), encoding="utf-8")
    values = list(noise_by_sample.values())
    if len(values) >= 8 and len({label for _, label in values}) >= 2:
        noises = torch.stack([noise for noise, _ in values])
        labels = torch.tensor([label for _, label in values])
        minimum_class = min(int((labels == label).sum()) for label in torch.unique(labels))
        if minimum_class >= 2:
            leakage = leakage_metrics(noises, labels, seed=int(cfg.seed))
            leakage.update(
                train_temporal_probe(
                    noises,
                    labels,
                    seed=int(cfg.seed),
                    log_dir=output_dir / "temporal_probe_training",
                )
            )
            (output_dir / "noise_leakage.json").write_text(
                json.dumps(leakage, indent=2), encoding="utf-8"
            )
    print(json.dumps(numeric, indent=2))


if __name__ == "__main__":
    main()
