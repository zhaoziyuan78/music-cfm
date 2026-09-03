"""Evaluate generated artifacts without paired counterfactual references."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from cfmusic.conditioning.schema import CONDITION_SCHEMA_VERSION
from cfmusic.config import CONFIG_DIR, prepare_config
from cfmusic.data.midi_io import validate_midi
from cfmusic.evaluation.clamp2 import (
    clamp2_style_metrics,
    extract_clamp2_embeddings,
    style_prompt,
)
from cfmusic.evaluation.content_preservation import (
    descriptor_preservation,
    midi_quality_metrics,
)
from cfmusic.evaluation.noise_leakage import leakage_metrics, train_temporal_probe
from cfmusic.memory import peak_memory_gib, reset_peak_memory
from cfmusic.progress import track


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_peak_memory(device)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_files]
    incompatible = [
        str(path)
        for path, record in zip(metadata_files, records, strict=True)
        if record.get("condition_schema_version") != CONDITION_SCHEMA_VERSION
    ]
    if incompatible:
        raise ValueError(
            "Generated artifacts use an obsolete condition schema; regenerate them before "
            f"evaluation (first incompatible artifact: {incompatible[0]})"
        )
    clamp_cfg = cfg.evaluation.clamp2
    clamp_embeddings: dict[str, np.ndarray] = {}
    style_embeddings: list[np.ndarray] = []
    if bool(clamp_cfg.enabled):
        card = json.loads(
            (paths["processed_dir"] / data_name / "dataset_card.json").read_text(encoding="utf-8")
        )
        factorial_axes = {
            str(record["factorial_intervention"])
            for record in records
            if record.get("factorial_intervention") is not None
        }
        if len(factorial_axes) > 1:
            raise ValueError("One evaluation run cannot mix factorial intervention axes")
        vocabulary_key = (
            f"{next(iter(factorial_axes))}_vocabulary" if factorial_axes else "style_vocabulary"
        )
        style_labels = [str(value) for value in card[vocabulary_key]]
        midi_files: dict[str, Path] = {}
        for index, (metadata_path, _metadata) in enumerate(
            zip(metadata_files, records, strict=True)
        ):
            generated = metadata_path.parent / "counterfactual.mid"
            if validate_midi(generated).valid:
                midi_files[f"generated-{index:08d}"] = generated
        template = str(clamp_cfg.style_template)
        prompts = {
            f"style-{index:04d}": style_prompt(label, template)
            for index, label in enumerate(style_labels)
        }
        clamp_embeddings = extract_clamp2_embeddings(
            repository=Path(str(clamp_cfg.repository)).expanduser().resolve(),
            midi_files=midi_files,
            texts=prompts,
            python_executable=(
                str(clamp_cfg.python_executable)
                if clamp_cfg.get("python_executable") is not None
                else None
            ),
            cache_dir=Path(str(clamp_cfg.cache_dir)).expanduser().resolve(),
        )
        style_embeddings = [
            clamp_embeddings[f"style-{index:04d}"] for index in range(len(style_labels))
        ]
    rows: list[dict[str, object]] = []
    noise_by_sample: dict[str, tuple[torch.Tensor, int]] = {}
    invalid_generated_midis = 0
    evaluation_progress = track(
        metadata_files,
        description="Evaluate counterfactuals",
        total=len(metadata_files),
        unit="transition",
    )
    for artifact_index, (metadata_path, metadata) in enumerate(
        zip(evaluation_progress, records, strict=True)
    ):
        directory = metadata_path.parent
        source, generated = directory / "source.mid", directory / "counterfactual.mid"
        metrics = artifact_midi_validity(source, generated)
        source_is_valid = bool(metrics["source_midi_valid"])
        generated_is_valid = bool(metrics["generated_midi_valid"])
        if not generated_is_valid:
            invalid_generated_midis += 1
        if generated_is_valid:
            metrics.update(midi_quality_metrics(generated))
        if source_is_valid and generated_is_valid:
            metrics.update(descriptor_preservation(source, generated))
        if generated_is_valid and style_embeddings:
            metrics.update(
                clamp2_style_metrics(
                    clamp_embeddings[f"generated-{artifact_index:08d}"],
                    style_embeddings=style_embeddings,
                    source_style_id=int(metadata["source_style_id"]),
                    target_style_id=int(metadata["target_style_id"]),
                )
            )
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
