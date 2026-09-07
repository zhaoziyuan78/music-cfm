"""Build the report for the distribution-matched counterfactual diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_PROJECT = Path("/home/gus.xia/ziyuan/music-scm")
DEFAULT_STORAGE = Path("/l/users/gus.xia/ziyuan/music-scm")
EXPERIMENT = "e34_proportional_300"


def _markdown_table(frame: pd.DataFrame, *, digits: int = 6) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.{digits}f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _mean(frame: pd.DataFrame, name: str) -> float | None:
    if name not in frame:
        return None
    values = pd.to_numeric(frame[name], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _metric_table(current: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "clamp2_target_style_success": "Target-style success",
        "clamp2_target_minus_source": "Target-source margin",
        "clamp2_target_similarity": "Target-style similarity",
        "clamp2_source_similarity": "Source-style similarity",
        "descriptor_cosine": "Descriptor cosine",
        "pitch_class_histogram_cosine": "Pitch-class cosine",
        "melody_contour_cosine": "Melody-contour cosine",
        "note_density_ratio": "Note-density ratio",
        "tempo_ratio": "Tempo ratio",
        "source_midi_valid": "Source MIDI valid",
        "generated_midi_valid": "Generated MIDI valid",
        "generated_note_count": "Generated note count",
        "generated_duration_beats": "Generated duration (beats)",
        "generated_note_density": "Generated note density",
        "generated_unique_pitch_classes": "Generated unique pitch classes",
        "generated_pitch_range": "Generated pitch range",
        "generated_nonpositive_duration_ratio": "Nonpositive duration ratio",
    }
    rows: list[dict[str, object]] = []
    current_pairs = current.groupby(["source_style_id", "target_style_id"]).size()
    for name, label in metrics.items():
        value = _mean(current, name)
        if value is None:
            continue
        baseline_value = _mean(baseline, name)
        baseline_by_pair = (
            baseline.groupby(["source_style_id", "target_style_id"])[name].mean()
            if name in baseline
            else pd.Series(dtype=float)
        )
        missing_pairs = [pair for pair in current_pairs.index if pair not in baseline_by_pair]
        reweighted_baseline = (
            sum(
                float(baseline_by_pair.loc[pair]) * int(count)
                for pair, count in current_pairs.items()
            )
            / len(current)
            if not missing_pairs and len(baseline_by_pair)
            else None
        )
        rows.append(
            {
                "metric": label,
                "proportional_300": value,
                "balanced_300_baseline": baseline_value,
                "delta_vs_balanced": value - baseline_value
                if baseline_value is not None
                else math.nan,
                "pair_mix_reweighted_baseline": reweighted_baseline,
                "delta_vs_reweighted": value - reweighted_baseline
                if reweighted_baseline is not None
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _noise_table(current: dict[str, Any], baseline: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, value in current.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        baseline_value = baseline.get(name)
        rows.append(
            {
                "metric": name,
                "proportional_300": float(value),
                "balanced_60_noise_baseline": float(baseline_value)
                if isinstance(baseline_value, (int, float))
                else math.nan,
                "descriptive_delta": float(value) - float(baseline_value)
                if isinstance(baseline_value, (int, float))
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON mapping: {path}")
    return payload


def build_report(args: argparse.Namespace) -> None:
    artifact_root = args.artifact_root.expanduser().resolve()
    baseline_root = args.baseline_root.expanduser().resolve()
    report_root = args.report_root.expanduser().resolve()
    manifest = _load_json(artifact_root / "generation_manifest.json")
    results = pd.read_csv(artifact_root / "evaluation" / "per_transition_results.csv")
    baseline = pd.read_csv(baseline_root / "evaluation" / "per_transition_results.csv")
    if len(results) != 300:
        raise ValueError(f"Expected exactly 300 proportional transitions, found {len(results)}")
    if results["sample_id"].astype(str).nunique() != 300:
        raise ValueError("The proportional diagnostic must use 300 unique source samples")
    if (results["source_style_id"] == results["target_style_id"]).any():
        raise ValueError("A proportional target unexpectedly equals its source style")

    card = _load_json(args.dataset_card.expanduser().resolve())
    labels = [str(value) for value in card["style_vocabulary"]]
    latent_frame = pd.read_parquet(
        args.latent_index.expanduser().resolve(),
        columns=["split", "sample_id", "style_id"],
    )
    population = latent_frame.loc[latent_frame["split"] == "test"].drop_duplicates("sample_id")
    population_counts = population["style_id"].astype(int).value_counts().sort_index()
    source_counts = results["source_style_id"].astype(int).value_counts().sort_index()
    target_counts = results["target_style_id"].astype(int).value_counts().sort_index()

    distribution_rows = []
    total_population = int(population_counts.sum())
    for style_id, label in enumerate(labels):
        population_count = int(population_counts.get(style_id, 0))
        source_count = int(source_counts.get(style_id, 0))
        target_count = int(target_counts.get(style_id, 0))
        distribution_rows.append(
            {
                "genre": label,
                "test_unique_midis": population_count,
                "dataset_share": population_count / total_population,
                "source_count": source_count,
                "source_share": source_count / len(results),
                "target_count": target_count,
                "target_share": target_count / len(results),
            }
        )
    distribution = pd.DataFrame(distribution_rows)

    pair_counts = pd.crosstab(results["source_style"], results["target_style"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    pair_counts.index.name = "source"
    pair_table = pair_counts.reset_index()

    pair_metrics = (
        results.groupby(["source_style", "target_style"], sort=True)
        .agg(
            n=("sample_id", "size"),
            target_style_success=("clamp2_target_style_success", "mean"),
            target_minus_source_similarity=("clamp2_target_minus_source", "mean"),
            descriptor_cosine=("descriptor_cosine", "mean"),
            pitch_class_cosine=("pitch_class_histogram_cosine", "mean"),
            melody_contour_cosine=("melody_contour_cosine", "mean"),
        )
        .reset_index()
    )
    metric_table = _metric_table(results, baseline)

    report_root.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(report_root / "sample_distribution.csv", index=False)
    pair_table.to_csv(report_root / "transition_count_matrix.csv", index=False)
    pair_metrics.to_csv(report_root / "per_pair_metrics.csv", index=False)
    metric_table.to_csv(report_root / "aggregate_metrics.csv", index=False)

    noise = _load_json(artifact_root / "evaluation" / "noise_leakage.json")
    baseline_noise_path = baseline_root / "evaluation" / "noise_leakage.json"
    baseline_noise = _load_json(baseline_noise_path) if baseline_noise_path.is_file() else {}
    noise_table = _noise_table(noise, baseline_noise)
    noise_table.to_csv(report_root / "noise_metrics.csv", index=False)

    def json_value(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    summary = {
        "experiment": EXPERIMENT,
        "sampling_seed": int(args.seed),
        "sampling_population": "XMIDI test split unique sample_id",
        "transitions": len(results),
        "unique_sources": results["sample_id"].astype(str).nunique(),
        "source_style_counts": manifest.get("source_style_counts", {}),
        "transition_pair_counts": manifest.get("transition_pair_counts", {}),
        "metrics": {
            str(row["metric"]): {
                "proportional_300": json_value(row["proportional_300"]),
                "balanced_300_baseline": json_value(row["balanced_300_baseline"]),
                "delta_vs_balanced": json_value(row["delta_vs_balanced"]),
                "pair_mix_reweighted_baseline": json_value(row["pair_mix_reweighted_baseline"]),
                "delta_vs_reweighted": json_value(row["delta_vs_reweighted"]),
            }
            for row in metric_table.to_dict("records")
        },
        "noise_leakage": noise,
        "balanced_baseline_noise_leakage": baseline_noise,
    }
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    success = _mean(results, "clamp2_target_style_success")
    margin = _mean(results, "clamp2_target_minus_source")
    valid = _mean(results, "generated_midi_valid")
    report = f"""# Proportional XMIDI counterfactual diagnostic

## Experimental design

- Model: current `{manifest.get("transport_weights", "raw")}` E34 CFM checkpoint with CFG.
- Evaluation population: XMIDI test split, deduplicated by `sample_id`.
- Sources: exactly 300 unique MIDI samples, allocated to genres by largest-remainder quotas from the empirical label distribution.
- Targets: one target per source. Within every source genre, the other five genres are allocated by their empirical frequencies after excluding and renormalizing the source genre.
- Seed: {args.seed}.
- Some ordered genre pairs can have zero observations by design, especially for rare source genres.

## Sampling distribution

{_markdown_table(distribution, digits=4)}

## Transition count matrix

{_markdown_table(pair_table, digits=0)}

## Aggregate evaluation

- Valid generated MIDI: {valid:.2%}
- Target-style success: {success:.2%}
- Mean target-minus-source CLaMP2 margin: {margin:.6f}

{_markdown_table(metric_table)}

The balanced baseline is the existing 300-transition E34 experiment (10 source samples per genre, each sent to all five other genres). `pair_mix_reweighted_baseline` applies this experiment's 28 observed pair counts to the old experiment's per-pair means. Both deltas are descriptive rather than paired: this experiment uses 300 unique sources, while the baseline reuses 60 sources across five targets.

## Per observed genre pair

{_markdown_table(pair_metrics)}

Pair rows with very small `n` are reported for completeness and should not be interpreted as stable pair-level estimates.

## Noise diagnostics

{_markdown_table(noise_table)}

The old balanced experiment has only 60 unique abducted noises because each source noise is shared across five targets; this experiment has 300 unique noises. Noise deltas are therefore descriptive, and rare-class probe estimates (especially jazz with five sources) have high uncertainty.

## Files

- `sample_distribution.csv`: empirical population, source, and target proportions.
- `transition_count_matrix.csv`: all source-target counts, including zero-count pairs.
- `per_pair_metrics.csv`: metrics for the ordered pairs that occurred.
- `aggregate_metrics.csv`: proportional experiment and descriptive balanced-baseline comparison.
- `noise_metrics.csv`: noise-probe metrics and descriptive baseline comparison.
- `summary.json`: machine-readable metrics plus noise-probe results.
"""
    (report_root / "report.md").write_text(report, encoding="utf-8")
    print(f"Report: {report_root / 'report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_STORAGE / "artifacts" / "diagnostics" / EXPERIMENT / "xmidi",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_STORAGE / "artifacts" / "e34_cfm_clamp2_prompt_cfg_roundtrip" / "xmidi",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_PROJECT / "reports" / "diagnostics" / EXPERIMENT,
    )
    parser.add_argument(
        "--latent-index",
        type=Path,
        default=DEFAULT_STORAGE / "data" / "latents" / "xmidi" / "index_clamp2_prompt.parquet",
    )
    parser.add_argument(
        "--dataset-card",
        type=Path,
        default=DEFAULT_STORAGE / "data" / "processed" / "xmidi" / "dataset_card.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    build_report(parse_args())
