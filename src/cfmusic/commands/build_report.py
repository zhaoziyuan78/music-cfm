"""Aggregate experiment artifacts into reproducible result tables and figures."""

from __future__ import annotations

import json

import hydra
import pandas as pd
from omegaconf import DictConfig

from cfmusic.config import CONFIG_DIR, prepare_config
from cfmusic.progress import track
from cfmusic.reporting.tables import write_result_table


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    paths = prepare_config(cfg)
    experiments = [str(value) for value in cfg.report.experiments] or [str(cfg.experiment.name)]
    frames: list[pd.DataFrame] = []
    leakage_rows: list[dict[str, object]] = []
    for experiment in track(
        experiments,
        description="Collect experiment results",
        total=len(experiments),
        unit="experiment",
    ):
        for path in (paths["artifacts_dir"] / experiment).rglob("per_transition_results.csv"):
            frame = pd.read_csv(path)
            frame["experiment"] = experiment
            frames.append(frame)
        for path in (paths["artifacts_dir"] / experiment).rglob("noise_leakage.json"):
            leakage_rows.append(
                {"experiment": experiment, **json.loads(path.read_text(encoding="utf-8"))}
            )
    if not frames:
        raise FileNotFoundError(f"No evaluated experiment results found for {experiments}")
    all_results = pd.concat(frames, ignore_index=True)
    numeric = list(all_results.select_dtypes("number").columns)
    main_results = all_results.groupby("experiment", as_index=False)[numeric].mean()
    reports = paths["reports_dir"]
    write_result_table(main_results, reports / "main_results")
    all_results.to_csv(reports / "per_transition_results.csv", index=False)
    pd.DataFrame(leakage_rows).to_csv(reports / "noise_leakage.csv", index=False)
    for name, columns in {
        "inversion_vs_nfe": ["experiment", "inverse_nfe", "mse"],
        "style_content_pareto": [
            "experiment",
            "clamp2_target_minus_source",
            "pitch_class_histogram_cosine",
            "melody_contour_cosine",
        ],
        "counterfactual_consistency": ["experiment", "mse", "mae", "cosine"],
        "cross_seed_ambiguity": ["experiment"],
    }.items():
        available = [column for column in columns if column in all_results]
        all_results[available].to_csv(reports / f"{name}.csv", index=False)
    (reports / "dataset_cards").mkdir(parents=True, exist_ok=True)
    cards = list(paths["processed_dir"].glob("*/dataset_card.json"))
    for card in track(cards, description="Collect dataset cards", total=len(cards), unit="card"):
        (
            reports / "dataset_cards" / card.name.replace("dataset_card", card.parent.name)
        ).write_text(card.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Report written to {reports}")


if __name__ == "__main__":
    main()
