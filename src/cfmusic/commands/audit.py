"""Summarize existing processed dataset audits."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from cfmusic.config import CONFIG_DIR, prepare_config
from cfmusic.progress import track


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    paths = prepare_config(cfg)
    datasets = list(cfg.datasets)
    for dataset in track(
        datasets, description="Read dataset audits", total=len(datasets), unit="dataset"
    ):
        card = paths["processed_dir"] / str(dataset) / "dataset_card.json"
        if not card.exists():
            raise FileNotFoundError(f"Missing audit card for {dataset}: run prepare first")
        payload = json.loads(card.read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
