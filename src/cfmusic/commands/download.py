"""Hydra CLI for licensed, resumable dataset downloads."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from cfmusic.config import CONFIG_DIR, prepare_config
from cfmusic.download.licenses import DatasetLicense, require_license_acknowledgement
from cfmusic.download.registry import create_source
from cfmusic.progress import track


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    paths = prepare_config(cfg)
    datasets = list(cfg.datasets)
    for dataset in track(
        datasets, description="Download datasets", total=len(datasets), unit="dataset"
    ):
        name = str(dataset)
        if name not in cfg.download.datasets:
            raise KeyError(f"Unknown dataset {name!r}")
        dataset_cfg = cfg.download.datasets[name]
        license_cfg = dataset_cfg.license
        require_license_acknowledgement(
            name,
            DatasetLicense(
                str(license_cfg.name),
                bool(license_cfg.requires_acknowledgement),
                license_cfg.get("commercial_use"),
            ),
            accept=bool(cfg.license.accept),
            acknowledge_unknown=bool(cfg.license.acknowledge_unknown),
        )
        destination = paths["raw_dir"] / name
        result = create_source(name, dataset_cfg).download(
            destination,
            resume=bool(cfg.download.resume),
            force=bool(cfg.download.force),
            dry_run=bool(cfg.download.dry_run),
        )
        if cfg.download.dry_run:
            print(f"DRY RUN {name}: {result.source} -> {destination}")
            continue
        manifest = destination / "download_manifest.json"
        manifest.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"Downloaded {name}; manifest: {manifest}")


if __name__ == "__main__":
    main()
