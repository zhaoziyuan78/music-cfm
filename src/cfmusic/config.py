"""Hydra configuration helpers shared by command entry points."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from cfmusic.paths import ensure_paths, resolved_paths

_PACKAGE_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_PROJECT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
CONFIG_DIR = str(_PACKAGE_CONFIG_DIR if _PACKAGE_CONFIG_DIR.is_dir() else _PROJECT_CONFIG_DIR)


def prepare_config(cfg: DictConfig) -> dict[str, Path]:
    """Resolve, create, and print all configured directories."""
    OmegaConf.resolve(cfg)
    paths = resolved_paths(cfg)
    ensure_paths(paths)
    if int(os.environ.get("RANK", "0")) == 0:
        print("Resolved paths:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    return paths


def config_mapping(cfg: DictConfig) -> dict[str, object]:
    """Convert a DictConfig to a string-keyed mapping for checkpoint metadata."""
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Expected a mapping configuration")
    return {str(key): item for key, item in value.items()}
