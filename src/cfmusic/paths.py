"""Path resolution and reproducible run metadata."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def resolve_path(path: str | Path) -> Path:
    """Resolve an absolute or cwd-relative path without requiring it to exist."""
    candidate = Path(os.path.expandvars(str(path))).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


def resolved_paths(cfg: DictConfig) -> dict[str, Path]:
    """Return all configured path values as absolute Paths."""
    return {str(key): resolve_path(value) for key, value in cfg.paths.items()}


def ensure_paths(paths: Mapping[str, Path]) -> None:
    """Create configured output directories."""
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def _git_state(project_root: Path) -> dict[str, object]:
    def run(args: list[str]) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_files": (run(["status", "--porcelain"]) or "").splitlines(),
    }


def save_run_context(cfg: DictConfig, run_dir: str | Path) -> None:
    """Persist resolved configuration, runtime environment, and git state."""
    target = resolve_path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, target / "config_resolved.yaml", resolve=True)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "torch_version": _optional_torch_version(),
    }
    (target / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
    )
    project_root = resolve_path(cfg.paths.project_root)
    (target / "git_state.json").write_text(
        json.dumps(_git_state(project_root), indent=2, sort_keys=True), encoding="utf-8"
    )


def _optional_torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return None
