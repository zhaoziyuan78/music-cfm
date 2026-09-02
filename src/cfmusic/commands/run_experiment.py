"""Execute the configured training stage as a reproducible subprocess."""

from __future__ import annotations

import subprocess
import sys

import hydra
from omegaconf import DictConfig

from cfmusic.config import CONFIG_DIR


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    stage = str(cfg.experiment.stage)
    module = {
        "codec": "cfmusic.commands.train_codec",
        "transport": "cfmusic.commands.train_transport",
        "abduction": "cfmusic.commands.finetune_abduction",
    }.get(stage)
    if module is None:
        raise ValueError(f"Unknown experiment stage: {stage}")
    command = [sys.executable, "-m", module, f"experiment={cfg.experiment.name}"]
    if cfg.codec_checkpoint is not None:
        command.append(f"codec_checkpoint={cfg.codec_checkpoint}")
    if cfg.transport_checkpoint is not None:
        command.append(f"transport_checkpoint={cfg.transport_checkpoint}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
