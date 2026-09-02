#!/usr/bin/env bash
#SBATCH --job-name=cfmusic-transport
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail
uv run python -m cfmusic.commands.train_transport experiment=e20_cfm_base "$@"
