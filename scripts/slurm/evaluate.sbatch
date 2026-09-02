#!/usr/bin/env bash
#SBATCH --job-name=cfmusic-eval
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail
uv run python -m cfmusic.commands.evaluate "$@"
