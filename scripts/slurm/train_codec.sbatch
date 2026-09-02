#!/usr/bin/env bash
#SBATCH --job-name=cfmusic-codec
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail
experiment_name="${CFMUSIC_CODEC_EXPERIMENT:-e00_xmidi_codec}"
uv run python -m cfmusic.commands.train_codec experiment="$experiment_name" "$@"
