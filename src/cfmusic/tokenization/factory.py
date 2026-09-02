"""Tokenizer construction shared by preprocessing, training, and inference."""

from __future__ import annotations

from omegaconf import DictConfig

from cfmusic.tokenization.bar_event import BarEventTokenizer, TokenizerConfig
from cfmusic.tokenization.beat import BeatTokenizer, BeatTokenizerConfig

MidiTokenizer = BarEventTokenizer | BeatTokenizer


def tokenizer_from_config(
    cfg: DictConfig, *, max_sequence_length: int | None = None
) -> MidiTokenizer:
    configured_length = int(cfg.max_sequence_length)
    effective_length = (
        configured_length
        if max_sequence_length is None
        else min(configured_length, max_sequence_length)
    )
    tokenizer_type = str(cfg.get("type", "bar_event"))
    if tokenizer_type == "beat":
        return BeatTokenizer(
            BeatTokenizerConfig(
                implementation_version=str(cfg.get("implementation_version", "cfmusic-beat-v1")),
                steps_per_beat=int(cfg.get("steps_per_beat", 4)),
                max_sequence_length=effective_length,
                preserve_velocity=bool(cfg.get("preserve_velocity", True)),
            )
        )
    if tokenizer_type != "bar_event":
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")
    return BarEventTokenizer(
        TokenizerConfig(
            int(cfg.steps_per_beat),
            int(cfg.velocity_bins),
            int(cfg.tempo_bins),
            float(cfg.tempo_min),
            float(cfg.tempo_max),
            int(cfg.max_duration_beats),
            effective_length,
        )
    )
