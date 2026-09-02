"""Deterministic vocabulary for bar-event MIDI tokenization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VocabularyConfig:
    steps_per_beat: int = 4
    velocity_bins: int = 32
    tempo_bins: int = 32
    max_duration_beats: int = 16


class EventVocabulary:
    def __init__(self, config: VocabularyConfig | None = None) -> None:
        config = config or VocabularyConfig()
        self.config = config
        steps_per_bar = config.steps_per_beat * 4
        duration_steps = config.steps_per_beat * config.max_duration_beats
        tokens = ["PAD", "BOS", "EOS", "UNK", "BAR"]
        tokens += [f"POSITION_{index}" for index in range(steps_per_bar)]
        tokens += [f"PROGRAM_{index}" for index in range(16)] + ["PROGRAM_DRUM"]
        tokens += [f"PITCH_{index}" for index in range(128)]
        tokens += [f"DRUM_PITCH_{index}" for index in range(128)]
        tokens += [f"VELOCITY_{index}" for index in range(config.velocity_bins)]
        tokens += [f"DURATION_{index}" for index in range(duration_steps)]
        tokens += [f"TEMPO_{index}" for index in range(config.tempo_bins)]
        tokens += ["TIME_SIGNATURE_4_4"]
        self.tokens = tuple(tokens)
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def id(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["UNK"])

    def token(self, index: int) -> str:
        return self.tokens[index] if 0 <= index < len(self.tokens) else "UNK"

    @property
    def pad_id(self) -> int:
        return self.id("PAD")

    @property
    def bos_id(self) -> int:
        return self.id("BOS")

    @property
    def eos_id(self) -> int:
        return self.id("EOS")
