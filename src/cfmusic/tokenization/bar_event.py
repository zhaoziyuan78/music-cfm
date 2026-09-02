"""Internal bar-event tokenizer with fixed note-event grammar."""

from __future__ import annotations

import math
from dataclasses import dataclass

import miditoolkit

from cfmusic.tokenization.vocabulary import EventVocabulary, VocabularyConfig


def encoded_segment_length(num_bars: int, num_notes: int, max_sequence_length: int) -> int:
    """Compute bar-event length without allocating the token sequence."""

    if num_bars <= 0 or num_notes < 0 or max_sequence_length <= 0:
        raise ValueError("bars/maximum length must be positive and notes must be non-negative")
    return min(max_sequence_length, 2 + 3 * num_bars + 5 * num_notes)


@dataclass(frozen=True)
class TokenizerConfig:
    steps_per_beat: int = 4
    velocity_bins: int = 32
    tempo_bins: int = 32
    tempo_min: float = 30.0
    tempo_max: float = 240.0
    max_duration_beats: int = 16
    max_sequence_length: int = 2048


class BarEventTokenizer:
    def __init__(self, config: TokenizerConfig | None = None) -> None:
        config = config or TokenizerConfig()
        self.config = config
        self.vocabulary = EventVocabulary(
            VocabularyConfig(
                config.steps_per_beat,
                config.velocity_bins,
                config.tempo_bins,
                config.max_duration_beats,
            )
        )

    def _tempo_bin(self, tempo: float) -> int:
        fraction = (tempo - self.config.tempo_min) / (self.config.tempo_max - self.config.tempo_min)
        return min(
            self.config.tempo_bins - 1, max(0, round(fraction * (self.config.tempo_bins - 1)))
        )

    def _tempo_value(self, index: int) -> float:
        return self.config.tempo_min + index / max(1, self.config.tempo_bins - 1) * (
            self.config.tempo_max - self.config.tempo_min
        )

    def encode(
        self, midi: miditoolkit.MidiFile, *, start_bar: int = 0, num_bars: int | None = None
    ) -> list[int]:
        tpb = midi.ticks_per_beat
        ticks_per_step = tpb / self.config.steps_per_beat
        ticks_per_bar = tpb * 4
        all_notes: list[tuple[int, int, int, int, int, int, bool]] = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                bar = note.start // ticks_per_bar
                if bar < start_bar or (num_bars is not None and bar >= start_bar + num_bars):
                    continue
                position = round((note.start % ticks_per_bar) / ticks_per_step)
                position = min(self.config.steps_per_beat * 4 - 1, max(0, position))
                duration = round((note.end - note.start) / ticks_per_step)
                duration = min(
                    self.config.steps_per_beat * self.config.max_duration_beats, max(1, duration)
                )
                velocity = min(
                    self.config.velocity_bins - 1,
                    (note.velocity - 1) * self.config.velocity_bins // 127,
                )
                program = 16 if instrument.is_drum else instrument.program // 8
                all_notes.append(
                    (
                        bar - start_bar,
                        position,
                        program,
                        note.pitch,
                        velocity,
                        duration,
                        instrument.is_drum,
                    )
                )
        bars = max((note[0] for note in all_notes), default=0) + 1 if num_bars is None else num_bars
        tempo = midi.tempo_changes[0].tempo if midi.tempo_changes else 120.0
        tokens = [self.vocabulary.bos_id]
        for bar in range(bars):
            tokens.append(self.vocabulary.id("BAR"))
            tokens.append(self.vocabulary.id(f"TEMPO_{self._tempo_bin(float(tempo))}"))
            tokens.append(self.vocabulary.id("TIME_SIGNATURE_4_4"))
            bar_notes = sorted(
                (note for note in all_notes if note[0] == bar),
                key=lambda note: (note[1], note[2], note[3]),
            )
            for _, position, program, pitch, velocity, duration, is_drum in bar_notes:
                tokens.extend(
                    [
                        self.vocabulary.id(f"POSITION_{position}"),
                        self.vocabulary.id("PROGRAM_DRUM" if is_drum else f"PROGRAM_{program}"),
                        self.vocabulary.id(f"DRUM_PITCH_{pitch}" if is_drum else f"PITCH_{pitch}"),
                        self.vocabulary.id(f"VELOCITY_{velocity}"),
                        self.vocabulary.id(f"DURATION_{duration - 1}"),
                    ]
                )
        tokens.append(self.vocabulary.eos_id)
        if len(tokens) > self.config.max_sequence_length:
            tokens = [*tokens[: self.config.max_sequence_length - 1], self.vocabulary.eos_id]
        return tokens

    def encoded_length(self, *, num_bars: int, num_notes: int) -> int:
        """Return encoded length without materializing tokens.

        Each segment has BOS/EOS, three structural events per bar, and five
        events per note.  This is used by preprocessing, where only the
        manifest length and note constraints are needed.
        """

        return encoded_segment_length(num_bars, num_notes, self.config.max_sequence_length)

    def decode(self, token_ids: list[int], *, ticks_per_beat: int = 480) -> miditoolkit.MidiFile:
        midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
        tracks: dict[tuple[int, bool], miditoolkit.Instrument] = {}
        bar = -1
        index = 0
        ticks_per_step = ticks_per_beat / self.config.steps_per_beat
        current_tempo = 120.0
        while index < len(token_ids):
            token = self.vocabulary.token(token_ids[index])
            if token == "BAR":
                bar += 1
            elif token.startswith("TEMPO_"):
                current_tempo = self._tempo_value(int(token.rsplit("_", 1)[1]))
                time = max(0, bar) * ticks_per_beat * 4
                if not midi.tempo_changes or midi.tempo_changes[-1].tempo != current_tempo:
                    midi.tempo_changes.append(miditoolkit.TempoChange(current_tempo, time))
            elif token == "TIME_SIGNATURE_4_4" and not midi.time_signature_changes:
                midi.time_signature_changes.append(
                    miditoolkit.TimeSignature(4, 4, max(0, bar) * ticks_per_beat * 4)
                )
            elif token.startswith("POSITION_") and index + 4 < len(token_ids):
                sequence = [self.vocabulary.token(value) for value in token_ids[index : index + 5]]
                if self._valid_note_sequence(sequence):
                    position = int(sequence[0].rsplit("_", 1)[1])
                    is_drum = sequence[1] == "PROGRAM_DRUM"
                    program = 0 if is_drum else int(sequence[1].rsplit("_", 1)[1]) * 8
                    pitch = int(sequence[2].rsplit("_", 1)[1])
                    velocity_bin = int(sequence[3].rsplit("_", 1)[1])
                    duration_steps = int(sequence[4].rsplit("_", 1)[1]) + 1
                    velocity = min(
                        127, max(1, round((velocity_bin + 0.5) * 127 / self.config.velocity_bins))
                    )
                    start = round(
                        (max(0, bar) * 4 * self.config.steps_per_beat + position) * ticks_per_step
                    )
                    end = start + max(1, round(duration_steps * ticks_per_step))
                    track = tracks.setdefault(
                        (program, is_drum), miditoolkit.Instrument(program=program, is_drum=is_drum)
                    )
                    track.notes.append(miditoolkit.Note(velocity, pitch, start, end))
                    index += 4
            index += 1
        midi.instruments = sorted(
            tracks.values(), key=lambda item: (not item.is_drum, item.program)
        )
        if not midi.tempo_changes:
            midi.tempo_changes = [miditoolkit.TempoChange(current_tempo, 0)]
        if not midi.time_signature_changes:
            midi.time_signature_changes = [miditoolkit.TimeSignature(4, 4, 0)]
        return midi

    @staticmethod
    def _valid_note_sequence(tokens: list[str]) -> bool:
        if len(tokens) != 5:
            return False
        position, program, pitch, velocity, duration = tokens
        drum_match = program == "PROGRAM_DRUM" and pitch.startswith("DRUM_PITCH_")
        pitched_match = (
            program.startswith("PROGRAM_")
            and program != "PROGRAM_DRUM"
            and pitch.startswith("PITCH_")
        )
        return (
            position.startswith("POSITION_")
            and (drum_match or pitched_match)
            and velocity.startswith("VELOCITY_")
            and duration.startswith("DURATION_")
        )

    def quantization_tolerance_ticks(self, ticks_per_beat: int) -> int:
        return math.ceil(ticks_per_beat / self.config.steps_per_beat / 2)
