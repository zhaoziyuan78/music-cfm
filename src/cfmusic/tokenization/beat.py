"""Direct MIDI implementation of the BEAT multitrack tokenizer.

This is a project-native adapter for the tokenizer portion of BEAT-code. It
keeps BEAT's uniform four-step beat patterns and exact 593-token layout, while
retaining real MIDI velocities (the official decoder already supports them).
No BEAT model or checkpoint is required.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import miditoolkit

from cfmusic.tokenization.beat_vocabulary import BeatVocabulary


@dataclass(frozen=True)
class BeatTokenizerConfig:
    implementation_version: str = "cfmusic-beat-v1"
    steps_per_beat: int = 4
    max_sequence_length: int = 2048
    preserve_velocity: bool = True


class BeatTokenizer:
    """Encode MIDI segments as BEAT/PAT/PIT/VEL token sequences."""

    def __init__(self, config: BeatTokenizerConfig | None = None) -> None:
        self.config = config or BeatTokenizerConfig()
        if self.config.implementation_version != "cfmusic-beat-v1":
            raise ValueError(
                f"Unsupported BEAT implementation version: {self.config.implementation_version}"
            )
        if self.config.steps_per_beat != 4:
            raise ValueError("The official BEAT vocabulary requires exactly 4 steps per beat")
        self.vocabulary = BeatVocabulary()

    @staticmethod
    def _pattern_id(states: list[int]) -> int:
        value = 0
        for state in states:
            value = value * 3 + state
        return value

    @staticmethod
    def _pattern_states(pattern_id: int) -> list[int]:
        states = [0, 0, 0, 0]
        value = min(80, max(0, pattern_id))
        for index in range(3, -1, -1):
            states[index] = value % 3
            value //= 3
        return states

    def _encode_tokens(
        self,
        midi: miditoolkit.MidiFile,
        *,
        start_bar: int,
        num_bars: int | None,
    ) -> list[int]:
        ticks_per_beat = midi.ticks_per_beat
        ticks_per_bar = ticks_per_beat * 4
        ticks_per_step = ticks_per_beat / self.config.steps_per_beat
        if num_bars is None:
            last_tick = max(note.end for track in midi.instruments for note in track.notes)
            num_bars = max(1, math.ceil(last_tick / ticks_per_bar) - start_bar)
        segment_start = start_bar * ticks_per_bar
        total_beats = num_bars * 4
        total_steps = total_beats * self.config.steps_per_beat

        # (beat, program, is_drum, pitch_index) -> four tri-state cells.
        patterns: dict[tuple[int, int, bool, int], list[int]] = {}
        velocity_sums: dict[tuple[int, int, bool, int], int] = {}
        velocity_counts: dict[tuple[int, int, bool, int], int] = {}
        for instrument in midi.instruments:
            program = int(instrument.program)
            for note in instrument.notes:
                pitch_index = int(note.pitch) - self.vocabulary.minimum_pitch
                if not 0 <= pitch_index < self.vocabulary.pitch_count:
                    continue
                absolute_start = round(note.start / ticks_per_step)
                relative_start = absolute_start - round(segment_start / ticks_per_step)
                covered_steps: Iterable[int]
                if instrument.is_drum:
                    covered_steps = [relative_start]
                else:
                    absolute_end = max(absolute_start + 1, round(note.end / ticks_per_step))
                    relative_end = absolute_end - round(segment_start / ticks_per_step)
                    covered_steps = range(max(0, relative_start), min(total_steps, relative_end))
                for step in covered_steps:
                    if not 0 <= step < total_steps:
                        continue
                    beat, position = divmod(step, self.config.steps_per_beat)
                    key = (beat, program, bool(instrument.is_drum), pitch_index)
                    states = patterns.setdefault(key, [0] * self.config.steps_per_beat)
                    # A segment cannot represent a note onset before time zero.
                    # Retrigger a carried note at the boundary so decoding is
                    # self-contained and encode→decode→encode is idempotent.
                    onset = step == max(0, relative_start)
                    if onset:
                        states[position] = 1
                    elif states[position] == 0:
                        states[position] = 2
                    velocity_sums[key] = velocity_sums.get(key, 0) + int(note.velocity)
                    velocity_counts[key] = velocity_counts.get(key, 0) + 1

        tempo = 120.0
        for change in midi.tempo_changes:
            if change.time <= segment_start:
                tempo = float(change.tempo)
            else:
                break
        signature = "4/4"
        for change in midi.time_signature_changes:
            if change.time <= segment_start:
                signature = f"{change.numerator}/{change.denominator}"
            else:
                break
        tokens = [
            self.vocabulary.bos_id,
            self.vocabulary.time_signature_token(signature),
            self.vocabulary.tempo_token(tempo),
        ]
        for bar in range(num_bars):
            tokens.append(self.vocabulary.bar_id)
            for beat_in_bar in range(4):
                beat = bar * 4 + beat_in_bar
                beat_keys = [key for key in patterns if key[0] == beat]
                if not beat_keys:
                    tokens.append(self.vocabulary.rest_id)
                    continue
                tokens.append(self.vocabulary.beat_id)
                tracks = sorted({(key[1], key[2]) for key in beat_keys}, key=lambda x: (x[1], x[0]))
                for program, is_drum in tracks:
                    tokens.append(self.vocabulary.drum_instrument_id if is_drum else 297 + program)
                    track_keys = sorted(
                        (key for key in beat_keys if key[1] == program and key[2] == is_drum),
                        key=lambda key: key[3],
                        reverse=not is_drum,
                    )
                    previous_pitch: int | None = None
                    for key in track_keys:
                        pitch = key[3]
                        if is_drum:
                            tokens.append(505 + pitch)
                        else:
                            pitch_code = pitch if previous_pitch is None else previous_pitch - pitch
                            tokens.append(209 + pitch_code)
                            previous_pitch = pitch
                        tokens.append(self._pattern_id(patterns[key]))
                        velocity = (
                            round(velocity_sums[key] / velocity_counts[key])
                            if self.config.preserve_velocity
                            else 64
                        )
                        tokens.append(81 + min(127, max(1, velocity)))
        tokens.append(self.vocabulary.eos_id)
        return tokens

    def encoded_segment_lengths(
        self,
        midi: miditoolkit.MidiFile,
        segments: list[tuple[int, int]],
    ) -> list[int]:
        """Count many overlapping segments after one pass over the MIDI notes."""
        ticks_per_step = midi.ticks_per_beat / self.config.steps_per_beat
        beat_items: dict[int, set[tuple[int, bool, int]]] = {}
        for instrument in midi.instruments:
            for note in instrument.notes:
                pitch_index = int(note.pitch) - self.vocabulary.minimum_pitch
                if not 0 <= pitch_index < self.vocabulary.pitch_count:
                    continue
                start_step = round(note.start / ticks_per_step)
                active_beats: Iterable[int]
                if instrument.is_drum:
                    active_beats = [start_step // self.config.steps_per_beat]
                else:
                    end_step = max(start_step + 1, round(note.end / ticks_per_step))
                    active_beats = range(
                        start_step // self.config.steps_per_beat,
                        (end_step - 1) // self.config.steps_per_beat + 1,
                    )
                item = (int(instrument.program), bool(instrument.is_drum), pitch_index)
                for beat in active_beats:
                    beat_items.setdefault(beat, set()).add(item)

        lengths: list[int] = []
        for start_bar, num_bars in segments:
            # BOS + TS + TEM + EOS, one BAR per measure.
            length = 4 + num_bars
            for beat in range(start_bar * 4, (start_bar + num_bars) * 4):
                items = beat_items.get(beat)
                if not items:
                    length += 1  # REST
                    continue
                tracks = {(program, is_drum) for program, is_drum, _pitch in items}
                length += 1 + len(tracks) + 3 * len(items)  # BEAT + INS + triples
            lengths.append(length)
        return lengths

    def encode_untruncated(
        self,
        midi: miditoolkit.MidiFile,
        *,
        start_bar: int = 0,
        num_bars: int | None = None,
    ) -> list[int]:
        return self._encode_tokens(midi, start_bar=start_bar, num_bars=num_bars)

    def encode(
        self,
        midi: miditoolkit.MidiFile,
        *,
        start_bar: int = 0,
        num_bars: int | None = None,
    ) -> list[int]:
        tokens = self.encode_untruncated(midi, start_bar=start_bar, num_bars=num_bars)
        if len(tokens) > self.config.max_sequence_length:
            return [*tokens[: self.config.max_sequence_length - 1], self.vocabulary.eos_id]
        return tokens

    def decode(self, token_ids: list[int], *, ticks_per_beat: int = 480) -> miditoolkit.MidiFile:
        tempo = 120.0
        beats: list[dict[tuple[int, bool], list[tuple[int, int, int]]]] = []
        current_beat: dict[tuple[int, bool], list[tuple[int, int, int]]] | None = None
        current_track: tuple[int, bool] | None = None
        previous_pitch: int | None = None
        index = 0
        while index < len(token_ids):
            token_id = int(token_ids[index])
            if 438 <= token_id < 453:
                tempo = self.vocabulary.tempo_value(token_id)
            elif token_id == self.vocabulary.beat_id:
                current_beat = {}
                beats.append(current_beat)
                current_track = None
                previous_pitch = None
            elif token_id == self.vocabulary.rest_id:
                beats.append({})
                current_beat = None
                current_track = None
                previous_pitch = None
            elif 297 <= token_id < 425 and current_beat is not None:
                current_track = (token_id - 297, False)
                current_beat.setdefault(current_track, [])
                previous_pitch = None
            elif token_id == self.vocabulary.drum_instrument_id and current_beat is not None:
                current_track = (0, True)
                current_beat.setdefault(current_track, [])
                previous_pitch = None
            elif (
                209 <= token_id < 297
                and index + 2 < len(token_ids)
                and 0 <= token_ids[index + 1] < 81
                and 81 <= token_ids[index + 2] < 209
                and current_beat is not None
                and current_track is not None
                and not current_track[1]
            ):
                pitch_code = token_id - 209
                pitch = pitch_code if previous_pitch is None else previous_pitch - pitch_code
                if 0 <= pitch < 88:
                    current_beat[current_track].append(
                        (pitch, int(token_ids[index + 1]), int(token_ids[index + 2]) - 81)
                    )
                    previous_pitch = pitch
                index += 2
            elif (
                505 <= token_id < 593
                and index + 2 < len(token_ids)
                and 0 <= token_ids[index + 1] < 81
                and 81 <= token_ids[index + 2] < 209
                and current_beat is not None
                and current_track is not None
                and current_track[1]
            ):
                current_beat[current_track].append(
                    (token_id - 505, int(token_ids[index + 1]), int(token_ids[index + 2]) - 81)
                )
                index += 2
            index += 1

        total_steps = len(beats) * self.config.steps_per_beat
        rolls: dict[tuple[int, bool, int], tuple[list[int], list[int]]] = {}
        for beat_index, beat in enumerate(beats):
            for (program, is_drum), items in beat.items():
                for pitch, pattern_id, velocity in items:
                    key = (program, is_drum, pitch)
                    states, velocities = rolls.setdefault(
                        key, ([0] * total_steps, [0] * total_steps)
                    )
                    for position, state in enumerate(self._pattern_states(pattern_id)):
                        step = beat_index * self.config.steps_per_beat + position
                        states[step] = state
                        velocities[step] = velocity

        midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
        midi.tempo_changes = [miditoolkit.TempoChange(tempo, 0)]
        midi.time_signature_changes = [miditoolkit.TimeSignature(4, 4, 0)]
        instruments: dict[tuple[int, bool], miditoolkit.Instrument] = {}
        ticks_per_step = ticks_per_beat // self.config.steps_per_beat
        for (program, is_drum, pitch_index), (states, velocities) in rolls.items():
            instrument = instruments.setdefault(
                (program, is_drum),
                miditoolkit.Instrument(program=program, is_drum=is_drum),
            )
            active_start: int | None = None
            active_velocities: list[int] = []

            def finish(
                end_step: int,
                target_instrument: miditoolkit.Instrument = instrument,
                target_pitch: int = pitch_index,
            ) -> None:
                nonlocal active_start, active_velocities
                if active_start is not None and end_step > active_start:
                    velocity = (
                        round(sum(active_velocities) / len(active_velocities))
                        if active_velocities
                        else 64
                    )
                    target_instrument.notes.append(
                        miditoolkit.Note(
                            velocity=min(127, max(1, velocity)),
                            pitch=target_pitch + self.vocabulary.minimum_pitch,
                            start=active_start * ticks_per_step,
                            end=end_step * ticks_per_step,
                        )
                    )
                active_start = None
                active_velocities = []

            for step, state in enumerate(states):
                if state == 1:
                    finish(step)
                    active_start = step
                elif state == 0:
                    finish(step)
                    continue
                elif active_start is None:
                    active_start = step
                if state and velocities[step] > 0:
                    active_velocities.append(velocities[step])
            finish(total_steps)
        for instrument in instruments.values():
            instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        midi.instruments = sorted(
            instruments.values(), key=lambda instrument: (instrument.is_drum, instrument.program)
        )
        return midi

    def quantization_tolerance_ticks(self, ticks_per_beat: int) -> int:
        return math.ceil(ticks_per_beat / self.config.steps_per_beat / 2)
