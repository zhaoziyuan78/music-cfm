"""Defensive MIDI parsing and audit statistics."""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import miditoolkit

from cfmusic.data.canonicalize import canonical_event_hash
from cfmusic.data.schema import MidiStatistics, ValidationResult


def _validate_midi(midi: miditoolkit.MidiFile) -> list[miditoolkit.Note]:
    if midi.ticks_per_beat <= 0 or midi.ticks_per_beat > 100_000:
        raise ValueError(f"Invalid ticks_per_beat={midi.ticks_per_beat}")
    notes = [note for instrument in midi.instruments for note in instrument.notes]
    if not notes:
        raise ValueError("MIDI has no notes")
    if len(notes) > 2_000_000:
        raise ValueError("MIDI has an unreasonable number of notes")
    for instrument in midi.instruments:
        if not 0 <= instrument.program <= 127:
            raise ValueError(f"Invalid program {instrument.program}")
        for note in instrument.notes:
            values = (note.start, note.end, note.pitch, note.velocity)
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError("Non-finite note value")
            if note.start < 0 or note.end <= note.start:
                raise ValueError("Invalid note interval")
            if not 0 <= note.pitch <= 127 or not 1 <= note.velocity <= 127:
                raise ValueError("Pitch or velocity outside MIDI range")
    max_tick = max(note.end for note in notes)
    if max_tick > midi.ticks_per_beat * 60 * 60 * 24:
        raise ValueError("MIDI duration exceeds 24 hours at one beat per second")
    return notes


def load_midi(path: Path) -> miditoolkit.MidiFile:
    """Parse MIDI and reject structurally dangerous values."""
    midi = miditoolkit.MidiFile(str(path))
    _validate_midi(midi)
    return midi


def validate_midi(path: Path) -> ValidationResult:
    try:
        load_midi(path)
    except (OSError, ValueError, EOFError, KeyError, IndexError) as error:
        return ValidationResult(False, f"{type(error).__name__}: {error}")
    return ValidationResult(True)


def _audit_statistics(
    midi: miditoolkit.MidiFile,
    notes: list[miditoolkit.Note],
    *,
    exact_file_sha256: str,
) -> MidiStatistics:
    tempo_values = [float(change.tempo) for change in midi.tempo_changes] or [120.0]
    if not all(math.isfinite(value) for value in tempo_values) or any(
        value <= 0 for value in tempo_values
    ):
        raise ValueError("Invalid tempo map")
    signatures = midi.time_signature_changes
    for signature in signatures:
        if signature.numerator <= 0 or signature.denominator <= 0:
            raise ValueError("Invalid time signature")
    signature_text = (
        f"{signatures[0].numerator}/{signatures[0].denominator}" if signatures else "4/4"
    )
    tempo_mean = sum(tempo_values) / len(tempo_values)
    max_tick = max(note.end for note in notes)
    duration = max_tick / midi.ticks_per_beat * 60.0 / tempo_mean
    return MidiStatistics(
        exact_file_sha256=exact_file_sha256,
        canonical_hash=canonical_event_hash(midi),
        num_notes=len(notes),
        num_tracks=len(midi.instruments),
        duration_seconds=duration,
        time_signature=signature_text,
        tempo_mean=tempo_mean,
        is_drum=all(instrument.is_drum for instrument in midi.instruments),
    )


def load_and_audit_midi(path: Path) -> tuple[miditoolkit.MidiFile, MidiStatistics]:
    """Read a file once, then parse, validate, hash, and audit it in memory."""

    payload = path.read_bytes()
    midi = miditoolkit.MidiFile(file=io.BytesIO(payload))
    notes = _validate_midi(midi)
    statistics = _audit_statistics(
        midi,
        notes,
        exact_file_sha256=hashlib.sha256(payload).hexdigest(),
    )
    return midi, statistics


def audit_midi(path: Path) -> MidiStatistics:
    return load_and_audit_midi(path)[1]
