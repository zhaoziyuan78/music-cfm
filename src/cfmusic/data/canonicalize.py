"""Deterministic canonical event serialization independent of track order."""

from __future__ import annotations

import hashlib
import json

import miditoolkit

CANONICAL_TICKS_PER_BEAT = 480


def _rescale(value: int, source_tpb: int) -> int:
    return round(value * CANONICAL_TICKS_PER_BEAT / source_tpb)


def canonical_events(midi: miditoolkit.MidiFile) -> dict[str, object]:
    """Serialize musically relevant MIDI state using stable ordering."""
    instruments: list[dict[str, object]] = []
    ordered = sorted(
        midi.instruments, key=lambda item: (not item.is_drum, item.program // 8, item.program)
    )
    for instrument in ordered:
        notes = sorted(
            instrument.notes,
            key=lambda note: (
                note.start,
                instrument.program,
                note.pitch,
                note.end - note.start,
                note.velocity,
            ),
        )
        instruments.append(
            {
                "is_drum": bool(instrument.is_drum),
                "program": int(instrument.program),
                "notes": [
                    [
                        _rescale(note.start, midi.ticks_per_beat),
                        _rescale(note.end, midi.ticks_per_beat),
                        int(note.pitch),
                        int(note.velocity),
                    ]
                    for note in notes
                ],
            }
        )
    tempos = sorted(
        [
            [_rescale(change.time, midi.ticks_per_beat), round(float(change.tempo), 6)]
            for change in midi.tempo_changes
        ]
    )
    signatures = sorted(
        [
            [_rescale(change.time, midi.ticks_per_beat), change.numerator, change.denominator]
            for change in midi.time_signature_changes
        ]
    )
    return {
        "ticks_per_beat": CANONICAL_TICKS_PER_BEAT,
        "instruments": instruments,
        "tempos": tempos,
        "time_signatures": signatures,
    }


def canonical_event_hash(midi: miditoolkit.MidiFile) -> str:
    payload = json.dumps(canonical_events(midi), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
