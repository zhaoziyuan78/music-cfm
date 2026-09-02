"""Drum-family onset preservation metrics."""

from __future__ import annotations

from pathlib import Path

from cfmusic.data.midi_io import load_midi


def drum_onset_f1(source: Path, generated: Path, tolerance_ticks: int = 30) -> float:
    left = [
        (note.pitch, note.start)
        for track in load_midi(source).instruments
        if track.is_drum
        for note in track.notes
    ]
    right = [
        (note.pitch, note.start)
        for track in load_midi(generated).instruments
        if track.is_drum
        for note in track.notes
    ]
    used: set[int] = set()
    matches = 0
    for pitch, onset in left:
        candidates = [
            (abs(onset - other_onset), index)
            for index, (other_pitch, other_onset) in enumerate(right)
            if index not in used
            and pitch == other_pitch
            and abs(onset - other_onset) <= tolerance_ticks
        ]
        if candidates:
            _, index = min(candidates)
            used.add(index)
            matches += 1
    precision = matches / max(1, len(right))
    recall = matches / max(1, len(left))
    return 2 * precision * recall / max(1e-12, precision + recall)
