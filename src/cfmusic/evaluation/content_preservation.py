"""Symbolic MIDI descriptors and separately reported preservation metrics."""

from __future__ import annotations

from pathlib import Path

import miditoolkit
import numpy as np

from cfmusic.data.midi_io import load_midi


def symbolic_descriptors(midi_or_path: miditoolkit.MidiFile | Path) -> np.ndarray:
    midi = load_midi(midi_or_path) if isinstance(midi_or_path, Path) else midi_or_path
    notes = [(note, instrument) for instrument in midi.instruments for note in instrument.notes]
    pitches = np.array([note.pitch for note, _ in notes], dtype=np.float64)
    velocities = np.array([note.velocity for note, _ in notes], dtype=np.float64)
    durations = np.array(
        [(note.end - note.start) / midi.ticks_per_beat for note, _ in notes], dtype=np.float64
    )
    onsets = np.array([note.start / midi.ticks_per_beat for note, _ in notes], dtype=np.float64)
    pitch_class = np.bincount((pitches.astype(int) % 12), minlength=12).astype(float)
    pitch_class /= max(1.0, pitch_class.sum())
    sorted_pitches = pitches[np.argsort(onsets)] if len(notes) else pitches
    intervals = np.diff(sorted_pitches).clip(-24, 24).astype(int) + 24
    interval_hist = np.bincount(intervals, minlength=49).astype(float)
    interval_hist /= max(1.0, interval_hist.sum())
    duration_hist, _ = np.histogram(durations, bins=[0, 0.125, 0.25, 0.5, 1, 2, 4, 8, np.inf])
    duration_hist = duration_hist / max(1, duration_hist.sum())
    onset_hist = np.bincount((np.round((onsets % 1) * 16).astype(int) % 16), minlength=16).astype(
        float
    )
    onset_hist /= max(1.0, onset_hist.sum())
    programs = np.zeros(17)
    drum_hist = np.zeros(128)
    for note, instrument in notes:
        programs[16 if instrument.is_drum else instrument.program // 8] += 1
        if instrument.is_drum:
            drum_hist[note.pitch] += 1
    programs /= max(1.0, programs.sum())
    drum_hist /= max(1.0, drum_hist.sum())
    max_beat = max((note.end for note, _ in notes), default=0) / midi.ticks_per_beat
    density = len(notes) / max(1.0, max_beat)
    simultaneous = len(notes) / max(1, len(np.unique(onsets)))
    tempo = np.mean([change.tempo for change in midi.tempo_changes] or [120.0])
    scalar = np.array(
        [
            density,
            simultaneous,
            velocities.mean() if len(velocities) else 0,
            velocities.std() if len(velocities) else 0,
            tempo,
            pitches.mean() if len(pitches) else 0,
            pitches.std() if len(pitches) else 0,
            ((onsets * 4) % 1 > 0.05).mean() if len(onsets) else 0,
        ]
    )
    return np.concatenate(
        [pitch_class, interval_hist, duration_hist, onset_hist, programs, drum_hist, scalar]
    )


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ right / max(1e-12, np.linalg.norm(left) * np.linalg.norm(right)))


def descriptor_preservation(source: Path, generated: Path) -> dict[str, float]:
    source_midi, generated_midi = load_midi(source), load_midi(generated)
    source_features = symbolic_descriptors(source_midi)
    generated_features = symbolic_descriptors(generated_midi)
    source_notes = [note for track in source_midi.instruments for note in track.notes]
    generated_notes = [note for track in generated_midi.instruments for note in track.notes]
    return {
        "descriptor_cosine": cosine_similarity(source_features, generated_features),
        "pitch_class_histogram_cosine": cosine_similarity(
            source_features[:12], generated_features[:12]
        ),
        "note_density_ratio": len(generated_notes) / max(1, len(source_notes)),
        "tempo_ratio": float(np.mean([x.tempo for x in generated_midi.tempo_changes] or [120]))
        / float(np.mean([x.tempo for x in source_midi.tempo_changes] or [120])),
    }
