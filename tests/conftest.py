from pathlib import Path

import miditoolkit
import pytest


@pytest.fixture
def tiny_midi_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.mid"
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    midi.tempo_changes = [miditoolkit.TempoChange(120.0, 0)]
    midi.time_signature_changes = [miditoolkit.TimeSignature(4, 4, 0)]
    piano = miditoolkit.Instrument(program=0, is_drum=False)
    for bar in range(2):
        for step, pitch in enumerate((60, 64, 67, 72)):
            start = bar * 1920 + step * 480
            piano.notes.append(
                miditoolkit.Note(velocity=64 + step, pitch=pitch, start=start, end=start + 360)
            )
    midi.instruments = [piano]
    midi.dump(str(path))
    return path
