from pathlib import Path

import miditoolkit

from cfmusic.commands.evaluate import artifact_midi_validity


def _write_midi(path: Path, *, with_note: bool) -> None:
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    instrument = miditoolkit.Instrument(program=0)
    if with_note:
        instrument.notes.append(
            miditoolkit.Note(velocity=80, pitch=60, start=0, end=480)
        )
    midi.instruments.append(instrument)
    midi.dump(str(path))


def test_empty_generated_midi_is_recorded_instead_of_raising(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    generated = tmp_path / "counterfactual.mid"
    _write_midi(source, with_note=True)
    _write_midi(generated, with_note=False)

    metrics = artifact_midi_validity(source, generated)

    assert metrics["source_midi_valid"] == 1.0
    assert metrics["generated_midi_valid"] == 0.0
    assert "MIDI has no notes" in str(metrics["generated_midi_error"])
