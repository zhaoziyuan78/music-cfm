from pathlib import Path

import miditoolkit
import numpy as np

from cfmusic.evaluation.clamp2 import clamp2_style_metrics, midi_to_mtf, style_prompt


def test_clamp2_style_prompt_and_similarity_metrics() -> None:
    prompts = [style_prompt("classical"), style_prompt("rock")]
    metrics = clamp2_style_metrics(
        np.array([0.0, 1.0]),
        style_embeddings=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        source_style_id=0,
        target_style_id=1,
    )

    assert prompts == [
        "This is a piece of classical music.",
        "This is a piece of rock music.",
    ]
    assert metrics["clamp2_target_style_success"] == 1.0
    assert metrics["clamp2_target_minus_source"] == 1.0


def test_midi_to_mtf_writes_clamp2_performance_format(tmp_path: Path) -> None:
    midi_path = tmp_path / "piece.mid"
    mtf_path = tmp_path / "piece.mtf"
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    instrument = miditoolkit.Instrument(program=0)
    instrument.notes.append(miditoolkit.Note(velocity=80, pitch=60, start=0, end=480))
    midi.instruments.append(instrument)
    midi.dump(str(midi_path))

    midi_to_mtf(midi_path, mtf_path)
    contents = mtf_path.read_text(encoding="utf-8")
    assert contents.startswith("ticks_per_beat 480\n")
    assert "note_on" in contents
