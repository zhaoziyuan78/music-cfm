from pathlib import Path

import miditoolkit

from cfmusic.data.midi_io import load_midi
from cfmusic.evaluation.reconstruction import multiset_f1, symbolic_note_events
from cfmusic.tokenization.beat import BeatTokenizer, BeatTokenizerConfig
from cfmusic.tokenization.grammar import EventGrammar


def test_beat_vocabulary_matches_official_layout() -> None:
    vocabulary = BeatTokenizer().vocabulary
    assert len(vocabulary) == 593
    assert vocabulary.id("PAT_0") == 0
    assert vocabulary.id("VEL_0") == 81
    assert vocabulary.id("PIT_0") == 209
    assert vocabulary.id("INS_0") == 297
    assert vocabulary.beat_id == 425
    assert vocabulary.bar_id == 426
    assert vocabulary.eos_id == 427
    assert vocabulary.bos_id == 428
    assert vocabulary.pad_id == 429
    assert vocabulary.rest_id == 503
    assert vocabulary.drum_instrument_id == 504
    assert vocabulary.id("DRUM_PIT_0") == 505


def test_beat_token_roundtrip_and_length(tiny_midi_path: Path) -> None:
    tokenizer = BeatTokenizer(BeatTokenizerConfig(max_sequence_length=2048))
    source = load_midi(tiny_midi_path)
    tokens = tokenizer.encode_untruncated(source, num_bars=2)
    lengths = tokenizer.encoded_segment_lengths(source, [(0, 2)])

    reconstructed = tokenizer.decode(tokens, ticks_per_beat=source.ticks_per_beat)
    roundtrip_tokens = tokenizer.encode_untruncated(reconstructed, num_bars=2)

    assert lengths == [len(tokens)]
    assert roundtrip_tokens == tokens
    assert EventGrammar(tokenizer.vocabulary).invalid_rate(tokens) == 0.0
    source_notes = source.instruments[0].notes
    reconstructed_notes = reconstructed.instruments[0].notes
    assert [(note.pitch, note.velocity) for note in reconstructed_notes] == [
        (note.pitch, note.velocity) for note in source_notes
    ]


def test_beat_preserves_programs_drums_and_real_velocity() -> None:
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    midi.tempo_changes = [miditoolkit.TempoChange(120, 0)]
    piano = miditoolkit.Instrument(program=40)
    piano.notes = [miditoolkit.Note(velocity=91, pitch=64, start=0, end=480)]
    drums = miditoolkit.Instrument(program=0, is_drum=True)
    drums.notes = [miditoolkit.Note(velocity=117, pitch=36, start=240, end=300)]
    midi.instruments = [piano, drums]
    tokenizer = BeatTokenizer()

    tokens = tokenizer.encode(midi, num_bars=1)
    decoded = tokenizer.decode(tokens)

    by_track = {(track.program, track.is_drum): track for track in decoded.instruments}
    assert by_track[(40, False)].notes[0].velocity == 91
    assert by_track[(40, False)].notes[0].pitch == 64
    assert by_track[(0, True)].notes[0].velocity == 117
    assert by_track[(0, True)].notes[0].pitch == 36


def test_beat_truncation_preserves_eos() -> None:
    tokenizer = BeatTokenizer(BeatTokenizerConfig(max_sequence_length=8))
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    instrument = miditoolkit.Instrument(program=0)
    instrument.notes = [miditoolkit.Note(velocity=64, pitch=60, start=0, end=480)]
    midi.instruments = [instrument]

    tokens = tokenizer.encode(midi, num_bars=1)

    assert len(tokens) == 8
    assert tokens[-1] == tokenizer.vocabulary.eos_id


def test_beat_reconstruction_events_include_timing_but_ignore_velocity() -> None:
    tokenizer = BeatTokenizer()
    vocabulary = tokenizer.vocabulary
    reference = [
        vocabulary.bos_id,
        vocabulary.time_signature_token("4/4"),
        vocabulary.tempo_token(120),
        vocabulary.bar_id,
        vocabulary.beat_id,
        vocabulary.id("INS_0"),
        vocabulary.id("PIT_39"),
        vocabulary.id("PAT_27"),
        vocabulary.id("VEL_64"),
        vocabulary.rest_id,
        vocabulary.rest_id,
        vocabulary.rest_id,
        vocabulary.eos_id,
    ]
    velocity_only_change = reference.copy()
    velocity_only_change[8] = vocabulary.id("VEL_100")
    timing_change = reference.copy()
    timing_change[4], timing_change[5:9] = (
        vocabulary.rest_id,
        [
            vocabulary.beat_id,
            vocabulary.id("INS_0"),
            vocabulary.id("PIT_39"),
            vocabulary.id("PAT_27"),
        ],
    )
    timing_change.insert(9, vocabulary.id("VEL_64"))

    reference_events = symbolic_note_events(vocabulary, reference)
    assert (
        multiset_f1(reference_events, symbolic_note_events(vocabulary, velocity_only_change)) == 1
    )
    assert multiset_f1(reference_events, symbolic_note_events(vocabulary, timing_change)) == 0
