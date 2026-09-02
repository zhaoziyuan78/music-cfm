"""Token-level reconstruction metrics with musical timing context."""

from __future__ import annotations

from collections import Counter

from cfmusic.tokenization.beat_vocabulary import BeatVocabulary
from cfmusic.tokenization.vocabulary import EventVocabulary

Vocabulary = EventVocabulary | BeatVocabulary


def trim_token_sequence(tokens: list[int], *, eos_id: int, pad_id: int) -> list[int]:
    """Remove batch padding and stop at the first end-of-sequence token."""
    result: list[int] = []
    for token in tokens:
        if token == pad_id:
            break
        result.append(token)
        if token == eos_id:
            break
    return result


def aligned_token_accuracy(reference: list[int], prediction: list[int]) -> float:
    """Position-wise accuracy, penalizing missing or extra predicted tokens."""
    aligned = min(len(reference), len(prediction))
    matches = sum(reference[index] == prediction[index] for index in range(aligned))
    return matches / max(1, max(len(reference), len(prediction)))


def symbolic_note_events(vocabulary: Vocabulary, sequence: list[int]) -> list[tuple[int, ...]]:
    """Extract time- and instrument-aware note events, excluding velocity.

    Velocity is deliberately excluded from event matching: a structurally
    correct note should count as a match even when its dynamics differ. Raw
    token accuracy continues to account for every VEL token.
    """
    if getattr(vocabulary, "scheme", None) == "beat":
        return _beat_note_events(sequence)
    return _bar_note_events(vocabulary, sequence)


def _beat_note_events(sequence: list[int]) -> list[tuple[int, ...]]:
    events: list[tuple[int, ...]] = []
    bar = -1
    beat = -1
    instrument = -1
    is_drum = False
    previous_pitch: int | None = None
    index = 0
    while index < len(sequence):
        token = int(sequence[index])
        if token == 426:  # BAR
            bar += 1
            beat = -1
            instrument = -1
            previous_pitch = None
        elif token in {425, 503}:  # BEAT or REST
            beat += 1
            instrument = -1
            previous_pitch = None
        elif 297 <= token < 425:  # INS
            instrument = token - 297
            is_drum = False
            previous_pitch = None
        elif token == 504:  # INS_DRUM
            instrument = 128
            is_drum = True
            previous_pitch = None
        elif (
            209 <= token < 297
            and index + 2 < len(sequence)
            and 0 <= sequence[index + 1] < 81
            and 81 <= sequence[index + 2] < 209
            and instrument >= 0
            and not is_drum
        ):
            pitch_code = token - 209
            pitch = pitch_code if previous_pitch is None else previous_pitch - pitch_code
            if 0 <= pitch < 88:
                events.append((bar, beat, instrument, pitch, int(sequence[index + 1])))
                previous_pitch = pitch
            index += 2
        elif (
            505 <= token < 593
            and index + 2 < len(sequence)
            and 0 <= sequence[index + 1] < 81
            and 81 <= sequence[index + 2] < 209
            and instrument == 128
        ):
            events.append((bar, beat, instrument, token - 505, int(sequence[index + 1])))
            index += 2
        index += 1
    return events


def _bar_note_events(vocabulary: Vocabulary, sequence: list[int]) -> list[tuple[int, ...]]:
    events: list[tuple[int, ...]] = []
    bar = -1
    for index, token in enumerate(sequence):
        if vocabulary.token(token) == "BAR":
            bar += 1
        if index + 4 >= len(sequence):
            continue
        names = [vocabulary.token(value) for value in sequence[index : index + 5]]
        if (
            names[0].startswith("POSITION_")
            and names[1].startswith("PROGRAM_")
            and (names[2].startswith("PITCH_") or names[2].startswith("DRUM_PITCH_"))
            and names[3].startswith("VELOCITY_")
            and names[4].startswith("DURATION_")
        ):
            # Exclude the velocity token (index + 3), but retain bar, onset,
            # program, pitch, and duration.
            events.append(
                (
                    bar,
                    int(sequence[index]),
                    int(sequence[index + 1]),
                    int(sequence[index + 2]),
                    int(sequence[index + 4]),
                )
            )
    return events


def multiset_f1(reference: list[tuple[int, ...]], prediction: list[tuple[int, ...]]) -> float:
    """F1 for duplicate-aware symbolic event bags."""
    reference_counts = Counter(reference)
    prediction_counts = Counter(prediction)
    matches = sum((reference_counts & prediction_counts).values())
    precision = matches / max(1, sum(prediction_counts.values()))
    recall = matches / max(1, sum(reference_counts.values()))
    return 2.0 * precision * recall / max(1e-12, precision + recall)
