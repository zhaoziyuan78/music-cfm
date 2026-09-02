"""Bar-aligned segment enumeration."""

from __future__ import annotations

from dataclasses import dataclass

import miditoolkit


@dataclass(frozen=True)
class BarSegment:
    index: int
    start_bar: int
    num_bars: int


def enumerate_segments(
    midi: miditoolkit.MidiFile,
    *,
    num_bars: int,
    hop_bars: int,
    drop_incomplete_final_segment: bool,
) -> list[BarSegment]:
    if num_bars <= 0 or hop_bars <= 0:
        raise ValueError("num_bars and hop_bars must be positive")
    max_tick = max(note.end for instrument in midi.instruments for note in instrument.notes)
    ticks_per_bar = midi.ticks_per_beat * 4
    total_bars = max(1, (max_tick + ticks_per_bar - 1) // ticks_per_bar)
    starts = list(range(0, total_bars, hop_bars))
    output: list[BarSegment] = []
    for start in starts:
        available = total_bars - start
        if available < num_bars and drop_incomplete_final_segment:
            continue
        output.append(BarSegment(len(output), start, min(num_bars, available)))
    return output


def segment_note_counts(midi: miditoolkit.MidiFile, segments: list[BarSegment]) -> list[int]:
    """Count note onsets per segment in O(notes + bars + segments)."""

    if not segments:
        return []
    total_bars = max(segment.start_bar + segment.num_bars for segment in segments)
    prefix = [0] * (total_bars + 1)
    ticks_per_bar = midi.ticks_per_beat * 4
    for instrument in midi.instruments:
        for note in instrument.notes:
            bar = note.start // ticks_per_bar
            if 0 <= bar < total_bars:
                prefix[bar + 1] += 1
    for index in range(total_bars):
        prefix[index + 1] += prefix[index]
    return [
        prefix[segment.start_bar + segment.num_bars] - prefix[segment.start_bar]
        for segment in segments
    ]
