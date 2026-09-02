from pathlib import Path

from cfmusic.data.midi_io import load_midi
from cfmusic.tokenization.bar_event import BarEventTokenizer


def test_quantized_token_roundtrip(tiny_midi_path: Path) -> None:
    tokenizer = BarEventTokenizer()
    midi = load_midi(tiny_midi_path)
    first = tokenizer.encode(midi, num_bars=2)
    decoded = tokenizer.decode(first)
    second = tokenizer.encode(decoded, num_bars=2)
    assert first == second
