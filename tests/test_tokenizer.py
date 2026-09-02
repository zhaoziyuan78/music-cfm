from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cfmusic.commands.train_codec import codec_from_config, tokenizer_from_config
from cfmusic.data.midi_io import load_midi
from cfmusic.tokenization.bar_event import BarEventTokenizer
from cfmusic.tokenization.grammar import EventGrammar


def test_codec_tokenizer_respects_a_smaller_model_limit() -> None:
    config = OmegaConf.create(
        {
            "steps_per_beat": 4,
            "velocity_bins": 32,
            "tempo_bins": 32,
            "tempo_min": 30.0,
            "tempo_max": 240.0,
            "max_duration_beats": 16,
            "max_sequence_length": 2048,
        }
    )
    tokenizer = tokenizer_from_config(config, max_sequence_length=1024)
    assert tokenizer.config.max_sequence_length == 1024


def test_codec_factory_rejects_an_uncapped_tokenizer() -> None:
    root = Path(__file__).resolve().parents[1]
    tokenizer_config = OmegaConf.load(root / "configs/tokenizer/bar_event.yaml")
    codec_config = OmegaConf.load(root / "configs/codec/drum_transformer_vae.yaml")

    with pytest.raises(ValueError, match="Tokenizer maximum length"):
        codec_from_config(codec_config, tokenizer_from_config(tokenizer_config))


def test_tokenizer_emits_valid_grammar(tiny_midi_path: Path) -> None:
    tokenizer = BarEventTokenizer()
    tokens = tokenizer.encode(load_midi(tiny_midi_path), num_bars=2)
    grammar = EventGrammar(tokenizer.vocabulary)
    assert tokens[0] == tokenizer.vocabulary.bos_id
    assert tokens[-1] == tokenizer.vocabulary.eos_id
    assert grammar.invalid_rate(tokens) == 0.0
