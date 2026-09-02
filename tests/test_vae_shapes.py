import pytest
import torch

from cfmusic.codec.transformer_vae import TransformerVAE
from cfmusic.tokenization.bar_event import BarEventTokenizer
from cfmusic.tokenization.beat import BeatTokenizer


def test_vae_shapes_and_finite_backward() -> None:
    tokenizer = BarEventTokenizer()
    model = TransformerVAE(
        vocab_size=len(tokenizer.vocabulary),
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        ff_multiplier=2,
        dropout=0,
        latent_tokens=2,
        latent_dim=8,
        max_sequence_length=32,
        vocabulary=tokenizer.vocabulary,
        gradient_checkpointing=True,
    )
    tokens = torch.tensor([[1, 4, tokenizer.vocabulary.id("TIME_SIGNATURE_4_4"), 2, 0]])
    mask = tokens.ne(0)
    posterior = model.encode_distribution(tokens, mask)
    assert posterior.mean.shape == (1, 2, 8)
    logits = model.decode_teacher_forced(tokens[:, :-1], posterior.sample())
    assert logits.shape == (1, 4, len(tokenizer.vocabulary))
    logits.mean().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_decoder_token_dropout_preserves_special_tokens() -> None:
    tokenizer = BarEventTokenizer()
    model = TransformerVAE(
        vocab_size=len(tokenizer.vocabulary),
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        ff_multiplier=2,
        dropout=0,
        latent_tokens=2,
        latent_dim=4,
        max_sequence_length=16,
        vocabulary=tokenizer.vocabulary,
        decoder_token_dropout=0.999,
    ).train()
    captured: list[torch.Tensor] = []
    handle = model.token_embedding.register_forward_hook(
        lambda _module, arguments, _output: captured.append(arguments[0].detach().clone())
    )
    tokens = torch.tensor([[1, 4, 5, 6, 2]])
    torch.manual_seed(3)
    model(tokens, tokens.ne(0))
    handle.remove()

    decoder_tokens = captured[-1]
    assert decoder_tokens[0, 0].item() == model.bos_id
    assert decoder_tokens[0, 1:].eq(model.unk_id).all()


def test_cached_generation_matches_full_prefix_generation() -> None:
    tokenizer = BarEventTokenizer()
    model = TransformerVAE(
        vocab_size=len(tokenizer.vocabulary),
        d_model=32,
        encoder_layers=1,
        decoder_layers=2,
        num_heads=4,
        ff_multiplier=2,
        dropout=0,
        latent_tokens=2,
        latent_dim=8,
        max_sequence_length=32,
        vocabulary=tokenizer.vocabulary,
    ).eval()
    latent = torch.randn(2, 2, 8)

    cached = model.generate(latent, max_length=24, use_cache=True, show_progress=False)
    full_prefix = model.generate(latent, max_length=24, use_cache=False, show_progress=False)

    assert torch.equal(cached, full_prefix)


def test_beat_generation_emits_exactly_four_beats_per_requested_bar() -> None:
    tokenizer = BeatTokenizer()
    vocabulary = tokenizer.vocabulary
    model = TransformerVAE(
        vocab_size=len(vocabulary),
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        ff_multiplier=2,
        dropout=0,
        latent_tokens=2,
        latent_dim=4,
        max_sequence_length=32,
        vocabulary=vocabulary,
        pad_id=vocabulary.pad_id,
        bos_id=vocabulary.bos_id,
        eos_id=vocabulary.eos_id,
        unk_id=vocabulary.unk_id,
    ).eval()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output.bias[vocabulary.bar_id] = 10
        model.output.bias[vocabulary.rest_id] = 9

    tokens = model.generate(
        torch.zeros(1, 2, 4),
        max_length=32,
        min_bars=1,
        max_bars=1,
        show_progress=False,
    )[0].tolist()

    assert tokens.count(vocabulary.bar_id) == 1
    assert tokens.count(vocabulary.rest_id) == 4
    assert tokens[-1] == vocabulary.eos_id


def test_beat_generation_masks_silent_patterns_zero_velocity_and_invalid_intervals() -> None:
    tokenizer = BeatTokenizer()
    vocabulary = tokenizer.vocabulary
    model = TransformerVAE(
        vocab_size=len(vocabulary),
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        ff_multiplier=2,
        dropout=0,
        latent_tokens=2,
        latent_dim=4,
        max_sequence_length=16,
        vocabulary=vocabulary,
        pad_id=vocabulary.pad_id,
        bos_id=vocabulary.bos_id,
        eos_id=vocabulary.eos_id,
        unk_id=vocabulary.unk_id,
    ).eval()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output.bias[vocabulary.beat_id] = 15
        model.output.bias[vocabulary.id("INS_0")] = 14
        model.output.bias[vocabulary.id("PIT_0")] = 20
        model.output.bias[vocabulary.id("PAT_0")] = 20
        model.output.bias[vocabulary.id("PAT_1")] = 19
        model.output.bias[vocabulary.id("VEL_0")] = 20
        model.output.bias[vocabulary.id("VEL_1")] = 19

    tokens = model.generate(torch.zeros(1, 2, 4), max_length=12, show_progress=False)[0].tolist()

    assert tokens[4:9] == [
        vocabulary.beat_id,
        vocabulary.id("INS_0"),
        vocabulary.id("PIT_0"),
        vocabulary.id("PAT_1"),
        vocabulary.id("VEL_1"),
    ]
    # With absolute pitch zero, no positive descending interval is valid and
    # interval zero would duplicate the same pitch. The next token must leave
    # the track rather than emit a triple that the decoder would discard.
    assert not 209 <= tokens[9] < 297


@pytest.mark.parametrize("probability", [-0.1, 1.0])
def test_decoder_token_dropout_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="decoder_token_dropout"):
        TransformerVAE(vocab_size=16, decoder_token_dropout=probability)
