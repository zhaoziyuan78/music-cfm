"""Transformer VAE with latent-query posterior and causal cross-attention decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from cfmusic.codec.base import DiagonalGaussian
from cfmusic.codec.transformer_blocks import LatentQueryPool, causal_mask
from cfmusic.progress import track
from cfmusic.tokenization.beat_vocabulary import BeatVocabulary
from cfmusic.tokenization.grammar import EventGrammar
from cfmusic.tokenization.vocabulary import EventVocabulary


class TransformerVAE(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int = 512,
        encoder_layers: int = 8,
        decoder_layers: int = 8,
        num_heads: int = 8,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        latent_tokens: int = 32,
        latent_dim: int = 256,
        max_sequence_length: int = 2048,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        unk_id: int = 3,
        vocabulary: EventVocabulary | BeatVocabulary | None = None,
        gradient_checkpointing: bool = False,
        decoder_token_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= decoder_token_dropout < 1.0:
            raise ValueError("decoder_token_dropout must be in [0, 1)")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.latent_tokens = latent_tokens
        self.latent_dim = latent_dim
        self.max_sequence_length = max_sequence_length
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = pad_id, bos_id, eos_id, unk_id
        self.gradient_checkpointing = gradient_checkpointing
        self.decoder_token_dropout = decoder_token_dropout
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Parameter(torch.randn(max_sequence_length, d_model) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_model * ff_multiplier, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, encoder_layers, enable_nested_tensor=False
        )
        self.latent_pool = LatentQueryPool(d_model, latent_tokens, num_heads, dropout)
        self.posterior = nn.Linear(d_model, latent_dim * 2)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, d_model * ff_multiplier, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.grammar = EventGrammar(vocabulary) if vocabulary is not None else None

    def _embed(self, tokens: Tensor) -> Tensor:
        if tokens.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {tokens.shape[1]} exceeds maximum {self.max_sequence_length}"
            )
        return self.token_embedding(tokens) + self.position_embedding[: tokens.shape[1]]

    def encode_distribution(self, tokens: Tensor, attention_mask: Tensor) -> DiagonalGaussian:
        padding_mask = ~attention_mask.bool()
        states = self._embed(tokens)
        if self.gradient_checkpointing and self.training:
            for layer in self.encoder.layers:
                states = checkpoint(
                    layer,
                    states,
                    src_key_padding_mask=padding_mask,
                    use_reentrant=False,
                )
            if self.encoder.norm is not None:
                states = self.encoder.norm(states)
        else:
            states = self.encoder(states, src_key_padding_mask=padding_mask)
        pooled = self.latent_pool(states, padding_mask)
        mean, logvar = self.posterior(pooled).chunk(2, dim=-1)
        return DiagonalGaussian(mean, logvar.clamp(-20.0, 10.0))

    def encode_mean(self, tokens: Tensor, attention_mask: Tensor) -> Tensor:
        return self.encode_distribution(tokens, attention_mask).mean

    def decode_teacher_forced(self, tokens: Tensor, latent: Tensor) -> Tensor:
        target = self._embed(tokens)
        memory = self.latent_projection(latent)
        target_mask = causal_mask(tokens.shape[1], tokens.device)
        padding_mask = tokens.eq(self.pad_id)
        if self.gradient_checkpointing and self.training:
            decoded = target
            for layer in self.decoder.layers:
                decoded = checkpoint(
                    layer,
                    decoded,
                    memory,
                    tgt_mask=target_mask,
                    tgt_key_padding_mask=padding_mask,
                    tgt_is_causal=True,
                    use_reentrant=False,
                )
            if self.decoder.norm is not None:
                decoded = self.decoder.norm(decoded)
        else:
            decoded = self.decoder(
                target,
                memory,
                tgt_mask=target_mask,
                tgt_key_padding_mask=padding_mask,
                tgt_is_causal=True,
            )
        return self.output(decoded)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    @staticmethod
    def _project_attention(
        attention: nn.MultiheadAttention, value: Tensor, projection: int
    ) -> Tensor:
        """Project one Q/K/V component and expose the head dimension."""

        weight = attention.in_proj_weight
        if weight is None:
            raise TypeError("Incremental decoding requires packed Q/K/V projection weights")
        width = attention.embed_dim
        start, stop = projection * width, (projection + 1) * width
        bias = attention.in_proj_bias
        projected = F.linear(
            value,
            weight[start:stop],
            bias[start:stop] if bias is not None else None,
        )
        return projected.reshape(
            projected.shape[0], projected.shape[1], attention.num_heads, attention.head_dim
        ).transpose(1, 2)

    def _cached_attention(
        self,
        attention: nn.MultiheadAttention,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Tensor:
        projected_query = self._project_attention(attention, query, 0)
        attended = F.scaled_dot_product_attention(
            projected_query,
            keys,
            values,
            dropout_p=float(attention.dropout) if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)

    def _decode_incremental_step(
        self,
        token: Tensor,
        position: int,
        memory: Tensor,
        layer_caches: list[tuple[Tensor, Tensor] | None],
        memory_caches: list[tuple[Tensor, Tensor] | None],
        cache_capacity: int,
    ) -> Tensor:
        """Decode one position with true projected K/V caches.

        ``nn.TransformerDecoder`` does not expose a generation cache.  Running it
        on the complete prefix for every new event makes decoding cubic in sequence
        length. Caching projected self-attention keys/values and the projected
        latent memory reduces generation to the expected quadratic attention cost.
        """
        state = self.token_embedding(token) + self.position_embedding[position : position + 1]
        for layer_index, layer in enumerate(self.decoder.layers):
            normalized = layer.norm1(state)
            cached = layer_caches[layer_index]
            new_key = self._project_attention(layer.self_attn, normalized, 1)
            new_value = self._project_attention(layer.self_attn, normalized, 2)
            if cached is None:
                key_cache = new_key.new_empty(
                    new_key.shape[0], new_key.shape[1], cache_capacity, new_key.shape[3]
                )
                value_cache = new_value.new_empty(
                    new_value.shape[0], new_value.shape[1], cache_capacity, new_value.shape[3]
                )
                cached = (key_cache, value_cache)
                layer_caches[layer_index] = cached
            key_cache, value_cache = cached
            key_cache[:, :, position : position + 1].copy_(new_key)
            value_cache[:, :, position : position + 1].copy_(new_value)
            attended = self._cached_attention(
                layer.self_attn,
                normalized,
                key_cache[:, :, : position + 1],
                value_cache[:, :, : position + 1],
            )
            state = state + layer.dropout1(attended)
            memory_cache = memory_caches[layer_index]
            if memory_cache is None:
                memory_cache = (
                    self._project_attention(layer.multihead_attn, memory, 1),
                    self._project_attention(layer.multihead_attn, memory, 2),
                )
                memory_caches[layer_index] = memory_cache
            attended_memory = self._cached_attention(
                layer.multihead_attn,
                layer.norm2(state),
                memory_cache[0],
                memory_cache[1],
            )
            state = state + layer.dropout2(attended_memory)
            feed_forward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(layer.norm3(state))))
            )
            state = state + layer.dropout3(feed_forward)
        if self.decoder.norm is not None:
            state = self.decoder.norm(state)
        return self.output(state[:, -1])

    def forward(self, tokens: Tensor, attention_mask: Tensor) -> tuple[Tensor, DiagonalGaussian]:
        posterior = self.encode_distribution(tokens, attention_mask)
        latent = posterior.sample() if self.training else posterior.mean
        decoder_tokens = tokens[:, :-1]
        if self.training and self.decoder_token_dropout > 0.0:
            eligible = (
                decoder_tokens.ne(self.pad_id)
                & decoder_tokens.ne(self.bos_id)
                & decoder_tokens.ne(self.eos_id)
            )
            dropped = torch.rand(decoder_tokens.shape, device=decoder_tokens.device).lt(
                self.decoder_token_dropout
            )
            decoder_tokens = decoder_tokens.masked_fill(eligible & dropped, self.unk_id)
        return self.decode_teacher_forced(decoder_tokens, latent), posterior

    @torch.no_grad()
    def generate(
        self,
        latent: Tensor,
        *,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_length: int = 2048,
        max_bars: int | None = None,
        min_bars: int | None = None,
        use_cache: bool = True,
        show_progress: bool = True,
        progress_description: str = "Decode tokens",
    ) -> Tensor:
        if strategy not in {"greedy", "top_p"}:
            raise ValueError(f"Unknown generation strategy: {strategy}")
        if min_bars is not None and max_bars is not None and min_bars > max_bars:
            raise ValueError("min_bars cannot exceed max_bars")
        tokens = torch.full(
            (latent.shape[0], 1), self.bos_id, dtype=torch.long, device=latent.device
        )
        finished = torch.zeros(latent.shape[0], dtype=torch.bool, device=latent.device)
        vocabulary_tokens = self.grammar.vocabulary.tokens if self.grammar is not None else ()
        bar_id = (
            self.grammar.vocabulary.id("BAR")
            if self.grammar is not None and "BAR" in vocabulary_tokens
            else None
        )
        beat_id = (
            self.grammar.vocabulary.id("BEAT")
            if self.grammar is not None and "BEAT" in vocabulary_tokens
            else None
        )
        rest_id = (
            self.grammar.vocabulary.id("REST")
            if self.grammar is not None and "REST" in vocabulary_tokens
            else None
        )
        beat_scheme = (
            self.grammar is not None and getattr(self.grammar.vocabulary, "scheme", None) == "beat"
        )
        bar_counts = torch.zeros_like(finished, dtype=torch.long)
        beats_in_bar = torch.zeros_like(finished, dtype=torch.long)
        inside_bar = torch.zeros_like(finished)
        # BEAT melodic pitches are absolute for the first pitch after INS and
        # descending relative intervals thereafter. Track this state so the
        # generator never selects an interval that decodes below MIDI A0, or a
        # drum pitch while inside a melodic instrument (and vice versa).
        track_kind = torch.zeros_like(finished, dtype=torch.int8)
        previous_pitch = torch.full_like(bar_counts, -1)
        memory = self.latent_projection(latent) if use_cache else None
        layer_caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.decoder.layers)
        memory_caches: list[tuple[Tensor, Tensor] | None] = [None] * len(self.decoder.layers)
        steps = range(min(max_length, self.max_sequence_length) - 1)
        decoding_steps = (
            track(
                steps,
                description=progress_description,
                total=len(steps),
                unit="token",
                leave=False,
                position=2,
            )
            if show_progress
            else steps
        )
        for _ in decoding_steps:
            if use_cache:
                assert memory is not None
                logits = self._decode_incremental_step(
                    tokens[:, -1:],
                    tokens.shape[1] - 1,
                    memory,
                    layer_caches,
                    memory_caches,
                    max(1, len(steps)),
                )
            else:
                logits = self.decode_teacher_forced(tokens, latent)[:, -1]
            logits = logits / max(temperature, 1e-6)
            if self.grammar is not None:
                logits = logits.masked_fill(
                    ~self.grammar.mask(tokens[:, -1]), torch.finfo(logits.dtype).min
                )
            if beat_scheme:
                minimum = torch.finfo(logits.dtype).min
                pitched_track = track_kind.eq(1)
                drum_track = track_kind.eq(2)
                logits[pitched_track, 505:593] = minimum
                logits[drum_track, 209:297] = minimum
                continuing_pitch = pitched_track & previous_pitch.ge(0)
                if bool(continuing_pitch.any()):
                    intervals = torch.arange(88, device=logits.device)
                    invalid_interval = intervals[None, :].gt(previous_pitch[:, None])
                    invalid_interval |= intervals[None, :].eq(0)
                    pitched_logits = logits[:, 209:297]
                    pitched_logits.masked_fill_(
                        continuing_pitch[:, None] & invalid_interval, minimum
                    )
            if beat_id is not None and rest_id is not None and bar_id is not None:
                incomplete_bar = inside_bar & beats_in_bar.lt(4)
                complete_bar = inside_bar & beats_in_bar.ge(4)
                logits[incomplete_bar, bar_id] = torch.finfo(logits.dtype).min
                logits[incomplete_bar, self.eos_id] = torch.finfo(logits.dtype).min
                logits[complete_bar, beat_id] = torch.finfo(logits.dtype).min
                logits[complete_bar, rest_id] = torch.finfo(logits.dtype).min
            if min_bars is not None:
                logits[bar_counts.lt(min_bars), self.eos_id] = torch.finfo(logits.dtype).min
            if strategy == "greedy":
                next_token = logits.argmax(-1)
            else:
                sorted_logits, sorted_indices = logits.sort(descending=True)
                probabilities = sorted_logits.softmax(-1)
                cumulative = probabilities.cumsum(-1)
                remove = cumulative - probabilities > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                choice = torch.multinomial(sorted_logits.softmax(-1), 1)
                next_token = sorted_indices.gather(-1, choice).squeeze(-1)
            if max_bars is not None and bar_id is not None:
                bar_limit_reached = next_token.eq(bar_id) & bar_counts.ge(max_bars)
                next_token = torch.where(bar_limit_reached, self.eos_id, next_token)
            next_token = torch.where(finished, self.pad_id, next_token)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            if bar_id is not None:
                bar_counts += next_token.eq(bar_id)
                started_bar = next_token.eq(bar_id)
                beats_in_bar = torch.where(started_bar, 0, beats_in_bar)
                inside_bar |= started_bar
            if beat_id is not None and rest_id is not None:
                beats_in_bar += next_token.eq(beat_id) | next_token.eq(rest_id)
            if beat_scheme:
                assert bar_id is not None and beat_id is not None and rest_id is not None
                reset_track = (
                    next_token.eq(bar_id) | next_token.eq(beat_id) | next_token.eq(rest_id)
                )
                track_kind = torch.where(reset_track, 0, track_kind)
                previous_pitch = torch.where(reset_track, -1, previous_pitch)
                melodic_instrument = next_token.ge(297) & next_token.lt(425)
                drum_instrument = next_token.eq(504)
                track_kind = torch.where(melodic_instrument, 1, track_kind)
                track_kind = torch.where(drum_instrument, 2, track_kind)
                previous_pitch = torch.where(
                    melodic_instrument | drum_instrument, -1, previous_pitch
                )
                pitched_token = next_token.ge(209) & next_token.lt(297) & track_kind.eq(1)
                pitch_code = next_token - 209
                absolute_pitch = torch.where(
                    previous_pitch.lt(0), pitch_code, previous_pitch - pitch_code
                )
                previous_pitch = torch.where(pitched_token, absolute_pitch, previous_pitch)
            finished |= next_token.eq(self.eos_id)
            if bool(finished.all()):
                break
        return tokens
