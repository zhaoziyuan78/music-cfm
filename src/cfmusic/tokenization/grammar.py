"""Finite-state masks for grammar-constrained token generation."""

from __future__ import annotations

from itertools import pairwise

import torch

from cfmusic.tokenization.beat_vocabulary import BeatVocabulary
from cfmusic.tokenization.vocabulary import EventVocabulary


class EventGrammar:
    def __init__(self, vocabulary: EventVocabulary | BeatVocabulary) -> None:
        self.vocabulary = vocabulary
        self.beat_scheme = getattr(vocabulary, "scheme", None) == "beat"
        categories = [self.category(token) for token in vocabulary.tokens]
        allowed_by_previous = [self.allowed_categories(token) for token in vocabulary.tokens]
        self._cpu_mask_table = torch.tensor(
            [
                [candidate_category in allowed for candidate_category in categories]
                for allowed in allowed_by_previous
            ],
            dtype=torch.bool,
        )
        if self.beat_scheme:
            # The reference tokenizer emits a pitch triple only for an active
            # pattern and MIDI note velocities are strictly positive. Keeping
            # PAT_0/VEL_0 out of generation prevents syntactically valid triples
            # that decode to silence or an artificial fallback velocity.
            for token_id, category in enumerate(categories):
                if category in {"PIT", "DRUM_PIT"}:
                    self._cpu_mask_table[token_id, vocabulary.id("PAT_0")] = False
                elif category == "PAT":
                    self._cpu_mask_table[token_id, vocabulary.id("VEL_0")] = False
        self._device_mask_tables: dict[torch.device, torch.Tensor] = {}

    @staticmethod
    def category(token: str) -> str:
        if token == "INS_DRUM":
            return token
        for prefix in (
            "DRUM_PIT_",
            "PAT_",
            "VEL_",
            "PIT_",
            "INS_",
            "TS_",
            "TEM_",
            "POSITION_",
            "PROGRAM_",
            "PITCH_",
            "DRUM_PITCH_",
            "VELOCITY_",
            "DURATION_",
            "TEMPO_",
        ):
            if token.startswith(prefix):
                return prefix.removesuffix("_")
        return token

    def allowed_categories(self, previous_token: str) -> set[str]:
        category = self.category(previous_token)
        if self.beat_scheme:
            beat_mapping = {
                "BOS": {"TS"},
                "TS": {"TEM"},
                "TEM": {"BAR"},
                "BAR": {"BEAT", "REST"},
                "BEAT": {"INS", "INS_DRUM"},
                "REST": {"BEAT", "REST", "BAR", "EOS"},
                "INS": {"PIT"},
                "INS_DRUM": {"DRUM_PIT"},
                "PIT": {"PAT"},
                "DRUM_PIT": {"PAT"},
                "PAT": {"VEL"},
                "VEL": {
                    "PIT",
                    "DRUM_PIT",
                    "INS",
                    "INS_DRUM",
                    "BEAT",
                    "REST",
                    "BAR",
                    "EOS",
                },
                "EOS": {"PAD"},
                "PAD": {"PAD"},
            }
            return beat_mapping.get(category, {"BAR", "EOS"})
        mapping = {
            "BOS": {"BAR"},
            "BAR": {"TEMPO", "TIME_SIGNATURE_4_4", "POSITION", "EOS"},
            "TEMPO": {"TIME_SIGNATURE_4_4", "POSITION", "BAR", "EOS"},
            "TIME_SIGNATURE_4_4": {"POSITION", "BAR", "EOS", "TEMPO"},
            "POSITION": {"PROGRAM"},
            "PROGRAM": {"PITCH", "DRUM_PITCH"},
            "PITCH": {"VELOCITY"},
            "DRUM_PITCH": {"VELOCITY"},
            "VELOCITY": {"DURATION"},
            "DURATION": {"POSITION", "BAR", "EOS"},
        }
        return mapping.get(category, {"BAR", "EOS"})

    def mask(self, previous_ids: torch.Tensor) -> torch.Tensor:
        table = self._device_mask_tables.get(previous_ids.device)
        if table is None:
            table = self._cpu_mask_table.to(previous_ids.device)
            self._device_mask_tables[previous_ids.device] = table
        return table[previous_ids]

    def invalid_rate(self, token_ids: list[int]) -> float:
        if len(token_ids) < 2:
            return 1.0
        invalid = 0
        for previous, current in pairwise(token_ids):
            allowed = self.allowed_categories(self.vocabulary.token(previous))
            invalid += self.category(self.vocabulary.token(current)) not in allowed
        return invalid / (len(token_ids) - 1)
