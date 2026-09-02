"""BEAT's fixed 593-token vocabulary.

The layout follows the official BEAT reference implementation:
https://github.com/Lekai-Qian/BEAT-code/blob/main/beat/vocab.py
"""

from __future__ import annotations


class BeatVocabulary:
    scheme = "beat"
    pattern_steps = 4
    pattern_count = 3**pattern_steps
    minimum_pitch = 21
    pitch_count = 88

    def __init__(self) -> None:
        tokens = [f"PAT_{index}" for index in range(81)]
        tokens += [f"VEL_{index}" for index in range(128)]
        tokens += [f"PIT_{index}" for index in range(88)]
        tokens += [f"INS_{index}" for index in range(128)]
        tokens += ["BEAT", "BAR", "EOS", "BOS", "PAD"]
        tokens += [
            "TS_2_2",
            "TS_2_4",
            "TS_3_4",
            "TS_3_8",
            "TS_4_4",
            "TS_6_8",
            "TS_9_8",
            "TS_UNK",
        ]
        tokens += [f"TEM_{index}" for index in range(15)]
        # BEAT reserves IDs 453..502. ID 453 is used as decoder input masking
        # during VAE training, while retaining its official numeric position.
        tokens += ["UNK", *[f"RESERVED_{index}" for index in range(1, 50)]]
        tokens += ["REST", "INS_DRUM"]
        tokens += [f"DRUM_PIT_{index}" for index in range(88)]
        if len(tokens) != 593:
            raise AssertionError(f"Invalid BEAT vocabulary size: {len(tokens)}")
        self.tokens = tuple(tokens)
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def token(self, index: int) -> str:
        return self.tokens[index] if 0 <= index < len(self.tokens) else "UNK"

    @property
    def pad_id(self) -> int:
        return 429

    @property
    def bos_id(self) -> int:
        return 428

    @property
    def eos_id(self) -> int:
        return 427

    @property
    def unk_id(self) -> int:
        return 453

    @property
    def beat_id(self) -> int:
        return 425

    @property
    def bar_id(self) -> int:
        return 426

    @property
    def rest_id(self) -> int:
        return 503

    @property
    def drum_instrument_id(self) -> int:
        return 504

    @staticmethod
    def time_signature_token(signature: str) -> int:
        signatures = {
            "2/2": 0,
            "2/4": 1,
            "3/4": 2,
            "3/8": 3,
            "4/4": 4,
            "6/8": 5,
            "9/8": 6,
        }
        return 430 + signatures.get(signature, 7)

    @staticmethod
    def tempo_token(tempo: float) -> int:
        rounded = round(tempo)
        bin_index = 0 if rounded < 40 else min((rounded - 40) // 20 + 1, 14)
        return 438 + bin_index

    @staticmethod
    def tempo_value(token_id: int) -> float:
        bin_index = min(14, max(0, token_id - 438))
        return 30.0 if bin_index == 0 else float(40 + (bin_index - 1) * 20 + 10)
