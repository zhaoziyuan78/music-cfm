#!/usr/bin/env python3
"""Measure BEAT sequence lengths and deterministic MIDI round-trip fidelity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfmusic.data.midi_io import load_midi
from cfmusic.progress import track
from cfmusic.tokenization.beat import BeatTokenizer, BeatTokenizerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    manifest = pd.read_parquet(
        args.data_root / "processed" / args.dataset / "manifest.parquet",
        columns=["source_midi_path", "start_bar", "num_bars", "valid"],
    )
    candidates = manifest.loc[manifest["valid"]]
    selected = candidates.sample(n=min(args.samples, len(candidates)), random_state=args.seed)
    tokenizer = BeatTokenizer(BeatTokenizerConfig(max_sequence_length=args.max_sequence_length))
    lengths: list[int] = []
    exact_roundtrips = 0
    exact_structure_roundtrips = 0
    matching_tokens = 0
    compared_tokens = 0
    velocity_absolute_error = 0
    compared_velocity_tokens = 0
    progress = track(
        selected.itertuples(index=False),
        description=f"Evaluate BEAT tokenizer ({args.dataset})",
        total=len(selected),
        unit="segment",
    )
    for row in progress:
        midi = load_midi(Path(str(row.source_midi_path)))
        tokens = tokenizer.encode_untruncated(
            midi, start_bar=int(row.start_bar), num_bars=int(row.num_bars)
        )
        reconstructed = tokenizer.decode(tokens, ticks_per_beat=midi.ticks_per_beat)
        reconstructed_tokens = tokenizer.encode_untruncated(
            reconstructed, num_bars=int(row.num_bars)
        )
        lengths.append(len(tokens))
        exact_roundtrips += int(tokens == reconstructed_tokens)
        structure = [-1 if 81 <= token < 209 else token for token in tokens]
        reconstructed_structure = [
            -1 if 81 <= token < 209 else token for token in reconstructed_tokens
        ]
        exact_structure_roundtrips += int(structure == reconstructed_structure)
        aligned = min(len(tokens), len(reconstructed_tokens))
        matching_tokens += sum(
            tokens[index] == reconstructed_tokens[index] for index in range(aligned)
        )
        for original, reconstructed_token in zip(
            tokens[:aligned], reconstructed_tokens[:aligned], strict=True
        ):
            if 81 <= original < 209 and 81 <= reconstructed_token < 209:
                velocity_absolute_error += abs(original - reconstructed_token)
                compared_velocity_tokens += 1
        compared_tokens += max(len(tokens), len(reconstructed_tokens))
        progress.set_postfix(
            exact=f"{exact_roundtrips / len(lengths):.3f}",
            p95=int(np.percentile(lengths, 95)),
            refresh=False,
        )
    values = np.asarray(lengths)
    result = {
        "dataset": args.dataset,
        "samples": len(lengths),
        "exact_token_roundtrip_rate": exact_roundtrips / max(1, len(lengths)),
        "exact_structure_roundtrip_rate": exact_structure_roundtrips / max(1, len(lengths)),
        "roundtrip_token_accuracy": matching_tokens / max(1, compared_tokens),
        "roundtrip_velocity_mae": velocity_absolute_error / max(1, compared_velocity_tokens),
        "sequence_length": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "maximum": int(values.max()),
        },
        "over_max_sequence_length": int((values > args.max_sequence_length).sum()),
        "over_max_sequence_length_rate": float((values > args.max_sequence_length).mean()),
    }
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(payload)


if __name__ == "__main__":
    main()
