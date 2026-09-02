#!/usr/bin/env python3
"""Evaluate a codec checkpoint on a deterministic validation subset."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path
from statistics import mean

import pandas as pd
import torch
import torch.nn.functional as functional
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from cfmusic.codec.transformer_vae import TransformerVAE
from cfmusic.commands.train_codec import codec_from_config
from cfmusic.config import CONFIG_DIR
from cfmusic.data.collate import collate_token_batch
from cfmusic.data.datasets import MidiTokenDataset
from cfmusic.evaluation.reconstruction import (
    aligned_token_accuracy,
    multiset_f1,
    symbolic_note_events,
    trim_token_sequence,
)
from cfmusic.memory import autocast_context, peak_memory_gib, reset_peak_memory
from cfmusic.progress import track
from cfmusic.tokenization.factory import MidiTokenizer, tokenizer_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-samples", type=int, default=4)
    parser.add_argument("--generation-max-length", type=int, default=2560)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summed_token_nll(logits: Tensor, targets: Tensor, *, pad_id: int) -> float:
    return float(
        functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=pad_id,
            reduction="sum",
        )
    )


def evaluate_teacher_forcing(
    model: TransformerVAE,
    loader: Iterable[Mapping[str, Tensor | list[str] | str | int]],
    *,
    device: torch.device,
    precision: str,
    pad_id: int,
) -> dict[str, float]:
    total_nll = 0.0
    total_shuffled_nll = 0.0
    total_zero_nll = 0.0
    total_correct = 0
    total_tokens = 0
    posterior_means: list[Tensor] = []
    posterior_logvars: list[Tensor] = []
    progress = track(
        loader,
        description="Teacher-forced validation",
        total=len(loader) if isinstance(loader, Sized) else None,
        unit="batch",
    )
    for batch in progress:
        tokens_value = batch["tokens"]
        mask_value = batch["attention_mask"]
        if not isinstance(tokens_value, Tensor) or not isinstance(mask_value, Tensor):
            raise TypeError("Invalid codec batch")
        tokens = tokens_value.to(device, non_blocking=True)
        attention_mask = mask_value.to(device, non_blocking=True)
        with torch.inference_mode(), autocast_context(device, precision):
            posterior = model.encode_distribution(tokens, attention_mask)
            logits = model.decode_teacher_forced(tokens[:, :-1], posterior.mean)
            shuffled_logits = model.decode_teacher_forced(
                tokens[:, :-1], posterior.mean.roll(1, dims=0)
            )
            zero_logits = model.decode_teacher_forced(
                tokens[:, :-1], torch.zeros_like(posterior.mean)
            )
        targets = tokens[:, 1:]
        valid = targets.ne(pad_id)

        total_nll += summed_token_nll(logits, targets, pad_id=pad_id)
        total_shuffled_nll += summed_token_nll(shuffled_logits, targets, pad_id=pad_id)
        total_zero_nll += summed_token_nll(zero_logits, targets, pad_id=pad_id)
        total_correct += int((logits.argmax(-1).eq(targets) & valid).sum())
        total_tokens += int(valid.sum())
        posterior_means.append(posterior.mean.float().cpu())
        posterior_logvars.append(posterior.logvar.float().cpu())
        progress.set_postfix(tokens=total_tokens, refresh=False)
    means = torch.cat(posterior_means)
    logvars = torch.cat(posterior_logvars)
    raw_kl = -0.5 * (1 + logvars - means.square() - logvars.exp())
    active_units = int(means.var(dim=0, unbiased=False).gt(1e-2).sum())
    ce = total_nll / max(1, total_tokens)
    shuffled_ce = total_shuffled_nll / max(1, total_tokens)
    zero_ce = total_zero_nll / max(1, total_tokens)
    metrics = {
        "token_ce": ce,
        "shuffled_latent_token_ce": shuffled_ce,
        "zero_latent_token_ce": zero_ce,
        "shuffled_latent_ce_increase": shuffled_ce - ce,
        "zero_latent_ce_increase": zero_ce - ce,
        "token_perplexity": math.exp(min(ce, math.log(1e6))),
        "teacher_forced_token_accuracy": total_correct / max(1, total_tokens),
        "raw_kl_per_dimension": float(raw_kl.mean()),
        "posterior_mean_variance": float(means.var(dim=0, unbiased=False).mean()),
        "active_latent_dimensions": float(active_units),
        "latent_dimensions": float(means.shape[1] * means.shape[2]),
        "evaluated_tokens": float(total_tokens),
        "evaluated_samples": float(means.shape[0]),
    }
    return metrics


def evaluate_generation(
    model: TransformerVAE,
    dataset: MidiTokenDataset,
    tokenizer: MidiTokenizer,
    *,
    device: torch.device,
    precision: str,
    samples: int,
    max_length: int,
) -> dict[str, float]:
    if samples == 0:
        return {"autoregressive_samples": 0.0}
    candidates: list[dict[str, Tensor | str | int]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        tokens = item["tokens"]
        if isinstance(tokens, Tensor) and tokens.numel() <= max_length:
            candidates.append(item)
        if len(candidates) >= samples:
            break
    if not candidates:
        return {"autoregressive_samples": 0.0}
    batch = collate_token_batch(candidates, tokenizer.vocabulary.pad_id)
    tokens_value = batch["tokens"]
    mask_value = batch["attention_mask"]
    if not isinstance(tokens_value, Tensor) or not isinstance(mask_value, Tensor):
        raise TypeError("Invalid generation batch")
    tokens = tokens_value.to(device)
    attention_mask = mask_value.to(device)
    with torch.inference_mode(), autocast_context(device, precision):
        latent = model.encode_mean(tokens, attention_mask)
        generated_items: list[Tensor | None] = [None] * len(candidates)
        bars_to_indices: dict[int, list[int]] = {}
        for index, item in enumerate(candidates):
            bars_to_indices.setdefault(int(item["num_bars"]), []).append(index)
        for bars, indices in bars_to_indices.items():
            generated_group = model.generate(
                latent[indices],
                strategy="greedy",
                max_length=max_length,
                min_bars=bars,
                max_bars=bars,
                show_progress=True,
                progress_description="Autoregressive validation",
            ).cpu()
            for index, generated_item in zip(indices, generated_group, strict=True):
                generated_items[index] = generated_item
    token_accuracies: list[float] = []
    event_f1s: list[float] = []
    invalid_rates: list[float] = []
    length_ratios: list[float] = []
    eos = 0
    if model.grammar is None:
        raise RuntimeError("Codec generation requires an event grammar")
    for reference_tensor, prediction_tensor in zip(tokens_value, generated_items, strict=True):
        if prediction_tensor is None:
            raise RuntimeError("Missing an autoregressive validation result")
        reference = trim_token_sequence(
            reference_tensor.tolist(),
            eos_id=tokenizer.vocabulary.eos_id,
            pad_id=tokenizer.vocabulary.pad_id,
        )
        prediction = trim_token_sequence(
            prediction_tensor.tolist(),
            eos_id=tokenizer.vocabulary.eos_id,
            pad_id=tokenizer.vocabulary.pad_id,
        )
        token_accuracies.append(aligned_token_accuracy(reference, prediction))
        event_f1s.append(
            multiset_f1(
                symbolic_note_events(tokenizer.vocabulary, reference),
                symbolic_note_events(tokenizer.vocabulary, prediction),
            )
        )
        invalid_rates.append(model.grammar.invalid_rate(prediction))
        length_ratios.append(len(prediction) / max(1, len(reference)))
        eos += int(tokenizer.vocabulary.eos_id in prediction)
    return {
        "autoregressive_samples": float(len(candidates)),
        "autoregressive_token_accuracy": mean(token_accuracies),
        "autoregressive_note_event_f1": mean(event_f1s),
        "autoregressive_invalid_transition_rate": mean(invalid_rates),
        "autoregressive_eos_rate": eos / len(candidates),
        "autoregressive_length_ratio": mean(length_ratios),
    }


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0 or args.generation_samples < 0:
        raise ValueError("Sample and batch counts must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_config = checkpoint["config"]
    codec_config = OmegaConf.create(checkpoint_config)
    embedded_tokenizer = codec_config.get("tokenizer")
    tokenizer_config = (
        OmegaConf.create(embedded_tokenizer)
        if embedded_tokenizer is not None
        else OmegaConf.load(Path(CONFIG_DIR) / "tokenizer" / "bar_event.yaml")
    )
    if not isinstance(tokenizer_config, DictConfig):
        raise TypeError("Tokenizer configuration must be a mapping")
    tokenizer = tokenizer_from_config(
        tokenizer_config, max_sequence_length=int(codec_config.max_sequence_length)
    )
    manifest_path = args.data_root / "processed" / args.dataset / "manifest.parquet"
    frame = pd.read_parquet(manifest_path)
    validation = frame.loc[(frame["split"] == "validation") & frame["valid"]]
    if validation.empty:
        raise RuntimeError(f"No valid validation rows in {manifest_path}")
    selected = validation.sample(n=min(args.samples, len(validation)), random_state=args.seed)
    dataset = MidiTokenDataset(selected, tokenizer, split="validation")
    generation_dataset = dataset
    if args.generation_samples > 0:
        generation_candidates = validation.loc[
            validation["token_count"].astype(int) <= args.generation_max_length
        ]
        if not generation_candidates.empty:
            generation_selected = generation_candidates.sample(
                n=min(args.generation_samples, len(generation_candidates)),
                random_state=args.seed + 17,
            )
            generation_dataset = MidiTokenDataset(
                generation_selected, tokenizer, split="validation"
            )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda items: collate_token_batch(items, tokenizer.vocabulary.pad_id),
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.workers),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_peak_memory(device)
    variants: dict[str, dict[str, Tensor]] = {"raw": checkpoint["model"]}
    ema_payload = checkpoint.get("ema_model")
    if isinstance(ema_payload, dict) and isinstance(ema_payload.get("shadow"), dict):
        variants["ema"] = ema_payload["shadow"]
    variant_results: dict[str, dict[str, float]] = {}
    results: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "split": "validation",
        "checkpoint_train_state": checkpoint.get("train_state", {}),
        "variants": variant_results,
    }
    for name, state in variants.items():
        print(f"Evaluating {args.dataset} with {name} weights")
        model = codec_from_config(codec_config, tokenizer)
        model.load_state_dict(state)
        model.to(device).eval()
        teacher_metrics = evaluate_teacher_forcing(
            model,
            loader,
            device=device,
            precision=args.precision,
            pad_id=tokenizer.vocabulary.pad_id,
        )
        generation_metrics = evaluate_generation(
            model,
            generation_dataset,
            tokenizer,
            device=device,
            precision=args.precision,
            samples=args.generation_samples,
            max_length=args.generation_max_length,
        )
        variant_results[name] = {**teacher_metrics, **generation_metrics}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["peak_gpu_memory_gib"] = peak_memory_gib(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
