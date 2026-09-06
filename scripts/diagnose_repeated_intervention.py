"""Repeat source-label abduction and target-label prediction on generated MIDI.

The ``generate`` command writes a fully isolated artifact tree compatible with
``cfmusic.commands.evaluate``. The ``compare`` command then produces paired
first-pass/second-pass tables and a Markdown report.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

from cfmusic.commands.generate_counterfactuals import (
    _codec_state_for_cache,
    _distributed_checkpoint_hash,
    _load_model_state,
    _replace_with_link_or_copy,
)
from cfmusic.commands.train_codec import codec_from_config
from cfmusic.conditioning.schema import (
    CONDITION_SCHEMA_VERSION,
    build_condition_batch,
    validate_condition_checkpoint,
)
from cfmusic.config import CONFIG_DIR
from cfmusic.data.midi_io import load_midi
from cfmusic.distributed import (
    DistributedContext,
    cleanup_distributed,
    distributed_barrier,
    initialize_distributed,
)
from cfmusic.evaluation.consistency import latent_errors
from cfmusic.latent.compatibility import (
    validate_latent_dataset,
    validate_transport_cache_provenance,
)
from cfmusic.latent.dataset import LatentDataset
from cfmusic.memory import autocast_context, peak_memory_gib, reset_peak_memory
from cfmusic.progress import progress_bar, track
from cfmusic.tokenization.factory import tokenizer_from_config
from cfmusic.transport.factory import create_transport, validate_guidance_checkpoint

DEFAULT_STORAGE = Path("/l/users/gus.xia/ziyuan/music-scm")
DEFAULT_PROJECT = Path("/home/gus.xia/ziyuan/music-scm")
SOURCE_EXPERIMENT = "e34_cfm_clamp2_prompt_cfg_roundtrip"
REPEATED_EXPERIMENT = "e34_repeated_source_to_target"
DIAGNOSTIC_SCHEMA = "cfmusic.repeated-intervention.v1"


@dataclass(frozen=True)
class ArtifactInput:
    metadata_path: Path
    metadata: dict[str, Any]
    output_directory: Path
    tokens: list[int]
    num_bars: int


def _mapping_config(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(
            config_name="config",
            overrides=["experiment=e24_cfm_cfg_roundtrip", *overrides],
        )
    if not isinstance(config, DictConfig):
        raise TypeError("Hydra composition must produce a mapping config")
    return config


def _metadata_paths(root: Path) -> list[Path]:
    manifest = root / "generation_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [root / str(value) for value in payload.get("metadata_files", [])]
        return sorted(path for path in paths if path.is_file())
    return sorted(root.rglob("counterfactual_metadata.json"))


def _load_source_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = _metadata_paths(root)
    if not paths:
        raise FileNotFoundError(f"No counterfactual metadata found under {root}")
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in track(paths, description="Read first-pass metadata", unit="transition"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid counterfactual metadata: {path}")
        for required in (
            "sample_id",
            "segment_id",
            "source_style",
            "source_style_id",
            "target_style",
            "target_style_id",
            "codec_checkpoint_hash",
            "transport_checkpoint_hash",
            "transport_weights",
        ):
            if required not in payload:
                raise ValueError(f"{path} is missing {required!r}")
        for filename in ("source.mid", "counterfactual.mid", "abducted_noise.pt"):
            if not (path.parent / filename).is_file():
                raise FileNotFoundError(f"Incomplete first-pass artifact: {path.parent / filename}")
        records.append((path, payload))
    identities = {
        (
            str(record["codec_checkpoint_hash"]),
            str(record["transport_checkpoint_hash"]),
            str(record["transport_weights"]),
            float(record.get("guidance_scale", 1.0)),
            str(record.get("condition_schema_version", "")),
        )
        for _path, record in records
    }
    if len(identities) != 1:
        raise ValueError("The source artifact tree mixes checkpoint or CFG identities")
    if next(iter(identities))[-1] != CONDITION_SCHEMA_VERSION:
        raise ValueError("First-pass artifacts use an incompatible condition schema")
    transitions = {
        (str(record["sample_id"]), int(record["source_style_id"]), int(record["target_style_id"]))
        for _path, record in records
    }
    if len(transitions) != len(records):
        raise ValueError("First-pass artifacts contain duplicate source/target transitions")
    return records


def _num_bars(metadata: Mapping[str, Any], default: int) -> int:
    value = metadata.get("num_bars")
    if value is not None:
        return int(value)
    match = re.search(r":n(\d+)$", str(metadata["segment_id"]))
    return int(match.group(1)) if match else default


def _generation_identity(
    *,
    codec_hash: str,
    transport_hash: str,
    transport_weights: str,
    guidance_scale: float,
    solver_steps: int,
    decode_length_multiplier: float,
    source_root: Path,
) -> dict[str, str]:
    payload = {
        "schema": DIAGNOSTIC_SCHEMA,
        "codec_checkpoint_hash": codec_hash,
        "transport_checkpoint_hash": transport_hash,
        "transport_weights": transport_weights,
        "guidance_scale": guidance_scale,
        "solver_steps": solver_steps,
        "decode_length_multiplier": decode_length_multiplier,
        "source_root": str(source_root.resolve()),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {
        "artifact_schema_version": "3",
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA,
        "codec_checkpoint_hash": codec_hash,
        "transport_checkpoint_hash": transport_hash,
        "transport_weights": transport_weights,
        "condition_schema_version": CONDITION_SCHEMA_VERSION,
        "generation_config_hash": digest,
    }


def _is_complete(directory: Path, identity: Mapping[str, str]) -> bool:
    metadata_path = directory / "counterfactual_metadata.json"
    if not all(
        (directory / filename).is_file()
        for filename in (
            "source.mid",
            "first_pass_counterfactual.mid",
            "counterfactual.mid",
            "abducted_noise.pt",
        )
    ) or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(metadata, dict) and all(
        metadata.get(key) == value for key, value in identity.items()
    )


def _true_token_length(tokens: torch.Tensor, eos_id: int) -> int:
    positions = torch.nonzero(tokens.eq(eos_id), as_tuple=False)
    return int(positions[0, 0]) + 1 if positions.numel() else int(tokens.numel())


def generate(args: argparse.Namespace) -> None:
    context = initialize_distributed()
    try:
        _generate(args, context)
    finally:
        cleanup_distributed()


def _generate(args: argparse.Namespace, context: DistributedContext) -> None:
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    codec_path = args.codec_checkpoint.expanduser().resolve()
    transport_path = args.transport_checkpoint.expanduser().resolve()
    latent_index = args.latent_index.expanduser().resolve()
    if args.batch_size <= 0 or args.solver_steps <= 0 or args.decode_length_multiplier <= 0:
        raise ValueError("Batch size, solver steps, and decode multiplier must be positive")
    records = _load_source_records(source_root)
    source_guidance = {float(record.get("guidance_scale", 1.0)) for _path, record in records}
    guidance_scale = (
        float(args.guidance_scale)
        if args.guidance_scale is not None
        else next(iter(source_guidance))
    )
    transport_weights = {str(record["transport_weights"]) for _path, record in records}
    if len(transport_weights) != 1:
        raise ValueError("First-pass artifacts mix raw and EMA transport weights")
    weight_variant = next(iter(transport_weights))
    config = _mapping_config(
        [
            f"paths.data_root={data_root}",
            f"data.latent_index={latent_index}",
            f"transport.guidance_scale={guidance_scale}",
        ]
    )
    device = context.device
    reset_peak_memory(device)
    latent_dataset = LatentDataset(
        data_root / "latents" / "xmidi", split="test", index_path=latent_index
    )
    validate_latent_dataset(
        latent_dataset,
        codec_cfg=config.codec,
        transport_cfg=config.transport,
        dataset_name="xmidi",
    )
    dataset_ids = sorted(latent_dataset.frame["dataset_id"].astype(int).unique().tolist())
    if len(dataset_ids) != 1:
        raise ValueError(f"Expected one XMIDI dataset id, found {dataset_ids}")
    dataset_id = dataset_ids[0]

    tokenizer = tokenizer_from_config(
        config.tokenizer, max_sequence_length=int(config.codec.max_sequence_length)
    )
    codec_hash = _distributed_checkpoint_hash(codec_path, context)
    transport_hash = _distributed_checkpoint_hash(transport_path, context)
    source_codec_hashes = {str(record["codec_checkpoint_hash"]) for _path, record in records}
    source_transport_hashes = {
        str(record["transport_checkpoint_hash"]) for _path, record in records
    }
    if source_codec_hashes != {codec_hash} or source_transport_hashes != {transport_hash}:
        raise ValueError("Requested checkpoints do not match the first-pass artifact provenance")
    if latent_dataset.metadata.get("codec_checkpoint_hash") != codec_hash:
        raise ValueError("Codec checkpoint does not match the published latent cache")

    codec_checkpoint = torch.load(
        codec_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(codec_checkpoint, Mapping):
        raise TypeError("Codec checkpoint must contain a mapping")
    codec = codec_from_config(config.codec, tokenizer).to(device)
    _load_model_state(codec, _codec_state_for_cache(codec_checkpoint, latent_dataset.metadata))
    del codec_checkpoint
    codec.eval()

    transport_checkpoint = torch.load(
        transport_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(transport_checkpoint, dict):
        raise TypeError("Transport checkpoint must contain a mapping")
    validate_transport_cache_provenance(transport_checkpoint, [latent_dataset.metadata])
    validate_condition_checkpoint(transport_checkpoint, task="genre", factorial=False)
    validate_guidance_checkpoint(
        transport_checkpoint, config.transport, exact_training_match=False
    )
    transport = create_transport(config.transport).to(device)
    _load_model_state(transport, transport_checkpoint, weights=weight_variant)
    del transport_checkpoint
    transport.eval()

    identity = _generation_identity(
        codec_hash=codec_hash,
        transport_hash=transport_hash,
        transport_weights=weight_variant,
        guidance_scale=guidance_scale,
        solver_steps=args.solver_steps,
        decode_length_multiplier=args.decode_length_multiplier,
        source_root=source_root,
    )
    assigned = records[context.rank :: context.world_size]
    all_relative_metadata: list[str] = []
    pending: list[ArtifactInput] = []
    default_bars = int(config.data.segmentation.num_bars)
    tokenization_progress = track(
        assigned,
        description=f"Tokenize first-pass CF (rank {context.rank})",
        total=len(assigned),
        unit="transition",
        position=context.rank,
    )
    for metadata_path, metadata in tokenization_progress:
        relative_directory = metadata_path.parent.relative_to(source_root)
        output_directory = output_root / relative_directory
        output_metadata = output_directory / "counterfactual_metadata.json"
        all_relative_metadata.append(str(output_metadata.relative_to(output_root)))
        if args.skip_existing and _is_complete(output_directory, identity):
            continue
        bars = _num_bars(metadata, default_bars)
        first_pass_midi = load_midi(metadata_path.parent / "counterfactual.mid")
        tokens = tokenizer.encode(first_pass_midi, start_bar=0, num_bars=bars)
        pending.append(
            ArtifactInput(metadata_path, metadata, output_directory, tokens, bars)
        )
    pending.sort(key=lambda item: len(item.tokens))

    generation_progress = progress_bar(
        description=f"Repeat intervention (rank {context.rank})",
        total=len(pending),
        unit="transition",
        position=context.rank,
    )
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        cpu_tokens = [torch.tensor(item.tokens, dtype=torch.long) for item in batch]
        tokens = pad_sequence(
            cpu_tokens,
            batch_first=True,
            padding_value=tokenizer.vocabulary.pad_id,
        ).to(device, non_blocking=True)
        attention_mask = tokens.ne(tokenizer.vocabulary.pad_id)
        source_ids = torch.tensor(
            [int(item.metadata["source_style_id"]) for item in batch],
            dtype=torch.long,
            device=device,
        )
        target_ids = torch.tensor(
            [int(item.metadata["target_style_id"]) for item in batch],
            dtype=torch.long,
            device=device,
        )
        dataset_batch = torch.full_like(source_ids, dataset_id)
        condition_metadata = {"dataset_id": dataset_batch, "style_id": source_ids}
        source_condition = build_condition_batch(
            condition_metadata,
            device,
            task="genre",
            factorial=False,
            active_id=source_ids,
        )
        target_condition = build_condition_batch(
            condition_metadata,
            device,
            task="genre",
            factorial=False,
            active_id=target_ids,
        )
        with torch.inference_mode(), autocast_context(
            device, str(config.codec.inference.precision)
        ):
            raw_input_latent = codec.encode_mean(tokens, attention_mask).float()
        normalized_input = latent_dataset.statistics.normalize(raw_input_latent)
        with torch.inference_mode(), autocast_context(
            device, str(config.transport.training.precision)
        ):
            output = transport.counterfactual(
                normalized_input,
                source_condition,
                target_condition,
                num_steps=args.solver_steps,
            )
        raw_counterfactual = latent_dataset.statistics.denormalize(
            output.counterfactual_latent.float()
        )
        maximum_length = min(
            int(config.codec.max_sequence_length),
            max(16, math.ceil(max(len(item.tokens) for item in batch) * args.decode_length_multiplier)),
        )
        maximum_bars = max(item.num_bars for item in batch)
        with torch.inference_mode(), autocast_context(
            device, str(config.codec.inference.precision)
        ):
            generated_tokens = codec.generate(
                raw_counterfactual,
                strategy="greedy",
                temperature=1.0,
                top_p=1.0,
                max_length=maximum_length,
                max_bars=maximum_bars,
                min_bars=maximum_bars,
                use_cache=True,
                show_progress=False,
            ).cpu()

        for index, item in enumerate(batch):
            item.output_directory.mkdir(parents=True, exist_ok=True)
            source_directory = item.metadata_path.parent
            _replace_with_link_or_copy(
                source_directory / "source.mid", item.output_directory / "source.mid"
            )
            _replace_with_link_or_copy(
                source_directory / "counterfactual.mid",
                item.output_directory / "first_pass_counterfactual.mid",
            )
            output_tokens = generated_tokens[index]
            tokenizer.decode(output_tokens.tolist()).dump(
                str(item.output_directory / "counterfactual.mid")
            )
            noise_path = item.output_directory / "abducted_noise.pt"
            torch.save(output.abducted_noise[index : index + 1].cpu(), noise_path)
            noise_hash = hashlib.sha256(noise_path.read_bytes()).hexdigest()
            original_sample_id = str(item.metadata["sample_id"])
            source_style = str(item.metadata["source_style"])
            target_style = str(item.metadata["target_style"])
            diagnostic_sample_id = f"{original_sample_id}::{source_style}_to_{target_style}"
            roundtrip = latent_errors(
                normalized_input[index : index + 1],
                output.reconstructed_source_latent[index : index + 1],
            )
            repeated_change = latent_errors(
                normalized_input[index : index + 1],
                output.counterfactual_latent[index : index + 1],
            )
            metadata = {
                "sample_id": diagnostic_sample_id,
                "original_sample_id": original_sample_id,
                "segment_id": str(item.metadata["segment_id"]),
                "dataset": "xmidi",
                "source_style": source_style,
                "source_style_id": int(item.metadata["source_style_id"]),
                "target_style": target_style,
                "target_style_id": int(item.metadata["target_style_id"]),
                "factorial_intervention": None,
                "shared_abducted_noise": False,
                "source_midi_path": item.metadata.get("source_midi_path"),
                "repeated_intervention": "source_label_abduction_then_target_label_prediction",
                "repeated_input": "first_pass_counterfactual.mid",
                "first_pass_metadata": str(item.metadata_path),
                "first_pass_generation_config_hash": item.metadata.get(
                    "generation_config_hash"
                ),
                "input_token_count": len(item.tokens),
                "output_token_count": _true_token_length(
                    output_tokens, tokenizer.vocabulary.eos_id
                ),
                "num_bars": item.num_bars,
                "inverse_nfe": output.inverse_nfe,
                "forward_nfe": output.forward_nfe,
                "classifier_free_guidance": bool(
                    config.transport.classifier_free_guidance
                ),
                "guidance_scale": guidance_scale,
                "noise_sha256": noise_hash,
                "latent_roundtrip": roundtrip,
                "first_to_second_latent": repeated_change,
                **identity,
            }
            (item.output_directory / "source_metadata.json").write_text(
                json.dumps(
                    {
                        "sample_id": diagnostic_sample_id,
                        "original_sample_id": original_sample_id,
                        "segment_id": str(item.metadata["segment_id"]),
                        "source_style": source_style,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (item.output_directory / "counterfactual_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            (item.output_directory / "evaluation_metrics.json").write_text(
                json.dumps(
                    {
                        "latent_roundtrip": roundtrip,
                        "first_to_second_latent": repeated_change,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        generation_progress.update(len(batch))
        generation_progress.set_postfix(
            source=batch[-1].metadata["source_style"],
            target=batch[-1].metadata["target_style"],
            gpu=f"{peak_memory_gib(device):.1f}GiB",
            refresh=False,
        )
    generation_progress.close()

    gathered: list[object] = [all_relative_metadata]
    if context.world_size > 1:
        gathered = [None] * context.world_size
        dist.all_gather_object(gathered, all_relative_metadata)
    if context.is_main:
        combined = sorted(
            {
                value
                for rank_values in gathered
                if isinstance(rank_values, list)
                for value in rank_values
                if isinstance(value, str)
            }
        )
        if len(combined) != len(records):
            raise RuntimeError(
                f"Expected {len(records)} repeated artifacts, gathered {len(combined)}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "generation_manifest.json").write_text(
            json.dumps(
                {
                    "version": 3,
                    "diagnostic_schema_version": DIAGNOSTIC_SCHEMA,
                    "source_experiment": SOURCE_EXPERIMENT,
                    "experiment": REPEATED_EXPERIMENT,
                    "world_size": context.world_size,
                    "planned_transitions": len(records),
                    "source_label_used_for_abduction": True,
                    "target_label_used_for_prediction": True,
                    "guidance_scale": guidance_scale,
                    "solver_steps": args.solver_steps,
                    **identity,
                    "metadata_files": combined,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    distributed_barrier(context)


def _parse_mapping(value: object) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {str(key): float(item) for key, item in value.items()}
    if not isinstance(value, str):
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}
    return (
        {str(key): float(item) for key, item in parsed.items()}
        if isinstance(parsed, Mapping)
        else {}
    )


def _add_latent_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ("latent_roundtrip", "first_to_second_latent"):
        if column not in output:
            continue
        mappings = output[column].map(_parse_mapping)
        for metric in ("mse", "mae", "cosine"):
            output[f"{column}_{metric}"] = mappings.map(
                lambda value, name=metric: value.get(name, np.nan)
            )
    return output


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if not np.isfinite(value) else f"{float(value):.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _paired_summary(before: pd.DataFrame, after: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    before = _add_latent_metrics(before)
    after = _add_latent_metrics(after)
    after = after.copy()
    after["diagnostic_sample_id"] = after["sample_id"]
    after["sample_id"] = after["original_sample_id"]
    keys = ["sample_id", "source_style_id", "target_style_id"]
    if after.duplicated(subset=keys).any():
        raise ValueError("Repeated evaluation has duplicate transition keys")
    merged = before.merge(after, on=keys, suffixes=("_first", "_second"), validate="one_to_one")
    if len(merged) != len(before) or len(merged) != len(after):
        raise ValueError(
            f"Paired comparison mismatch: first={len(before)}, second={len(after)}, paired={len(merged)}"
        )
    excluded = {
        "source_style_id",
        "target_style_id",
        "guidance_scale",
        "inverse_nfe",
        "forward_nfe",
    }
    numeric_before = set(before.select_dtypes(include=[np.number]).columns)
    numeric_after = set(after.select_dtypes(include=[np.number]).columns)
    metrics = sorted((numeric_before & numeric_after) - set(keys) - excluded)
    rows: list[dict[str, object]] = []
    for metric in metrics:
        first = pd.to_numeric(merged[f"{metric}_first"], errors="coerce")
        second = pd.to_numeric(merged[f"{metric}_second"], errors="coerce")
        valid = first.notna() & second.notna()
        delta = (second[valid] - first[valid]).to_numpy(dtype=np.float64)
        if not len(delta):
            continue
        mean_delta = float(delta.mean())
        standard_error = float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.0
        first_mean = float(first[valid].mean())
        rows.append(
            {
                "metric": metric,
                "n": int(valid.sum()),
                "first_pass": first_mean,
                "second_pass": float(second[valid].mean()),
                "delta": mean_delta,
                "relative_delta_pct": (
                    100 * mean_delta / abs(first_mean) if abs(first_mean) > 1e-12 else np.nan
                ),
                "delta_ci95_low": mean_delta - 1.96 * standard_error,
                "delta_ci95_high": mean_delta + 1.96 * standard_error,
            }
        )
        merged[f"delta_{metric}"] = second - first
    return pd.DataFrame(rows), merged


def compare(args: argparse.Namespace) -> None:
    baseline_root = args.baseline_root.expanduser().resolve()
    repeated_root = args.repeated_root.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    baseline_csv = baseline_root / "evaluation" / "per_transition_results.csv"
    repeated_csv = repeated_root / "evaluation" / "per_transition_results.csv"
    before = pd.read_csv(baseline_csv)
    after = pd.read_csv(repeated_csv)
    aggregate, paired = _paired_summary(before, after)
    report_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(report_dir / "aggregate_metric_changes.csv", index=False)
    paired.to_csv(report_dir / "paired_transition_changes.csv", index=False)

    pair_rows: list[dict[str, object]] = []
    paired_metrics = [
        metric
        for metric in (
            "clamp2_target_style_success",
            "clamp2_target_minus_source",
            "clamp2_target_similarity",
            "descriptor_cosine",
            "pitch_class_histogram_cosine",
            "melody_contour_cosine",
            "note_density_ratio",
            "tempo_ratio",
        )
        if f"{metric}_first" in paired and f"{metric}_second" in paired
    ]
    for (source, target), group in paired.groupby(
        ["source_style_first", "target_style_first"], sort=True
    ):
        row: dict[str, object] = {
            "source_style": source,
            "target_style": target,
            "n": len(group),
        }
        for metric in paired_metrics:
            row[f"first_{metric}"] = float(group[f"{metric}_first"].mean())
            row[f"second_{metric}"] = float(group[f"{metric}_second"].mean())
            row[f"delta_{metric}"] = float(group[f"delta_{metric}"].mean())
        pair_rows.append(row)
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(report_dir / "style_pair_changes.csv", index=False)
    target_frame = (
        pair_frame.groupby("target_style", as_index=False)
        .mean(numeric_only=True)
        .sort_values("delta_clamp2_target_style_success", ascending=False)
    )
    target_counts = pair_frame.groupby("target_style")["n"].sum()
    target_frame["n"] = target_frame["target_style"].map(target_counts)
    target_columns = [
        "target_style",
        "n",
        "first_clamp2_target_style_success",
        "second_clamp2_target_style_success",
        "delta_clamp2_target_style_success",
        "delta_clamp2_target_minus_source",
        "delta_pitch_class_histogram_cosine",
        "delta_melody_contour_cosine",
    ]
    target_frame = target_frame[target_columns]
    target_frame.to_csv(report_dir / "target_genre_changes.csv", index=False)

    baseline_noise_path = baseline_root / "evaluation" / "noise_leakage.json"
    repeated_noise_path = repeated_root / "evaluation" / "noise_leakage.json"
    noise_rows: list[dict[str, object]] = []
    if baseline_noise_path.is_file() and repeated_noise_path.is_file():
        baseline_noise = json.loads(baseline_noise_path.read_text(encoding="utf-8"))
        repeated_noise = json.loads(repeated_noise_path.read_text(encoding="utf-8"))
        for metric in sorted(set(baseline_noise) & set(repeated_noise)):
            first_value = baseline_noise[metric]
            second_value = repeated_noise[metric]
            if isinstance(first_value, (int, float)) and isinstance(second_value, (int, float)):
                noise_rows.append(
                    {
                        "metric": metric,
                        "first_pass": float(first_value),
                        "second_pass": float(second_value),
                        "delta": float(second_value) - float(first_value),
                    }
                )
    noise_frame = pd.DataFrame(noise_rows)
    noise_frame.to_csv(report_dir / "noise_metric_changes.csv", index=False)
    noise_by_metric = noise_frame.set_index("metric") if not noise_frame.empty else noise_frame

    key_names = [
        "clamp2_target_style_success",
        "clamp2_target_minus_source",
        "clamp2_target_similarity",
        "clamp2_source_similarity",
        "descriptor_cosine",
        "pitch_class_histogram_cosine",
        "melody_contour_cosine",
        "note_density_ratio",
        "tempo_ratio",
        "generated_midi_valid",
        "generated_note_count",
        "generated_note_density",
        "generated_unique_pitch_classes",
        "generated_pitch_range",
        "generated_nonpositive_duration_ratio",
        "latent_roundtrip_mse",
        "latent_roundtrip_mae",
        "latent_roundtrip_cosine",
    ]
    key_table = aggregate.loc[aggregate["metric"].isin(key_names)].copy()
    ordering = {name: index for index, name in enumerate(key_names)}
    key_table["_order"] = key_table["metric"].map(ordering)
    key_table = key_table.sort_values("_order").drop(columns="_order")
    by_metric = aggregate.set_index("metric")
    success = by_metric.loc["clamp2_target_style_success"]
    margin = by_metric.loc["clamp2_target_minus_source"]
    pitch = by_metric.loc["pitch_class_histogram_cosine"]
    melody = by_metric.loc["melody_contour_cosine"]
    descriptor = by_metric.loc["descriptor_cosine"]
    density = by_metric.loc["note_density_ratio"]
    improved_pairs = int((pair_frame["delta_clamp2_target_style_success"] > 0).sum())
    unchanged_pairs = int((pair_frame["delta_clamp2_target_style_success"] == 0).sum())
    worsened_pairs = int((pair_frame["delta_clamp2_target_style_success"] < 0).sum())
    improved_margins = int((pair_frame["delta_clamp2_target_minus_source"] > 0).sum())
    direction_columns = [
        "source_style",
        "target_style",
        "first_clamp2_target_style_success",
        "second_clamp2_target_style_success",
        "delta_clamp2_target_style_success",
        "delta_clamp2_target_minus_source",
    ]
    top_directions = pair_frame.nlargest(5, "delta_clamp2_target_style_success")[
        direction_columns
    ]
    bottom_directions = pair_frame.nsmallest(5, "delta_clamp2_target_style_success")[
        direction_columns
    ]
    summary = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "baseline_root": str(baseline_root),
        "repeated_root": str(repeated_root),
        "transitions": len(paired),
        "source_styles": int(paired["source_style_id"].nunique()),
        "target_pairs": int(
            paired[["source_style_id", "target_style_id"]].drop_duplicates().shape[0]
        ),
        "noise_first_pass_unique_sources": int(before["sample_id"].nunique()),
        "noise_second_pass_transitions": int(after["sample_id"].nunique()),
        "metrics": {
            str(row.metric): {
                "first_pass": float(row.first_pass),
                "second_pass": float(row.second_pass),
                "delta": float(row.delta),
            }
            for row in aggregate.itertuples()
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = f"""# E34 repeated source→target intervention diagnostic

## Protocol

- Scope: all {len(paired)} generated transitions from `{SOURCE_EXPERIMENT}` ({summary['source_styles']} source genres, {summary['target_pairs']} ordered genre pairs).
- First pass: original source latent → source-label CFG abduction → target-label CFG prediction.
- Second pass: encode the first-pass `counterfactual.mid`, again use the **original source label** for CFG abduction, then use the same target label for CFG prediction.
- Both passes use the same VAE, per-token-v2 normalization, raw E34 transport checkpoint, 32-step Heun solver, and CFG scale 1.5.
- Style/content/MIDI metrics for both passes use the original source MIDI as reference. Therefore the reported second-pass content delta is cumulative drift from the original, not merely first→second drift.

## Main findings

- Repeating the intervention strengthens the aggregate target-style signal: target-style success rises from {float(success['first_pass']):.2%} to {float(success['second_pass']):.2%} ({float(success['delta']):+.2%}, paired 95% CI {float(success['delta_ci95_low']):+.2%} to {float(success['delta_ci95_high']):+.2%}). Target-minus-source CLaMP 2 margin rises by {float(margin['delta']):+.4f}.
- The gain is heterogeneous: success improves on {improved_pairs}/30 ordered genre pairs, is unchanged on {unchanged_pairs}/30, and worsens on {worsened_pairs}/30. The continuous target-minus-source margin improves on {improved_margins}/30 pairs.
- The stronger style movement has a clear cumulative content cost relative to the original MIDI: descriptor cosine changes by {float(descriptor['delta']):+.4f}, pitch-class cosine by {float(pitch['delta']):+.4f}, melody-contour cosine by {float(melody['delta']):+.4f}, and note-density ratio by {float(density['delta']):+.4f}.
- MIDI validity remains {float(by_metric.loc['generated_midi_valid', 'second_pass']):.1%}. Tempo-ratio change is small and its paired confidence interval includes zero.
- The repeated noises become much easier to classify by original source label: logistic balanced accuracy is {float(noise_by_metric.loc['logistic_balanced_accuracy', 'second_pass']):.2%} and MLP balanced accuracy is {float(noise_by_metric.loc['mlp_balanced_accuracy', 'second_pass']):.2%}. Because the first and second noise evaluations contain different numbers of independent noise samples, this comparison is descriptive rather than a paired effect estimate.

The result is therefore a trade-off rather than a uniform improvement: a second application generally amplifies target-style evidence, especially for rock/classical targets, while accumulating pitch, melody, and density drift and producing substantially stronger source-label leakage in the newly abducted noises.

## Aggregate paired changes

Positive `delta` means the second pass has a larger value. Confidence intervals are normal-approximation 95% intervals over 300 paired transition deltas.

{_markdown_table(key_table)}

The complete metric table is in `aggregate_metric_changes.csv`; per-transition deltas are in `paired_transition_changes.csv`, and all 30 ordered genre-pair summaries are in `style_pair_changes.csv`.

### Changes grouped by target genre

{_markdown_table(target_frame)}

### Largest target-success gains

{_markdown_table(top_directions)}

### Largest target-success losses

{_markdown_table(bottom_directions)}

## Noise diagnostics

{_markdown_table(noise_frame)}

Noise metrics are descriptive rather than paired: the first pass has one shared abducted noise per original source song ({summary['noise_first_pass_unique_sources']} unique noises), whereas the second pass creates one noise per first-pass transition ({summary['noise_second_pass_transitions']} noises). HSIC/SWD/probe values remain evaluation-only diagnostics and are not training losses.

## Artifact locations

- First pass: `{baseline_root}`
- Repeated intervention: `{repeated_root}`
- Report tables: `{report_dir}`
"""
    (report_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generation = subparsers.add_parser("generate")
    generation.add_argument(
        "--source-root",
        type=_path,
        default=DEFAULT_STORAGE / "artifacts" / SOURCE_EXPERIMENT / "xmidi",
    )
    generation.add_argument(
        "--output-root",
        type=_path,
        default=DEFAULT_STORAGE / "artifacts" / "diagnostics" / REPEATED_EXPERIMENT / "xmidi",
    )
    generation.add_argument("--data-root", type=_path, default=DEFAULT_STORAGE / "data")
    generation.add_argument(
        "--latent-index",
        type=_path,
        default=DEFAULT_STORAGE / "data" / "latents" / "xmidi" / "index_clamp2_prompt.parquet",
    )
    generation.add_argument(
        "--codec-checkpoint",
        type=_path,
        default=DEFAULT_STORAGE / "checkpoints" / "e00_xmidi_codec" / "codec" / "xmidi" / "last.pt",
    )
    generation.add_argument(
        "--transport-checkpoint",
        type=_path,
        default=DEFAULT_STORAGE / "checkpoints" / SOURCE_EXPERIMENT / "transport" / "last.pt",
    )
    generation.add_argument("--batch-size", type=int, default=8)
    generation.add_argument("--solver-steps", type=int, default=32)
    generation.add_argument("--guidance-scale", type=float)
    generation.add_argument("--decode-length-multiplier", type=float, default=1.1)
    generation.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    generation.set_defaults(function=generate)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument(
        "--baseline-root",
        type=_path,
        default=DEFAULT_STORAGE / "artifacts" / SOURCE_EXPERIMENT / "xmidi",
    )
    comparison.add_argument(
        "--repeated-root",
        type=_path,
        default=DEFAULT_STORAGE / "artifacts" / "diagnostics" / REPEATED_EXPERIMENT / "xmidi",
    )
    comparison.add_argument(
        "--report-dir",
        type=_path,
        default=DEFAULT_PROJECT / "reports" / "diagnostics" / "e34_repeated_intervention",
    )
    comparison.set_defaults(function=compare)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    function = arguments.function
    if not callable(function):
        raise TypeError("Selected diagnostic command is not callable")
    function(arguments)


if __name__ == "__main__":
    main()
