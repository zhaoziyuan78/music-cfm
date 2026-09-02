"""Generate shared-noise unpaired counterfactual MIDI artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

import hydra
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from cfmusic.codec.checkpoint import checkpoint_hash
from cfmusic.commands.train_codec import codec_from_config
from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.config import CONFIG_DIR, prepare_config
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
from cfmusic.training.checkpointing import checkpoint_model_state
from cfmusic.transport.factory import create_transport


def select_source_indices(
    frame: pd.DataFrame,
    *,
    strata: list[str],
    max_per_stratum: int,
    max_total: int | None,
    unique_sources: bool,
    seed: int,
) -> list[int]:
    """Select a balanced, deterministic subset without loading latent shards."""
    if max_per_stratum <= 0:
        raise ValueError("max_sources_per_style must be positive")
    missing = [column for column in strata if column not in frame]
    if missing:
        raise ValueError(f"Latent index is missing stratification columns: {missing}")
    candidates = frame
    if unique_sources:
        if "sample_id" not in candidates:
            raise ValueError("unique_sources requires sample_id in the latent index")
        candidates = candidates.loc[~candidates["sample_id"].astype(str).duplicated()]

    groups: dict[tuple[int, ...], list[int]] = {}
    group_columns: str | list[str] = strata[0] if len(strata) == 1 else strata
    for raw_key, indices in candidates.groupby(group_columns, sort=True).groups.items():
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        key = tuple(int(value) for value in values)
        groups[key] = [int(index) for index in indices]
    randomizer = random.Random(seed)
    for key in sorted(groups):
        randomizer.shuffle(groups[key])
        groups[key] = groups[key][:max_per_stratum]

    limit = sum(len(indices) for indices in groups.values())
    if max_total is not None:
        if max_total <= 0:
            raise ValueError("max_total_sources must be positive or null")
        limit = min(limit, max_total)
    selected: list[int] = []
    depth = 0
    ordered_keys = sorted(groups)
    while len(selected) < limit:
        added = False
        for key in ordered_keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    # Nearby indices tend to share an mmap shard, avoiding repeated shard reloads.
    return sorted(selected)


def concatenate_conditions(conditions: list[ConditionBatch]) -> ConditionBatch:
    if not conditions:
        raise ValueError("Cannot concatenate an empty condition list")

    def combine(name: str) -> torch.Tensor | None:
        values = [getattr(condition, name) for condition in conditions]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Inconsistent optional condition field: {name}")
        return torch.cat(cast(list[torch.Tensor], values))

    dataset_id = combine("dataset_id")
    task_id = combine("task_id")
    style_id = combine("style_id")
    assert dataset_id is not None and task_id is not None and style_id is not None
    return ConditionBatch(
        dataset_id,
        task_id,
        style_id,
        combine("genre_id"),
        combine("emotion_id"),
    )


def _safe_component(value: str) -> str:
    return value.replace("/", "-").replace("\\", "-")


def _replace_with_link_or_copy(source: Path, destination: Path) -> None:
    """Reuse immutable per-source artifacts across target directories."""
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _artifact_matches_generation(metadata_path: Path, expected_identity: Mapping[str, str]) -> bool:
    """Only reuse artifacts produced by the exact current model/config identity."""

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and all(
        payload.get(key) == value for key, value in expected_identity.items()
    )


def target_style_ids(
    source: int,
    styles: list[int],
    *,
    policy: str,
    targets_per_source: int | None,
    seed: int,
    labels: list[str] | None = None,
    explicit_pairs: list[object] | None = None,
) -> list[int]:
    alternatives = [style for style in styles if style != source]
    if policy == "all_other":
        targets = alternatives
    elif policy == "cyclic":
        targets = [styles[(styles.index(source) + 1) % len(styles)]]
    elif policy == "sampled_k":
        randomizer = random.Random(seed)
        targets = randomizer.sample(alternatives, min(len(alternatives), targets_per_source or 1))
    elif policy == "explicit_pairs":
        if labels is None or explicit_pairs is None:
            raise ValueError("explicit_pairs requires a label vocabulary and transition list")
        source_name = labels[source]
        targets = []
        for pair in explicit_pairs:
            if isinstance(pair, str):
                parts = [part.strip() for part in pair.split("->")]
                if len(parts) != 2:
                    raise ValueError(f"Invalid explicit transition: {pair!r}")
                source_label, target_label = parts
            elif isinstance(pair, DictConfig):
                source_label, target_label = str(pair.source), str(pair.target)
            else:
                raise TypeError(
                    "Explicit transitions must be 'source -> target' strings or mappings"
                )
            if source_label == source_name:
                if target_label not in labels:
                    raise ValueError(f"Unknown explicit target label: {target_label!r}")
                targets.append(labels.index(target_label))
    else:
        raise ValueError(f"Unknown target policy: {policy}")
    return targets[:targets_per_source] if targets_per_source is not None else targets


def _load_model_state(
    model: torch.nn.Module, checkpoint: Mapping[str, object], *, weights: str = "raw"
) -> None:
    state = checkpoint_model_state(checkpoint, weights=weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"Checkpoint is missing model parameters: {missing}")
    invalid = [name for name in unexpected if not name.startswith("noise_projector.")]
    if invalid:
        raise ValueError(f"Unexpected checkpoint parameters: {invalid}")


def _codec_state_for_cache(
    checkpoint: Mapping[str, object], cache_metadata: Mapping[str, object]
) -> dict[str, object]:
    weights = str(cache_metadata.get("codec_weights", "raw"))
    if weights == "raw":
        state = checkpoint.get("model")
    elif weights == "ema":
        ema = checkpoint.get("ema_model")
        state = ema.get("shadow") if isinstance(ema, Mapping) else None
    else:
        raise ValueError(f"Unknown cached codec weight variant: {weights!r}")
    if not isinstance(state, Mapping):
        raise TypeError(f"Codec checkpoint has no valid {weights} model state")
    return {"model": dict(state)}


def _distributed_checkpoint_hash(checkpoint_path: Path, context: DistributedContext) -> str:
    """Hash a shared checkpoint once instead of rereading it on every GPU rank."""

    digest: str | None = checkpoint_hash(checkpoint_path) if context.is_main else None
    if context.world_size > 1:
        values: list[object] = [digest]
        dist.broadcast_object_list(values, src=0, device=context.device)
        received = values[0]
        if not isinstance(received, str):
            raise TypeError("Invalid codec checkpoint hash broadcast")
        digest = received
    if digest is None:
        raise RuntimeError("Codec checkpoint hash was not constructed")
    return digest


def _generate(cfg: DictConfig, context: DistributedContext) -> None:
    if cfg.codec_checkpoint is None or cfg.transport_checkpoint is None:
        raise ValueError("codec_checkpoint and transport_checkpoint are both required")
    paths = prepare_config(cfg)
    device = context.device
    reset_peak_memory(device)
    data_name = str(cfg.data.name)
    latent_dataset = LatentDataset(paths["latent_dir"] / data_name, split="test")
    if len(latent_dataset) == 0:
        latent_dataset = LatentDataset(paths["latent_dir"] / data_name, split="validation")
    validate_latent_dataset(
        latent_dataset,
        codec_cfg=cfg.codec,
        transport_cfg=cfg.transport,
        dataset_name=data_name,
    )
    cache_metadata = [latent_dataset.metadata]
    tokenizer = tokenizer_from_config(
        cfg.tokenizer, max_sequence_length=int(cfg.codec.max_sequence_length)
    )
    tokenizer_digest = hashlib.sha256(
        json.dumps(asdict(tokenizer.config), sort_keys=True).encode()
    ).hexdigest()
    if latent_dataset.metadata.get("tokenizer_hash") != tokenizer_digest:
        raise ValueError("Configured tokenizer does not match the cached XMIDI latents")
    codec = codec_from_config(cfg.codec, tokenizer).to(device)
    codec_checkpoint_path = Path(str(cfg.codec_checkpoint)).expanduser().resolve()
    cached_checkpoint_hash = latent_dataset.metadata.get("codec_checkpoint_hash")
    codec_checkpoint_digest = _distributed_checkpoint_hash(codec_checkpoint_path, context)
    if cached_checkpoint_hash != codec_checkpoint_digest:
        raise ValueError("codec_checkpoint does not match the checkpoint used by cache_latents")
    codec_checkpoint = torch.load(
        codec_checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(codec_checkpoint, Mapping):
        raise TypeError("Codec checkpoint must contain a mapping")
    _load_model_state(codec, _codec_state_for_cache(codec_checkpoint, latent_dataset.metadata))
    del codec_checkpoint
    codec.eval()
    transport = create_transport(cfg.transport).to(device)
    transport_checkpoint_path = Path(str(cfg.transport_checkpoint)).expanduser().resolve()
    transport_checkpoint_digest = _distributed_checkpoint_hash(transport_checkpoint_path, context)
    transport_checkpoint = torch.load(
        transport_checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(transport_checkpoint, dict):
        raise TypeError("Transport checkpoint must contain a mapping")
    validate_transport_cache_provenance(transport_checkpoint, cache_metadata)
    transport_weights = str(cfg.counterfactual.get("transport_weights", "ema"))
    _load_model_state(transport, transport_checkpoint, weights=transport_weights)
    del transport_checkpoint
    transport.eval()
    generation_config_hash = hashlib.sha256(
        json.dumps(
            {
                "solver_steps": int(cfg.transport.solver.num_steps),
                "decode_length_multiplier": float(
                    cfg.counterfactual.get("decode_length_multiplier", 1.1)
                ),
                "deterministic_decode": bool(cfg.counterfactual.get("deterministic_decode", True)),
                "tokenizer_hash": tokenizer_digest,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    generation_identity = {
        "artifact_schema_version": "3",
        "codec_checkpoint_hash": codec_checkpoint_digest,
        "transport_checkpoint_hash": transport_checkpoint_digest,
        "transport_weights": transport_weights,
        "generation_config_hash": generation_config_hash,
    }
    card = json.loads(
        (paths["processed_dir"] / data_name / "dataset_card.json").read_text(encoding="utf-8")
    )
    labels = list(card["style_vocabulary"])
    styles = sorted(latent_dataset.frame["style_id"].astype(int).unique().tolist())
    factorial = bool(cfg.experiment.get("factorial", False))
    intervention = str(cfg.counterfactual.factorial_intervention)
    if factorial and intervention == "genre":
        strata = ["genre_id"]
    elif factorial and intervention == "emotion":
        strata = ["emotion_id"]
    elif factorial and intervention == "joint":
        strata = ["genre_id", "emotion_id"]
    elif factorial:
        raise ValueError(f"Unknown factorial intervention: {intervention}")
    else:
        strata = ["style_id"]
    maximum_total = cfg.counterfactual.get("max_total_sources")
    global_selected_indices = select_source_indices(
        latent_dataset.frame,
        strata=strata,
        max_per_stratum=int(cfg.counterfactual.max_sources_per_style),
        max_total=int(maximum_total) if maximum_total is not None else None,
        unique_sources=bool(cfg.counterfactual.get("unique_sources", True)),
        seed=int(cfg.seed),
    )
    if not global_selected_indices:
        raise RuntimeError("No counterfactual source samples were selected")
    selected_indices = global_selected_indices[context.rank :: context.world_size]

    selected_frame = latent_dataset.frame.iloc[selected_indices]
    manifest_columns = [
        "sample_id",
        "segment_id",
        "source_midi_path",
        "token_count",
        "start_bar",
        "num_bars",
    ]
    selected_sample_ids = set(selected_frame["sample_id"].astype(str))
    selected_segment_ids = (
        set(selected_frame["segment_id"].astype(str)) if "segment_id" in selected_frame else set()
    )
    filters: list[list[tuple[str, str, list[str]]]] = [
        [("sample_id", "in", sorted(selected_sample_ids))]
    ]
    if selected_segment_ids:
        filters.append([("segment_id", "in", sorted(selected_segment_ids))])
    manifest = pd.read_parquet(
        paths["processed_dir"] / data_name / "manifest.parquet",
        columns=manifest_columns,
        filters=filters,
    )
    relevant_manifest = manifest.loc[
        manifest["sample_id"].astype(str).isin(selected_sample_ids)
        | manifest["segment_id"].astype(str).isin(selected_segment_ids)
    ]
    manifest_by_sample: dict[str, dict[str, object]] = {}
    manifest_by_segment: dict[str, dict[str, object]] = {}
    for record in relevant_manifest.to_dict("records"):
        sample_key = str(record["sample_id"])
        segment_key = str(record["segment_id"])
        manifest_by_sample.setdefault(sample_key, record)
        manifest_by_segment[segment_key] = record

    requested_targets = (
        int(cfg.counterfactual.targets_per_source)
        if cfg.counterfactual.targets_per_source is not None
        else None
    )
    explicit_pairs = list(cfg.counterfactual.explicit_pairs)

    def planned_targets(index: int) -> int:
        row = latent_dataset.frame.iloc[index]
        if not factorial:
            return len(
                target_style_ids(
                    int(row["style_id"]),
                    styles,
                    policy=str(cfg.counterfactual.target_policy),
                    targets_per_source=requested_targets,
                    seed=int(cfg.seed) + index,
                    labels=labels,
                    explicit_pairs=explicit_pairs,
                )
            )
        genre_labels = list(card["genre_vocabulary"])
        emotion_labels = list(card["emotion_vocabulary"])
        if intervention in {"genre", "emotion"}:
            axis_source = int(row[f"{intervention}_id"])
            axis_labels = genre_labels if intervention == "genre" else emotion_labels
            return len(
                target_style_ids(
                    axis_source,
                    list(range(len(axis_labels))),
                    policy=str(cfg.counterfactual.target_policy),
                    targets_per_source=requested_targets,
                    seed=int(cfg.seed) + index,
                    labels=axis_labels,
                    explicit_pairs=explicit_pairs,
                )
            )
        count = (len(genre_labels) - 1) * (len(emotion_labels) - 1)
        return min(count, requested_targets) if requested_targets is not None else count

    planned_transition_total = sum(planned_targets(index) for index in selected_indices)
    global_transition_total = sum(planned_targets(index) for index in global_selected_indices)
    artifact_root = paths["artifacts_dir"] / str(cfg.experiment.name) / data_name
    transition_progress = progress_bar(
        description=f"Generate counterfactuals (rank {context.rank})",
        total=planned_transition_total,
        unit="transition",
        position=1,
    )
    source_progress = track(
        selected_indices,
        description=f"Generate selected sources (rank {context.rank})",
        total=len(selected_indices),
        unit="source",
        position=0,
    )
    completed_metadata: list[str] = []
    for index in source_progress:
        item = latent_dataset[index]
        source_style = int(item["style_id"])
        latent = item["latent"]
        dataset_id = int(item["dataset_id"])
        if not isinstance(latent, torch.Tensor):
            raise TypeError("Invalid latent sample")
        latent = latent[None].to(device)
        dataset_tensor = torch.tensor([dataset_id], device=device)
        task_tensor = torch.zeros(1, dtype=torch.long, device=device)
        transitions: list[tuple[ConditionBatch, str, str, int]] = []
        if factorial:
            if "genre_id" not in item or "emotion_id" not in item:
                raise ValueError("Factorial generation requires cached genre_id and emotion_id")
            source_genre, source_emotion = int(item["genre_id"]), int(item["emotion_id"])
            genre_labels = list(card["genre_vocabulary"])
            emotion_labels = list(card["emotion_vocabulary"])
            source_condition = ConditionBatch(
                dataset_tensor,
                task_tensor,
                torch.zeros(1, dtype=torch.long, device=device),
                torch.tensor([source_genre], device=device),
                torch.tensor([source_emotion], device=device),
            )
            if intervention in {"genre", "emotion"}:
                axis_source = source_genre if intervention == "genre" else source_emotion
                axis_labels = genre_labels if intervention == "genre" else emotion_labels
                target_ids = target_style_ids(
                    axis_source,
                    list(range(len(axis_labels))),
                    policy=str(cfg.counterfactual.target_policy),
                    targets_per_source=requested_targets,
                    seed=int(cfg.seed) + index,
                    labels=axis_labels,
                    explicit_pairs=explicit_pairs,
                )
                for target_id in target_ids:
                    target_genre = target_id if intervention == "genre" else source_genre
                    target_emotion = target_id if intervention == "emotion" else source_emotion
                    transitions.append(
                        (
                            ConditionBatch(
                                dataset_tensor,
                                task_tensor,
                                torch.zeros(1, dtype=torch.long, device=device),
                                torch.tensor([target_genre], device=device),
                                torch.tensor([target_emotion], device=device),
                            ),
                            axis_labels[axis_source],
                            axis_labels[target_id],
                            target_id,
                        )
                    )
            elif intervention == "joint":
                candidates = [
                    (genre, emotion)
                    for genre in range(len(genre_labels))
                    for emotion in range(len(emotion_labels))
                    if genre != source_genre and emotion != source_emotion
                ]
                if requested_targets is not None:
                    candidates = candidates[:requested_targets]
                for target_genre, target_emotion in candidates:
                    transitions.append(
                        (
                            ConditionBatch(
                                dataset_tensor,
                                task_tensor,
                                torch.zeros(1, dtype=torch.long, device=device),
                                torch.tensor([target_genre], device=device),
                                torch.tensor([target_emotion], device=device),
                            ),
                            f"{genre_labels[source_genre]}+{emotion_labels[source_emotion]}",
                            f"{genre_labels[target_genre]}+{emotion_labels[target_emotion]}",
                            target_genre * len(emotion_labels) + target_emotion,
                        )
                    )
            else:
                raise ValueError(f"Unknown factorial intervention: {intervention}")
        else:
            source_condition = ConditionBatch(
                dataset_tensor, task_tensor, torch.tensor([source_style], device=device)
            )
            target_ids = target_style_ids(
                source_style,
                styles,
                policy=str(cfg.counterfactual.target_policy),
                targets_per_source=requested_targets,
                seed=int(cfg.seed) + index,
                labels=labels,
                explicit_pairs=explicit_pairs,
            )
            source_name = labels[source_style] if source_style < len(labels) else str(source_style)
            for target_style in target_ids:
                target_name = (
                    labels[target_style] if target_style < len(labels) else str(target_style)
                )
                transitions.append(
                    (
                        ConditionBatch(
                            dataset_tensor, task_tensor, torch.tensor([target_style], device=device)
                        ),
                        source_name,
                        target_name,
                        target_style,
                    )
                )
        sample_id = str(item["sample_id"])
        latent_row = latent_dataset.frame.iloc[index]
        segment_id = str(latent_row.get("segment_id", sample_id))
        source_record = manifest_by_segment.get(segment_id) or manifest_by_sample.get(sample_id)
        source_path = (
            Path(str(source_record["source_midi_path"])) if source_record is not None else None
        )
        artifact_id = _safe_component(segment_id)
        pending: list[tuple[ConditionBatch, str, str, int, Path]] = []
        for target_condition, source_name, target_name, target_style in transitions:
            safe_transition = _safe_component(f"{source_name}_to_{target_name}")
            sample_dir = artifact_root / safe_transition / artifact_id
            metadata_path = sample_dir / "counterfactual_metadata.json"
            completed_metadata.append(str(metadata_path.relative_to(artifact_root)))
            if (
                bool(cfg.counterfactual.get("skip_existing", True))
                and metadata_path.exists()
                and (sample_dir / "counterfactual.mid").exists()
                and (sample_dir / "source.mid").exists()
                and (sample_dir / "same_style_reconstruction.mid").exists()
                and (sample_dir / "abducted_noise.pt").exists()
                and _artifact_matches_generation(metadata_path, generation_identity)
            ):
                transition_progress.update(1)
                continue
            pending.append((target_condition, source_name, target_name, target_style, sample_dir))
        if not pending:
            source_progress.set_postfix(generated=transition_progress.n, refresh=False)
            continue

        transition_progress.set_postfix(
            source=pending[0][1], target=pending[0][2], stage="transport", refresh=True
        )
        solver_steps = int(cfg.transport.solver.num_steps)
        transport_precision = str(cfg.transport.training.get("precision", "fp32"))
        with (
            torch.inference_mode(),
            autocast_context(device, transport_precision),
        ):
            first_output = transport.counterfactual(
                latent,
                source_condition,
                pending[0][0],
                num_steps=solver_steps,
            )
            target_latents = [first_output.counterfactual_latent]
            target_batch_size = int(cfg.counterfactual.get("target_batch_size", 8))
            for start in range(1, len(pending), target_batch_size):
                condition_batch = concatenate_conditions(
                    [entry[0] for entry in pending[start : start + target_batch_size]]
                )
                batch_count = condition_batch.batch_size
                repeated_noise = first_output.abducted_noise.expand(batch_count, -1, -1)
                predicted = transport.predict(
                    repeated_noise,
                    condition_batch,
                    num_steps=solver_steps,
                )
                target_latents.extend(predicted[index : index + 1] for index in range(batch_count))

        source_token_count: object | None = latent_row.get("token_count")
        source_num_bars: object | None = latent_row.get("num_bars")
        if source_record is not None:
            if source_token_count is None or pd.isna(source_token_count):
                source_token_count = source_record.get("token_count")
            if source_num_bars is None or pd.isna(source_num_bars):
                source_num_bars = source_record.get("num_bars")
        maximum_decode_length = tokenizer.config.max_sequence_length
        if source_token_count is not None and not pd.isna(source_token_count):
            maximum_decode_length = min(
                maximum_decode_length,
                max(
                    16,
                    math.ceil(
                        int(str(source_token_count))
                        * float(cfg.counterfactual.get("decode_length_multiplier", 1.1))
                    ),
                ),
            )
        maximum_bars = (
            int(str(source_num_bars))
            if source_num_bars is not None and not pd.isna(source_num_bars)
            else None
        )
        transition_progress.set_postfix(
            source=pending[0][1], target=f"batch({len(pending)})", stage="decode", refresh=True
        )
        normalized_latents = [first_output.reconstructed_source_latent, *target_latents]
        decoded_midis: list[object] = []
        decode_batch_size = int(cfg.counterfactual.get("decode_batch_size", 8))
        for start in range(0, len(normalized_latents), decode_batch_size):
            normalized_batch = torch.cat(
                normalized_latents[start : start + decode_batch_size], dim=0
            )
            raw_latent = latent_dataset.statistics.denormalize(normalized_batch)
            with (
                torch.inference_mode(),
                autocast_context(device, str(cfg.codec.inference.precision)),
            ):
                token_tensor = codec.generate(
                    raw_latent,
                    strategy="greedy",
                    temperature=1.0,
                    top_p=1.0,
                    max_length=maximum_decode_length,
                    max_bars=maximum_bars,
                    min_bars=maximum_bars,
                    use_cache=True,
                    show_progress=bool(cfg.counterfactual.get("show_decode_progress", False)),
                    progress_description="Decode counterfactual batch",
                )
            decoded_midis.extend(tokenizer.decode(tokens.tolist()) for tokens in token_tensor.cpu())

        first_dir = pending[0][4]
        first_dir.mkdir(parents=True, exist_ok=True)
        if source_path is None or not source_path.exists() or source_record is None:
            raise FileNotFoundError(f"Missing source MIDI for selected segment {segment_id}")
        source_midi = load_midi(source_path)
        source_tokens = tokenizer.encode(
            source_midi,
            start_bar=int(str(source_record["start_bar"])),
            num_bars=int(str(source_record["num_bars"])),
        )
        tokenizer.decode(source_tokens).dump(str(first_dir / "source.mid"))
        reconstructed_path = first_dir / "same_style_reconstruction.mid"
        decoded_midis[0].dump(str(reconstructed_path))  # type: ignore[attr-defined]
        noise_path = first_dir / "abducted_noise.pt"
        torch.save(first_output.abducted_noise.cpu(), noise_path)
        noise_digest = hashlib.sha256(noise_path.read_bytes()).hexdigest()
        roundtrip = latent_errors(latent, first_output.reconstructed_source_latent)

        for transition_index, (
            _target_condition,
            source_name,
            target_name,
            target_style,
            sample_dir,
        ) in enumerate(pending):
            transition_progress.set_postfix(
                source=source_name, target=target_name, stage="write", refresh=True
            )
            sample_dir.mkdir(parents=True, exist_ok=True)
            if sample_dir != first_dir:
                if (first_dir / "source.mid").exists():
                    _replace_with_link_or_copy(first_dir / "source.mid", sample_dir / "source.mid")
                _replace_with_link_or_copy(
                    reconstructed_path, sample_dir / "same_style_reconstruction.mid"
                )
                _replace_with_link_or_copy(noise_path, sample_dir / "abducted_noise.pt")
            decoded_midis[transition_index + 1].dump(  # type: ignore[attr-defined]
                str(sample_dir / "counterfactual.mid")
            )
            metadata = {
                "sample_id": sample_id,
                "segment_id": segment_id,
                "dataset": data_name,
                "source_style": source_name,
                "source_style_id": source_style,
                "target_style": target_name,
                "target_style_id": target_style,
                "factorial_intervention": intervention if factorial else None,
                "shared_abducted_noise": True,
                "source_midi_path": str(source_path) if source_path else None,
                "inverse_nfe": first_output.inverse_nfe,
                "forward_nfe": first_output.forward_nfe,
                "noise_sha256": noise_digest,
                "latent_roundtrip": roundtrip,
                **generation_identity,
            }
            (sample_dir / "source_metadata.json").write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "segment_id": segment_id,
                        "source_style": source_name,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (sample_dir / "counterfactual_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            (sample_dir / "evaluation_metrics.json").write_text(
                json.dumps(metadata["latent_roundtrip"], indent=2), encoding="utf-8"
            )
            transition_progress.update(1)
        source_progress.set_postfix(generated=transition_progress.n, refresh=False)
    transition_progress.close()
    local_peak_memory = peak_memory_gib(device)
    gathered_metadata: list[object] = [completed_metadata]
    peak_memory = local_peak_memory
    if context.world_size > 1:
        gathered_metadata = [None] * context.world_size
        dist.all_gather_object(gathered_metadata, completed_metadata)
        peak_tensor = torch.tensor(local_peak_memory, dtype=torch.float64, device=device)
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        peak_memory = float(peak_tensor.item())
    if context.is_main:
        all_metadata = [
            path
            for rank_metadata in gathered_metadata
            if isinstance(rank_metadata, list)
            for path in rank_metadata
            if isinstance(path, str)
        ]
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "generation_manifest.json").write_text(
            json.dumps(
                {
                    "version": 3,
                    "world_size": context.world_size,
                    "selected_sources": len(global_selected_indices),
                    "planned_transitions": global_transition_total,
                    **generation_identity,
                    "metadata_files": sorted(set(all_metadata)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"Counterfactual artifacts: {artifact_root}; "
            f"sources={len(global_selected_indices)}; transitions={global_transition_total}; "
            f"peak GPU memory: {peak_memory:.2f} GiB"
        )
    distributed_barrier(context)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    context = initialize_distributed()
    try:
        _generate(cfg, context)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
