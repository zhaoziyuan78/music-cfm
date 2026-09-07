"""Build a confidence-filtered CLaMP 2 label overlay from real XMIDI segments.

The command never changes cached latent tensors.  It selects one existing
latent window per song, embeds that exact BEAT-decoded window, and writes a
small parquet overlay that keeps the original shard/offset references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO

import miditoolkit
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from cfmusic.evaluation.clamp2 import (
    CLAMP2_WEIGHT_FILENAMES,
    extract_clamp2_embeddings,
    midi_to_mtf,
    style_prompt,
)
from cfmusic.progress import progress_bar, track
from cfmusic.tokenization.beat import BeatTokenizer, BeatTokenizerConfig

LABEL_SOURCE = "clamp2_segment_prompt_ensemble_v2"
OVERLAY_SCHEMA_VERSION = "cfmusic.latent-label-overlay.v1"
PROMPT_TEMPLATES = (
    "This is a piece of {style} music.",
    "The musical genre is {style}.",
    "This MIDI performance is in the {style} genre.",
    "A {style} instrumental track.",
    "This composition has a {style} style.",
    "Music best described as {style}.",
    "An example of {style} music.",
    "The style of this music is {style}.",
)


def _parse_args() -> argparse.Namespace:
    project_root = Path(os.environ.get("CFMUSIC_PROJECT_ROOT", Path.cwd()))
    data_root = Path(os.environ.get("CFMUSIC_DATA_ROOT", "/l/users/gus.xia/ziyuan/music-scm/data"))
    artifacts_root = Path(
        os.environ.get("CFMUSIC_ARTIFACTS_DIR", "/l/users/gus.xia/ziyuan/music-scm/artifacts")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=data_root / "processed" / "xmidi" / "manifest.parquet",
    )
    parser.add_argument(
        "--dataset-card",
        type=Path,
        default=data_root / "processed" / "xmidi" / "dataset_card.json",
    )
    parser.add_argument(
        "--base-latent-index",
        type=Path,
        default=data_root / "latents" / "xmidi" / "index.parquet",
    )
    parser.add_argument(
        "--base-cache-metadata",
        type=Path,
        default=data_root / "latents" / "xmidi" / "cache_metadata.json",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(os.environ.get("CLAMP2_REPOSITORY", project_root / "external" / "clamp2")),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "CLAMP2_CACHE_DIR",
                "/l/users/gus.xia/ziyuan/music-scm/checkpoints/clamp2/huggingface",
            )
        ),
    )
    parser.add_argument(
        "--relabeled-manifest",
        type=Path,
        default=data_root / "processed" / "xmidi" / "manifest_clamp2_segment_v2.parquet",
    )
    parser.add_argument(
        "--relabeled-dataset-card",
        type=Path,
        default=data_root / "processed" / "xmidi" / "dataset_card_clamp2_segment_v2.json",
    )
    parser.add_argument(
        "--relabeled-latent-index",
        type=Path,
        default=data_root / "latents" / "xmidi" / "index_clamp2_segment_v2.parquet",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=artifacts_root / "diagnostics" / "clamp2_xmidi_segment_relabel_v2",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=project_root / "reports" / "diagnostics" / "clamp2_xmidi_segment_relabel_v2",
    )
    parser.add_argument("--keep-fraction", type=float, default=0.6)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--conversion-workers", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--keep-intermediates", action="store_true")
    return parser.parse_args()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("CLaMP 2 embeddings must be a rank-2 matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norms <= 0):
        raise ValueError("CLaMP 2 embeddings must be a finite non-zero matrix")
    return array / norms


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _configuration_fingerprint(args: argparse.Namespace, cache_hash: str) -> str:
    payload = {
        "label_source": LABEL_SOURCE,
        "base_dataset_manifest_hash": cache_hash,
        "keep_fraction": args.keep_fraction,
        "prompt_templates": PROMPT_TEMPLATES,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _reuse_complete_outputs(args: argparse.Namespace, fingerprint: str) -> bool:
    sidecar = args.relabeled_latent_index.with_suffix(".metadata.json")
    required = (
        args.relabeled_manifest,
        args.relabeled_dataset_card,
        args.relabeled_latent_index,
        sidecar,
        args.artifact_dir / "segment_label_mapping.parquet",
        args.artifact_dir / "relabel_config.json",
        args.report_dir / "summary.json",
        args.report_dir / "report.md",
    )
    present = [path.is_file() for path in required]
    if not any(present):
        return False
    if not all(present):
        missing = [str(path) for path, exists in zip(required, present, strict=True) if not exists]
        print(
            "Incomplete segment relabel publication will be rebuilt; missing: " + ", ".join(missing)
        )
        return False
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("configuration_fingerprint") != fingerprint:
        raise ValueError("Existing segment label overlay uses a different configuration")
    print(f"Reusing complete segment label overlay: {args.relabeled_latent_index}")
    return True


_TOKENIZER: BeatTokenizer | None = None


def _convert_segment(item: tuple[str, int, int, str]) -> str:
    global _TOKENIZER
    source_value, start_bar, num_bars, destination_value = item
    destination = Path(destination_value)
    if destination.is_file():
        return destination_value
    if _TOKENIZER is None:
        _TOKENIZER = BeatTokenizer(BeatTokenizerConfig(max_sequence_length=2560))
    source = miditoolkit.MidiFile(source_value)
    tokens = _TOKENIZER.encode_untruncated(source, start_bar=int(start_bar), num_bars=int(num_bars))
    if len(tokens) > _TOKENIZER.config.max_sequence_length:
        raise ValueError(f"Cached segment unexpectedly exceeds BEAT limit: {len(tokens)} tokens")
    segment_midi = _TOKENIZER.decode(tokens, ticks_per_beat=source.ticks_per_beat)
    temporary_midi = destination.with_suffix(".mid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    segment_midi.dump(str(temporary_midi))
    try:
        midi_to_mtf(temporary_midi, destination)
    finally:
        temporary_midi.unlink(missing_ok=True)
    return destination_value


def _tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return "<worker produced no log>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _extract_music_embeddings(
    frame: pd.DataFrame,
    *,
    repository: Path,
    cache_dir: Path,
    work_dir: Path,
    num_gpus: int,
    batch_size: int,
) -> np.ndarray:
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    visible = [value for value in visible if value]
    if visible and len(visible) < num_gpus:
        raise ValueError(f"Requested {num_gpus} GPUs but CUDA_VISIBLE_DEVICES exposes {visible}")
    assignments: list[list[dict[str, str]]] = [[] for _ in range(num_gpus)]
    for index, record in enumerate(frame.itertuples(index=False)):
        assignments[index % num_gpus].append(
            {"key": str(record.embedding_key), "mtf_path": str(record.mtf_path)}
        )
    processes: list[tuple[subprocess.Popen[bytes], BinaryIO, Path]] = []
    for rank, entries in enumerate(assignments):
        file_list = work_dir / "file_lists" / f"rank-{rank}.jsonl"
        file_list.parent.mkdir(parents=True, exist_ok=True)
        file_list.write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
        )
        output_dir = work_dir / "embeddings" / f"rank-{rank}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if sum(1 for _ in output_dir.glob("*.npy")) == len(entries):
            continue
        log_path = work_dir / "logs" / f"rank-{rank}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream: BinaryIO = log_path.open("ab")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = visible[rank] if visible else str(rank)
        environment["HF_HOME"] = str(cache_dir)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "cfmusic.commands.clamp2_batched_worker",
                "--repository",
                str(repository),
                "--file-list",
                str(file_list.resolve()),
                "--output-dir",
                str(output_dir.resolve()),
                "--batch-size",
                str(batch_size),
            ],
            cwd=repository / "code",
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_stream, log_path))

    def completed() -> int:
        return sum(
            sum(1 for _ in (work_dir / "embeddings" / f"rank-{rank}").glob("*.npy"))
            for rank in range(num_gpus)
        )

    bar = progress_bar(
        description=f"Embed representative XMIDI segments ({num_gpus} GPUs)",
        total=len(frame),
        initial=completed(),
        unit="segment",
    )
    try:
        while any(process.poll() is None for process, _, _ in processes):
            count = completed()
            bar.update(max(0, count - bar.n))
            time.sleep(5)
        count = completed()
        bar.update(max(0, count - bar.n))
    finally:
        bar.close()
        for _, process_stream, _ in processes:
            process_stream.close()
    failures = [(process.returncode, log) for process, _, log in processes if process.returncode]
    if failures:
        code, log = failures[0]
        raise RuntimeError(f"CLaMP 2 segment worker exited with code {code}:\n{_tail(log)}")
    if completed() != len(frame):
        raise RuntimeError(f"CLaMP 2 produced only {completed()}/{len(frame)} embeddings")
    packed = np.empty((len(frame), 768), dtype=np.float32)
    for index, record in enumerate(
        track(
            frame.itertuples(index=False),
            description="Pack segment embeddings",
            total=len(frame),
            unit="segment",
        )
    ):
        rank = index % num_gpus
        packed[index] = np.asarray(
            np.load(work_dir / "embeddings" / f"rank-{rank}" / f"{record.embedding_key}.npy"),
            dtype=np.float32,
        ).reshape(-1)
    return _normalize_rows(packed)


def _prompt_ensemble(labels: list[str], *, repository: Path, cache_dir: Path) -> np.ndarray:
    texts = {
        f"prompt-{style_id:02d}-{template_id:02d}": style_prompt(label, template)
        for style_id, label in enumerate(labels)
        for template_id, template in enumerate(PROMPT_TEMPLATES)
    }
    embeddings = extract_clamp2_embeddings(
        repository=repository,
        midi_files={},
        texts=texts,
        cache_dir=cache_dir,
    )
    ensemble = np.stack(
        [
            np.stack(
                [
                    embeddings[f"prompt-{style_id:02d}-{template_id:02d}"]
                    for template_id in range(len(PROMPT_TEMPLATES))
                ]
            ).mean(axis=0)
            for style_id in range(len(labels))
        ]
    )
    return _normalize_rows(ensemble)


def _calibrate_logit_bias(similarities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=similarities.shape[1]).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Validation split must contain every original XMIDI genre")
    weights = 1.0 / counts[labels]
    weights /= weights.mean()

    def objective(raw_bias: np.ndarray) -> float:
        bias = raw_bias - raw_bias.mean()
        logits = 50.0 * (similarities.astype(np.float64) + bias)
        negative_log_likelihood = logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]
        return float(
            np.average(negative_log_likelihood, weights=weights) + 1e-3 * np.square(50 * bias).sum()
        )

    result = minimize(
        objective,
        np.zeros(similarities.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(-0.1, 0.1)] * similarities.shape[1],
    )
    if not result.success:
        raise RuntimeError(f"CLaMP 2 logit-bias calibration failed: {result.message}")
    bias = result.x - result.x.mean()
    return bias.astype(np.float32)


def _representative_segments(manifest: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    details = manifest[
        [
            "sample_id",
            "segment_id",
            "split",
            "segment_index",
            "start_bar",
            "num_bars",
            "source_midi_path",
            "genre_id",
            "genre_label",
        ]
    ]
    eligible = index.merge(
        details,
        on=["sample_id", "segment_id", "split"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_original"),
    ).sort_values(["sample_id", "segment_index"], ignore_index=True)
    if len(eligible) != len(index):
        raise ValueError(f"Manifest matches only {len(eligible)}/{len(index)} cached segments")
    position = eligible.groupby("sample_id", sort=False).cumcount()
    count = eligible.groupby("sample_id", sort=False)["segment_id"].transform("size")
    selected = eligible.loc[position == (count - 1) // 2].reset_index(drop=True)
    if selected["sample_id"].duplicated().any() or len(selected) != index["sample_id"].nunique():
        raise RuntimeError("Failed to select exactly one cached segment per XMIDI song")
    selected.insert(0, "embedding_key", [f"segment-{value:06d}" for value in range(len(selected))])
    return selected


def main() -> None:
    args = _parse_args()
    if not 0.5 <= args.keep_fraction <= 0.7:
        raise ValueError("keep-fraction must stay in the recommended [0.5, 0.7] range")
    if args.num_gpus <= 0 or args.conversion_workers <= 0 or args.embedding_batch_size <= 0:
        raise ValueError("GPU/worker/batch counts must be positive")
    repository = args.repository.expanduser().resolve()
    for filename in CLAMP2_WEIGHT_FILENAMES:
        if not (repository / "code" / filename).is_file():
            raise FileNotFoundError(f"Missing CLaMP 2 checkpoint: {repository / 'code' / filename}")
    cache_metadata = json.loads(args.base_cache_metadata.read_text(encoding="utf-8"))
    base_hash = str(cache_metadata["dataset_manifest_hash"])
    fingerprint = _configuration_fingerprint(args, base_hash)
    if _reuse_complete_outputs(args, fingerprint):
        return

    card = json.loads(args.dataset_card.read_text(encoding="utf-8"))
    labels = [str(value) for value in card["genre_vocabulary"]]
    if len(labels) != 6 or labels != [str(value) for value in card["style_vocabulary"]]:
        raise ValueError("XMIDI must expose the same six genre/style labels")
    manifest = pd.read_parquet(args.manifest)
    base_index = pd.read_parquet(args.base_latent_index)
    representatives = _representative_segments(manifest, base_index)
    if (
        not representatives["genre_id"]
        .astype(int)
        .equals(representatives["genre_id_original"].astype(int))
    ):
        raise ValueError("Manifest and latent index disagree on original XMIDI genre labels")
    work_dir = args.artifact_dir / "work"
    mtf_dir = work_dir / "mtf"
    representatives["mtf_path"] = [
        str(mtf_dir / f"{key}.mtf") for key in representatives["embedding_key"]
    ]
    conversion_items = [
        (str(source), int(start), int(bars), str(destination))
        for source, start, bars, destination in representatives[
            ["source_midi_path", "start_bar", "num_bars", "mtf_path"]
        ].itertuples(index=False, name=None)
    ]
    with ProcessPoolExecutor(max_workers=args.conversion_workers) as executor:
        iterator = executor.map(_convert_segment, conversion_items, chunksize=32)
        for _ in track(
            iterator,
            description="Decode cached BEAT segments for CLaMP 2",
            total=len(conversion_items),
            unit="segment",
        ):
            pass

    prompt_embeddings = _prompt_ensemble(labels, repository=repository, cache_dir=args.cache_dir)
    music_embeddings = _extract_music_embeddings(
        representatives,
        repository=repository,
        cache_dir=args.cache_dir,
        work_dir=work_dir,
        num_gpus=args.num_gpus,
        batch_size=args.embedding_batch_size,
    )
    similarities = music_embeddings @ prompt_embeddings.T
    validation_mask = representatives["split"].astype(str).eq("validation").to_numpy()
    original_ids = representatives["genre_id_original"].astype(int).to_numpy()
    bias = _calibrate_logit_bias(similarities[validation_mask], original_ids[validation_mask])
    calibrated = similarities + bias[None]
    predicted = calibrated.argmax(axis=1).astype(np.int64)
    ordered = np.sort(calibrated, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]

    mapping = representatives[
        [
            "embedding_key",
            "sample_id",
            "segment_id",
            "split",
            "segment_index",
            "start_bar",
            "num_bars",
            "source_midi_path",
            "genre_id_original",
            "genre_label",
        ]
    ].rename(
        columns={
            "genre_id_original": "original_genre_id",
            "genre_label": "original_genre_label",
        }
    )
    mapping["clamp2_genre_id"] = predicted
    mapping["clamp2_genre_label"] = [labels[value] for value in predicted]
    mapping["nearest_prompt_margin"] = margins
    mapping["original_label_correct"] = original_ids == predicted
    for style_id, label in enumerate(labels):
        mapping[f"ensemble_similarity_{label}"] = similarities[:, style_id]
        mapping[f"calibrated_similarity_{label}"] = calibrated[:, style_id]
    train = mapping["split"].astype(str).eq("train")
    mapping["confidence_percentile"] = 1.0
    mapping.loc[train, "confidence_percentile"] = (
        mapping.loc[train]
        .groupby("clamp2_genre_id", observed=True)["nearest_prompt_margin"]
        .rank(method="average", pct=True)
    )
    mapping["selected_for_overlay"] = (~train) | (
        mapping["confidence_percentile"] > 1.0 - args.keep_fraction
    )
    mapping["label_confidence_weight"] = (0.25 + 0.75 * mapping["confidence_percentile"]).astype(
        np.float32
    )
    selected_mapping = mapping.loc[mapping["selected_for_overlay"]].copy()

    label_lookup = selected_mapping.set_index("segment_id")
    selected_ids = set(selected_mapping["segment_id"].astype(str))
    relabeled_index = base_index.loc[base_index["segment_id"].astype(str).isin(selected_ids)].copy()
    relabeled_index["original_style_id"] = relabeled_index["style_id"]
    relabeled_index["original_genre_id"] = relabeled_index["genre_id"]
    relabeled_index["style_id"] = (
        relabeled_index["segment_id"].map(label_lookup["clamp2_genre_id"]).astype(np.int64)
    )
    relabeled_index["genre_id"] = relabeled_index["style_id"]
    relabeled_index["label_confidence_weight"] = (
        relabeled_index["segment_id"]
        .map(label_lookup["label_confidence_weight"])
        .astype(np.float32)
    )
    relabeled_index["nearest_prompt_margin"] = (
        relabeled_index["segment_id"].map(label_lookup["nearest_prompt_margin"]).astype(np.float32)
    )
    relabeled_index["label_source"] = LABEL_SOURCE
    relabeled_index = relabeled_index.sort_values(["shard", "offset"], ignore_index=True)

    relabeled_manifest = manifest.loc[manifest["segment_id"].astype(str).isin(selected_ids)].copy()
    relabeled_manifest["original_style_id"] = relabeled_manifest["style_id"]
    relabeled_manifest["original_style_label"] = relabeled_manifest["style_label"]
    relabeled_manifest["original_genre_id"] = relabeled_manifest["genre_id"]
    relabeled_manifest["original_genre_label"] = relabeled_manifest["genre_label"]
    relabeled_manifest["style_id"] = (
        relabeled_manifest["segment_id"].map(label_lookup["clamp2_genre_id"]).astype(np.int64)
    )
    relabeled_manifest["genre_id"] = relabeled_manifest["style_id"]
    label_names = dict(enumerate(labels))
    relabeled_manifest["style_label"] = relabeled_manifest["style_id"].map(label_names)
    relabeled_manifest["genre_label"] = relabeled_manifest["style_label"]
    relabeled_manifest["label_source"] = LABEL_SOURCE

    assignment_digest = hashlib.sha256()
    assignment_digest.update(fingerprint.encode())
    for segment_id, label_id in relabeled_index[["segment_id", "style_id"]].itertuples(
        index=False, name=None
    ):
        assignment_digest.update(f"{segment_id}\0{label_id}\n".encode())
    assignment_hash = assignment_digest.hexdigest()
    configuration = {
        "label_source": LABEL_SOURCE,
        "configuration_fingerprint": fingerprint,
        "label_assignment_hash": assignment_hash,
        "base_manifest": str(args.manifest.resolve()),
        "base_latent_index": str(args.base_latent_index.resolve()),
        "base_dataset_manifest_hash": base_hash,
        "keep_fraction": args.keep_fraction,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "prompt_logit_bias": dict(zip(labels, bias.tolist(), strict=True)),
        "representative_segment_policy": "middle_cached_segment_per_unique_song",
    }
    overlay_metadata = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "label_source": LABEL_SOURCE,
        "label_assignment_hash": assignment_hash,
        "configuration_fingerprint": fingerprint,
        "base_dataset_manifest_hash": base_hash,
        "rows": len(relabeled_index),
        "unique_samples": int(relabeled_index["sample_id"].nunique()),
        "keep_fraction": args.keep_fraction,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "prompt_logit_bias": bias.tolist(),
        "genre_vocabulary": labels,
    }
    relabeled_card = dict(card)
    relabeled_card.update(configuration)
    relabeled_card["num_selected_segments"] = len(relabeled_manifest)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(mapping, args.artifact_dir / "segment_label_mapping.parquet")
    _write_json(configuration, args.artifact_dir / "relabel_config.json")
    with (args.artifact_dir / "prompt_ensemble.npy").open("wb") as stream:
        np.save(stream, prompt_embeddings)
    _write_parquet(relabeled_manifest, args.relabeled_manifest)
    _write_json(relabeled_card, args.relabeled_dataset_card)
    _write_parquet(relabeled_index, args.relabeled_latent_index)
    _write_json(overlay_metadata, args.relabeled_latent_index.with_suffix(".metadata.json"))

    args.report_dir.mkdir(parents=True, exist_ok=True)
    confusion = pd.crosstab(mapping["original_genre_label"], mapping["clamp2_genre_label"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    confusion.to_csv(args.report_dir / "original_to_segment_prompt_confusion.csv")
    counts = (
        selected_mapping.groupby(["split", "clamp2_genre_label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=labels, fill_value=0)
    )
    counts.to_csv(args.report_dir / "selected_counts_by_split.csv")
    per_class_ceiling = {
        label: float(
            mapping.loc[mapping["original_genre_id"] == style_id, "original_label_correct"].mean()
        )
        for style_id, label in enumerate(labels)
    }
    summary = {
        **configuration,
        "num_base_latent_segments": len(base_index),
        "num_representative_segments": len(mapping),
        "num_selected_overlay_segments": len(relabeled_index),
        "num_selected_train_segments": int((selected_mapping["split"] == "train").sum()),
        "segment_original_label_top1": float(mapping["original_label_correct"].mean()),
        "segment_original_label_top1_per_class": per_class_ceiling,
        "mean_margin": float(mapping["nearest_prompt_margin"].mean()),
        "median_margin": float(mapping["nearest_prompt_margin"].median()),
        "selected_counts_by_split": counts.to_dict(orient="index"),
    }
    _write_json(summary, args.report_dir / "summary.json")
    table = "\n".join(
        f"| {label} | {per_class_ceiling[label]:.2%} | "
        f"{int((selected_mapping['clamp2_genre_label'] == label).sum()):,} |"
        for label in labels
    )
    report = f"""# XMIDI segment-level CLaMP 2 relabel

One existing 8-bar latent segment was selected per unique song and labeled with an
{len(PROMPT_TEMPLATES)}-template CLaMP 2 prompt ensemble. Class logit biases were calibrated on
the original XMIDI validation labels. The training split retains the top
{args.keep_fraction:.0%} prompt-margin examples independently within each predicted class;
validation and test representatives remain available for ceiling diagnostics.

| Genre | Original-label top-1 ceiling | Selected segments |
|---|---:|---:|
{table}

- Representative segments: **{len(mapping):,}**
- Selected overlay segments: **{len(relabeled_index):,}**
- Overall original-label top-1 ceiling: **{summary["segment_original_label_top1"]:.2%}**
- Mean / median calibrated top-1 margin: **{summary["mean_margin"]:.4f} / {summary["median_margin"]:.4f}**

The overlay preserves every original latent `shard` and `offset`; no VAE weight or latent tensor
is changed. The assignment hash is included in CFM checkpoint provenance.
"""
    _write_text(report, args.report_dir / "report.md")
    if not args.keep_intermediates:
        shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Segment label overlay: {args.relabeled_latent_index.resolve()}")
    print(f"Relabel diagnostics: {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
