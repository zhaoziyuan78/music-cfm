"""Build full-XMIDI CLaMP 2 genre centroids and evaluate counterfactuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from cfmusic.data.midi_io import validate_midi
from cfmusic.evaluation.clamp2 import CLAMP2_WEIGHT_FILENAMES, midi_to_mtf
from cfmusic.progress import progress_bar, track


def _parse_args() -> argparse.Namespace:
    project_root = Path(os.environ.get("CFMUSIC_PROJECT_ROOT", Path.cwd()))
    data_root = Path(
        os.environ.get("CFMUSIC_DATA_ROOT", "/l/users/gus.xia/ziyuan/music-scm/data")
    )
    artifacts_root = Path(
        os.environ.get(
            "CFMUSIC_ARTIFACTS_DIR", "/l/users/gus.xia/ziyuan/music-scm/artifacts"
        )
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
        "--repository",
        type=Path,
        default=Path(
            os.environ.get("CLAMP2_REPOSITORY", project_root / "external" / "clamp2")
        ),
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
        "--artifact-dir",
        type=Path,
        default=artifacts_root / "diagnostics" / "clamp2_xmidi_genre_centroids",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=project_root / "reports" / "diagnostics" / "clamp2_xmidi_genre_centroids",
    )
    parser.add_argument(
        "--counterfactual",
        action="append",
        default=[],
        metavar="NAME=ARTIFACT_ROOT",
        help="Evaluated generation root containing generation_manifest.json; repeatable",
    )
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--conversion-workers", type=int, default=32)
    return parser.parse_args()


def _convert_one(item: tuple[Path, Path]) -> None:
    source, destination = item
    if not destination.is_file():
        midi_to_mtf(source, destination)


def _write_if_absent(path: Path, content: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to overwrite incompatible diagnostic state: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_counterfactual_roots(values: list[str]) -> dict[str, Path]:
    if not values:
        values = [
            "stage2_current=/l/users/gus.xia/ziyuan/music-scm/artifacts/e22_cfm_exoreg/xmidi",
            "stage1=/l/users/gus.xia/ziyuan/music-scm/artifacts/diagnostics/"
            "stage1_checkpoint/e20_cfm_base/xmidi",
        ]
    roots: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Invalid --counterfactual value: {value!r}")
        if name in roots:
            raise ValueError(f"Duplicate counterfactual result name: {name}")
        roots[name] = Path(path).expanduser().resolve()
    return roots


def _unique_xmidi_sources(manifest: Path) -> pd.DataFrame:
    columns = [
        "sample_id",
        "source_midi_path",
        "split",
        "genre_label",
        "genre_id",
        "segment_index",
    ]
    frame = pd.read_parquet(manifest, columns=columns)
    unique = (
        frame.sort_values(["sample_id", "segment_index"])
        .drop_duplicates("sample_id")
        .sort_values(["genre_id", "sample_id"], ignore_index=True)
    )
    if unique["sample_id"].duplicated().any() or unique["genre_id"].nunique() != 6:
        raise ValueError("XMIDI source-song index is not unique with six genres")
    unique.insert(0, "embedding_key", [f"xmidi-{index:06d}" for index in range(len(unique))])
    return unique


def _counterfactual_records(name: str, root: Path) -> pd.DataFrame:
    manifest_path = root / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for index, relative in enumerate(manifest["metadata_files"]):
        metadata_path = root / str(relative)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        midi_path = metadata_path.parent / "counterfactual.mid"
        validity = validate_midi(midi_path)
        records.append(
            {
                "result_name": name,
                "embedding_key": f"cf-{name}-{index:06d}" if validity.valid else None,
                "sample_id": metadata["sample_id"],
                "segment_id": metadata["segment_id"],
                "source_style": metadata["source_style"],
                "source_style_id": int(metadata["source_style_id"]),
                "target_style": metadata["target_style"],
                "target_style_id": int(metadata["target_style_id"]),
                "midi_path": str(midi_path),
                "midi_valid": bool(validity.valid),
                "midi_error": validity.reason,
                "transport_checkpoint_hash": metadata["transport_checkpoint_hash"],
                "transport_weights": metadata["transport_weights"],
            }
        )
    frame = pd.DataFrame(records)
    if len(frame) != int(manifest["planned_transitions"]):
        raise ValueError(f"Incomplete generation manifest under {root}")
    return frame


def _runtime_links(runtime_dir: Path, repository: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for filename in CLAMP2_WEIGHT_FILENAMES:
        source = repository / "code" / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing CLaMP 2 weight: {source}")
        link = runtime_dir / filename
        if not link.exists():
            link.symlink_to(source)


def _count_embeddings(artifact_dir: Path, num_gpus: int) -> int:
    return sum(
        sum(1 for path in (artifact_dir / "embeddings" / f"rank-{rank}").glob("*.npy"))
        for rank in range(num_gpus)
    )


def _run_workers(
    *,
    artifact_dir: Path,
    repository: Path,
    cache_dir: Path,
    num_gpus: int,
    batch_size: int,
    expected: int,
) -> None:
    processes: list[tuple[subprocess.Popen[bytes], object, Path]] = []
    for rank in range(num_gpus):
        output_dir = artifact_dir / "embeddings" / f"rank-{rank}"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_list = artifact_dir / "file_lists" / f"rank-{rank}.jsonl"
        expected_rank = sum(1 for line in file_list.read_text(encoding="utf-8").splitlines() if line)
        if sum(1 for _ in output_dir.glob("*.npy")) == expected_rank:
            continue
        runtime_dir = artifact_dir / "runtime" / f"rank-{rank}"
        _runtime_links(runtime_dir, repository)
        log_path = runtime_dir / "batched_extractor.log"
        stream = log_path.open("ab")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(rank)
        environment["HF_HOME"] = str(cache_dir)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "cfmusic.commands.clamp2_batched_worker",
                "--repository",
                str(repository),
                "--file-list",
                str(file_list),
                "--output-dir",
                str(output_dir),
                "--batch-size",
                str(batch_size),
            ],
            cwd=runtime_dir,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, stream, log_path))
    initial = _count_embeddings(artifact_dir, num_gpus)
    bar = progress_bar(
        description="Extract full-XMIDI CLaMP 2 embeddings",
        total=expected,
        initial=initial,
        unit="midi",
    )
    try:
        while processes and any(process.poll() is None for process, _, _ in processes):
            count = _count_embeddings(artifact_dir, num_gpus)
            bar.update(max(0, count - bar.n))
            time.sleep(5)
        count = _count_embeddings(artifact_dir, num_gpus)
        bar.update(max(0, count - bar.n))
    finally:
        bar.close()
        for _, stream, _ in processes:
            stream.close()
    failures = [(process.returncode, log) for process, _, log in processes if process.returncode]
    if failures:
        code, log = failures[0]
        tail = "\n".join(
            log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        )
        raise RuntimeError(f"Batched CLaMP 2 worker exited with code {code}:\n{tail}")
    count = _count_embeddings(artifact_dir, num_gpus)
    if count != expected:
        raise RuntimeError(f"Expected {expected} embeddings but found {count}")


def _normalized_feature(path: Path) -> np.ndarray:
    feature = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Invalid CLaMP 2 embedding: {path}")
    return feature / norm


def _load_embeddings(
    frame: pd.DataFrame, *, artifact_dir: Path, key_to_rank: dict[str, int], description: str
) -> np.ndarray:
    keys = [str(value) for value in frame["embedding_key"]]
    first = _normalized_feature(
        artifact_dir / "embeddings" / f"rank-{key_to_rank[keys[0]]}" / f"{keys[0]}.npy"
    )
    embeddings = np.empty((len(keys), len(first)), dtype=np.float32)
    for index, key in enumerate(
        track(keys, description=description, total=len(keys), unit="embedding")
    ):
        embeddings[index] = _normalized_feature(
            artifact_dir / "embeddings" / f"rank-{key_to_rank[key]}" / f"{key}.npy"
        )
    return embeddings


def genre_centroids(
    embeddings: np.ndarray, labels: np.ndarray, *, num_genres: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized class centroids, unnormalized sums, and counts."""

    sums = np.stack(
        [embeddings[labels == index].sum(axis=0, dtype=np.float64) for index in range(num_genres)]
    )
    counts = np.bincount(labels, minlength=num_genres)
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    if np.any(counts < 2) or np.any(norms <= 0):
        raise ValueError("Every genre centroid needs at least two valid embeddings")
    return (sums / norms).astype(np.float32), sums, counts


def nearest_centroid_predictions(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    class_sums: np.ndarray | None = None,
    class_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict nearest cosine centroid, optionally leave one sample out."""

    similarities = embeddings @ centroids.T
    if labels is not None:
        if class_sums is None or class_counts is None:
            raise ValueError("Leave-one-out predictions require class sums and counts")
        own = class_sums[labels] - embeddings
        own /= np.linalg.norm(own, axis=1, keepdims=True)
        similarities[np.arange(len(labels)), labels] = np.einsum("ij,ij->i", embeddings, own)
    return similarities.argmax(axis=1), similarities


def _classification_metrics(
    truth: np.ndarray, predicted: np.ndarray, labels: list[str]
) -> dict[str, object]:
    matrix = confusion_matrix(truth, predicted, labels=np.arange(len(labels)))
    detail = classification_report(
        truth,
        predicted,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float((truth == predicted).mean()),
        "balanced_accuracy": float(np.mean(np.diag(matrix) / matrix.sum(axis=1))),
        "macro_f1": float(detail["macro avg"]["f1-score"]),
        "per_genre_recall": {
            label: float(matrix[index, index] / matrix[index].sum())
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
    }


def _save_confusion(
    matrix: list[list[int]], labels: list[str], *, path: Path, title: str
) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    normalized = values / values.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Nearest full-XMIDI CLaMP 2 centroid")
    axis.set_ylabel("XMIDI genre label")
    axis.set_title(title)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_results(
    *,
    sources: pd.DataFrame,
    source_embeddings: np.ndarray,
    counterfactuals: dict[str, pd.DataFrame],
    artifact_dir: Path,
    report_dir: Path,
    key_to_rank: dict[str, int],
    labels: list[str],
    config: dict[str, object],
) -> None:
    if report_dir.exists() and any(report_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing diagnostic report: {report_dir}")
    report_dir.mkdir(parents=True)
    truth = sources["genre_id"].to_numpy(dtype=np.int64)
    centroids, class_sums, class_counts = genre_centroids(
        source_embeddings, truth, num_genres=len(labels)
    )
    direct_pred, _ = nearest_centroid_predictions(source_embeddings, centroids)
    loo_pred, loo_sim = nearest_centroid_predictions(
        source_embeddings,
        centroids,
        labels=truth,
        class_sums=class_sums,
        class_counts=class_counts,
    )
    direct_metrics = _classification_metrics(truth, direct_pred, labels)
    loo_metrics = _classification_metrics(truth, loo_pred, labels)
    split_metrics = {
        str(split): _classification_metrics(truth[indices], loo_pred[indices], labels)
        for split in sorted(sources["split"].unique())
        if (indices := sources["split"].eq(split).to_numpy()).any()
    }
    source_predictions = sources[
        ["sample_id", "source_midi_path", "split", "genre_id", "genre_label"]
    ].copy()
    source_predictions["direct_predicted_genre_id"] = direct_pred
    source_predictions["leave_one_out_predicted_genre_id"] = loo_pred
    source_predictions["leave_one_out_correct"] = loo_pred == truth
    for index, label in enumerate(labels):
        source_predictions[f"loo_similarity_{label}"] = loo_sim[:, index]
    source_predictions.to_parquet(report_dir / "xmidi_per_source_predictions.parquet", index=False)
    pd.DataFrame(
        loo_metrics["confusion_matrix"], index=labels, columns=labels
    ).to_csv(report_dir / "xmidi_leave_one_out_confusion_matrix.csv")
    np.savez(
        report_dir / "genre_centroids.npz",
        centroids=centroids,
        class_sums=class_sums,
        class_counts=class_counts,
        labels=np.asarray(labels),
    )
    _save_confusion(
        loo_metrics["confusion_matrix"],
        labels,
        path=report_dir / "xmidi_leave_one_out_confusion_matrix.png",
        title="XMIDI leave-one-out centroid classification",
    )

    counterfactual_metrics: dict[str, object] = {}
    for name, frame in counterfactuals.items():
        valid = frame[frame["midi_valid"]].copy()
        embeddings = _load_embeddings(
            valid,
            artifact_dir=artifact_dir,
            key_to_rank=key_to_rank,
            description=f"Load {name} counterfactual embeddings",
        )
        predicted, similarities = nearest_centroid_predictions(embeddings, centroids)
        valid["predicted_genre_id"] = predicted
        valid["predicted_genre"] = [labels[index] for index in predicted]
        valid["target_success"] = predicted == valid["target_style_id"].to_numpy()
        valid["source_retained"] = predicted == valid["source_style_id"].to_numpy()
        valid["target_centroid_similarity"] = similarities[
            np.arange(len(valid)), valid["target_style_id"].to_numpy(dtype=np.int64)
        ]
        valid["source_centroid_similarity"] = similarities[
            np.arange(len(valid)), valid["source_style_id"].to_numpy(dtype=np.int64)
        ]
        valid["target_minus_source_centroid_similarity"] = (
            valid["target_centroid_similarity"] - valid["source_centroid_similarity"]
        )
        for index, label in enumerate(labels):
            valid[f"similarity_{label}"] = similarities[:, index]
        invalid = frame[~frame["midi_valid"]].copy()
        combined = pd.concat([valid, invalid], ignore_index=True, sort=False)
        combined.to_csv(report_dir / f"{name}_counterfactual_predictions.csv", index=False)
        valid_success = float(valid["target_success"].mean())
        overall_success = float(valid["target_success"].sum() / len(frame))
        by_target = {
            str(target): float(group["target_success"].mean())
            for target, group in valid.groupby("target_style", sort=True)
        }
        counterfactual_metrics[name] = {
            "total": len(frame),
            "valid_midi": len(valid),
            "valid_midi_rate": float(len(valid) / len(frame)),
            "target_centroid_success_valid_only": valid_success,
            "target_centroid_success_invalid_as_failure": overall_success,
            "source_centroid_retention": float(valid["source_retained"].mean()),
            "mean_target_minus_source_centroid_similarity": float(
                valid["target_minus_source_centroid_similarity"].mean()
            ),
            "success_by_target_genre": by_target,
        }

    summary = {
        **config,
        "centroid_counts": {
            label: int(class_counts[index]) for index, label in enumerate(labels)
        },
        "xmidi_direct": direct_metrics,
        "xmidi_leave_one_out": loo_metrics,
        "xmidi_leave_one_out_by_split": split_metrics,
        "counterfactuals": counterfactual_metrics,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    centroid_rows = "\n".join(
        f"| {label} | {class_counts[index]} | {loo_metrics['per_genre_recall'][label]:.4f} |"
        for index, label in enumerate(labels)
    )
    counterfactual_rows = "\n".join(
        f"| {name} | {values['valid_midi']}/{values['total']} | "
        f"{values['target_centroid_success_valid_only']:.4f} | "
        f"{values['target_centroid_success_invalid_as_failure']:.4f} | "
        f"{values['source_centroid_retention']:.4f} | "
        f"{values['mean_target_minus_source_centroid_similarity']:+.4f} |"
        for name, values in counterfactual_metrics.items()
    )
    report = f"""# Full-XMIDI CLaMP 2 genre-centroid diagnostic

## Method

All {len(sources):,} unique XMIDI source songs are embedded with the official CLaMP 2 MIDI
encoder. For each genre, the arithmetic mean embedding is computed from every song carrying that
XMIDI label and L2-normalized. Classification selects the centroid with maximum cosine similarity.

The direct score uses the complete centroid. The leave-one-out score removes each evaluated song
from its own genre centroid before classification, avoiding self-inclusion while retaining the
requested all-dataset centroid estimator.

## XMIDI classification

- Direct all-centroid accuracy: **{direct_metrics['accuracy']:.4f}**
- Leave-one-out accuracy: **{loo_metrics['accuracy']:.4f}**
- Leave-one-out balanced accuracy: **{loo_metrics['balanced_accuracy']:.4f}**
- Leave-one-out macro F1: **{loo_metrics['macro_f1']:.4f}**

| XMIDI genre | Songs in centroid | Leave-one-out recall |
|---|---:|---:|
{centroid_rows}

## Generated counterfactuals

Success means that the nearest full-XMIDI centroid is the requested target genre.

| Result set | Valid MIDI | Success (valid) | Success (all) | Source retained | Target-source margin |
|---|---:|---:|---:|---:|---:|
{counterfactual_rows}

See `summary.json`, `xmidi_per_source_predictions.parquet`, and each
`*_counterfactual_predictions.csv` for complete results. This directory is separate from every
previous diagnostic and from the main experiment reports.
"""
    (report_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.num_gpus <= 0 or args.batch_size <= 0 or args.conversion_workers <= 0:
        raise ValueError("GPU, batch, and conversion worker counts must be positive")
    if args.artifact_dir.exists() and not (args.artifact_dir / "diagnostic_config.json").is_file():
        raise FileExistsError(
            f"Refusing to reuse an unrelated non-empty artifact directory: {args.artifact_dir}"
        )
    card = json.loads(args.dataset_card.read_text(encoding="utf-8"))
    labels = [str(value) for value in card["genre_vocabulary"]]
    if len(labels) != 6:
        raise ValueError(f"Expected six XMIDI genres, got {labels}")
    roots = _parse_counterfactual_roots(args.counterfactual)
    sources = _unique_xmidi_sources(args.manifest)
    counterfactuals = {
        name: _counterfactual_records(name, root) for name, root in roots.items()
    }
    config = {
        "method": "normalized mean of all unique-song CLaMP 2 MIDI embeddings",
        "distance": "cosine",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "num_unique_xmidi_sources": len(sources),
        "counterfactual_roots": {name: str(root) for name, root in roots.items()},
        "num_gpus": args.num_gpus,
        "batch_size": args.batch_size,
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_if_absent(
        args.artifact_dir / "diagnostic_config.json",
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
    )
    source_index = args.artifact_dir / "xmidi_unique_sources.parquet"
    if not source_index.is_file():
        sources.to_parquet(source_index, index=False)
    elif not pd.read_parquet(source_index).equals(sources):
        raise ValueError(f"Refusing to overwrite incompatible source index: {source_index}")

    jobs: list[tuple[str, Path]] = [
        (str(row.embedding_key), Path(str(row.source_midi_path)))
        for row in sources.itertuples(index=False)
    ]
    for frame in counterfactuals.values():
        jobs.extend(
            (str(row.embedding_key), Path(str(row.midi_path)))
            for row in frame[frame["midi_valid"]].itertuples(index=False)
        )
    key_to_rank: dict[str, int] = {}
    conversions: list[tuple[Path, Path]] = []
    entries_by_rank: list[list[dict[str, str]]] = [[] for _ in range(args.num_gpus)]
    for index, (key, midi_path) in enumerate(jobs):
        rank = index % args.num_gpus
        key_to_rank[key] = rank
        mtf_path = args.artifact_dir / "inputs" / f"rank-{rank}" / f"{key}.mtf"
        conversions.append((midi_path, mtf_path))
        entries_by_rank[rank].append({"key": key, "mtf_path": str(mtf_path)})
    for rank, entries in enumerate(entries_by_rank):
        content = "".join(json.dumps(entry) + "\n" for entry in entries)
        _write_if_absent(args.artifact_dir / "file_lists" / f"rank-{rank}.jsonl", content)

    with ProcessPoolExecutor(max_workers=args.conversion_workers) as executor:
        iterator = executor.map(_convert_one, conversions, chunksize=4)
        for _ in track(
            iterator,
            description="Convert full XMIDI and counterfactual MIDI to MTF",
            total=len(conversions),
            unit="midi",
        ):
            pass
    _run_workers(
        artifact_dir=args.artifact_dir.resolve(),
        repository=args.repository.resolve(),
        cache_dir=args.cache_dir.resolve(),
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        expected=len(jobs),
    )
    source_embeddings = _load_embeddings(
        sources,
        artifact_dir=args.artifact_dir,
        key_to_rank=key_to_rank,
        description="Load full-XMIDI embeddings",
    )
    _write_results(
        sources=sources,
        source_embeddings=source_embeddings,
        counterfactuals=counterfactuals,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        key_to_rank=key_to_rank,
        labels=labels,
        config=config,
    )
    print(f"CLaMP 2 centroid diagnostic written to {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
