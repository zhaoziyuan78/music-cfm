"""Diagnose XMIDI genre labels against nearest CLaMP 2 text prompts.

This command deliberately samples unique source songs rather than preprocessed
segments.  It keeps converted MTF files and embeddings in a resumable artifact
directory while writing compact, human-readable results to a separate report
directory.
"""

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

from cfmusic.evaluation.clamp2 import (
    CLAMP2_WEIGHT_FILENAMES,
    midi_to_mtf,
    style_prompt,
)
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
        default=artifacts_root / "diagnostics" / "clamp2_xmidi_genre",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=project_root / "reports" / "diagnostics" / "clamp2_xmidi_genre",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples-per-genre", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--conversion-workers", type=int, default=32)
    parser.add_argument(
        "--prompt-template", default="This is a piece of {style} music."
    )
    return parser.parse_args()


def balanced_source_sample(
    frame: pd.DataFrame, *, samples_per_genre: int, seed: int
) -> pd.DataFrame:
    """Return a deterministic, genre-balanced sample of unique XMIDI songs."""

    required = {
        "sample_id",
        "source_midi_path",
        "split",
        "genre_label",
        "genre_id",
        "segment_index",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"XMIDI manifest is missing required columns: {sorted(missing)}")
    unique = (
        frame.sort_values(["sample_id", "segment_index"])
        .drop_duplicates("sample_id")
        .reset_index(drop=True)
    )
    groups: list[pd.DataFrame] = []
    for genre_id, group in unique.groupby("genre_id", sort=True):
        if len(group) < samples_per_genre:
            raise ValueError(
                f"Genre {genre_id} has only {len(group)} unique songs; "
                f"cannot sample {samples_per_genre}"
            )
        groups.append(group.sample(n=samples_per_genre, random_state=seed + int(genre_id)))
    sampled = pd.concat(groups, ignore_index=True).sort_values(
        ["genre_id", "sample_id"], ignore_index=True
    )
    counts = sampled.groupby("genre_id").size()
    if len(counts) != 6 or not counts.eq(samples_per_genre).all():
        raise ValueError(f"Expected six balanced genres, got {counts.to_dict()}")
    sampled.insert(0, "embedding_key", [f"music-{index:06d}" for index in range(len(sampled))])
    return sampled


def _convert_one(item: tuple[Path, Path]) -> None:
    source, destination = item
    if not destination.is_file():
        midi_to_mtf(source, destination)


def _tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return "<extractor produced no log>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _run_extractors(
    *,
    repository: Path,
    cache_dir: Path,
    artifact_dir: Path,
    expected_by_rank: list[list[str]],
) -> None:
    extractor = repository / "code" / "extract_clamp2.py"
    if not extractor.is_file():
        raise FileNotFoundError(f"Missing official CLaMP 2 extractor: {extractor}")
    for filename in CLAMP2_WEIGHT_FILENAMES:
        if not (repository / "code" / filename).is_file():
            raise FileNotFoundError(f"Missing CLaMP 2 weight: {repository / 'code' / filename}")

    processes: list[tuple[subprocess.Popen[bytes], object, Path]] = []
    total_expected = sum(len(values) for values in expected_by_rank)
    for rank, expected in enumerate(expected_by_rank):
        output_dir = artifact_dir / "embeddings" / f"rank-{rank}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if all((output_dir / f"{key}.npy").is_file() for key in expected):
            continue
        runtime_dir = artifact_dir / "runtime" / f"rank-{rank}"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        for filename in CLAMP2_WEIGHT_FILENAMES:
            link = runtime_dir / filename
            if not link.exists():
                link.symlink_to(repository / "code" / filename)
        log_path = runtime_dir / "extractor.log"
        log_stream = log_path.open("ab")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(rank)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(repository / "code"), environment.get("PYTHONPATH", "")])
        )
        environment["HF_HOME"] = str(cache_dir)
        process = subprocess.Popen(
            [
                sys.executable,
                str(extractor),
                str(artifact_dir / "inputs" / f"rank-{rank}"),
                str(output_dir),
                "--normalize",
            ],
            cwd=runtime_dir,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_stream, log_path))

    def completed_count() -> int:
        return sum(
            (artifact_dir / "embeddings" / f"rank-{rank}" / f"{key}.npy").is_file()
            for rank, keys in enumerate(expected_by_rank)
            for key in keys
        )

    bar = progress_bar(
        description="Extract CLaMP 2 embeddings (4 GPUs)",
        total=total_expected,
        initial=completed_count(),
        unit="file",
    )
    try:
        while processes and any(process.poll() is None for process, _, _ in processes):
            count = completed_count()
            bar.update(max(0, count - bar.n))
            time.sleep(2)
        count = completed_count()
        bar.update(max(0, count - bar.n))
    finally:
        bar.close()
        for _, stream, _ in processes:
            stream.close()
    failures = [(process.returncode, log) for process, _, log in processes if process.returncode]
    if failures:
        code, log = failures[0]
        raise RuntimeError(f"CLaMP 2 extractor exited with code {code}:\n{_tail(log)}")
    missing = [
        key
        for rank, keys in enumerate(expected_by_rank)
        for key in keys
        if not (artifact_dir / "embeddings" / f"rank-{rank}" / f"{key}.npy").is_file()
    ]
    if missing:
        raise RuntimeError(f"CLaMP 2 omitted {len(missing)} embeddings, including {missing[:10]}")


def _normalized_feature(path: Path) -> np.ndarray:
    feature = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Invalid CLaMP 2 embedding: {path}")
    return feature / norm


def _write_reports(
    *,
    sampled: pd.DataFrame,
    labels: list[str],
    similarities: np.ndarray,
    report_dir: Path,
    metadata: dict[str, object],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    true_ids = sampled["genre_id"].to_numpy(dtype=np.int64)
    predicted_ids = similarities.argmax(axis=1)
    predictions = sampled[
        ["embedding_key", "sample_id", "source_midi_path", "split", "genre_id", "genre_label"]
    ].copy()
    predictions["predicted_genre_id"] = predicted_ids
    predictions["predicted_genre"] = [labels[value] for value in predicted_ids]
    predictions["correct"] = predicted_ids == true_ids
    predictions["true_prompt_similarity"] = similarities[np.arange(len(sampled)), true_ids]
    masked = similarities.copy()
    masked[np.arange(len(sampled)), true_ids] = -np.inf
    predictions["true_minus_best_other"] = (
        predictions["true_prompt_similarity"].to_numpy() - masked.max(axis=1)
    )
    for genre_id, label in enumerate(labels):
        predictions[f"similarity_{label}"] = similarities[:, genre_id]
    predictions.to_csv(report_dir / "per_source_predictions.csv", index=False)

    matrix = confusion_matrix(true_ids, predicted_ids, labels=np.arange(len(labels)))
    confusion = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion.index.name = "xmidi_label"
    confusion.columns.name = "nearest_prompt"
    confusion.to_csv(report_dir / "confusion_matrix.csv")
    normalized = confusion.div(confusion.sum(axis=1), axis=0)
    normalized.to_csv(report_dir / "confusion_matrix_row_normalized.csv")

    mean_similarity = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for genre_id, label in enumerate(labels):
        mean_similarity.loc[label] = similarities[true_ids == genre_id].mean(axis=0)
    mean_similarity.index.name = "xmidi_label"
    mean_similarity.columns.name = "prompt"
    mean_similarity.to_csv(report_dir / "mean_prompt_similarity.csv")

    detail = classification_report(
        true_ids,
        predicted_ids,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    per_genre = pd.DataFrame(detail).transpose()
    per_genre.to_csv(report_dir / "classification_report.csv")
    accuracy = float((predicted_ids == true_ids).mean())
    top2 = np.argpartition(similarities, -2, axis=1)[:, -2:]
    top2_accuracy = float(np.any(top2 == true_ids[:, None], axis=1).mean())
    summary = {
        **metadata,
        "num_sources": len(sampled),
        "samples_per_genre": int(sampled.groupby("genre_id").size().iloc[0]),
        "nearest_prompt_accuracy": accuracy,
        "balanced_accuracy": float(np.mean(np.diag(matrix) / matrix.sum(axis=1))),
        "macro_f1": float(detail["macro avg"]["f1-score"]),
        "top2_accuracy": top2_accuracy,
        "mean_true_minus_best_other": float(predictions["true_minus_best_other"].mean()),
        "per_genre_nearest_prompt_accuracy": {
            label: float(matrix[index, index] / matrix[index].sum())
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
        "genre_vocabulary": labels,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized.to_numpy(), vmin=0.0, vmax=1.0, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Nearest CLaMP 2 genre prompt")
    axis.set_ylabel("XMIDI genre label")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = float(normalized.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Row-normalized fraction")
    figure.tight_layout()
    figure.savefig(report_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    rows = "\n".join(
        f"| {label} | {matrix[index].sum()} | {matrix[index, index] / matrix[index].sum():.3f} |"
        for index, label in enumerate(labels)
    )
    markdown = f"""# XMIDI genre labels vs. CLaMP 2 prompts

This diagnostic compares each original MIDI embedding with six text embeddings and uses
the nearest prompt as the CLaMP 2 genre prediction. It measures agreement with XMIDI's
existing labels; it does not treat either labeling source as unquestionable ground truth.

## Protocol

- Split: `{metadata['split']}`
- Sampling: {summary['samples_per_genre']} unique source songs per genre, {summary['num_sources']} total
- Prompt template: `{metadata['prompt_template']}`
- Random seed: {metadata['seed']}
- CLaMP 2 extraction: official released checkpoint, normalized embeddings, cosine similarity

## Agreement

- Top-1 nearest-prompt accuracy: **{accuracy:.3f}**
- Balanced accuracy: **{summary['balanced_accuracy']:.3f}**
- Macro F1: **{summary['macro_f1']:.3f}**
- Top-2 accuracy: **{top2_accuracy:.3f}**
- Mean true-prompt margin over best alternative: **{summary['mean_true_minus_best_other']:.4f}**

| XMIDI genre | Samples | Nearest-prompt accuracy |
|---|---:|---:|
{rows}

See `confusion_matrix.png`, `mean_prompt_similarity.csv`, and
`per_source_predictions.csv` for the full diagnostic.
"""
    (report_dir / "report.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.samples_per_genre <= 0 or args.num_gpus <= 0:
        raise ValueError("samples-per-genre and num-gpus must be positive")
    card = json.loads(args.dataset_card.read_text(encoding="utf-8"))
    labels = [str(label) for label in card["genre_vocabulary"]]
    if len(labels) != 6:
        raise ValueError(f"Expected six XMIDI genres, got {labels}")
    columns = [
        "sample_id",
        "source_midi_path",
        "split",
        "genre_label",
        "genre_id",
        "segment_index",
    ]
    frame = pd.read_parquet(args.manifest, columns=columns)
    frame = frame[frame["split"].eq(args.split)]
    sampled = balanced_source_sample(
        frame, samples_per_genre=args.samples_per_genre, seed=args.seed
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.artifact_dir / "sampled_sources.csv", index=False)
    selection_hash = hashlib.sha256(
        sampled[["sample_id", "genre_id"]].to_csv(index=False).encode()
    ).hexdigest()
    configuration = {
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "samples_per_genre": args.samples_per_genre,
        "seed": args.seed,
        "num_gpus": args.num_gpus,
        "prompt_template": args.prompt_template,
        "selection_hash": selection_hash,
    }
    config_path = args.artifact_dir / "diagnostic_config.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != configuration:
        raise ValueError(
            f"Refusing to mix a different diagnostic configuration in {args.artifact_dir}"
        )
    config_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")

    expected_by_rank: list[list[str]] = [[] for _ in range(args.num_gpus)]
    conversions: list[tuple[Path, Path]] = []
    key_to_rank: dict[str, int] = {}
    for index, row in sampled.iterrows():
        rank = index % args.num_gpus
        key = str(row["embedding_key"])
        key_to_rank[key] = rank
        expected_by_rank[rank].append(key)
        conversions.append(
            (
                Path(str(row["source_midi_path"])),
                args.artifact_dir / "inputs" / f"rank-{rank}" / f"{key}.mtf",
            )
        )
    for genre_id, label in enumerate(labels):
        key = f"prompt-{genre_id:02d}"
        expected_by_rank[0].append(key)
        prompt_path = args.artifact_dir / "inputs" / "rank-0" / f"{key}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(style_prompt(label, args.prompt_template), encoding="utf-8")
    for rank in range(args.num_gpus):
        (args.artifact_dir / "inputs" / f"rank-{rank}").mkdir(parents=True, exist_ok=True)

    # MIDI parsing and MTF serialization are Python-heavy; separate processes
    # avoid the GIL bottleneck seen with a thread pool on long XMIDI songs.
    with ProcessPoolExecutor(max_workers=args.conversion_workers) as executor:
        iterator = executor.map(_convert_one, conversions, chunksize=1)
        for _ in track(
            iterator,
            description="Convert sampled XMIDI to CLaMP 2 MTF",
            total=len(conversions),
            unit="midi",
        ):
            pass
    _run_extractors(
        repository=args.repository.resolve(),
        cache_dir=args.cache_dir.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
        expected_by_rank=expected_by_rank,
    )

    prompt_embeddings = np.stack(
        [
            _normalized_feature(args.artifact_dir / "embeddings" / "rank-0" / f"prompt-{i:02d}.npy")
            for i in range(len(labels))
        ]
    )
    music_embeddings = np.stack(
        [
            _normalized_feature(
                args.artifact_dir
                / "embeddings"
                / f"rank-{key_to_rank[str(key)]}"
                / f"{key}.npy"
            )
            for key in sampled["embedding_key"]
        ]
    )
    similarities = music_embeddings @ prompt_embeddings.T
    _write_reports(
        sampled=sampled,
        labels=labels,
        similarities=similarities,
        report_dir=args.report_dir.resolve(),
        metadata=configuration,
    )
    print(f"CLaMP 2 XMIDI genre diagnostic written to {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
