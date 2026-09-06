"""Relabel every XMIDI song with its nearest CLaMP 2 genre prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cfmusic.progress import progress_bar

LABEL_SOURCE = "clamp2_nearest_genre_prompt"
OVERLAY_SCHEMA_VERSION = "cfmusic.latent-label-overlay.v1"


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
    centroid_artifacts = artifacts_root / "diagnostics" / "clamp2_xmidi_genre_centroids"
    prompt_artifacts = artifacts_root / "diagnostics" / "clamp2_xmidi_genre"
    classifier_artifacts = artifacts_root / "diagnostics" / "clamp2_xmidi_genre_classifier"
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
        "--source-index",
        type=Path,
        default=centroid_artifacts / "xmidi_unique_sources.parquet",
    )
    parser.add_argument(
        "--packed-embeddings",
        type=Path,
        default=classifier_artifacts / "xmidi_clamp2_embeddings.npy",
    )
    parser.add_argument(
        "--prompt-artifact-dir",
        type=Path,
        default=prompt_artifacts,
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
        "--relabeled-manifest",
        type=Path,
        default=data_root / "processed" / "xmidi" / "manifest_clamp2_prompt.parquet",
    )
    parser.add_argument(
        "--relabeled-dataset-card",
        type=Path,
        default=data_root / "processed" / "xmidi" / "dataset_card_clamp2_prompt.json",
    )
    parser.add_argument(
        "--relabeled-latent-index",
        type=Path,
        default=data_root / "latents" / "xmidi" / "index_clamp2_prompt.parquet",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=artifacts_root / "diagnostics" / "clamp2_xmidi_prompt_relabel",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=project_root / "reports" / "diagnostics" / "clamp2_xmidi_prompt_relabel",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("CLaMP 2 embeddings must be a rank-2 matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norms <= 0):
        raise ValueError("CLaMP 2 embeddings must be a finite rank-2 matrix")
    return array / norms


def nearest_prompt_labels(
    music_embeddings: np.ndarray, prompt_embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest prompt IDs, all similarities, and top-1 margins."""

    music = _normalize_rows(music_embeddings)
    prompts = _normalize_rows(prompt_embeddings)
    similarities = music @ prompts.T
    predicted = similarities.argmax(axis=1).astype(np.int64)
    ordered = np.sort(similarities, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    return predicted, similarities, margins


def _assignment_hash(mapping: pd.DataFrame, prompt_hashes: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(prompt_hashes, separators=(",", ":")).encode())
    for sample_id, genre_id in mapping[["sample_id", "clamp2_genre_id"]].itertuples(
        index=False, name=None
    ):
        digest.update(str(sample_id).encode())
        digest.update(b"\0")
        digest.update(int(genre_id).to_bytes(2, byteorder="little", signed=False))
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing relabel output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing relabel output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _apply_manifest_labels(
    manifest: pd.DataFrame, mapping: pd.DataFrame, labels: list[str]
) -> pd.DataFrame:
    lookup = mapping.set_index("sample_id")["clamp2_genre_id"]
    new_ids = manifest["sample_id"].map(lookup)
    if new_ids.isna().any():
        missing = manifest.loc[new_ids.isna(), "sample_id"].astype(str).unique()[:10]
        raise ValueError(f"CLaMP 2 labels are missing for XMIDI samples: {missing.tolist()}")
    output = manifest.copy()
    for source, destination in (
        ("style_id", "original_style_id"),
        ("style_label", "original_style_label"),
        ("genre_id", "original_genre_id"),
        ("genre_label", "original_genre_label"),
    ):
        if destination in output:
            raise ValueError(f"Input manifest is already relabeled: {destination}")
        output[destination] = output[source]
    ids = new_ids.astype(np.int64)
    names = ids.map(dict(enumerate(labels)))
    output["style_id"] = ids
    output["genre_id"] = ids
    output["style_label"] = names
    output["genre_label"] = names
    output["label_source"] = LABEL_SOURCE
    return output


def _apply_latent_labels(index: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    lookup = mapping.set_index("sample_id")["clamp2_genre_id"]
    new_ids = index["sample_id"].map(lookup)
    if new_ids.isna().any():
        missing = index.loc[new_ids.isna(), "sample_id"].astype(str).unique()[:10]
        raise ValueError(f"CLaMP 2 labels are missing for latent samples: {missing.tolist()}")
    output = index.copy()
    if "original_style_id" in output or "original_genre_id" in output:
        raise ValueError("Input latent index is already relabeled")
    output["original_style_id"] = output["style_id"]
    output["original_genre_id"] = output["genre_id"]
    output["style_id"] = new_ids.astype(np.int64)
    output["genre_id"] = new_ids.astype(np.int64)
    output["label_source"] = LABEL_SOURCE
    return output


def _verify_complete_outputs(
    *,
    required_paths: list[Path],
    sidecar_path: Path,
    assignment_hash: str,
) -> bool:
    present = [path.exists() for path in required_paths]
    if not any(present):
        return False
    if not all(present):
        missing = [str(path) for path, exists in zip(required_paths, present, strict=True) if not exists]
        raise FileExistsError(
            "Refusing to mix partial XMIDI relabel outputs; missing: " + ", ".join(missing)
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("label_assignment_hash") != assignment_hash:
        raise ValueError("Existing XMIDI label overlay was built from different assignments")
    return True


def main() -> None:
    args = _parse_args()
    progress = progress_bar(
        description="Relabel full XMIDI with CLaMP 2 prompts", total=6, unit="phase"
    )
    progress.set_postfix_str("load source embeddings")
    card = json.loads(args.dataset_card.read_text(encoding="utf-8"))
    labels = [str(value) for value in card["genre_vocabulary"]]
    if len(labels) != 6 or labels != [str(value) for value in card["style_vocabulary"]]:
        raise ValueError("XMIDI must expose the same six style and genre labels")
    sources = pd.read_parquet(args.source_index).sort_values(
        "embedding_key", ignore_index=True
    )
    if sources["sample_id"].duplicated().any() or len(sources) != 107_975:
        raise ValueError("Expected 107,975 unique XMIDI source songs")
    music_embeddings = np.load(args.packed_embeddings, mmap_mode="r")
    if music_embeddings.shape != (len(sources), 768):
        raise ValueError(
            f"Packed CLaMP 2 matrix {music_embeddings.shape} does not match source index"
        )
    progress.update(1)
    progress.set_postfix_str("compare genre prompts")
    prompt_paths = [
        args.prompt_artifact_dir / "embeddings" / "rank-0" / f"prompt-{index:02d}.npy"
        for index in range(len(labels))
    ]
    prompt_embeddings = np.stack(
        [np.asarray(np.load(path), dtype=np.float32).reshape(-1) for path in prompt_paths]
    )
    predicted, similarities, margins = nearest_prompt_labels(
        music_embeddings, prompt_embeddings
    )
    mapping = sources[
        ["embedding_key", "sample_id", "source_midi_path", "split", "genre_id", "genre_label"]
    ].copy()
    mapping = mapping.rename(
        columns={"genre_id": "original_genre_id", "genre_label": "original_genre_label"}
    )
    mapping["clamp2_genre_id"] = predicted
    mapping["clamp2_genre_label"] = [labels[index] for index in predicted]
    mapping["nearest_prompt_margin"] = margins
    mapping["label_changed"] = mapping["original_genre_id"].to_numpy() != predicted
    for index, label in enumerate(labels):
        mapping[f"prompt_similarity_{label}"] = similarities[:, index]
    prompt_hashes = [_sha256(path) for path in prompt_paths]
    assignment_hash = _assignment_hash(mapping, prompt_hashes)
    progress.update(1)
    sidecar_path = args.relabeled_latent_index.with_suffix(".metadata.json")
    required_paths = [
        args.relabeled_manifest,
        args.relabeled_dataset_card,
        args.relabeled_latent_index,
        sidecar_path,
        args.artifact_dir / "source_label_mapping.parquet",
        args.artifact_dir / "relabel_config.json",
        args.report_dir / "summary.json",
        args.report_dir / "report.md",
    ]
    if _verify_complete_outputs(
        required_paths=required_paths,
        sidecar_path=sidecar_path,
        assignment_hash=assignment_hash,
    ):
        progress.update(4)
        progress.close()
        print(f"Reusing complete XMIDI CLaMP 2 relabel outputs: {args.relabeled_latent_index}")
        return

    progress.set_postfix_str("load manifest and latent index")
    manifest = pd.read_parquet(args.manifest)
    base_latent_index = pd.read_parquet(args.base_latent_index)
    progress.update(1)
    progress.set_postfix_str("apply source labels to segments")
    relabeled_manifest = _apply_manifest_labels(manifest, mapping, labels)
    relabeled_latent_index = _apply_latent_labels(base_latent_index, mapping)
    cache_metadata = json.loads(args.base_cache_metadata.read_text(encoding="utf-8"))
    prompt_config = json.loads(
        (args.prompt_artifact_dir / "diagnostic_config.json").read_text(encoding="utf-8")
    )
    overlay_metadata = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "label_source": LABEL_SOURCE,
        "label_assignment_hash": assignment_hash,
        "base_dataset_manifest_hash": cache_metadata["dataset_manifest_hash"],
        "rows": len(relabeled_latent_index),
        "unique_samples": int(relabeled_latent_index["sample_id"].nunique()),
        "prompt_template": str(prompt_config["prompt_template"]),
        "genre_vocabulary": labels,
    }
    relabeled_card = dict(card)
    relabeled_card.update(
        {
            "label_source": LABEL_SOURCE,
            "label_assignment_hash": assignment_hash,
            "base_manifest": str(args.manifest.resolve()),
            "base_manifest_sha256": _sha256(args.manifest),
            "prompt_template": str(prompt_config["prompt_template"]),
            "unique_source_label_counts": {
                labels[index]: int((predicted == index).sum()) for index in range(len(labels))
            },
        }
    )
    configuration = {
        "label_source": LABEL_SOURCE,
        "label_assignment_hash": assignment_hash,
        "source_index": str(args.source_index.resolve()),
        "source_index_sha256": _sha256(args.source_index),
        "packed_embeddings": str(args.packed_embeddings.resolve()),
        "prompt_embedding_sha256": dict(zip(labels, prompt_hashes, strict=True)),
        "prompt_template": str(prompt_config["prompt_template"]),
        "base_manifest": str(args.manifest.resolve()),
        "base_latent_index": str(args.base_latent_index.resolve()),
        "relabeled_manifest": str(args.relabeled_manifest.resolve()),
        "relabeled_latent_index": str(args.relabeled_latent_index.resolve()),
    }
    progress.update(1)
    progress.set_postfix_str("write relabeled indexes")

    _atomic_parquet(mapping, args.artifact_dir / "source_label_mapping.parquet")
    _atomic_json(configuration, args.artifact_dir / "relabel_config.json")
    _atomic_parquet(relabeled_manifest, args.relabeled_manifest)
    _atomic_json(relabeled_card, args.relabeled_dataset_card)
    _atomic_parquet(relabeled_latent_index, args.relabeled_latent_index)
    _atomic_json(overlay_metadata, sidecar_path)
    progress.update(1)
    progress.set_postfix_str("write audit report")

    source_counts = {
        label: int((predicted == index).sum()) for index, label in enumerate(labels)
    }
    latent_counts_frame = (
        relabeled_latent_index.groupby(["split", "style_id"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(len(labels)), fill_value=0)
    )
    latent_counts = {
        str(split): {
            labels[index]: int(row[index]) for index in range(len(labels))
        }
        for split, row in latent_counts_frame.iterrows()
    }
    old_to_new = pd.crosstab(
        mapping["original_genre_label"],
        mapping["clamp2_genre_label"],
    ).reindex(index=labels, columns=labels, fill_value=0)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    old_to_new.to_csv(args.report_dir / "original_to_clamp2_prompt_confusion.csv")
    mapping.groupby(["split", "clamp2_genre_label"], observed=True).size().unstack(
        fill_value=0
    ).reindex(columns=labels, fill_value=0).to_csv(
        args.report_dir / "source_label_counts_by_split.csv"
    )
    summary = {
        **configuration,
        "num_unique_sources": len(mapping),
        "num_manifest_segments": len(relabeled_manifest),
        "num_latent_segments": len(relabeled_latent_index),
        "changed_sources": int(mapping["label_changed"].sum()),
        "changed_source_rate": float(mapping["label_changed"].mean()),
        "old_new_agreement": float((mapping["original_genre_id"] == predicted).mean()),
        "mean_nearest_prompt_margin": float(margins.mean()),
        "median_nearest_prompt_margin": float(np.median(margins)),
        "source_label_counts": source_counts,
        "latent_segment_label_counts": latent_counts,
        "original_to_new_source_counts": old_to_new.to_dict(orient="index"),
    }
    _atomic_json(summary, args.report_dir / "summary.json")
    count_rows = "\n".join(
        f"| {label} | {source_counts[label]:,} | "
        f"{source_counts[label] / len(mapping):.2%} |"
        for label in labels
    )
    report = f"""# XMIDI labels reassigned by CLaMP 2 genre prompts

Every one of the {len(mapping):,} unique XMIDI songs is assigned the genre whose normalized
CLaMP 2 text embedding has the highest cosine similarity to its normalized CLaMP 2 MIDI
embedding. The exact prompt template is `{prompt_config['prompt_template']}`.

| New genre | Unique songs | Fraction |
|---|---:|---:|
{count_rows}

- Original/new agreement: **{summary['old_new_agreement']:.4f}**
- Relabeled songs: **{summary['changed_sources']:,} ({summary['changed_source_rate']:.2%})**
- Mean nearest-vs-second-nearest prompt margin: **{summary['mean_nearest_prompt_margin']:.4f}**

The original label columns are retained as `original_*`. VAE weights and latent tensors are
unchanged. CFM uses `{args.relabeled_latent_index}` as a lightweight label overlay; its assignment
hash enters transport checkpoint provenance so checkpoints trained with old labels cannot resume.
"""
    report_path = args.report_dir / "report.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite relabel report: {report_path}")
    report_path.write_text(report, encoding="utf-8")
    progress.update(1)
    progress.close()
    print(f"Relabeled manifest: {args.relabeled_manifest.resolve()}")
    print(f"Relabeled latent index: {args.relabeled_latent_index.resolve()}")
    print(f"Relabel report: {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
