from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cfmusic.commands.clamp2_batched_worker import _segments
from cfmusic.commands.diagnose_clamp2_centroids import (
    genre_centroids,
    nearest_centroid_predictions,
)
from cfmusic.commands.diagnose_clamp2_classifier import (
    ClampGenreClassifier,
    _candidate_weights,
)
from cfmusic.commands.diagnose_clamp2_genre import balanced_source_sample
from cfmusic.commands.relabel_xmidi_clamp2 import nearest_prompt_labels


def _manifest(songs_per_genre: int = 5) -> pd.DataFrame:
    rows = []
    for genre_id in range(6):
        for song in range(songs_per_genre):
            for segment in range(2):
                rows.append(
                    {
                        "sample_id": f"g{genre_id}-s{song}",
                        "source_midi_path": f"/data/g{genre_id}-s{song}.mid",
                        "split": "test",
                        "genre_label": f"genre-{genre_id}",
                        "genre_id": genre_id,
                        "segment_index": segment,
                    }
                )
    return pd.DataFrame(rows)


def test_balanced_source_sample_uses_unique_songs_and_is_deterministic() -> None:
    frame = _manifest()
    first = balanced_source_sample(frame, samples_per_genre=3, seed=7)
    second = balanced_source_sample(frame, samples_per_genre=3, seed=7)

    assert first.equals(second)
    assert len(first) == 18
    assert first["sample_id"].is_unique
    assert first.groupby("genre_id").size().eq(3).all()
    assert first["embedding_key"].is_unique


def test_balanced_source_sample_rejects_undersized_genre() -> None:
    with pytest.raises(ValueError, match="cannot sample"):
        balanced_source_sample(_manifest(2), samples_per_genre=3, seed=7)


def test_clamp2_chunks_match_official_overlapping_final_chunk() -> None:
    patches = torch.arange(700 * 4).reshape(700, 4)

    chunks, weights = _segments(patches, patch_length=512)

    assert [len(chunk) for chunk in chunks] == [512, 512]
    assert torch.equal(chunks[0], patches[:512])
    assert torch.equal(chunks[1], patches[-512:])
    assert weights == [512, 188]


def test_nearest_centroid_leave_one_out() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]], dtype=np.float32
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    truth = np.asarray([0, 0, 1, 1])
    centroids, sums, counts = genre_centroids(embeddings, truth, num_genres=2)

    predicted, similarities = nearest_centroid_predictions(
        embeddings,
        centroids,
        labels=truth,
        class_sums=sums,
        class_counts=counts,
    )

    assert predicted.tolist() == truth.tolist()
    assert similarities.shape == (4, 2)


def test_clamp_genre_classifier_shapes_linear_and_mlp() -> None:
    features = torch.randn(7, 12)

    linear = ClampGenreClassifier(12, 6, hidden_dims=(), dropout=0.2)
    mlp = ClampGenreClassifier(12, 6, hidden_dims=(8, 4), dropout=0.2)

    assert linear(features).shape == (7, 6)
    assert mlp(features).shape == (7, 6)


def test_classifier_class_weights_are_normalized_and_balance_counts() -> None:
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2])

    unweighted = _candidate_weights(labels, num_classes=3, power=0.0)
    balanced = _candidate_weights(labels, num_classes=3, power=1.0)

    assert torch.allclose(unweighted, torch.ones(3))
    assert balanced.mean().item() == pytest.approx(1.0)
    assert balanced[2] > balanced[1] > balanced[0]


def test_nearest_prompt_relabel_normalizes_embeddings_and_returns_margin() -> None:
    music = np.asarray([[2.0, 0.0], [0.2, 0.8]], dtype=np.float32)
    prompts = np.asarray([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)

    predicted, similarities, margins = nearest_prompt_labels(music, prompts)

    assert predicted.tolist() == [0, 1]
    assert similarities.shape == (2, 2)
    assert np.all(margins > 0)
