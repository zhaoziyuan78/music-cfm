import pytest
import torch

from cfmusic.conditioning.schema import (
    CONDITION_SCHEMA_VERSION,
    build_condition_batch,
    condition_schema_provenance,
    validate_condition_checkpoint,
)
from cfmusic.training.transport_trainer import conditions_from_batch, contrasting_conditions


def _metadata() -> dict[str, torch.Tensor]:
    return {
        "dataset_id": torch.tensor([0, 0]),
        "style_id": torch.tensor([1, 2]),
        "genre_id": torch.tensor([1, 2]),
        "emotion_id": torch.tensor([3, 4]),
    }


def test_train_and_generation_use_identical_factual_condition() -> None:
    factual = conditions_from_batch(_metadata(), torch.device("cpu"), task="genre")
    generation = build_condition_batch(_metadata(), torch.device("cpu"), task="genre")

    for field in ("dataset_id", "task_id", "style_id"):
        assert torch.equal(getattr(factual, field), getattr(generation, field))
    assert factual.genre_id is None and factual.emotion_id is None


def test_generic_style_task_uses_only_style_slot() -> None:
    condition = build_condition_batch(_metadata(), torch.device("cpu"), task="style")

    torch.testing.assert_close(condition.style_id, torch.tensor([1, 2]))
    assert condition.genre_id is None and condition.emotion_id is None


def test_nonfactorial_emotion_uses_only_style_slot() -> None:
    condition = build_condition_batch(_metadata(), torch.device("cpu"), task="emotion")

    torch.testing.assert_close(condition.style_id, torch.tensor([3, 4]))
    assert condition.genre_id is None and condition.emotion_id is None
    assert condition.task_id.unique().item() == 1


@pytest.mark.parametrize("axis", ["genre", "emotion"])
def test_factorial_wrong_condition_changes_one_axis(axis: str) -> None:
    factual = build_condition_batch(
        _metadata(), torch.device("cpu"), task="factorial", factorial=True
    )
    wrong = contrasting_conditions(
        factual,
        {"genre_id": [0, 1, 2], "emotion_id": [0, 1, 2, 3, 4]},
        factorial=True,
        active_axis=axis,
    )

    assert wrong.style_id.equal(factual.style_id)
    changed_genre = not wrong.genre_id.equal(factual.genre_id)  # type: ignore[union-attr]
    changed_emotion = not wrong.emotion_id.equal(factual.emotion_id)  # type: ignore[union-attr]
    assert (changed_genre, changed_emotion) == (axis == "genre", axis == "emotion")


@pytest.mark.parametrize("axis", ["genre", "emotion"])
def test_factorial_generation_intervention_changes_one_axis(axis: str) -> None:
    metadata = _metadata()
    factual = build_condition_batch(metadata, torch.device("cpu"), task="factorial", factorial=True)
    target = build_condition_batch(
        metadata,
        torch.device("cpu"),
        task="factorial",
        factorial=True,
        genre_id=torch.tensor([0, 0]) if axis == "genre" else None,
        emotion_id=torch.tensor([0, 0]) if axis == "emotion" else None,
    )

    assert target.style_id.equal(factual.style_id)
    assert target.genre_id is not None and factual.genre_id is not None
    assert target.emotion_id is not None and factual.emotion_id is not None
    assert target.genre_id.equal(factual.genre_id) is (axis != "genre")
    assert target.emotion_id.equal(factual.emotion_id) is (axis != "emotion")


def test_old_condition_checkpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="old or unknown condition schema"):
        validate_condition_checkpoint({"provenance": {}}, task="genre", factorial=False)

    checkpoint = {
        "condition_schema_version": CONDITION_SCHEMA_VERSION,
        "provenance": condition_schema_provenance(task="genre", factorial=False),
    }
    validate_condition_checkpoint(checkpoint, task="genre", factorial=False)
