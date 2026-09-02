from pathlib import Path

import torch

from cfmusic.training.checkpointing import (
    checkpoint_model_state,
    load_checkpoint,
    resolve_resume_checkpoint,
    save_rolling_checkpoint,
)
from cfmusic.training.state import ExponentialMovingAverage, TrainState


def _training_objects() -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler


def test_rolling_checkpoint_overwrites_and_removes_legacy_steps(tmp_path: Path) -> None:
    model, optimizer, scheduler = _training_objects()
    legacy = tmp_path / "step-00000001.pt"
    torch.save({"legacy": True}, legacy)
    state = TrainState(global_step=7, epoch=2, batch_in_epoch=11, world_size=4)

    path = save_rolling_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        state=state,
        ema=None,
        config={"name": "test"},
        provenance={"source": "unit"},
    )

    assert path == tmp_path / "last.pt"
    assert path.is_file()
    assert not legacy.exists()
    assert list(tmp_path.glob("*.pt")) == [path]

    restored_model, restored_optimizer, restored_scheduler = _training_objects()
    restored = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    assert restored == state
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_resume_true_discovers_last_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"
    checkpoint.touch()

    assert resolve_resume_checkpoint(tmp_path, resume=True, announce=False) == checkpoint
    assert resolve_resume_checkpoint(tmp_path, resume=False, announce=False) is None


def test_checkpoint_model_state_selects_raw_or_ema_weights() -> None:
    model = torch.nn.Linear(3, 2)
    ema = ExponentialMovingAverage(model)
    checkpoint = {"model": model.state_dict(), "ema_model": ema.state_dict()}

    raw = checkpoint_model_state(checkpoint, weights="raw")
    averaged = checkpoint_model_state(checkpoint, weights="ema")

    assert raw.keys() == averaged.keys()
    assert all(torch.equal(raw[name], averaged[name]) for name in raw)
