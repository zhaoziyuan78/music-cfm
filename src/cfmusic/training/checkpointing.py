"""Atomic full-state checkpoints with RNG restoration."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from cfmusic.training.state import ExponentialMovingAverage, TrainState

ROLLING_CHECKPOINT_NAME = "last.pt"


def checkpoint_model_state(
    checkpoint: Mapping[str, object], *, weights: str
) -> Mapping[str, Tensor]:
    """Select raw or EMA model parameters from a training checkpoint."""

    if weights == "raw":
        state = checkpoint.get("model")
    elif weights == "ema":
        ema = checkpoint.get("ema_model")
        state = ema.get("shadow") if isinstance(ema, Mapping) else None
    else:
        raise ValueError(f"Unknown checkpoint weight variant: {weights!r}")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in state.items()
    ):
        raise TypeError(f"Checkpoint has no valid {weights} model state")
    return state


def resolve_resume_checkpoint(
    checkpoint_dir: Path,
    *,
    resume: bool,
    resume_from: Path | None = None,
    announce: bool = True,
    checkpoint_name: str = ROLLING_CHECKPOINT_NAME,
) -> Path | None:
    """Resolve an explicit checkpoint or the rolling checkpoint for ``resume=true``."""

    if resume_from is not None:
        path = resume_from.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        if announce:
            print(f"Resuming from explicit checkpoint: {path}")
        return path
    if not resume:
        return None
    path = checkpoint_dir / checkpoint_name
    if path.is_file():
        if announce:
            print(f"Resuming from rolling checkpoint: {path}")
        return path
    if announce:
        print(f"resume=true, but no rolling checkpoint exists at {path}; starting fresh")
    return None


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.GradScaler | None,
    state: TrainState,
    ema: ExponentialMovingAverage | None,
    config: Mapping[str, object],
    provenance: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "ema_model": ema.state_dict() if ema else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "gradient_scaler": scaler.state_dict() if scaler else None,
        "train_state": state.__dict__,
        "rng_states": {
            "torch": torch.get_rng_state(),
            # Save only this worker's active device.  A rank-zero rolling checkpoint
            # can then be resumed with either fewer or more visible GPUs.
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "config": dict(config),
        "provenance": dict(provenance),
    }
    if "condition_schema_version" in provenance:
        payload["condition_schema_version"] = provenance["condition_schema_version"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_rolling_checkpoint(
    checkpoint_dir: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.GradScaler | None,
    state: TrainState,
    ema: ExponentialMovingAverage | None,
    config: Mapping[str, object],
    provenance: Mapping[str, str],
) -> Path:
    """Atomically overwrite the sole intermediate checkpoint in a run directory."""

    path = checkpoint_dir / ROLLING_CHECKPOINT_NAME
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        state=state,
        ema=ema,
        config=config,
        provenance=provenance,
    )
    # Versions before rolling checkpoints wrote one full copy per interval.  Remove
    # those only after the new checkpoint is durable, so migration is crash-safe.
    for legacy in checkpoint_dir.glob("step-*.pt"):
        legacy.unlink(missing_ok=True)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.GradScaler | None = None,
    ema: ExponentialMovingAverage | None = None,
) -> TrainState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload["gradient_scaler"] is not None:
        scaler.load_state_dict(payload["gradient_scaler"])
    if ema is not None and payload["ema_model"] is not None:
        ema.load_state_dict(payload["ema_model"])
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        ema.to(model_device)
    rng = payload["rng_states"]
    torch.set_rng_state(rng["torch"])
    cuda_rng = rng.get("cuda")
    if torch.cuda.is_available() and cuda_rng is not None:
        if isinstance(cuda_rng, list):
            # Backward compatibility with checkpoints that stored every visible GPU.
            device_index = torch.cuda.current_device()
            cuda_rng = cuda_rng[min(device_index, len(cuda_rng) - 1)] if cuda_rng else None
        if isinstance(cuda_rng, torch.Tensor):
            torch.cuda.set_rng_state(cuda_rng, device=torch.cuda.current_device())
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    saved_state = dict(payload["train_state"])
    # Checkpoints produced before mid-epoch cursors were introduced remain valid.
    saved_state.setdefault("batch_in_epoch", 0)
    saved_state.setdefault("world_size", 1)
    return TrainState(**saved_state)
