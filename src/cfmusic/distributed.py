"""Small native-PyTorch distributed helpers."""

from __future__ import annotations

import os
import random
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    """Initialize torchrun process groups when WORLD_SIZE is larger than one."""
    # Surface a failed worker promptly instead of leaving peers blocked in an
    # NCCL collective until the long process-group timeout expires.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        if device.type == "cuda":
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            dist.init_process_group(backend="gloo")
    return DistributedContext(rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def distributed_model(
    model: nn.Module,
    context: DistributedContext,
    *,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap a module in native DDP while preserving the original module for checkpoints."""

    if context.world_size == 1:
        return model
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    output_device = context.local_rank if context.device.type == "cuda" else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        # Projector/schedule/position buffers are immutable. Re-broadcasting them on
        # every forward is pure communication overhead, especially in Stage 2.
        broadcast_buffers=False,
    )


def maybe_no_sync(model: nn.Module, *, synchronize: bool) -> AbstractContextManager[None]:
    """Skip DDP gradient synchronization on non-final accumulation micro-batches."""

    if isinstance(model, DistributedDataParallel) and not synchronize:
        return model.no_sync()
    return nullcontext()


def distributed_barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        device_ids = [context.local_rank] if context.device.type == "cuda" else None
        dist.barrier(device_ids=device_ids)


def distributed_max_int(value: int, context: DistributedContext) -> int:
    """Return a rank-wide maximum, or the local value outside distributed jobs."""

    if context.world_size == 1:
        return value
    maximum = torch.tensor(value, dtype=torch.int64, device=context.device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return int(maximum.item())


def differentiable_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather an equal-sized tensor while retaining cross-rank autograd."""

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    return torch.cat(tuple(dist_nn.all_gather(tensor)), dim=0)


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a non-differentiable tensor such as integer class labels."""

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def decorrelate_worker_rng(context: DistributedContext) -> None:
    """Keep rank zero's resumed RNG exact while giving other workers distinct streams."""

    if context.rank == 0:
        return
    seed = (torch.initial_seed() + 1_000_003 * context.rank) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if context.device.type == "cuda":
        torch.cuda.manual_seed(seed)


def set_data_epoch(batches: object, epoch: int) -> None:
    """Forward an epoch cursor to either a sampler or a batch sampler."""

    for attribute in ("sampler", "batch_sampler"):
        sampler = getattr(batches, attribute, None)
        setter = getattr(sampler, "set_epoch", None)
        if callable(setter):
            setter(epoch)
