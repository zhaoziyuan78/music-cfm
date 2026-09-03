import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cfmusic.distributed import all_gather_tensor, differentiable_all_gather
from cfmusic.losses.hsic import normalized_hsic
from cfmusic.losses.mmd import cross_class_mmd
from cfmusic.losses.sliced_wasserstein import sliced_wasserstein_standard_normal


def _exogeneity_loss(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (
        normalized_hsic(features, labels)
        + cross_class_mmd(features, labels)
        + sliced_wasserstein_standard_normal(features, num_projections=3, seed=17)
    )


def _worker(rank: int, world_size: int, rendezvous: str, output: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        parameter = torch.tensor(1.5, requires_grad=True)
        base = torch.arange(rank * 6 + 1, rank * 6 + 7, dtype=torch.float32).reshape(2, 3)
        labels = torch.full((2,), rank, dtype=torch.long)
        global_features = differentiable_all_gather(parameter * base)
        global_labels = all_gather_tensor(labels)
        loss = _exogeneity_loss(global_features, global_labels)
        loss.backward()
        assert parameter.grad is not None
        # This is the parameter-gradient reduction that DDP performs after the
        # differentiable gather routes each rank's global-loss contribution.
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)
        torch.save(
            {"loss": loss.detach(), "gradient": parameter.grad.detach()},
            Path(output) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_distributed_exogeneity_matches_single_full_batch(tmp_path: Path) -> None:
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except OSError as error:
        pytest.skip(f"local process-group sockets are unavailable: {error}")
    rendezvous = tmp_path / "rendezvous"
    mp.spawn(_worker, args=(2, str(rendezvous), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]

    full = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    full_labels = torch.tensor([0, 0, 1, 1])
    parameter = torch.tensor(1.5, requires_grad=True)
    expected_loss = _exogeneity_loss(parameter * full, full_labels)
    expected_loss.backward()
    for result in results:
        torch.testing.assert_close(result["loss"], expected_loss.detach())
        torch.testing.assert_close(result["gradient"], parameter.grad)
