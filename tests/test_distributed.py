import torch

from cfmusic.distributed import (
    DistributedContext,
    distributed_max_int,
    distributed_model,
    set_data_epoch,
)


class EpochAwareSampler:
    def __init__(self) -> None:
        self.epoch = -1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class LoaderLike:
    def __init__(self) -> None:
        self.sampler = EpochAwareSampler()
        self.batch_sampler = object()


def test_single_process_distributed_model_is_identity() -> None:
    model = torch.nn.Linear(2, 2)
    context = DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))

    assert distributed_model(model, context) is model
    assert distributed_max_int(37, context) == 37
    assert context.is_main


def test_set_data_epoch_updates_loader_sampler() -> None:
    loader = LoaderLike()
    set_data_epoch(loader, 7)
    assert loader.sampler.epoch == 7
