import torch

from cfmusic.solvers.ddim import ddim_timesteps, deterministic_ddim_update
from cfmusic.solvers.schedules import cosine_alpha_cumprod


def test_schedule_and_timestep_indexing() -> None:
    alpha = cosine_alpha_cumprod(100)
    assert torch.all(alpha[1:] <= alpha[:-1])
    descending = ddim_timesteps(100, 10, descending=True)
    assert descending[0] == 99 and descending[-1] == 0
    assert torch.all(descending[1:] < descending[:-1])


def test_ddim_update_is_deterministic() -> None:
    state, epsilon = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    first = deterministic_ddim_update(state, epsilon, 0.4, 0.6)
    second = deterministic_ddim_update(state, epsilon, 0.4, 0.6)
    assert torch.equal(first, second)
