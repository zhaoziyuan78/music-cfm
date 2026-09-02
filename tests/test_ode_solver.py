import math

import torch

from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.solvers.ode import FixedGridODESolver


def condition(batch: int) -> ConditionBatch:
    zeros = torch.zeros(batch, dtype=torch.long)
    return ConditionBatch(zeros, zeros, zeros)


def test_linear_ode_forward_backward_nfe_and_gradient() -> None:
    initial = torch.tensor([[[1.0]]], requires_grad=True)

    def function(state: torch.Tensor, time: torch.Tensor, _: ConditionBatch) -> torch.Tensor:
        return 0.5 * state + 0.25

    heun = FixedGridODESolver("heun")
    result = heun.integrate(
        function, initial, condition(1), t_start=0, t_end=1, num_steps=64, track_grad=True
    )
    expected = (1 + 0.5) * math.exp(0.5) - 0.5
    assert torch.allclose(result.state, torch.tensor([[[expected]]]), atol=2e-4)
    assert result.nfe == 128
    backward = heun.integrate(
        function, result.state, condition(1), t_start=1, t_end=0, num_steps=64, track_grad=True
    )
    assert torch.allclose(backward.state, initial, atol=3e-4)
    backward.state.sum().backward()
    assert initial.grad is not None and torch.isfinite(initial.grad).all()


def test_heun_converges_faster_than_euler() -> None:
    initial = torch.ones(1, 1, 1)
    expected = torch.full_like(initial, math.e)

    def function(state: torch.Tensor, time: torch.Tensor, _: ConditionBatch) -> torch.Tensor:
        return state

    euler = (
        FixedGridODESolver("euler")
        .integrate(
            function, initial, condition(1), t_start=0, t_end=1, num_steps=8, track_grad=False
        )
        .state
    )
    heun = (
        FixedGridODESolver("heun")
        .integrate(
            function, initial, condition(1), t_start=0, t_end=1, num_steps=8, track_grad=False
        )
        .state
    )
    assert (heun - expected).abs().item() < (euler - expected).abs().item()
