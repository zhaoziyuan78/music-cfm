import torch

from cfmusic.solvers.ddim_fpi import fixed_point_inversion_step


def test_fixed_point_residual_decreases() -> None:
    lower = torch.randn(2, 3, 4)
    initial = lower * 0.8
    result = fixed_point_inversion_step(
        lower,
        initial,
        lambda state: 0.1 * state,
        alpha_lower=0.9,
        alpha_upper=0.7,
        iterations=8,
        stop_on_convergence=False,
    )
    assert result.state.shape == lower.shape
    assert result.nfe == 8
    assert result.residuals[-1] < result.residuals[0]
