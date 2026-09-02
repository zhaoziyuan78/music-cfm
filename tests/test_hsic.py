import torch

from cfmusic.losses.hsic import normalized_hsic


def test_hsic_detects_mean_and_covariance_dependence() -> None:
    generator = torch.Generator().manual_seed(4)
    labels = torch.arange(4).repeat_interleave(32)
    independent = torch.randn(128, 8, generator=generator, requires_grad=True)
    dependent_mean = independent.detach() + labels[:, None] * 0.8
    scales = 1 + labels[:, None] * 0.4
    dependent_covariance = torch.randn(128, 8, generator=generator) * scales
    independent_score = normalized_hsic(independent, labels)
    assert normalized_hsic(dependent_mean, labels) > independent_score + 0.05
    assert normalized_hsic(dependent_covariance, labels) > independent_score
    independent_score.backward()
    assert independent.grad is not None and torch.isfinite(independent.grad).all()


def test_hsic_handles_unbalanced_and_tiny_batches() -> None:
    assert (
        normalized_hsic(torch.randn(1, 4, requires_grad=True), torch.zeros(1, dtype=torch.long))
        == 0
    )
    value = normalized_hsic(torch.randn(10, 4), torch.tensor([0] * 9 + [1]))
    assert torch.isfinite(value)
