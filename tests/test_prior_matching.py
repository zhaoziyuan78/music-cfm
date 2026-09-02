import torch

from cfmusic.losses.prior_matching import classwise_prior_matching


def test_prior_matching_is_differentiable() -> None:
    samples = torch.randn(64, 8, requires_grad=True)
    labels = torch.arange(4).repeat_interleave(16)
    loss = classwise_prior_matching(samples, labels)
    loss.backward()
    assert loss >= 0 and samples.grad is not None and torch.isfinite(samples.grad).all()
