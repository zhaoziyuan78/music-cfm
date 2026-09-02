import torch

from cfmusic.models.orthogonal_split import OrthogonalLatentSplit


def test_split_recomposes_exactly() -> None:
    split = OrthogonalLatentSplit(16, 0.25)
    latent = torch.randn(4, 3, 16)
    reconstructed = split.merge(*split.split(latent))
    assert (reconstructed - latent).abs().max() < 1e-5
    assert split.orthogonality_error() < 1e-6
