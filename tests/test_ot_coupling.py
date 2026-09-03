import torch

from cfmusic.transport.ot_coupling import couple_noise_to_data


def test_sinkhorn_rounding_keeps_a_one_to_one_noise_permutation() -> None:
    noise = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    latent = torch.randn_like(noise)
    styles = torch.tensor([0, 0, 0, 1, 1, 1])

    coupled = couple_noise_to_data(
        noise, latent, styles, solver="sinkhorn", cost_projection_dim=4
    ).noise

    for style in torch.unique(styles):
        selected = styles == style
        original_rows = {tuple(row.tolist()) for row in noise[selected].flatten(1)}
        coupled_rows = {tuple(row.tolist()) for row in coupled[selected].flatten(1)}
        assert coupled_rows == original_rows
