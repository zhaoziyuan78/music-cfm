from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from cfmusic.latent.compatibility import expected_transport_input_dim

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize(
    "experiment",
    [
        "e10_ddim_vanilla",
        "e11_ddim_fpi",
        "e12_ddim_exoreg",
        "e20_cfm_base",
        "e21_cfm_hsic",
        "e22_cfm_exoreg",
        "e23_otcfm_exoreg",
        "e30_xmidi_factorial",
        "e40_split_cfm",
        "e50_independent_flows",
        "e51_shuffled_labels",
    ],
)
def test_xmidi_transports_match_the_current_codec(experiment: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        config = compose(config_name="config", overrides=[f"experiment={experiment}"])

    assert str(config.data.name) == "xmidi"
    assert int(config.codec.latent_tokens) == 64
    assert int(config.codec.latent_dim) == 512
    assert expected_transport_input_dim(config.transport) == 512


def test_groove_transport_retains_its_32_x_256_codec_shape() -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        config = compose(config_name="config", overrides=["experiment=e33_groove_cfm"])

    assert str(config.data.name) == "groove"
    assert int(config.codec.latent_tokens) == 32
    assert int(config.codec.latent_dim) == 256
    assert expected_transport_input_dim(config.transport) == 256


@pytest.mark.parametrize(
    "experiment", ["e31_emopia_cfm", "e32_vgmidi_cfm", "e60_cross_domain_4q"]
)
def test_other_pitched_transports_also_use_the_512_dimensional_codec(
    experiment: str,
) -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        config = compose(config_name="config", overrides=[f"experiment={experiment}"])

    assert int(config.codec.latent_dim) == 512
    assert expected_transport_input_dim(config.transport) == 512
