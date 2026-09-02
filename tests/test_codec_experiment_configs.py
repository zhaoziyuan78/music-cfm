from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize(
    ("experiment", "dataset", "max_sequence_length"),
    [
        ("e00_xmidi_codec", "xmidi", 2560),
        ("e01_emopia_codec", "emopia", 2560),
        ("e02_vgmidi_codec", "vgmidi", 2560),
        ("e03_groove_codec", "groove", 512),
    ],
)
def test_codec_experiments_are_dataset_isolated(
    experiment: str, dataset: str, max_sequence_length: int
) -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        config = compose(config_name="config", overrides=[f"experiment={experiment}"])

    assert str(config.experiment.name) == experiment
    assert str(config.experiment.stage) == "codec"
    assert str(config.data.name) == dataset
    assert int(config.codec.max_sequence_length) == max_sequence_length
    assert str(config.codec.codec_scope.mode) == "per_dataset"
    assert "datasets" not in config.data


@pytest.mark.parametrize("experiment", ["e01_emopia_codec", "e02_vgmidi_codec"])
def test_small_pitched_codec_experiments_have_independent_schedule(experiment: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        config = compose(config_name="config", overrides=[f"experiment={experiment}"])

    assert int(config.codec.training.max_epochs) == 200
    assert int(config.codec.training.max_steps) == 20_000
    assert int(config.codec.training.warmup_steps) == 500
    assert int(config.codec.kl.warmup_steps) == 2_000
    assert float(config.codec.kl.beta_max) == 0.0001
