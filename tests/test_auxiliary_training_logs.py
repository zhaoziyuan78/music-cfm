from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from cfmusic.commands import train_evaluator
from cfmusic.evaluation.noise_leakage import train_temporal_probe


def _assert_logging_bundle(directory: Path, records: int) -> None:
    assert len((directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == records
    assert (directory / "metrics.csv").is_file()
    assert (directory / "training.log").is_file()
    assert (directory / "training_curves.png").is_file()
    assert list((directory / "tensorboard").glob("events.out.tfevents.*"))


def test_descriptor_mlp_writes_training_logs(tmp_path: Path, monkeypatch: object) -> None:
    descriptors = {
        "a.mid": np.array([0.0, 0.1, 0.2]),
        "b.mid": np.array([0.1, 0.2, 0.1]),
        "c.mid": np.array([0.9, 0.8, 0.9]),
        "d.mid": np.array([0.8, 0.9, 0.8]),
    }
    monkeypatch.setattr(
        train_evaluator,
        "symbolic_descriptors",
        lambda path: descriptors[path.name],
    )
    frame = pd.DataFrame(
        {
            "source_midi_path": list(descriptors),
            "style_id": [0, 0, 1, 1],
        }
    )
    config = OmegaConf.create(
        {
            "evaluator": {
                "hidden_layers": [4],
                "random_state": 0,
                "max_iter": 2,
                "checkpoint_interval": 1,
                "min_iter": 3,
                "early_stopping_patience": 3,
                "early_stopping_tolerance": 1e-4,
            }
        }
    )

    train_evaluator._train_descriptor_mlp(config, frame, frame, tmp_path, None)

    _assert_logging_bundle(tmp_path / "descriptor_mlp_training", records=2)
    assert (tmp_path / "last.joblib").is_file()


def test_temporal_probe_writes_training_logs(tmp_path: Path) -> None:
    noise = torch.randn(20, 4, 3)
    labels = torch.tensor([0, 1] * 10)

    train_temporal_probe(noise, labels, epochs=2, seed=0, log_dir=tmp_path)

    _assert_logging_bundle(tmp_path, records=2)
