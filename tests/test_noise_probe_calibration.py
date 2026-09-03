import numpy as np
import pytest

from cfmusic.evaluation.noise_leakage import _probe_scores

pytestmark = pytest.mark.filterwarnings("ignore:Stochastic Optimizer.*")


def test_iid_gaussian_linear_leakage_is_near_chance() -> None:
    random = np.random.default_rng(19)
    features = random.standard_normal((800, 6))
    labels = np.repeat(np.arange(4), 200)
    random.shuffle(labels)

    metrics = _probe_scores(features, labels, seed=11)

    assert abs(metrics["logistic_balanced_accuracy"] - 0.25) < 0.08
