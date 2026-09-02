import torch
from torch import nn

from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.training.transport_trainer import (
    contrasting_conditions,
    inverse_frequency_weights,
)
from cfmusic.transport.conditional_flow import cfm_loss


class StyleValueField(nn.Module):
    def forward(
        self, state: torch.Tensor, time: torch.Tensor, condition: ConditionBatch
    ) -> torch.Tensor:
        del time
        return condition.style_id.to(state)[:, None, None].expand_as(state)


def test_condition_contrast_compares_labels_at_the_same_flow_state() -> None:
    latent = torch.tensor([[[0.0]], [[1.0]]])
    zeros = torch.zeros(2, dtype=torch.long)
    condition = ConditionBatch(zeros, zeros, torch.tensor([0, 1]))
    negative = ConditionBatch(zeros, zeros, torch.tensor([1, 0]))

    losses = cfm_loss(
        StyleValueField(),
        latent,
        condition,
        noise=torch.zeros_like(latent),
        negative_condition=negative,
        condition_contrast_weight=1.0,
        condition_contrast_margin=0.25,
    )

    assert losses["cfm_loss"].item() == 0.0
    assert losses["condition_contrast_loss"].item() == 0.0
    assert losses["condition_gap"].item() == 1.0
    assert losses["condition_accuracy"].item() == 1.0
    assert losses["condition_correct_error"].item() == 0.0
    assert losses["condition_wrong_error"].item() == 1.0


def test_contrasting_conditions_stay_in_support_and_change_every_label() -> None:
    zeros = torch.zeros(6, dtype=torch.long)
    condition = ConditionBatch(zeros, zeros, torch.tensor([0, 1, 2, 0, 1, 2]))

    negative = contrasting_conditions(condition, {"style_id": [0, 1, 2]}, factorial=False)

    assert bool((negative.style_id != condition.style_id).all())
    assert set(negative.style_id.tolist()) <= {0, 1, 2}


def test_inverse_frequency_weights_are_normalized_and_reduce_imbalance() -> None:
    labels = [0] * 100 + [1] * 25
    weights = inverse_frequency_weights(labels, exponent=0.5)

    empirical_mean = sum(weights[label] for label in labels) / len(labels)
    assert abs(empirical_mean - 1.0) < 1e-6
    assert weights[1] > weights[0]
