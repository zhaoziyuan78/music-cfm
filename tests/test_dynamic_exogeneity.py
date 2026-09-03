import torch

from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.models.probes import DynamicNoiseProjector
from cfmusic.training.abduction_trainer import AbductionLossModule
from cfmusic.training.state import ExponentialMovingAverage
from cfmusic.training.transport_trainer import evaluate_raw_and_ema


class _MinimalTransport(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def training_loss(
        self, latent: torch.Tensor, _condition: object, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
        return {"loss": latent.square().mean() * self.scale}


def test_dynamic_projector_changes_directions_between_steps() -> None:
    projector = DynamicNoiseProjector(32, 8, num_views=2, seed=7)
    first = projector.projection_matrix(0, 0, device=torch.device("cpu"))
    second = projector.projection_matrix(1, 0, device=torch.device("cpu"))
    validation = projector.projection_matrix(0, 0, device=torch.device("cpu"), validation=True)

    assert not torch.equal(first, second)
    assert not torch.equal(first, validation)


def test_ema_context_restores_raw_parameters() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    ema = ExponentialMovingAverage(model, decay=0.9)
    raw = model.weight.detach().clone()
    ema.shadow["weight"].fill_(12)

    with ema.average_parameters(model):
        assert torch.equal(model.weight, torch.full_like(model.weight, 12))
    torch.testing.assert_close(model.weight, raw)


def test_validation_records_raw_and_ema_variants() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2)
    ema = ExponentialMovingAverage(model)
    ema.shadow["weight"].fill_(5)

    metrics = evaluate_raw_and_ema(model, ema, lambda: {"score": float(model.weight.item())})

    assert metrics == {"validation_raw_score": 2.0, "validation_ema_score": 5.0}
    assert model.weight.item() == 2.0


def test_adversary_remains_in_static_graph_between_abduction_steps() -> None:
    transport = _MinimalTransport()
    adversary = torch.nn.Linear(4, 2)
    objective = AbductionLossModule(
        transport, DynamicNoiseProjector(8, 4, num_views=2), adversary
    )
    condition = ConditionBatch(
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([0, 1]),
    )
    losses = objective(
        torch.randn(2, 2, 4),
        condition,
        run_abduction=False,
        inverse_steps=1,
        factorial_conditioning=False,
        regularization=0.0,
        hsic_weight=0.0,
        prior_weight=0.0,
        cross_class_weight=0.0,
        adversarial_weight=1.0,
        roundtrip_weight=0.0,
        cosine_weight=0.0,
        negative_condition=None,
        condition_contrast_weight=0.0,
        condition_contrast_margin=0.0,
        condition_contrast_samples=None,
        global_step=0,
    )
    losses["loss"].backward()

    assert all(parameter.grad is not None for parameter in adversary.parameters())
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in adversary.parameters())
