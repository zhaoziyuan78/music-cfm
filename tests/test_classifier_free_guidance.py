from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from cfmusic.conditioning.embeddings import AdditiveConditionEmbedding
from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.training.transport_trainer import TransportLossModule, roundtrip_schedule_scale
from cfmusic.transport.conditional_flow import ConditionalFlow, cfm_loss
from cfmusic.transport.factory import validate_guidance_checkpoint

CONFIGS = Path(__file__).parents[1] / "configs"


class MaskAwareConstantField(nn.Module):
    def forward(
        self, state: torch.Tensor, time: torch.Tensor, condition: ConditionBatch
    ) -> torch.Tensor:
        del time
        mask = (
            condition.condition_mask.to(state)
            if condition.condition_mask is not None
            else torch.ones(condition.batch_size, device=state.device, dtype=state.dtype)
        )
        return (condition.style_id.to(state) * mask)[:, None, None].expand_as(state)


class RecordingField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition_mask: torch.Tensor | None = None

    def forward(
        self, state: torch.Tensor, time: torch.Tensor, condition: ConditionBatch
    ) -> torch.Tensor:
        del time
        self.condition_mask = condition.condition_mask
        return torch.zeros_like(state)


class LearnableMaskedField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(
        self, state: torch.Tensor, time: torch.Tensor, condition: ConditionBatch
    ) -> torch.Tensor:
        del time
        mask = (
            condition.condition_mask.to(state)
            if condition.condition_mask is not None
            else torch.ones(condition.batch_size, device=state.device, dtype=state.dtype)
        )
        style = (condition.style_id.to(state) * mask)[:, None, None]
        return self.scale * state + style


def test_cfg_is_used_for_both_abduction_and_prediction() -> None:
    transport = ConditionalFlow(
        MaskAwareConstantField(),
        solver_method="heun",
        classifier_free_guidance=True,
        condition_dropout=0.1,
        guidance_scale=2.0,
    )
    latent = torch.randn(2, 3, 4)
    zeros = torch.zeros(2, dtype=torch.long)
    source = ConditionBatch(zeros, zeros, torch.ones(2, dtype=torch.long))
    target = ConditionBatch(zeros, zeros, torch.full((2,), 2, dtype=torch.long))

    output = transport.counterfactual(latent, source, target, num_steps=4)

    torch.testing.assert_close(output.reconstructed_source_latent, latent)
    torch.testing.assert_close(output.counterfactual_latent, latent + 2)
    assert output.inverse_nfe == 16
    assert output.forward_nfe == 32


def test_cfm_training_applies_per_sample_condition_dropout() -> None:
    torch.manual_seed(7)
    model = RecordingField()
    latent = torch.zeros(64, 2, 3)
    labels = torch.arange(64, dtype=torch.long) % 6
    condition = ConditionBatch(torch.zeros_like(labels), torch.zeros_like(labels), labels)

    losses = cfm_loss(
        model,
        latent,
        condition,
        noise=torch.zeros_like(latent),
        condition_dropout=0.25,
    )

    assert model.condition_mask is not None
    dropped = (model.condition_mask == 0).float().mean()
    torch.testing.assert_close(losses["condition_dropout_fraction"], dropped)
    assert 0 < float(dropped) < 1


def test_null_condition_retains_context_but_removes_style() -> None:
    embedding = AdditiveConditionEmbedding(
        num_datasets=1,
        num_tasks=1,
        num_styles=2,
        num_genres=1,
        num_emotions=1,
        embedding_dim=3,
    )
    with torch.no_grad():
        embedding.dataset.weight.fill_(1)
        embedding.task.weight.fill_(2)
        embedding.style.weight[0].fill_(4)
        embedding.style.weight[1].fill_(8)
        embedding.genre.weight.zero_()
        embedding.emotion.weight.zero_()
    zeros = torch.zeros(2, dtype=torch.long)
    condition = ConditionBatch(zeros, zeros, torch.tensor([0, 1]))

    conditional = embedding(condition)
    unconditional = embedding(condition.unconditional())

    torch.testing.assert_close(conditional, torch.tensor([[7.0] * 3, [11.0] * 3]))
    torch.testing.assert_close(unconditional, torch.tensor([[3.0] * 3, [3.0] * 3]))


def test_roundtrip_schedule_uses_warmup_and_linear_ramp() -> None:
    assert roundtrip_schedule_scale(4999, warmup_steps=5000, ramp_steps=5000) == 0
    assert roundtrip_schedule_scale(5000, warmup_steps=5000, ramp_steps=5000) == 1 / 5000
    assert roundtrip_schedule_scale(7499, warmup_steps=5000, ramp_steps=5000) == 0.5
    assert roundtrip_schedule_scale(9999, warmup_steps=5000, ramp_steps=5000) == 1
    assert roundtrip_schedule_scale(50_000, warmup_steps=5000, ramp_steps=5000) == 1


def test_unified_loss_backpropagates_through_guided_roundtrip() -> None:
    field = LearnableMaskedField()
    transport = ConditionalFlow(
        field,
        solver_method="heun",
        classifier_free_guidance=True,
        condition_dropout=0.1,
        guidance_scale=1.5,
    )
    objective = TransportLossModule(transport)
    latent = torch.randn(4, 2, 3)
    labels = torch.arange(4, dtype=torch.long) % 2
    condition = ConditionBatch(torch.zeros_like(labels), torch.zeros_like(labels), labels)

    losses = objective(
        latent,
        condition,
        None,
        None,
        condition_contrast_weight=0.0,
        condition_contrast_margin=0.0,
        condition_contrast_samples=None,
        roundtrip_weight=0.25,
        roundtrip_steps=2,
        roundtrip_samples=2,
        roundtrip_cosine_weight=0.1,
    )
    losses["loss"].backward()

    assert float(losses["roundtrip_loss"].detach()) > 0
    assert field.scale.grad is not None
    assert bool(torch.isfinite(field.scale.grad))


def test_unified_cfm_recipe_enables_cfg_roundtrip_and_no_noise_regularizer() -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        config = compose(config_name="config", overrides=["experiment=e24_cfm_cfg_roundtrip"])

    assert str(config.transport.type) == "cfm"
    assert bool(config.transport.classifier_free_guidance)
    assert float(config.transport.condition_dropout) == 0.1
    assert float(config.transport.guidance_scale) == 1.5
    assert bool(config.transport.roundtrip.enabled)
    assert float(config.transport.roundtrip.weight) == 0.25
    assert str(config.experiment.checkpoint_subdir) == "transport"
    assert not bool(config.independence.enabled)


def test_cfg_rejects_checkpoint_without_trained_null_branch() -> None:
    config = OmegaConf.create(
        {"classifier_free_guidance": True, "condition_dropout": 0.1, "guidance_scale": 1.5}
    )

    with pytest.raises(ValueError, match="checkpoint trained with positive condition dropout"):
        validate_guidance_checkpoint({"config": {}}, config, exact_training_match=False)


def test_resume_requires_same_cfg_training_settings_but_inference_scale_can_change() -> None:
    checkpoint = {
        "config": {
            "classifier_free_guidance": True,
            "condition_dropout": 0.1,
            "guidance_scale": 1.5,
        }
    }
    config = OmegaConf.create(
        {"classifier_free_guidance": True, "condition_dropout": 0.1, "guidance_scale": 2.0}
    )

    validate_guidance_checkpoint(checkpoint, config, exact_training_match=False)
    with pytest.raises(ValueError, match="guidance-scale"):
        validate_guidance_checkpoint(checkpoint, config, exact_training_match=True)
