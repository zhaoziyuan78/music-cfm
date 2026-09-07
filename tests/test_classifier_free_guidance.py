from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from cfmusic.conditioning.embeddings import AdditiveConditionEmbedding
from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.models.probes import DynamicNoiseProjector
from cfmusic.solvers.schedules import sample_flow_time
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


def test_split_cfg_keeps_abduction_unguided_and_repels_source_on_prediction() -> None:
    transport = ConditionalFlow(
        MaskAwareConstantField(),
        solver_method="heun",
        classifier_free_guidance=True,
        condition_dropout=0.1,
        abduction_guidance_scale=1.0,
        reconstruction_guidance_scale=1.0,
        prediction_guidance_scale=2.0,
        source_repulsion_scale=0.5,
    )
    latent = torch.randn(2, 3, 4)
    zeros = torch.zeros(2, dtype=torch.long)
    source = ConditionBatch(zeros, zeros, torch.ones(2, dtype=torch.long))
    target = ConditionBatch(zeros, zeros, torch.full((2,), 2, dtype=torch.long))

    output = transport.counterfactual(latent, source, target, num_steps=4)

    torch.testing.assert_close(output.abducted_noise, latent - 1)
    torch.testing.assert_close(output.reconstructed_source_latent, latent)
    torch.testing.assert_close(output.counterfactual_latent, latent + 2.5)
    assert output.inverse_nfe == 8
    assert output.forward_nfe == 32  # reconstruction 8 + three-branch prediction 24


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


def test_uniform_beta_mixture_concentrates_more_times_near_noise() -> None:
    torch.manual_seed(3)
    values = sample_flow_time(100_000, torch.device("cpu"), "uniform_beta_mixture")

    assert bool(((values >= 0) & (values <= 1)).all())
    assert 0.40 < float(values.mean()) < 0.435


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
        endpoint_weight=0.0,
        endpoint_steps=4,
        endpoint_samples_per_style=2,
        exogeneity_hsic_weight=0.0,
        exogeneity_prior_weight=0.0,
        exogeneity_cross_mmd_weight=0.0,
        exogeneity_cross_swd_weight=0.0,
        exogeneity_steps=4,
        exogeneity_samples_per_style=2,
        global_step=0,
        factorial_conditioning=False,
        factorial_active_axis=None,
    )
    losses["loss"].backward()

    assert float(losses["roundtrip_loss"].detach()) > 0
    assert field.scale.grad is not None
    assert bool(torch.isfinite(field.scale.grad))


def test_sparse_endpoint_and_exogeneity_losses_backpropagate_unguided() -> None:
    field = LearnableMaskedField()
    transport = ConditionalFlow(
        field,
        solver_method="heun",
        classifier_free_guidance=True,
        condition_dropout=0.1,
        prediction_guidance_scale=2.0,
    )
    objective = TransportLossModule(
        transport,
        DynamicNoiseProjector(8, 4, num_views=2, block_tokens=1, block_channels=2),
    )
    latent = torch.randn(4, 2, 4)
    labels = torch.tensor([0, 0, 1, 1])
    condition = ConditionBatch(torch.zeros_like(labels), torch.zeros_like(labels), labels)

    losses = objective(
        latent,
        condition,
        None,
        None,
        condition_contrast_weight=0.0,
        condition_contrast_margin=0.0,
        condition_contrast_samples=None,
        roundtrip_weight=0.0,
        roundtrip_steps=1,
        roundtrip_samples=None,
        roundtrip_cosine_weight=0.1,
        endpoint_weight=0.01,
        endpoint_steps=1,
        endpoint_samples_per_style=2,
        exogeneity_hsic_weight=0.01,
        exogeneity_prior_weight=0.01,
        exogeneity_cross_mmd_weight=0.01,
        exogeneity_cross_swd_weight=0.01,
        exogeneity_steps=1,
        exogeneity_samples_per_style=2,
        global_step=8,
        factorial_conditioning=False,
        factorial_active_axis=None,
    )
    losses["loss"].backward()

    assert float(losses["endpoint_swd"].detach()) > 0
    assert float(losses["noise_prior_swd"].detach()) > 0
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


def test_segment_otcfm_recipe_uses_update_recommendations() -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        config = compose(config_name="config", overrides=["experiment=e25_otcfm_segment_cfg"])

    assert str(config.transport.flow.path) == "ot"
    assert str(config.transport.flow.ot.solver) == "hungarian"
    assert int(config.transport.flow.ot.cost_projection_dim) == 128
    assert str(config.transport.flow.time_sampling) == "uniform_beta_mixture"
    assert float(config.transport.abduction_guidance_scale) == 1.0
    assert float(config.transport.reconstruction_guidance_scale) == 1.0
    assert float(config.transport.prediction_guidance_scale) == 2.0
    assert float(config.transport.source_repulsion_scale) == 0.5
    assert bool(config.transport.sampling.balance_by_style)
    assert bool(config.transport.sampling.unique_song_per_batch)
    assert bool(config.transport.endpoint_matching.enabled)
    assert bool(config.transport.exogeneity.enabled)
    assert float(config.transport.roundtrip.weight) == 0.05
    assert int(config.transport.roundtrip.inverse_steps) == 4


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
    validate_guidance_checkpoint(checkpoint, config, exact_training_match=True)

    config.condition_dropout = 0.2
    with pytest.raises(ValueError, match="condition-dropout"):
        validate_guidance_checkpoint(checkpoint, config, exact_training_match=True)
