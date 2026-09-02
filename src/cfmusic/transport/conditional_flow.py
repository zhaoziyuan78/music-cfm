"""Shared conditional flow matching and invertible counterfactual API."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.solvers.ode import FixedGridODESolver
from cfmusic.solvers.schedules import sample_flow_time
from cfmusic.transport.counterfactual import CounterfactualOutput
from cfmusic.transport.ot_coupling import couple_noise_to_data


def cfm_loss(
    model: nn.Module,
    latent: Tensor,
    condition: ConditionBatch,
    *,
    time_sampling: str = "uniform",
    noise: Tensor | None = None,
    negative_condition: ConditionBatch | None = None,
    condition_contrast_weight: float = 0.0,
    condition_contrast_margin: float = 0.0,
    condition_contrast_samples: int | None = None,
    sample_weight: Tensor | None = None,
) -> dict[str, Tensor]:
    base_noise = torch.randn_like(latent) if noise is None else noise
    time = sample_flow_time(latent.shape[0], latent.device, time_sampling)
    state = (1 - time[:, None, None]) * base_noise + time[:, None, None] * latent
    target = latent - base_noise
    prediction = model(state, time, condition)
    per_sample_error = functional.mse_loss(prediction, target, reduction="none").flatten(1).mean(1)
    if sample_weight is None:
        cfm = per_sample_error.mean()
    else:
        weights = sample_weight.to(per_sample_error)
        if weights.ndim != 1 or weights.shape[0] != latent.shape[0]:
            raise ValueError("sample_weight must contain one value per latent")
        cfm = (per_sample_error * weights).sum() / weights.sum().clamp_min(1e-8)

    contrast = latent.new_zeros(())
    condition_gap = latent.new_zeros(())
    condition_accuracy = latent.new_zeros(())
    condition_correct_error = latent.new_zeros(())
    condition_wrong_error = latent.new_zeros(())
    if negative_condition is not None and condition_contrast_weight > 0:
        count = latent.shape[0]
        if condition_contrast_samples is not None:
            count = min(count, max(1, condition_contrast_samples))
        # Conditions matter most near the Gaussian endpoint, where the state
        # itself contains the least factual-style information.  Concentrating
        # the auxiliary objective there avoids spending its capacity on easy
        # near-data examples while the primary CFM loss remains uniform in time.
        indices = torch.topk(time, count, largest=False, sorted=False).indices
        negative_prediction = model(
            state.index_select(0, indices),
            time.index_select(0, indices),
            negative_condition.index_select(indices),
        )
        negative_error = (
            functional.mse_loss(
                negative_prediction, target.index_select(0, indices), reduction="none"
            )
            .flatten(1)
            .mean(1)
        )
        correct_error = per_sample_error.index_select(0, indices)
        gaps = negative_error - correct_error
        contrast_values = functional.relu(condition_contrast_margin - gaps)
        if sample_weight is None:
            contrast = contrast_values.mean()
        else:
            contrast_weights = sample_weight.index_select(0, indices).to(contrast_values)
            contrast = (
                contrast_values * contrast_weights
            ).sum() / contrast_weights.sum().clamp_min(1e-8)
        condition_gap = gaps.mean()
        condition_accuracy = (gaps > 0).to(latent.dtype).mean()
        condition_correct_error = correct_error.mean()
        condition_wrong_error = negative_error.mean()
    loss = cfm + condition_contrast_weight * contrast
    return {
        "loss": loss,
        "cfm_loss": cfm,
        "condition_contrast_loss": contrast,
        "condition_gap": condition_gap,
        "condition_accuracy": condition_accuracy,
        "condition_correct_error": condition_correct_error,
        "condition_wrong_error": condition_wrong_error,
        "time_mean": time.mean(),
    }


class ConditionalFlow(nn.Module):
    def __init__(
        self,
        vector_field: nn.Module,
        *,
        solver_method: str = "heun",
        time_sampling: str = "uniform",
        ot_solver: str | None = None,
        ot_projection_dim: int = 128,
        ot_regularization: float = 0.05,
    ) -> None:
        super().__init__()
        self.vector_field = vector_field
        self.solver = FixedGridODESolver(solver_method)
        self.time_sampling = time_sampling
        self.ot_solver = ot_solver
        self.ot_projection_dim = ot_projection_dim
        self.ot_regularization = ot_regularization

    def training_loss(
        self,
        latent: Tensor,
        condition: ConditionBatch,
        *,
        negative_condition: ConditionBatch | None = None,
        condition_contrast_weight: float = 0.0,
        condition_contrast_margin: float = 0.0,
        condition_contrast_samples: int | None = None,
        sample_weight: Tensor | None = None,
    ) -> dict[str, Tensor]:
        noise = torch.randn_like(latent)
        fallback = 0.0
        if self.ot_solver is not None:
            result = couple_noise_to_data(
                noise,
                latent,
                condition.style_id,
                solver=self.ot_solver,
                cost_projection_dim=self.ot_projection_dim,
                regularization=self.ot_regularization,
            )
            noise = result.noise
            fallback = result.fallback_ratio
        losses = cfm_loss(
            self.vector_field,
            latent,
            condition,
            time_sampling=self.time_sampling,
            noise=noise,
            negative_condition=negative_condition,
            condition_contrast_weight=condition_contrast_weight,
            condition_contrast_margin=condition_contrast_margin,
            condition_contrast_samples=condition_contrast_samples,
            sample_weight=sample_weight,
        )
        losses["ot_fallback_ratio"] = latent.new_tensor(fallback)
        return losses

    def _integrate(
        self,
        state: Tensor,
        condition: ConditionBatch,
        *,
        t_start: float,
        t_end: float,
        num_steps: int,
        track_grad: bool,
    ) -> tuple[Tensor, int]:
        result = self.solver.integrate(
            self.vector_field,
            state,
            condition,
            t_start=t_start,
            t_end=t_end,
            num_steps=num_steps,
            track_grad=track_grad,
        )
        if result.nan_count:
            raise FloatingPointError(f"ODE integration produced {result.nan_count} NaNs")
        return result.state, result.nfe

    def abduct(
        self, latent: Tensor, condition: ConditionBatch, *, num_steps: int, track_grad: bool = False
    ) -> Tensor:
        return self._integrate(
            latent, condition, t_start=1.0, t_end=0.0, num_steps=num_steps, track_grad=track_grad
        )[0]

    def predict(
        self, noise: Tensor, condition: ConditionBatch, *, num_steps: int, track_grad: bool = False
    ) -> Tensor:
        return self._integrate(
            noise, condition, t_start=0.0, t_end=1.0, num_steps=num_steps, track_grad=track_grad
        )[0]

    def counterfactual(
        self,
        latent: Tensor,
        source_condition: ConditionBatch,
        target_condition: ConditionBatch,
        *,
        num_steps: int,
    ) -> CounterfactualOutput:
        noise, inverse_nfe = self._integrate(
            latent,
            source_condition,
            t_start=1.0,
            t_end=0.0,
            num_steps=num_steps,
            track_grad=False,
        )
        reconstructed, reconstruction_nfe = self._integrate(
            noise,
            source_condition,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            track_grad=False,
        )
        counterfactual, counterfactual_nfe = self._integrate(
            noise,
            target_condition,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            track_grad=False,
        )
        return CounterfactualOutput(
            latent,
            noise,
            reconstructed,
            counterfactual,
            source_condition,
            target_condition,
            inverse_nfe,
            reconstruction_nfe + counterfactual_nfe,
        )
