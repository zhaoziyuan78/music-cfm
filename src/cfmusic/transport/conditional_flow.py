"""Shared conditional flow matching and invertible counterfactual API."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from cfmusic.conditioning.schema import ConditionBatch, apply_condition_dropout
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
    condition_dropout: float = 0.0,
) -> dict[str, Tensor]:
    base_noise = torch.randn_like(latent) if noise is None else noise
    time = sample_flow_time(latent.shape[0], latent.device, time_sampling)
    state = (1 - time[:, None, None]) * base_noise + time[:, None, None] * latent
    target = latent - base_noise
    model_condition, dropout_fraction = apply_condition_dropout(condition, condition_dropout)
    prediction = model(state, time, model_condition)
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
        if condition_dropout > 0 or condition.condition_mask is not None:
            correct_prediction = model(
                state.index_select(0, indices),
                time.index_select(0, indices),
                condition.index_select(indices),
            )
            correct_error = (
                functional.mse_loss(
                    correct_prediction, target.index_select(0, indices), reduction="none"
                )
                .flatten(1)
                .mean(1)
            )
        else:
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
        "condition_dropout_fraction": dropout_fraction,
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
        classifier_free_guidance: bool = False,
        condition_dropout: float = 0.0,
        guidance_scale: float = 1.0,
        abduction_guidance_scale: float | None = None,
        reconstruction_guidance_scale: float | None = None,
        prediction_guidance_scale: float | None = None,
        source_repulsion_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.vector_field = vector_field
        self.solver = FixedGridODESolver(solver_method)
        self.time_sampling = time_sampling
        self.ot_solver = ot_solver
        self.ot_projection_dim = ot_projection_dim
        self.ot_regularization = ot_regularization
        self.classifier_free_guidance = classifier_free_guidance
        self.condition_dropout = condition_dropout if classifier_free_guidance else 0.0
        legacy_scale = guidance_scale if classifier_free_guidance else 1.0
        self.abduction_guidance_scale = (
            legacy_scale if abduction_guidance_scale is None else abduction_guidance_scale
        )
        self.reconstruction_guidance_scale = (
            legacy_scale if reconstruction_guidance_scale is None else reconstruction_guidance_scale
        )
        self.prediction_guidance_scale = (
            legacy_scale if prediction_guidance_scale is None else prediction_guidance_scale
        )
        self.source_repulsion_scale = source_repulsion_scale if classifier_free_guidance else 0.0
        # Kept as a compatibility alias for old diagnostics and checkpoint metadata.
        self.guidance_scale = self.prediction_guidance_scale
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")
        if self.classifier_free_guidance and self.condition_dropout <= 0:
            raise ValueError(
                "Classifier-free guidance training requires positive condition_dropout"
            )
        scales = (
            self.abduction_guidance_scale,
            self.reconstruction_guidance_scale,
            self.prediction_guidance_scale,
            self.source_repulsion_scale,
        )
        if any(scale < 0 for scale in scales):
            raise ValueError("Guidance and source-repulsion scales must be non-negative")

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
            condition_dropout=self.condition_dropout,
        )
        losses["ot_fallback_ratio"] = latent.new_tensor(fallback)
        return losses

    def _guided_vector_field(
        self,
        state: Tensor,
        time: Tensor,
        condition: ConditionBatch,
        *,
        guidance_scale: float,
        source_condition: ConditionBatch | None = None,
        source_repulsion_scale: float = 0.0,
    ) -> Tensor:
        """Evaluate null/target/source branches together for efficient guidance."""

        if not self.classifier_free_guidance or (
            guidance_scale == 1.0 and source_repulsion_scale == 0.0
        ):
            return self.vector_field(state, time, condition)
        if source_repulsion_scale > 0 and source_condition is None:
            raise ValueError("Source-repulsive guidance requires a source condition")
        if source_condition is not None and source_condition.batch_size != condition.batch_size:
            raise ValueError("Source and target guidance conditions must have equal batch size")

        conditions = [condition.unconditional(), condition]
        if source_repulsion_scale > 0:
            assert source_condition is not None
            conditions.append(source_condition)

        def combine(name: str) -> Tensor | None:
            values = [getattr(value, name) for value in conditions]
            if name == "condition_mask":
                # ``None`` is the compact representation of an all-conditional
                # branch, while ``unconditional()`` materializes an all-zero
                # mask. Expand the implicit ones before concatenating branches.
                masks = [
                    torch.ones_like(value.style_id)
                    if value.condition_mask is None
                    else value.condition_mask
                    for value in conditions
                ]
                return torch.cat(masks)
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(f"Inconsistent guidance condition field: {name}")
            return torch.cat([cast(Tensor, value) for value in values])

        dataset_id = combine("dataset_id")
        task_id = combine("task_id")
        style_id = combine("style_id")
        assert dataset_id is not None and task_id is not None and style_id is not None
        combined_condition = ConditionBatch(
            dataset_id,
            task_id,
            style_id,
            combine("genre_id"),
            combine("emotion_id"),
            combine("condition_mask"),
        )
        combined_prediction = self.vector_field(
            torch.cat([state] * len(conditions), dim=0),
            torch.cat([time] * len(conditions), dim=0),
            combined_condition,
        )
        branches = combined_prediction.chunk(len(conditions), dim=0)
        unconditional, conditional = branches[:2]
        guided = unconditional + guidance_scale * (conditional - unconditional)
        if source_repulsion_scale > 0:
            source = branches[2]
            guided = guided - source_repulsion_scale * (source - unconditional)
        return guided

    def _integrate(
        self,
        state: Tensor,
        condition: ConditionBatch,
        *,
        t_start: float,
        t_end: float,
        num_steps: int,
        track_grad: bool,
        guidance_scale: float = 1.0,
        source_condition: ConditionBatch | None = None,
        source_repulsion_scale: float = 0.0,
    ) -> tuple[Tensor, int]:
        guided = self.classifier_free_guidance and (
            guidance_scale != 1.0 or source_repulsion_scale > 0.0
        )
        branch_count = 1 + int(guided) + int(guided and source_repulsion_scale > 0)

        def vector_field(
            current_state: Tensor, current_time: Tensor, current_condition: ConditionBatch
        ) -> Tensor:
            return self._guided_vector_field(
                current_state,
                current_time,
                current_condition,
                guidance_scale=guidance_scale,
                source_condition=source_condition,
                source_repulsion_scale=source_repulsion_scale,
            )

        result = self.solver.integrate(
            vector_field if guided else self.vector_field,
            state,
            condition,
            t_start=t_start,
            t_end=t_end,
            num_steps=num_steps,
            track_grad=track_grad,
        )
        if result.nan_count:
            raise FloatingPointError(f"ODE integration produced {result.nan_count} NaNs")
        return result.state, result.nfe * branch_count

    def abduct(
        self,
        latent: Tensor,
        condition: ConditionBatch,
        *,
        num_steps: int,
        track_grad: bool = False,
        guidance_scale: float | None = None,
    ) -> Tensor:
        return self._integrate(
            latent,
            condition,
            t_start=1.0,
            t_end=0.0,
            num_steps=num_steps,
            track_grad=track_grad,
            guidance_scale=(
                self.abduction_guidance_scale if guidance_scale is None else guidance_scale
            ),
        )[0]

    def predict(
        self,
        noise: Tensor,
        condition: ConditionBatch,
        *,
        num_steps: int,
        track_grad: bool = False,
        guidance_scale: float | None = None,
        source_condition: ConditionBatch | None = None,
        source_repulsion_scale: float | None = None,
    ) -> Tensor:
        repulsion = (
            self.source_repulsion_scale
            if source_repulsion_scale is None
            else source_repulsion_scale
        )
        if source_condition is None:
            repulsion = 0.0
        return self._integrate(
            noise,
            condition,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            track_grad=track_grad,
            guidance_scale=(
                self.prediction_guidance_scale if guidance_scale is None else guidance_scale
            ),
            source_condition=source_condition,
            source_repulsion_scale=repulsion,
        )[0]

    def reconstruct(
        self,
        noise: Tensor,
        condition: ConditionBatch,
        *,
        num_steps: int,
        track_grad: bool = False,
    ) -> Tensor:
        return self._integrate(
            noise,
            condition,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            track_grad=track_grad,
            guidance_scale=self.reconstruction_guidance_scale,
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
            guidance_scale=self.abduction_guidance_scale,
        )
        reconstructed, reconstruction_nfe = self._integrate(
            noise,
            source_condition,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            track_grad=False,
            guidance_scale=self.reconstruction_guidance_scale,
        )
        counterfactual, counterfactual_nfe = self._integrate(
            noise,
            target_condition,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            track_grad=False,
            guidance_scale=self.prediction_guidance_scale,
            source_condition=source_condition,
            source_repulsion_scale=self.source_repulsion_scale,
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
