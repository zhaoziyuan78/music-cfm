"""Independent per-style flow negative control."""

from __future__ import annotations

import copy
from typing import cast

import torch
from torch import Tensor, nn

from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.transport.conditional_flow import ConditionalFlow
from cfmusic.transport.counterfactual import CounterfactualOutput


class IndependentStyleFlows(nn.Module):
    def __init__(self, prototype: ConditionalFlow, num_styles: int) -> None:
        super().__init__()
        self.flows = nn.ModuleList([copy.deepcopy(prototype) for _ in range(num_styles)])

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
        # This negative-control model selects a separate network by style, so a
        # within-network wrong-label comparison is not a meaningful objective.
        del negative_condition, condition_contrast_weight, condition_contrast_margin
        del condition_contrast_samples
        weighted_losses: list[Tensor] = []
        for style in torch.unique(condition.style_id):
            indices = torch.nonzero(condition.style_id == style, as_tuple=False).flatten()
            flow = cast(ConditionalFlow, self.flows[int(style)])
            result = flow.training_loss(
                latent.index_select(0, indices),
                condition.index_select(indices),
                sample_weight=(
                    sample_weight.index_select(0, indices) if sample_weight is not None else None
                ),
            )
            weighted_losses.append(result["loss"] * indices.numel() / latent.shape[0])
        loss = torch.stack(weighted_losses).sum()
        zero = loss.new_zeros(())
        return {
            "loss": loss,
            "cfm_loss": loss,
            "condition_contrast_loss": zero,
            "condition_gap": zero,
            "condition_accuracy": zero,
            "condition_correct_error": zero,
            "condition_wrong_error": zero,
        }

    def _map(
        self,
        state: Tensor,
        condition: ConditionBatch,
        *,
        operation: str,
        num_steps: int,
        track_grad: bool,
    ) -> Tensor:
        output = torch.empty_like(state)
        for style in torch.unique(condition.style_id):
            indices = torch.nonzero(condition.style_id == style, as_tuple=False).flatten()
            flow = cast(ConditionalFlow, self.flows[int(style)])
            method = flow.abduct if operation == "abduct" else flow.predict
            values = method(
                state.index_select(0, indices),
                condition.index_select(indices),
                num_steps=num_steps,
                track_grad=track_grad,
            )
            output.index_copy_(0, indices, values)
        return output

    def abduct(
        self, latent: Tensor, condition: ConditionBatch, *, num_steps: int, track_grad: bool = False
    ) -> Tensor:
        return self._map(
            latent, condition, operation="abduct", num_steps=num_steps, track_grad=track_grad
        )

    def predict(
        self, noise: Tensor, condition: ConditionBatch, *, num_steps: int, track_grad: bool = False
    ) -> Tensor:
        return self._map(
            noise, condition, operation="predict", num_steps=num_steps, track_grad=track_grad
        )

    def counterfactual(
        self,
        latent: Tensor,
        source_condition: ConditionBatch,
        target_condition: ConditionBatch,
        *,
        num_steps: int,
    ) -> CounterfactualOutput:
        noise = self.abduct(latent, source_condition, num_steps=num_steps)
        reconstructed = self.predict(noise, source_condition, num_steps=num_steps)
        counterfactual = self.predict(noise, target_condition, num_steps=num_steps)
        first_flow = cast(ConditionalFlow, self.flows[0])
        method_nfe = num_steps * (2 if first_flow.solver.method == "heun" else 1)
        return CounterfactualOutput(
            latent,
            noise,
            reconstructed,
            counterfactual,
            source_condition,
            target_condition,
            method_nfe,
            method_nfe * 2,
        )
