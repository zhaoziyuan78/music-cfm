"""Weak c/h split transport that exactly bypasses the conserved subspace."""

from __future__ import annotations

from torch import Tensor, nn

from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.models.orthogonal_split import OrthogonalLatentSplit
from cfmusic.transport.conditional_flow import ConditionalFlow
from cfmusic.transport.counterfactual import CounterfactualOutput


class SplitConditionalTransport(nn.Module):
    def __init__(self, split: OrthogonalLatentSplit, editable_transport: ConditionalFlow) -> None:
        super().__init__()
        self.splitter = split
        self.editable_transport = editable_transport

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
        _, editable = self.splitter.split(latent)
        return self.editable_transport.training_loss(
            editable,
            condition,
            negative_condition=negative_condition,
            condition_contrast_weight=condition_contrast_weight,
            condition_contrast_margin=condition_contrast_margin,
            condition_contrast_samples=condition_contrast_samples,
            sample_weight=sample_weight,
        )

    def abduct(
        self, latent: Tensor, condition: ConditionBatch, *, num_steps: int, track_grad: bool = False
    ) -> Tensor:
        conserved, editable = self.splitter.split(latent)
        noise = self.editable_transport.abduct(
            editable, condition, num_steps=num_steps, track_grad=track_grad
        )
        return self.splitter.merge(conserved, noise)

    def predict(
        self, state: Tensor, condition: ConditionBatch, *, num_steps: int, track_grad: bool = False
    ) -> Tensor:
        conserved, noise = self.splitter.split(state)
        editable = self.editable_transport.predict(
            noise, condition, num_steps=num_steps, track_grad=track_grad
        )
        return self.splitter.merge(conserved, editable)

    def counterfactual(
        self,
        latent: Tensor,
        source_condition: ConditionBatch,
        target_condition: ConditionBatch,
        *,
        num_steps: int,
    ) -> CounterfactualOutput:
        conserved, editable = self.splitter.split(latent)
        output = self.editable_transport.counterfactual(
            editable, source_condition, target_condition, num_steps=num_steps
        )
        return CounterfactualOutput(
            latent,
            self.splitter.merge(conserved, output.abducted_noise),
            self.splitter.merge(conserved, output.reconstructed_source_latent),
            self.splitter.merge(conserved, output.counterfactual_latent),
            source_condition,
            target_condition,
            output.inverse_nfe,
            output.forward_nfe,
        )
