import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from cfmusic.commands.generate_counterfactuals import (
    _artifact_matches_generation,
    concatenate_conditions,
    select_source_indices,
)
from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.transport.conditional_flow import ConditionalFlow


class ConstantField(nn.Module):
    def forward(
        self, state: torch.Tensor, time: torch.Tensor, condition: ConditionBatch
    ) -> torch.Tensor:
        return condition.style_id.to(state)[:, None, None].expand_as(state) + 1


def test_shared_noise_counterfactual_api() -> None:
    transport = ConditionalFlow(ConstantField(), solver_method="heun")
    latent = torch.randn(3, 2, 4)
    zeros = torch.zeros(3, dtype=torch.long)
    source = ConditionBatch(zeros, zeros, zeros)
    target = ConditionBatch(zeros, zeros, torch.ones(3, dtype=torch.long))
    output = transport.counterfactual(latent, source, target, num_steps=4)
    assert torch.allclose(output.reconstructed_source_latent, latent, atol=1e-6)
    assert torch.allclose(output.counterfactual_latent, latent + 1, atol=1e-6)
    assert output.inverse_nfe == 8 and output.forward_nfe == 16


def test_source_selection_is_balanced_unique_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "c", "d", "e", "f"],
            "style_id": [0, 0, 0, 0, 1, 1, 1],
        }
    )
    arguments = {
        "strata": ["style_id"],
        "max_per_stratum": 2,
        "max_total": 4,
        "unique_sources": True,
        "seed": 7,
    }

    selected = select_source_indices(frame, **arguments)

    assert selected == select_source_indices(frame, **arguments)
    selected_frame = frame.iloc[selected]
    assert selected_frame["sample_id"].nunique() == 4
    assert selected_frame["style_id"].value_counts().to_dict() == {0: 2, 1: 2}


def test_condition_concatenation_preserves_factorial_fields() -> None:
    first = ConditionBatch(
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([2]),
        torch.tensor([3]),
        torch.tensor([4]),
    )
    second = ConditionBatch(
        torch.tensor([5]),
        torch.tensor([6]),
        torch.tensor([7]),
        torch.tensor([8]),
        torch.tensor([9]),
    )

    combined = concatenate_conditions([first, second])

    assert combined.batch_size == 2
    assert combined.style_id.tolist() == [2, 7]
    assert combined.genre_id is not None and combined.genre_id.tolist() == [3, 8]
    assert combined.emotion_id is not None and combined.emotion_id.tolist() == [4, 9]


def test_existing_artifact_must_match_current_generation_identity(tmp_path: Path) -> None:
    metadata_path = tmp_path / "counterfactual_metadata.json"
    identity = {
        "codec_checkpoint_hash": "codec-new",
        "transport_checkpoint_hash": "transport-new",
        "generation_config_hash": "config-new",
    }
    metadata_path.write_text(
        json.dumps({**identity, "sample_id": "example"}), encoding="utf-8"
    )

    assert _artifact_matches_generation(metadata_path, identity)
    assert not _artifact_matches_generation(
        metadata_path, {**identity, "transport_checkpoint_hash": "transport-other"}
    )
