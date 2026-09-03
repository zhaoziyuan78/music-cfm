"""Four-worker smoke test for every native PyTorch training loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cfmusic.codec.transformer_vae import TransformerVAE
from cfmusic.conditioning.embeddings import AdditiveConditionEmbedding
from cfmusic.distributed import (
    cleanup_distributed,
    distributed_barrier,
    initialize_distributed,
)
from cfmusic.evaluation.style_effect import TokenStyleClassifier
from cfmusic.models.latent_vector_field import ConditionalVectorField
from cfmusic.models.probes import DynamicNoiseProjector
from cfmusic.reproducibility import seed_everything
from cfmusic.training.abduction_trainer import finetune_abduction_steps
from cfmusic.training.codec_trainer import train_codec_steps
from cfmusic.training.evaluator_trainer import train_token_evaluator
from cfmusic.training.schedules import warmup_cosine_scheduler
from cfmusic.training.transport_trainer import train_transport_steps
from cfmusic.transport.conditional_flow import ConditionalFlow


def _scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    return warmup_cosine_scheduler(optimizer, warmup_steps=0, max_steps=max_steps)


def _logging_bundle(run_dir: Path) -> dict[str, bool | int]:
    """Summarize artifacts shared by every iterative trainer."""

    return {
        "jsonl_records": len((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()),
        "csv_exists": (run_dir / "metrics.csv").is_file(),
        "text_log_exists": (run_dir / "training.log").is_file(),
        "curve_exists": (run_dir / "training_curves.png").is_file(),
        "tensorboard_events": len(list((run_dir / "tensorboard").glob("events.out.tfevents.*"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    context = initialize_distributed()
    try:
        seed_everything(17)
        if context.is_main:
            args.output.mkdir(parents=True, exist_ok=True)
        distributed_barrier(context)
        max_steps = 2 if args.resume else 1

        def resume_path(stage: str) -> Path | None:
            return args.output / stage / "last.pt" if args.resume else None

        codec = TransformerVAE(
            vocab_size=24,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            num_heads=4,
            ff_multiplier=2,
            dropout=0.0,
            latent_tokens=2,
            latent_dim=4,
            max_sequence_length=16,
        ).to(context.device)
        codec_optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3)
        token_batch = {
            "tokens": torch.tensor(
                [[1, 4, 5, 6, 7, 2], [1, 7, 6, 5, 4, 2], [1, 3, 8, 9, 3, 2], [1, 9, 8, 3, 9, 2]]
            ),
            "attention_mask": torch.ones(4, 6, dtype=torch.bool),
            "style_id": torch.tensor([0, 0, 1, 1]),
        }
        train_codec_steps(
            codec,
            [token_batch],
            optimizer=codec_optimizer,
            scheduler=_scheduler(codec_optimizer, max_steps),
            device=context.device,
            max_steps=max_steps,
            gradient_accumulation=1,
            gradient_clip_norm=1.0,
            precision="bf16",
            warmup_steps=1,
            beta_max=1e-3,
            free_bits_per_dim=0.0,
            checkpoint_dir=args.output / "codec",
            checkpoint_interval=1,
            config={"smoke": True},
            provenance={"source": "synthetic"},
            resume_from=resume_path("codec"),
            distributed=context,
            validation_batches=[token_batch] if context.is_main else None,
            validation_interval=1,
        )

        embedding = AdditiveConditionEmbedding(
            num_datasets=1,
            num_tasks=1,
            num_styles=2,
            num_genres=1,
            num_emotions=1,
            embedding_dim=16,
        )
        field = ConditionalVectorField(
            latent_dim=4,
            hidden_dim=16,
            layers=1,
            heads=4,
            mlp_ratio=2,
            dropout=0.0,
            condition_embedding=embedding,
            zero_init_output=False,
        )
        transport = ConditionalFlow(field, solver_method="euler").to(context.device)
        transport_optimizer = torch.optim.AdamW(transport.parameters(), lr=1e-3)
        latent_batch = {
            "latent": torch.arange(32, dtype=torch.float32).reshape(4, 2, 4).div(32),
            "dataset_id": torch.zeros(4, dtype=torch.long),
            "style_id": torch.tensor([0, 0, 1, 1]),
        }
        train_transport_steps(
            transport,
            [latent_batch],
            optimizer=transport_optimizer,
            scheduler=_scheduler(transport_optimizer, max_steps),
            device=context.device,
            max_steps=max_steps,
            gradient_accumulation=1,
            gradient_clip_norm=1.0,
            precision="bf16",
            checkpoint_dir=args.output / "transport",
            checkpoint_interval=1,
            config={"smoke": True},
            provenance={"source": "synthetic"},
            resume_from=resume_path("transport"),
            distributed=context,
        )

        projector = DynamicNoiseProjector(
            8, 4, num_views=2, seed=3, block_tokens=1, block_channels=2
        ).to(context.device)
        stage2_optimizer = torch.optim.AdamW(transport.parameters(), lr=1e-4)
        finetune_abduction_steps(
            transport,
            projector,
            None,
            [latent_batch],
            optimizer=stage2_optimizer,
            scheduler=_scheduler(stage2_optimizer, max_steps),
            device=context.device,
            max_steps=max_steps,
            abduction_interval=1,
            inverse_steps=1,
            hsic_weight=1e-3,
            prior_weight=1e-3,
            cross_class_weight=1e-3,
            adversarial_weight=0.0,
            roundtrip_weight=1e-2,
            cosine_weight=0.1,
            warmup_steps=0,
            ramp_steps=1,
            gradient_clip_norm=1.0,
            precision="bf16",
            checkpoint_dir=args.output / "stage2",
            checkpoint_interval=1,
            config={"smoke": True},
            provenance={"source": "synthetic"},
            resume_from=resume_path("stage2"),
            distributed=context,
        )

        evaluator = TokenStyleClassifier(
            vocab_size=24,
            num_classes=2,
            d_model=16,
            layers=1,
            heads=4,
            dropout=0.0,
            max_length=16,
        ).to(context.device)
        train_token_evaluator(
            evaluator,
            [token_batch],
            device=context.device,
            max_steps=max_steps,
            learning_rate=1e-3,
            weight_decay=0.0,
            checkpoint_dir=args.output / "evaluator",
            checkpoint_interval=1,
            config={"smoke": True},
            provenance={"source": "synthetic"},
            precision="bf16",
            resume_from=resume_path("evaluator"),
            distributed=context,
        )
        distributed_barrier(context)
        if context.is_main:
            stages = ["codec", "transport", "stage2", "evaluator"]
            result = {
                stage: {
                    "last_exists": (args.output / stage / "last.pt").is_file(),
                    "legacy_step_files": len(list((args.output / stage).glob("step-*.pt"))),
                    "global_step": int(
                        torch.load(
                            args.output / stage / "last.pt",
                            map_location="cpu",
                            weights_only=False,
                        )["train_state"]["global_step"]
                    ),
                    "logging": _logging_bundle(args.output / stage),
                }
                for stage in stages
            }
            result["codec"]["validation_logging"] = _logging_bundle(
                args.output / "codec" / "validation"
            )
            print(json.dumps(result, indent=2))
            if not all(item["last_exists"] for item in result.values()):
                raise RuntimeError("A DDP trainer did not write its rolling checkpoint")
            if not all(item["global_step"] == max_steps for item in result.values()):
                raise RuntimeError("A DDP trainer did not resume to the expected step")
            logging_bundles = [item["logging"] for item in result.values()]
            logging_bundles.append(result["codec"]["validation_logging"])
            if not all(
                bundle["jsonl_records"] == max_steps
                and bundle["csv_exists"]
                and bundle["text_log_exists"]
                and bundle["curve_exists"]
                and bundle["tensorboard_events"] > 0
                for bundle in logging_bundles
            ):
                raise RuntimeError("A DDP trainer did not write its complete logging bundle")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
