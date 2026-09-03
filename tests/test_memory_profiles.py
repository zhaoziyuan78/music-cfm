from pathlib import Path

from omegaconf import OmegaConf

CONFIGS = Path(__file__).parents[1] / "configs"


def test_latent_cache_profile_uses_four_gpu_io_defaults() -> None:
    config = OmegaConf.load(CONFIGS / "config.yaml")
    cache = config.latent_cache
    assert bool(cache.resume)
    assert int(cache.samples_per_shard) == 8192
    assert int(cache.batch_size_per_gpu) == 384
    assert str(cache.codec_weights) == "raw"
    assert int(cache.dataloader_workers) == 32
    assert int(cache.prefetch_factor) == 4
    assert int(cache.midi_cache_size) == 16
    assert int(cache.dataloader_timeout_seconds) == 180
    assert int(cache.rank_stall_timeout_seconds) == 300
    assert bool(cache.length_bucketing)
    assert not bool(cache.verify_after_write)


def test_codec_profiles_preserve_effective_batch_with_safe_micro_batches() -> None:
    expected_effective_batches = {"transformer_vae": 32, "drum_transformer_vae": 64}
    for name, effective_batch in expected_effective_batches.items():
        config = OmegaConf.load(CONFIGS / "codec" / f"{name}.yaml")
        training = config.training
        assert int(training.batch_size) * int(training.gradient_accumulation) == effective_batch
        assert bool(training.length_bucketing)
        assert bool(training.drop_overlength)
        assert 0.0 < float(training.decoder_token_dropout) < 1.0
        assert int(training.dataloader_workers) == 8
        assert str(training.precision) == "bf16"
        assert int(config.inference.batch_size) <= effective_batch
    pitched = OmegaConf.load(CONFIGS / "codec" / "transformer_vae.yaml")
    drums = OmegaConf.load(CONFIGS / "codec" / "drum_transformer_vae.yaml")
    assert int(pitched.training.batch_size) == 32
    assert bool(pitched.training.gradient_checkpointing)
    assert int(pitched.training.max_epochs) == 8
    assert int(pitched.kl.warmup_steps) == 20_000
    assert int(pitched.training.validation_samples) == 256
    assert int(drums.training.batch_size) == 64
    assert not bool(drums.training.gradient_checkpointing)
    assert int(drums.training.max_epochs) == 50
    assert int(drums.training.warmup_steps) == 1_000
    assert int(drums.kl.warmup_steps) == 2_000
    assert int(drums.training.validation_samples) == 256


def test_transport_and_evaluator_profiles_bound_micro_batches() -> None:
    for name in ("cfm", "ddim", "ot_cfm", "split_cfm"):
        config = OmegaConf.load(CONFIGS / "transport" / f"{name}.yaml")
        expected_batch = 512
        assert int(config.training.batch_size) == expected_batch
        assert int(config.training.batch_size) * int(config.training.gradient_accumulation) == (
            expected_batch
        )
        assert not bool(config.model.gradient_checkpointing)
        assert int(config.training.dataloader_workers) == 2
        assert int(config.training.max_steps) == 50_000
        expected_epochs = 60 if name in {"cfm", "ot_cfm"} else 40
        assert int(config.training.max_epochs) == expected_epochs
        if name == "split_cfm":
            assert int(config.split.original_latent_dim) == 512
            assert int(config.model.latent_dim) == 256
        else:
            assert int(config.model.latent_dim) == 512
    evaluator = OmegaConf.load(CONFIGS / "evaluator" / "transformer.yaml")
    assert int(evaluator.training.batch_size) == 32
    assert int(evaluator.training.gradient_accumulation) == 1
    assert int(evaluator.training.dataloader_workers) == 4
    assert int(evaluator.training.max_steps) == 10_000
    assert not bool(evaluator.gradient_checkpointing)


def test_abduction_profiles_use_balanced_batch_sixty_four() -> None:
    for name in ("exoreg", "hsic", "adversarial"):
        config = OmegaConf.load(CONFIGS / "independence" / f"{name}.yaml")
        assert int(config.sampler.classes_per_batch) * int(config.sampler.samples_per_class) == 64
        assert not bool(config.training.gradient_checkpointing)
        assert int(config.training.max_steps) == 12_000
        assert int(config.training.train_inverse_steps) == 4
        assert int(config.training.dataloader_workers) == 2


def test_cfm_profile_removes_invalid_wrong_condition_objective() -> None:
    config = OmegaConf.load(CONFIGS / "transport" / "cfm.yaml")

    assert int(config.model.hidden_dim) == 768
    assert int(config.model.layers) == 10
    assert not bool(config.condition_objective.enabled)
    assert float(config.condition_objective.weight) == 0
    assert float(config.training.class_balance_exponent) > 0


def test_ot_cfm_only_changes_the_flow_coupling() -> None:
    cfm = OmegaConf.load(CONFIGS / "transport" / "cfm.yaml")
    ot_cfm = OmegaConf.load(CONFIGS / "transport" / "ot_cfm.yaml")

    for field in (
        "model",
        "conditioning",
        "solver",
        "training",
        "condition_objective",
        "classifier_free_guidance",
        "condition_dropout",
        "guidance_scale",
    ):
        assert cfm[field] == ot_cfm[field]
    assert str(cfm.flow.path) == "independent"
    assert str(ot_cfm.flow.path) == "ot"
