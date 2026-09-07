"""Build CFM or DDIM transports from composed Hydra config."""

from __future__ import annotations

from collections.abc import Mapping

from omegaconf import DictConfig

from cfmusic.conditioning.embeddings import AdditiveConditionEmbedding
from cfmusic.models.latent_denoiser import ConditionalLatentDenoiser
from cfmusic.models.latent_vector_field import ConditionalVectorField
from cfmusic.models.orthogonal_split import OrthogonalLatentSplit
from cfmusic.transport.conditional_ddim import ConditionalDDIM
from cfmusic.transport.conditional_flow import ConditionalFlow
from cfmusic.transport.independent_flows import IndependentStyleFlows
from cfmusic.transport.split_transport import SplitConditionalTransport


def validate_guidance_checkpoint(
    checkpoint: Mapping[str, object],
    cfg: DictConfig,
    *,
    exact_training_match: bool,
) -> None:
    """Prevent CFG inference/resume with a checkpoint that never learned a null condition."""

    requested = bool(cfg.get("classifier_free_guidance", False))
    saved_config = checkpoint.get("config")
    saved = (
        bool(saved_config.get("classifier_free_guidance", False))
        if isinstance(saved_config, Mapping)
        else False
    )
    saved_dropout = (
        float(saved_config.get("condition_dropout", 0.0))
        if isinstance(saved_config, Mapping)
        else 0.0
    )
    requested_dropout = float(cfg.get("condition_dropout", 0.0))
    if requested and (not saved or saved_dropout <= 0):
        raise ValueError(
            "Classifier-free guidance requires a checkpoint trained with positive condition "
            "dropout; disable CFG for this checkpoint or train the unified CFG recipe"
        )
    if exact_training_match and (
        requested != saved or (requested and abs(requested_dropout - saved_dropout) > 1e-12)
    ):
        raise ValueError(
            "Cannot resume with different classifier-free-guidance or condition-dropout "
            "training settings; inference guidance scales may be changed safely"
        )


def _embedding(cfg: DictConfig) -> AdditiveConditionEmbedding:
    return AdditiveConditionEmbedding(
        num_datasets=int(cfg.num_datasets),
        num_tasks=int(cfg.num_tasks),
        num_styles=int(cfg.num_styles),
        num_genres=int(cfg.num_genres),
        num_emotions=int(cfg.num_emotions),
        embedding_dim=int(cfg.embedding_dim),
    )


def create_transport(
    cfg: DictConfig,
) -> ConditionalFlow | ConditionalDDIM | IndependentStyleFlows | SplitConditionalTransport:
    model_cfg = cfg.model
    embedding = _embedding(cfg.conditioning)
    model_type = ConditionalLatentDenoiser if str(cfg.type) == "ddim" else ConditionalVectorField
    backbone = model_type(
        latent_dim=int(model_cfg.latent_dim),
        hidden_dim=int(model_cfg.hidden_dim),
        layers=int(model_cfg.layers),
        heads=int(model_cfg.heads),
        mlp_ratio=int(model_cfg.mlp_ratio),
        dropout=float(model_cfg.dropout),
        condition_embedding=embedding,
        zero_init_output=bool(model_cfg.zero_init_output),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
    )
    if str(cfg.type) == "ddim":
        return ConditionalDDIM(
            backbone,
            train_timesteps=int(cfg.diffusion.train_timesteps),
            inversion_method=str(cfg.ddim_inversion.method),
            fpi_iterations=int(cfg.ddim_inversion.iterations),
            fpi_tolerance=float(cfg.ddim_inversion.tolerance),
            fpi_stop_on_convergence=bool(cfg.ddim_inversion.stop_on_convergence),
        )
    ot_cfg = cfg.flow.get("ot")
    legacy_guidance_scale = float(cfg.get("guidance_scale", 1.0))
    flow = ConditionalFlow(
        backbone,
        solver_method=str(cfg.solver.method),
        time_sampling=str(cfg.flow.time_sampling),
        ot_solver=str(ot_cfg.solver) if str(cfg.flow.path) == "ot" and ot_cfg else None,
        ot_projection_dim=int(ot_cfg.cost_projection_dim) if ot_cfg else 128,
        ot_regularization=float(ot_cfg.regularization) if ot_cfg else 0.05,
        classifier_free_guidance=bool(cfg.get("classifier_free_guidance", False)),
        condition_dropout=float(cfg.get("condition_dropout", 0.0)),
        guidance_scale=legacy_guidance_scale,
        abduction_guidance_scale=float(cfg.get("abduction_guidance_scale", legacy_guidance_scale)),
        reconstruction_guidance_scale=float(
            cfg.get("reconstruction_guidance_scale", legacy_guidance_scale)
        ),
        prediction_guidance_scale=float(
            cfg.get("prediction_guidance_scale", legacy_guidance_scale)
        ),
        source_repulsion_scale=float(cfg.get("source_repulsion_scale", 0.0)),
    )
    if bool(cfg.get("independent_per_style", False)):
        return IndependentStyleFlows(flow, int(cfg.conditioning.num_styles))
    if "split" in cfg and bool(cfg.split.enabled):
        splitter = OrthogonalLatentSplit(
            int(cfg.split.original_latent_dim), float(cfg.split.editable_fraction)
        )
        if splitter.editable_dim != int(model_cfg.latent_dim):
            raise ValueError("split editable dimension must match transport model latent_dim")
        return SplitConditionalTransport(splitter, flow)
    return flow
