"""Conditional epsilon denoiser sharing the latent DiT architecture."""

from cfmusic.models.latent_vector_field import ConditionalVectorField


class ConditionalLatentDenoiser(ConditionalVectorField):
    """Time-conditioned latent denoiser; time is normalized to [0, 1]."""
