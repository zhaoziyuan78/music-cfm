"""VAE objective and posterior-collapse diagnostics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor

from cfmusic.codec.base import DiagonalGaussian


def kl_beta(step: int, *, warmup_steps: int, beta_max: float) -> float:
    return beta_max * min(1.0, step / max(1, warmup_steps))


def vae_loss(
    logits: Tensor,
    targets: Tensor,
    posterior: DiagonalGaussian,
    *,
    pad_id: int,
    beta: float,
    free_bits_per_dim: float,
) -> dict[str, Tensor]:
    token_ce = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=pad_id
    )
    kl_per_dim = -0.5 * (1 + posterior.logvar - posterior.mean.square() - posterior.logvar.exp())
    kl = kl_per_dim.clamp_min(free_bits_per_dim).mean()
    raw_kl = kl_per_dim.mean()
    active_units = posterior.mean.var(dim=0, unbiased=False).gt(1e-2).float().sum()
    posterior_mean_variance = posterior.mean.var(dim=0, unbiased=False).mean()
    mutual_information_proxy = (
        0.5
        * torch.log1p(
            posterior.mean.var(dim=0, unbiased=False) / posterior.logvar.exp().mean(dim=0)
        )
        .sum(dim=-1)
        .mean()
    )
    loss = token_ce + beta * kl
    return {
        "loss": loss,
        "token_ce": token_ce,
        "kl": kl,
        "raw_kl": raw_kl,
        "active_units": active_units,
        "posterior_mean_variance": posterior_mean_variance,
        "mutual_information_proxy": mutual_information_proxy,
        "token_perplexity": token_ce.detach().clamp_max(math.log(1e6)).exp(),
    }
