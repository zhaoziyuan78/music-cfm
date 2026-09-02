"""Shared mixed-precision and CUDA peak-memory helpers."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import AbstractContextManager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def autocast_context(device: torch.device, precision: str) -> AbstractContextManager[None]:
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported precision: {precision!r}")
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@contextlib.contextmanager
def sdpa_kernel_context(device: torch.device, backend: str) -> Iterator[None]:
    """Select a stable SDPA kernel for short latent-transport sequences.

    BF16 fused SDPA can return plausible forward values while producing severely
    amplified backward gradients for trained AdaLN attention blocks on A100.  The
    math backend retains BF16 autocast for the surrounding model but uses its more
    stable attention implementation.
    """

    normalized = backend.lower()
    if normalized not in {"auto", "math"}:
        raise ValueError(f"Unsupported SDPA backend: {backend!r}")
    if device.type != "cuda" or normalized == "auto":
        yield
        return
    with sdpa_kernel(SDPBackend.MATH):
        yield


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def total_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / 1024**3
