import contextlib

import torch

from cfmusic.memory import (
    autocast_context,
    peak_memory_gib,
    sdpa_kernel_context,
    total_memory_gib,
)


def test_cpu_memory_helpers_are_safe_noops() -> None:
    device = torch.device("cpu")
    context = autocast_context(device, "bf16")
    assert isinstance(context, contextlib.nullcontext)
    assert peak_memory_gib(device) == 0
    assert total_memory_gib(device) == 0


def test_invalid_precision_is_rejected() -> None:
    try:
        autocast_context(torch.device("cpu"), "tf32")
    except ValueError as error:
        assert "Unsupported precision" in str(error)
    else:
        raise AssertionError("Invalid precision was accepted")


def test_cpu_sdpa_context_is_a_safe_noop() -> None:
    with sdpa_kernel_context(torch.device("cpu"), "math"):
        assert True


def test_invalid_sdpa_backend_is_rejected() -> None:
    try:
        with sdpa_kernel_context(torch.device("cpu"), "flash"):
            pass
    except ValueError as error:
        assert "Unsupported SDPA backend" in str(error)
    else:
        raise AssertionError("Invalid SDPA backend was accepted")
