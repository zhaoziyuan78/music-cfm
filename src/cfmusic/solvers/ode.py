"""Differentiable fixed-grid ODE solvers for matched forward/inverse integration."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from cfmusic.conditioning.schema import ConditionBatch
from cfmusic.progress import track

ODEFunction = Callable[[Tensor, Tensor, ConditionBatch], Tensor]


@dataclass
class SolverResult:
    state: Tensor
    nfe: int
    wall_time: float
    maximum_state_norm: float
    nan_count: int


class FixedGridODESolver:
    def __init__(self, method: str = "heun") -> None:
        if method not in {"euler", "heun", "rk4", "implicit_midpoint"}:
            raise ValueError(f"Unsupported ODE method: {method}")
        self.method = method

    def integrate(
        self,
        func: ODEFunction,
        initial_state: Tensor,
        condition: ConditionBatch,
        *,
        t_start: float,
        t_end: float,
        num_steps: int,
        track_grad: bool,
    ) -> SolverResult:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        context = torch.enable_grad() if track_grad else torch.no_grad()
        state = initial_state
        nfe = 0
        maximum_norm = float(state.detach().float().norm(dim=-1).max())
        started = time.perf_counter()
        grid = torch.linspace(
            t_start, t_end, num_steps + 1, device=state.device, dtype=torch.float32
        )
        indices = range(num_steps)
        integration_steps = (
            indices
            if track_grad
            else track(
                indices,
                description=f"ODE integrate {t_start:g}->{t_end:g}",
                total=num_steps,
                unit="step",
                leave=False,
                position=2,
            )
        )
        with context:
            for index in integration_steps:
                t0, t1 = grid[index], grid[index + 1]
                step = t1 - t0
                time0 = t0.expand(state.shape[0])
                if self.method == "euler":
                    state = state + step * func(state, time0, condition)
                    nfe += 1
                elif self.method == "heun":
                    k1 = func(state, time0, condition)
                    k2 = func(state + step * k1, t1.expand(state.shape[0]), condition)
                    state = state + step * 0.5 * (k1 + k2)
                    nfe += 2
                elif self.method == "rk4":
                    midpoint = (t0 + step / 2).expand(state.shape[0])
                    k1 = func(state, time0, condition)
                    k2 = func(state + step * k1 / 2, midpoint, condition)
                    k3 = func(state + step * k2 / 2, midpoint, condition)
                    k4 = func(state + step * k3, t1.expand(state.shape[0]), condition)
                    state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    nfe += 4
                else:
                    midpoint_state = state
                    midpoint_time = (t0 + step / 2).expand(state.shape[0])
                    for _ in range(4):
                        midpoint_state = (
                            state + step * func(midpoint_state, midpoint_time, condition) / 2
                        )
                        nfe += 1
                    state = state + step * func(midpoint_state, midpoint_time, condition)
                    nfe += 1
                maximum_norm = max(maximum_norm, float(state.detach().float().norm(dim=-1).max()))
        return SolverResult(
            state,
            nfe,
            time.perf_counter() - started,
            maximum_norm,
            int(torch.isnan(state).sum().item()),
        )
