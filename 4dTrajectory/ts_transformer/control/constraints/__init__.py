"""Constraint modules that ride the rollout's command hook (control/dynamics/hooks.py).

Each module is one `CommandHook`; `build_command_hook` picks it from the config and
gives it the batch's per-flight context. Nothing here is imported by the dynamics.
"""

from __future__ import annotations

import torch

from config import (
    CONTROL_HOOK_BARRIER,
    CONTROL_HOOK_NOMINAL_RESIDUAL,
    CONTROL_HOOK_OFF,
    HOOK_SATURATION_HARD,
    TSConfig,
)
from control.constraints.barrier_filter import BarrierFilter
from control.constraints.nominal_residual import NominalResidual
from control.dynamics.hooks import CommandHook

_HOOKS = {
    CONTROL_HOOK_BARRIER: BarrierFilter,
    CONTROL_HOOK_NOMINAL_RESIDUAL: NominalResidual,
}


def build_command_hook(
    config: TSConfig, dynamics: dict[str, torch.Tensor]
) -> CommandHook | None:
    """The configured hook for this batch, or None when the recipe runs without one."""
    if config.control_command_hook == CONTROL_HOOK_OFF:
        return None
    return _HOOKS[config.control_command_hook](
        config, dynamics, hard=config.control_hook_saturation == HOOK_SATURATION_HARD
    )


__all__ = ["BarrierFilter", "NominalResidual", "build_command_hook"]
