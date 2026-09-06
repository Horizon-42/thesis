"""Constraint modules that ride the rollout's command hook (control/dynamics/hooks.py).

Each module is one `CommandHook`; `build_command_hook` picks it from the config and
gives it the batch's per-flight context. Nothing here is imported by the dynamics.

Only the barrier is left: the nominal tracking law that shared this directory was never
adopted and is archived under `archive/nominal_law_hook_2026_09/`.
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
from control.dynamics.hooks import CommandHook

_HOOKS = {CONTROL_HOOK_BARRIER: BarrierFilter}


def build_command_hook(
    config: TSConfig, dynamics: dict[str, torch.Tensor]
) -> CommandHook | None:
    """The configured hook for this batch, or None when the recipe runs without one."""
    if config.control_command_hook == CONTROL_HOOK_OFF:
        return None
    if config.control_command_hook == CONTROL_HOOK_NOMINAL_RESIDUAL:
        # The value survives so the six control_hooks_20260906 configs — and the
        # checkpoints load_checkpoint rebuilds from them — still load; the law itself is
        # archive/nominal_law_hook_2026_09/ and cannot be flown from here.
        raise ValueError(
            "the nominal-law hook is archived (archive/nominal_law_hook_2026_09/); its "
            "numbers are in docs/2026-09-06_control_hooks_results.zh.md and the adopted "
            "hook is 'barrier'"
        )
    return _HOOKS[config.control_command_hook](
        config, dynamics, hard=config.control_hook_saturation == HOOK_SATURATION_HARD
    )


__all__ = ["BarrierFilter", "build_command_hook"]
