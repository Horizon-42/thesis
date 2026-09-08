"""Constraint modules that ride the rollout's command hook (control/dynamics/hooks.py).

Each module is one `CommandHook`; `build_command_hook` picks the modules the config's
vocabulary value names (`config.CONTROL_HOOK_MEMBERS`) and gives them the batch's
per-flight context. A value naming more than one is applied through `CompositeHook`, in
the order the table lists — the rollout still sees one hook. Nothing here is imported by
the dynamics.

Two modules are live: the lateral `barrier` (the adopted inference-time safety layer) and
the `speed-floor` (the stall margin, held through the thrust command). The nominal
tracking law that shared this directory was never adopted and is archived under
`archive/nominal_law_hook_2026_09/`.
"""

from __future__ import annotations

import torch

from config import (
    CONTROL_HOOK_BARRIER,
    CONTROL_HOOK_MEMBERS,
    CONTROL_HOOK_NOMINAL_RESIDUAL,
    CONTROL_HOOK_OFF,
    CONTROL_HOOK_SPEED_FLOOR,
    HOOK_SATURATION_HARD,
    TSConfig,
)
from control.constraints.barrier_filter import BarrierFilter
from control.constraints.composite import CompositeHook
from control.constraints.speed_floor import SpeedFloor
from control.dynamics.hooks import CommandHook

_HOOKS = {
    CONTROL_HOOK_BARRIER: BarrierFilter,
    CONTROL_HOOK_SPEED_FLOOR: SpeedFloor,
}
# Fail at import, the companion of the assertion under `CONTROL_HOOK_MEMBERS`: that one says
# every selectable hook names its modules, this one that every module named has a class.
assert set().union(*CONTROL_HOOK_MEMBERS.values()) <= set(_HOOKS), (
    "CONTROL_HOOK_MEMBERS names a module with no implementation here"
)


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
    hard = config.control_hook_saturation == HOOK_SATURATION_HARD
    members = tuple(
        _HOOKS[name](config, dynamics, hard=hard)
        for name in CONTROL_HOOK_MEMBERS[config.control_command_hook]
    )
    return members[0] if len(members) == 1 else CompositeHook(members)


__all__ = ["BarrierFilter", "CompositeHook", "SpeedFloor", "build_command_hook"]
