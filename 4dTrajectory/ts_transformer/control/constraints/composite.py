"""Several constraint modules on one rollout, applied in a fixed order.

``control/dynamics/hooks.py`` gives the rollout ONE hook per segment, and that is right:
"the command flown" has to be a single answer. Composition therefore happens here, not in
the rollout — each module in turn is handed the SAME state (the segment's start) and the
command the previous one returned, and the last answer is what the rollout integrates.

The order is part of the vocabulary value, not a free choice
(``config.CONTROL_HOOK_MEMBERS``). ``barrier+speed-floor`` means the barrier first: it sets
the bank and re-coordinates the load factor, and the speed floor then reads THAT load
factor for its stall speed and sets the thrust. Reversed, the floor would price a manoeuvre
the barrier is about to change. The two modules write disjoint channels — bank and load
against thrust — which is what makes that combination well defined and its reverse absent
from the vocabulary.

``barrier+trombone`` and ``barrier+speed-floor+trombone`` add a module that writes bank and
load as well, so the disjoint-channel argument alone does not cover them. What does is that
the two lateral modules have COMPLEMENTARY GATES: the barrier acts only where the on-final
gate says the aircraft is on the final, the trombone only where the predicted path is more
than 30 deg off the runway course (strictly inside "the gate is closed"). No step is ever
rewritten by both, so "the command flown" is still one module's answer and the order between
them cannot change it. The trombone last is then a free choice made by the value's spelling,
and it pays for it with a 25 deg turn cap, so the load factor it coordinates cannot raise the
stall speed past the margin the floor just held (``control/constraints/trombone.py``).

Diagnostics are merged rather than nested, so a hook record keeps one flat shape however
many modules ran. ``hook_steps`` is the one key they all report and it means the same thing
in each (every module is called on every segment of every row), so it is taken once; any
other collision is a genuine ambiguity and is refused at construction.
"""

from __future__ import annotations

import torch

from control.dynamics.hooks import HOOK_STEPS_KEY, CommandHook, RolloutStateView


class CompositeHook:
    """Apply ``hooks`` in order; the last command returned is the one flown."""

    def __init__(self, hooks: tuple[CommandHook, ...]):
        if len(hooks) < 2:
            raise ValueError("a composite hook composes at least two modules")
        self.hooks = hooks
        self.needs_reference = any(hook.needs_reference for hook in hooks)
        seen: set[str] = set()
        for hook in hooks:
            keys = set(hook.diagnostics()) - {HOOK_STEPS_KEY}
            clash = keys & seen
            if clash:
                raise ValueError(
                    f"{type(hook).__name__} reports diagnostics another module in this "
                    f"composite already reports: {sorted(clash)}; two meanings under one "
                    "key cannot be merged"
                )
            seen |= keys

    def __call__(
        self, state: RolloutStateView, command: torch.Tensor, segment_index: int
    ) -> torch.Tensor:
        for hook in self.hooks:
            command = hook(state, command, segment_index)
        return command

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return _merge(hook.diagnostics() for hook in self.hooks)

    def per_flight_diagnostics(self) -> dict[str, torch.Tensor]:
        return _merge(hook.per_flight_diagnostics() for hook in self.hooks)


def _merge(blocks) -> dict[str, torch.Tensor]:
    """Union the modules' counts; ``hook_steps`` is one shared count, not a sum."""
    merged: dict[str, torch.Tensor] = {}
    for block in blocks:
        for key, value in block.items():
            if key == HOOK_STEPS_KEY and key in merged:
                continue
            merged[key] = value
    return merged
