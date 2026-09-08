"""Two constraint modules on one rollout, applied in a fixed order.

``control/dynamics/hooks.py`` gives the rollout ONE hook per segment, and that is right:
"the command flown" has to be a single answer. Composition therefore happens here, not in
the rollout — each module in turn is handed the SAME state (the segment's start) and the
command the previous one returned, and the last answer is what the rollout integrates.

The order is part of the vocabulary value, not a free choice
(``config.CONTROL_HOOK_MEMBERS``). ``barrier+speed-floor`` means the barrier first: it sets
the bank and re-coordinates the load factor, and the speed floor then reads THAT load
factor for its stall speed and sets the thrust. Reversed, the floor would price a manoeuvre
the barrier is about to change. The two modules write disjoint channels — bank and load
against thrust — which is what makes this combination well defined and every other one
absent from the vocabulary.

Diagnostics are merged rather than nested, so a hook record keeps one flat shape whether
one module ran or two. ``hook_steps`` is the one key both report and it means the same
thing in both (every module is called on every segment of every row), so it is taken once;
any other collision is a genuine ambiguity and is refused at construction.
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
