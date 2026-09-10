"""Several constraint modules on one rollout, applied in a fixed order.

``outputs/dynamics/hooks.py`` gives the rollout ONE hook per segment, and that is right:
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
the two lateral modules have COMPLEMENTARY GATES, and the builder makes them so: the
trombone may engage only where the HARD on-final gate has not opened on the flight, and a
barrier composed with a trombone is built ``confine_to_hard_gate=True``, i.e. it acts only
INSIDE that hard gate (under ``soft`` its blend is multiplied by the hard gate; inside the
gate it is the standalone soft barrier to the bit). The soft gate alone was NOT the
complement — its 30–40° alignment shoulder is exactly the band where the trombone is
admitted, and there the barrier used to blend a correction the trombone then overwrote
(2026-09-09 review, A-4). No step is now rewritten by both, so "the command flown" is one
module's answer and the order between them cannot change it. The trombone last is then a
free choice made by the value's spelling, and it pays for it with a 15 deg turn cap, so the
load factor it coordinates cannot raise the stall speed past the margin the floor just held
(``outputs/constraints/trombone.py``).

**The hook-free reference rollout is a member of the composite, not a private trick of one
module.** ``needs_reference`` is the OR of the members', so under
``trombone_surplus_reference="reference-rollout"`` the value ``barrier+speed-floor+trombone``
gains a fourth thing the rollout does: the network's own schedule is integrated UNHOOKED as
well (``outputs/dynamics/hooks.py``), once, before the first hooked segment, and every
member sees it as ``RolloutStateView.reference``. **Cost**: one more integration of the same
schedule, paid ONCE and not per step — about 2x the SEGMENTED rollout's wall time and memory
(at predict the dense re-integration runs on top of that and is untouched, so the whole
predict step grows by less than that). The trombone then derives its own per-segment table
from it at ``segment_index == 0`` and indexes that table thereafter, so the per-step cost of
the axis is an index. Under the default ``beeline`` no member asks for it and nothing extra
is integrated.

Diagnostics are merged rather than nested, so a hook record keeps one flat shape however
many modules ran. ``hook_steps`` is the one key they all report and it means the same thing
in each (every module is called on every segment of every row), so it is taken once; any
other collision is a genuine ambiguity and is refused at construction. The labels
(``diagnostic_labels`` — which named variant a module ran) merge the same way and under the
same refusal: they are strings and share the counts' namespace, so a collision there is the
same ambiguity.
"""

from __future__ import annotations

from typing import Iterable, TypeVar

import torch

from ts_transformer.outputs.dynamics.hooks import HOOK_STEPS_KEY, CommandHook, RolloutStateView

#: Counts (tensors) and labels (strings) merge by the same rule, under the same refusal.
_Reported = TypeVar("_Reported", torch.Tensor, str)


class CompositeHook:
    """Apply ``hooks`` in order; the last command returned is the one flown."""

    def __init__(self, hooks: tuple[CommandHook, ...]):
        if len(hooks) < 2:
            raise ValueError("a composite hook composes at least two modules")
        self.hooks = hooks
        self.needs_reference = any(hook.needs_reference for hook in hooks)
        seen: set[str] = set()
        for hook in hooks:
            keys = (set(hook.diagnostics()) | set(hook.diagnostic_labels())) - {HOOK_STEPS_KEY}
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

    def diagnostic_labels(self) -> dict[str, str]:
        return _merge(hook.diagnostic_labels() for hook in self.hooks)


def _merge(blocks: Iterable[dict[str, _Reported]]) -> dict[str, _Reported]:
    """Union the modules' counts (or labels); ``hook_steps`` is one shared count, not a sum."""
    merged: dict[str, _Reported] = {}
    for block in blocks:
        for key, value in block.items():
            if key == HOOK_STEPS_KEY and key in merged:
                continue
            merged[key] = value
    return merged
