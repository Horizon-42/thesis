"""The command hook: how a constraint module sits inside the rollout.

A hook is called once per control segment, at the segment's start, with the physical
state the backend carries there and the command the network emitted for the segment,
and returns the command actually flown. The rollout then integrates the segment with
that command, so positions are always the dynamics' own — a hook changes what the
aircraft is told to do, never where it is. Gradients flow through the hook's use of the
state (ordinary autograd across segments) and through the command.

Backends expose their state to hooks through :class:`RolloutStateView` so a hook never
sees a backend's private layout (the lagged backends carry actuator states, the
point-mass ones do not).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


#: Every diagnostic a hook reports is namespaced with this, so a reader can pick the hook's
#: counts out of a diagnostic bag that also carries the objective's; the surfaces that
#: publish them (an epoch record, a prediction record) drop it.
HOOK_DIAGNOSTIC_PREFIX = "hook_"
#: The one count EVERY hook reports, and the denominator every other count it reports is a
#: share of (`train.fit_model`'s epoch record and `forecast._per_flight_hook_diagnostics`
#: both divide by it). Both live here because they are part of the protocol below, not of
#: any one module: a composite merges two modules' counts and has to know which key means
#: "steps" rather than "something to add up".
HOOK_STEPS_KEY = f"{HOOK_DIAGNOSTIC_PREFIX}steps"


@dataclass(frozen=True)
class RolloutStateView:
    """The physical state at a segment start, in the chart every module reads.

    ``chart`` is ``[B,7]``: ``e, n, u`` (m, threshold-anchored ENU), ``ve, vn, vu`` (m/s)
    and mass (kg) — ``aerodynamic_model.torch_transport_chart_dynamics``'s state.
    ``actuators`` is ``[B,3]`` (thrust fraction, bank, load factor) the aircraft is
    actually doing, or None for a backend without actuator states. ``duration_s`` is
    ``[B]``, how long the command returned for this segment will be held — a hook that
    reasons in rates must not ask for one faster than the hold can realise.
    ``reference`` is the same view of the UNHOOKED schedule — where the network's own
    commands would have the aircraft now — for hooks that declare ``needs_reference``
    (None otherwise). A segment's command alone does not say which path or speed it was
    trimmed for; the schedule's own rollout does.
    """

    chart: torch.Tensor
    actuators: torch.Tensor | None
    duration_s: torch.Tensor
    reference: RolloutStateView | None = None


class CommandHook(Protocol):
    # Every hook declares it: True on a hook that reads ``RolloutStateView.reference``; the
    # rollout then integrates the network's schedule unhooked alongside (an endpoint
    # rollout per segment more — about 2× the hooked rollout's wall time and memory).
    needs_reference: bool

    def __call__(
        self, state: RolloutStateView, command: torch.Tensor, segment_index: int
    ) -> torch.Tensor:
        """Return the ``[B,3]`` envelope-unit command flown for this segment."""

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Counts accumulated over every call since construction (not objectives)."""

    def per_flight_diagnostics(self) -> dict[str, torch.Tensor]:
        """The same counts, still per ROW: ``[B]`` tensors, one entry per flight.

        :meth:`diagnostics` is these summed over the batch — it is what an epoch record
        reports. A prediction RECORD is one flight, so the record surface reads this one:
        a batch share written onto every record would say the same thing about a flight the
        hook never touched and one it rewrote at every step.
        """
