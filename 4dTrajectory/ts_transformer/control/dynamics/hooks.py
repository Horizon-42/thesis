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
