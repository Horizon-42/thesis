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
    ``actuators`` is ``[B,3]`` — what the aircraft is actually doing in the contract's own
    columns (``control_thrust_parameterization``: the longitudinal command, bank, and a load
    factor or, under the path-angle contract, a path-angle target) — or None for a backend
    without actuator states. A module that wants the LOAD asks the contract's law
    (``outputs/constraints/vertical.VerticalChannel``), never reads column 2. ``duration_s`` is
    ``[B]``, how long the command returned for this segment will be held — a hook that
    reasons in rates must not ask for one faster than the hold can realise.
    ``remaining_s`` is ``[B]``: how much of the SCHEDULE is left from this segment's start,
    this hold included (the sum of this and every later segment's duration). Under a
    CTA-conditioned decoder the schedule's total IS the CTA, so it reads ``T_cta − t`` —
    the time the rest of the flight has to be flown in. A hook that must decide whether the
    path still ahead can absorb the time still ahead cannot get that from one hold, and the
    rollout knows every duration before the first segment is integrated.
    ``reference`` is the UNHOOKED schedule's own rollout, WHOLE: ``[B,N+1,7]`` in the same
    chart, one row per segment BOUNDARY — index 0 the anchor, index ``i`` the start of
    segment ``i``, index ``N`` the schedule's end. It is there for hooks that declare
    ``needs_reference`` (None otherwise). A segment's command alone does not say which path
    or speed it was trimmed for; the schedule's own rollout does. It arrives whole rather
    than one state at a time because the questions that need it are LOOK-AHEAD ones — how
    much path the network still intends to fly — and no per-segment state can answer those;
    a hook that only wants "where the network's commands would have the aircraft now" reads
    column ``segment_index``. It is the same tensor at every call, so a hook may derive its
    own table from it once (at ``segment_index == 0``) and index that thereafter.
    """

    chart: torch.Tensor
    actuators: torch.Tensor | None
    duration_s: torch.Tensor
    remaining_s: torch.Tensor
    reference: torch.Tensor | None = None


class CommandHook(Protocol):
    # Every hook declares it: True on a hook that reads ``RolloutStateView.reference``; the
    # rollout then integrates the network's schedule unhooked as well, once, before the
    # first segment — about 2× the hooked rollout's wall time and memory. It may be a
    # per-INSTANCE value where a module reads the reference only under one of its settings
    # (the trombone's ``trombone_surplus_reference``), so read it off the object.
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

    def diagnostic_labels(self) -> dict[str, str]:
        """Named settings reported NEXT TO the counts: strings, never divided by steps.

        A module with more than one way of computing the same quantity says which one it
        ran, so a record carries the law that produced its numbers instead of leaving a
        reader to infer it from which counts happen to be present. Empty for a module with
        no such choice, and empty for a module sitting on its historical default — a key
        that appears on every record is a key that distinguishes nothing, and adding one to
        the default would rewrite the records of every campaign already on disk.
        """


def per_flight_hook_diagnostics(command_hook: CommandHook) -> list[dict[str, float | str]]:
    """Each flight's own hook counts: ``steps``, then every other count as a share of it —
    the record surface of the protocol above, read by every forecast that runs a hook.

    The same normalisation ``train.fit_model`` writes into an epoch record
    (``value / hook_steps``), one level down — an epoch reports the batch, a prediction
    record reports the flight. Keys drop the ``hook_`` prefix and arrive camelCased like
    every other ``source`` field. A module's LABELS (which named variant of itself it ran)
    join the same bag under the same naming, undivided: they are strings, and a share of a
    name means nothing.
    """
    counts = {
        name: value.tolist()
        for name, value in command_hook.per_flight_diagnostics().items()
    }
    labels = {
        _camel_case(name.removeprefix(HOOK_DIAGNOSTIC_PREFIX)): value
        for name, value in command_hook.diagnostic_labels().items()
    }
    steps = counts.pop(HOOK_STEPS_KEY)
    return [
        {
            "steps": row_steps,
            **{
                _camel_case(name.removeprefix(HOOK_DIAGNOSTIC_PREFIX)): value[row] / row_steps
                for name, value in counts.items()
            },
            **labels,
        }
        for row, row_steps in enumerate(steps)
    ]


def _camel_case(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(word.title() for word in rest)
