"""Roll piecewise-constant controls forward through any step-based simulator.

An optimizer emits ``N`` piecewise-constant control segments spanning ``[0, T]``.
Replaying them means stepping a simulator across each segment — and that loop is
identical wherever it appears (the scenario optimizer's "simulator states", the
backend's CZML playback rollout, …). It lives here once, in neutral terms
(``GeodeticState`` in, ``GeodeticState`` out, over anything exposing
``step(state, control, dt)``); each caller maps the yielded samples onto its own
output type. This module sits in the ``aerodynamic_model`` layer and depends only
on ``common`` — it must not import the optimizer or backend above it.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Protocol, Sequence

from aerodynamic_model.common import GeodeticState


class SupportsStep(Protocol):
    """Any simulator that advances one state by ``dt`` under a constant control."""

    def step(self, state: GeodeticState, control: Any, dt: float) -> GeodeticState: ...


class RolloutSample(NamedTuple):
    t: float
    state: GeodeticState
    control: Any
    segment_index: int


def rollout_piecewise_constant(
    simulator: SupportsStep,
    initial_state: GeodeticState,
    controls: Sequence[Any],
    total_duration: float,
    *,
    integrator_dt: float,
    output_dt: float | None = None,
    truncate_on_envelope_exit: bool = False,
    on_truncate: Callable[[float, ValueError], None] | None = None,
) -> list[RolloutSample]:
    """Integrate ``controls`` (one per equal segment of ``[0, total_duration]``) forward.

    Segment ``i`` holds ``controls[i]`` over ``[i·Δ, (i+1)·Δ]`` with
    ``Δ = total_duration / len(controls)``. Within a segment the simulator is stepped at
    ``integrator_dt``, clamped so a step never crosses the segment boundary (or
    ``output_dt`` when that is finer). A sample is emitted at ``t=0``, then every
    ``output_dt`` and at every segment boundary; ``output_dt=None`` emits after every step.

    A step that leaves the flight envelope raises ``ValueError`` by default (fail loudly).
    Pass ``truncate_on_envelope_exit=True`` to instead return the partial samples (a viz
    aid — a real optimized solution lands on target and never exits); ``on_truncate`` is
    then called with ``(t, exc)`` so the truncation isn't silent.
    """
    segment_count = len(controls)
    if segment_count == 0:
        raise ValueError("controls must contain at least one segment")
    segment_duration = total_duration / segment_count
    step_dt_cap = integrator_dt if output_dt is None else min(integrator_dt, output_dt)

    state = initial_state
    t = 0.0
    samples = [RolloutSample(0.0, state, controls[0], 0)]

    for segment_index, control in enumerate(controls):
        segment_end = (segment_index + 1) * segment_duration
        next_output_t = t + output_dt if output_dt is not None else 0.0
        while t < segment_end - 1e-9:
            step_dt = min(step_dt_cap, segment_end - t)
            try:
                state = simulator.step(state, control, step_dt)
            except ValueError as exc:
                if not truncate_on_envelope_exit:
                    raise
                if on_truncate is not None:
                    on_truncate(t, exc)
                return samples
            t += step_dt
            at_segment_end = t >= segment_end - 1e-9
            if output_dt is None or t >= next_output_t - 1e-9 or at_segment_end:
                samples.append(RolloutSample(t, state, control, segment_index))
                next_output_t = t + output_dt if output_dt is not None else 0.0
    return samples
