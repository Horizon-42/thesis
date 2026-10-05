"""Where a flight ends (vocabulary §5.8, D79): the judge's tests, written once. The judge reads them on every state row
of a flight (numpy, `judge._outcome`), the batched executor on every cycle (torch, `executor.Executor.cycle`) and the
single-flight executor on every cycle (Python floats, `single.SingleExecutor.cycle`), so the executor ends a flight where
the judge ends it: an approach crossing of the runway in force R, ground contact, a lined-up crossing of another candidate
with G false (inside that runway's own limit), a dynamics failure (the stall cut-off included) — or the time limit, which
is the executor's own. Each test is written over operations that a Python float, a numpy array and a torch tensor share
(comparisons, ``abs``, ``&``, ``|``).
"""

from __future__ import annotations

from typing import Any

from flight_scenarios.fas_geometry import fas_course_geometry
from ts_transformer.instructions.airport import AirportGeometry, landing_cross_limit_m
from ts_transformer.instructions.spec import VocabularySpec


def runway_lateral_limit_m(geometry: AirportGeometry, index: int, spec: VocabularySpec) -> float:
    """How far off candidate ``index``'s centreline an approach crossing may lie and still be ON the runway: its final
    approach segment's full-scale course half-width at the threshold (FAA Order 8260.58D Formula 3-1-1,
    `flight_scenarios.fas_geometry`; 350 ft = 106.7 m on every candidate here), and never beyond the harvest's limit
    (`landing_cross_limit_m`: 1,000 m, half the spacing to a parallel)."""
    return min(fas_course_geometry(geometry.candidates[index].length_m).course_width_m,
               landing_cross_limit_m(geometry, index, spec.landing_cross_limit_m, spec.parallel_course_delta_deg))


def _false(value: Any) -> Any:
    """``not value`` for a bool, a numpy bool array and a torch bool tensor alike."""
    return value == False  # noqa: E712 — the one negation the three share


def plane_crossed(before_m: Any, past_m: Any) -> Any:
    """A candidate's threshold plane crossed between a row ``before_m`` before it and the next row ``past_m``."""
    return (before_m > 0.0) & (past_m <= 0.0)


def crossing_ends(cross_m: Any, off_course_deg: Any, go_around: Any, runway_in_force: Any, landing_limit_m: Any,
                  on_runway_m: Any, spec: VocabularySpec) -> tuple[Any, Any]:
    """A crossing of a candidate's threshold plane, interpolated ``cross_m`` right of its centreline, at the row after
    it ``off_course_deg`` off its course, under the go-around state ``go_around``; ``runway_in_force``: the candidate is R.
    ``(approach, other)``: an approach crossing of R (G false, lined up, inside the landing screen's lateral limit
    ``landing_limit_m``: what it is — landed, unstable, too high, off the runway — the judge reads) and a crossing of
    another candidate that ends the flight (G false, lined up with it, inside its runway limit ``on_runway_m``,
    `runway_lateral_limit_m`, at any height). While G is true no crossing is an event (D33)."""
    flying = _false(go_around) & (abs(off_course_deg) <= spec.lined_up_deg)
    approach = runway_in_force & flying & (abs(cross_m) <= landing_limit_m)
    other = _false(runway_in_force) & flying & (abs(cross_m) <= on_runway_m)
    return approach, other


def ground_contact(before_m: Any, height_above_threshold_m: Any) -> Any:
    """Below R's threshold elevation while before R's threshold."""
    return (before_m > 0.0) & (height_above_threshold_m < 0.0)


def dynamics_failure(finite: Any, speed_mps: Any, stalled: Any) -> Any:
    """The dynamics left: a state not finite (``finite`` false), no airspeed, or a cycle the dynamics' own stall cut-off
    bound in (``stalled``)."""
    return _false(finite) | (speed_mps <= 0.0) | stalled
