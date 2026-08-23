"""Segment specifications + per-segment constraint assembly.

This module is **provided wiring**: the data records that describe one approach leg, and the
function that turns a leg + its state nodes into a dict of violation arrays by calling the
primitives you implement in geometry/lateral/vertical. Read it to see how the pieces compose —
nothing here is a TODO (it will start working as soon as the primitives are done).

A trajectory is split into four kinds of leg (design-doc §3):

    FEEDER → INITIAL → INTERMEDIATE → FINAL_LPV
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import lateral, mathx, vertical
from . import geometry as geo
from .lateral import DEFAULT_K_MARGIN
from .state import altitudes, east, flight_path_angles, north


class SegmentKind(Enum):
    FEEDER = "feeder"
    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    FINAL_LPV = "final_lpv"


@dataclass(frozen=True)
class StepDown:
    """A step-down fix inside a (non-final) segment: at-or-above ``min_alt_m`` until ``s``."""

    s_from_start_m: float
    min_alt_m: float


# Glidepath vertical-window half-tolerances (metres) for the OPTIMIZED REFERENCE path — the
# SINGLE source of truth (the backend's ``build_constraint_segments`` reuses these, so there is
# only one default). Below is the dangerous side, hence tighter, but not so tight that an
# otherwise-flyable approach becomes infeasible.
DEFAULT_GLIDEPATH_BELOW_M = 60.0
DEFAULT_GLIDEPATH_ABOVE_M = 120.0

# FAA standard maximum intercept angle onto the final approach course at the PFAF (degrees,
# design-doc §4.4). Single source for the optimizer's intercept constraint and the backend's
# procedure data-validation warning.
STANDARD_INTERCEPT_MAX_DEG = 30.0


@dataclass(frozen=True)
class LpvFinalSpec:
    """The FAS-data-block geometry for the LPV final (all positions in the ``(n, e)`` frame).

    ``ltp_ne`` is the target/origin ``(0, 0)``. ``course_width_m`` = ``max(350 ft,
    tan(1.5°)·d_GARP(LTP))``. ``below_m``/``above_m`` are the glidepath window half-tolerances
    (keep ``below_m`` tight — below the glidepath is the dangerous side).

    ``d_faf_m``/``prefaf_floor_m`` drive the VERTICAL GATE for the flexible FAC join (README
    §4b): with ``d_faf_m`` set, the glidepath window binds only from the FAF toward the runway
    (``d ≤ d_faf_m``); upstream of the FAF the published pre-FAF floor ``prefaf_floor_m`` (the
    FAF crossing minimum, m MSL) applies instead. Both default to ``None`` — gate off, the
    window binds on every final node exactly as before.
    """

    ltp_ne: np.ndarray
    fpap_ne: np.ndarray
    garp_ne: np.ndarray
    course_width_m: float
    tdze_m: float
    tch_m: float
    gpa_deg: float = 3.0
    below_m: float = DEFAULT_GLIDEPATH_BELOW_M
    above_m: float = DEFAULT_GLIDEPATH_ABOVE_M
    d_faf_m: float | None = None
    # NOTE: as a SEGMENT ROW the pre-FAF floor is emitted only when d_faf_m gates it
    # (below); without d_faf_m the field still matters — the optimizer's
    # _first_leg_entry_floor_m reads it as the transition phase's entry floor.
    prefaf_floor_m: float | None = None


@dataclass
class SegmentSpec:
    """One approach leg and the constraints that apply along it.

    ``start_ne`` / ``end_ne`` are the leg endpoints in the ``(n, e)`` frame. For non-final legs
    set ``halfwidth_m`` (the RNP containment half-width) and any ``step_downs``; for the final
    leg set ``lpv``. ``k_margin`` is the lateral design-margin fraction.
    """

    kind: SegmentKind
    start_ne: np.ndarray
    end_ne: np.ndarray
    start_ident: str = ""
    end_ident: str = ""
    k_margin: float = DEFAULT_K_MARGIN
    halfwidth_m: float | None = None          # box corridor (non-final)
    step_downs: list[StepDown] = field(default_factory=list)
    base_floor_m: float = 0.0
    max_descent_deg: float | None = None
    lpv: LpvFinalSpec | None = None

    def __post_init__(self) -> None:
        self.start_ne = np.asarray(self.start_ne, dtype=float)
        self.end_ne = np.asarray(self.end_ne, dtype=float)
        if self.kind is SegmentKind.FINAL_LPV and self.lpv is None:
            raise ValueError("FINAL_LPV segment needs an LpvFinalSpec (.lpv)")
        if self.kind is not SegmentKind.FINAL_LPV and self.halfwidth_m is None:
            raise ValueError(f"{self.kind.value} segment needs halfwidth_m (the RNP corridor)")


def segment_violations_from_components(seg: SegmentSpec, n, e, h, gamma, *, include_lateral=True) -> dict:
    """Evaluate every constraint of ``seg`` from the state's **scalar components**.

    ``n``, ``e``, ``h``, ``gamma`` are the node coordinates as scalar components — a NumPy array
    of node values **or** a CasADi vector (the optimizer feeds the latter). Returns
    ``{name: violation}`` with the package convention ``≤ 0 ⇔ satisfied``. Pure composition of the
    backend-agnostic primitives — no new math, no NumPy-only ops, so it serves both backends.

    ``include_lateral=False`` drops the **non-final** box corridor (the centerline containment),
    keeping only the vertical floor + descent cap. The multiphase optimiser uses this so the
    pre-final horizontal path is free (only altitude + a FAF intercept angle are constrained); the
    final LPV corridor is the protected segment and is always kept.
    """
    out: dict = {}
    tag = f"{seg.kind.value}[{seg.start_ident}->{seg.end_ident}]"

    if seg.kind is SegmentKind.FINAL_LPV:
        lpv = seg.lpv
        right, left = lateral.lpv_corridor_violation(n, e, lpv, seg.k_margin)
        out[f"{tag}.lateral_right"] = right
        out[f"{tag}.lateral_left"] = left
        # Along-course distance BACK from the LTP (0 at threshold, + toward the PFAF), measured
        # on the SAME GARP→LTP axis the lateral corridor uses — one final-approach-course axis
        # for both lateral and vertical, whatever the leg's own start fix is.
        d_to_ltp = lateral.fac_distance_to_ltp(n, e, lpv)
        low, high = vertical.glidepath_window_violation(h, d_to_ltp, lpv)
        if lpv.d_faf_m is not None:
            # Vertical gate for the flexible FAC join (README §4b): PUBLISHED semantics — the
            # glidepath window binds only from the FAF toward the runway; upstream of the FAF
            # (the early-join stretch) the published pre-FAF floor applies instead. The lateral
            # join slides; the vertical rules stay tied to the published geography. Same
            # mathx.if_else machinery as the step-down staircase (both backends); -1.0 is the
            # always-satisfied metre-scale row for the inactive side.
            after_faf = d_to_ltp <= lpv.d_faf_m
            low = mathx.if_else(after_faf, low, -1.0)
            high = mathx.if_else(after_faf, high, -1.0)
            if lpv.prefaf_floor_m is not None:
                out[f"{tag}.prefaf_floor"] = mathx.if_else(
                    after_faf, -1.0, lpv.prefaf_floor_m - h
                )
        out[f"{tag}.glidepath_low"] = low
        out[f"{tag}.glidepath_high"] = high
    else:
        if include_lateral:
            # Report labels are FLIGHT-direction sides. cross_track's positive side is
            # LEFT of its axis; a box leg's axis (start→end) IS the flight direction, so
            # the +e_xt row is the LEFT edge here — the final's axis (GARP→LTP) OPPOSES
            # flight, which is why the LPV branch above binds the same pair the other
            # way round. Feasibility is unaffected (the rows are symmetric); only the
            # report's side naming was inverted for box legs.
            left, right = lateral.box_corridor_violation(
                n, e, seg.start_ne, seg.end_ne, seg.halfwidth_m, seg.k_margin
            )
            out[f"{tag}.lateral_right"] = right
            out[f"{tag}.lateral_left"] = left
        # From the segment start, projected onto the LEG CENTERLINE. With
        # include_lateral=False the path is free to leave that centerline, yet the
        # step-down staircase stays keyed to this projection — a wide off-axis path can
        # advance s past a step-down and shed its floor at the wrong geographic point.
        # Accepted modeling choice for the pre-final legs (the published floors are
        # coded along the procedure's own track).
        s = geo.along_track(n, e, seg.start_ne, seg.end_ne)
        out[f"{tag}.floor"] = vertical.moc_floor_violation(
            h, s, seg.step_downs, seg.base_floor_m
        )

    if seg.max_descent_deg is not None:
        out[f"{tag}.descent"] = vertical.descent_gradient_violation(gamma, seg.max_descent_deg)

    return out


def segment_violations(seg: SegmentSpec, nodes: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate every constraint of ``seg`` on its state ``nodes`` (a ``(k, 7)`` array).

    Thin NumPy wrapper: extracts the scalar columns and calls
    :func:`segment_violations_from_components`.
    """
    return segment_violations_from_components(
        seg, north(nodes), east(nodes), altitudes(nodes), flight_path_angles(nodes)
    )
