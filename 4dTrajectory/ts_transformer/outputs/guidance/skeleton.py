"""The procedure skeleton in a flight's own chart.

`flight_scenarios.procedure_final.procedure_skeleton` reads the coded RNAV(GPS) approach —
the final's fixes with their altitude floors, the glidepath, every published transition —
in geodetic coordinates. This module puts it where the plan path measures everything:
the flight's chart (`FlightSeries.frame`), and the runway axes about the threshold
(`geometry.final_approach_geometry`: ``d`` = distance to go along the course, ``xt`` =
cross-track, + right of the inbound course). The threshold is `FlightSeries.target_chart`,
never the chart origin (review C-11), and the course is the scenario target's — the one
`final_approach_arrays` hands every consumer of the on-final gate.

The document's own threshold (the CIFP landing threshold point) and the target the arrival
manifest carries are two renderings of one point that round differently — 0.05–0.22 m over
the runways in service, 2.98 m on KRDU 32, 39.45 m on KSMF 35R (the optimizer's measurement,
`scenario_optimization._require_procedure_threshold_agrees`). The skeleton is validated
against the target at the same 150 m tolerance rather than snapped onto the document: a
displaced threshold (KSJC 12L, 390 m) fails loudly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from geokit import METRES_PER_DEG_LAT, NM_M

from flight_scenarios.procedure_final import (
    DEFAULT_PROCEDURE_ROOT,
    ProcedureFix,
    procedure_skeleton,
)

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

#: RNP APCH initial / intermediate lateral accuracy: the box corridor a published pre-final
#: leg is flown inside. MUST match `aeroviz_backend.procedure_segments._DEFAULT_RNP_NM`
#: (the optimizer's constraint bridge is not on this package's import path).
RNP_HALF_WIDTH_M = 1.0 * NM_M
#: How far the document's threshold may sit from the manifest's target before the two are
#: not the same runway. MUST match `scenario_optimization._FRAME_ANCHOR_TOLERANCE_M`.
THRESHOLD_TOLERANCE_M = 150.0


def runway_axes(
    e: np.ndarray, n: np.ndarray, *, course_rad: float, target_e: float, target_n: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(d, xt)`` of chart positions about the threshold — the NumPy twin of
    `final_approach_geometry.runway_axes`, measured from `target_chart` (C-11)."""
    de = np.asarray(e, dtype=np.float64) - target_e
    dn = np.asarray(n, dtype=np.float64) - target_n
    ue, un = math.cos(course_rad), math.sin(course_rad)
    return -(de * ue + dn * un), de * un - dn * ue


@dataclass(frozen=True)
class ChartFix:
    """One coded fix in the flight's chart and in runway axes."""

    ident: str
    role: str
    e: float
    n: float
    d: float                        # distance to go along the course, m (+ before the threshold)
    xt: float                       # cross-track, m (+ right of the inbound course)
    altitude_min_m: float | None    # m MSL
    altitude_max_m: float | None    # m MSL
    speed_max_mps: float | None


@dataclass(frozen=True)
class RunwaySkeleton:
    """The coded approach onto this flight's runway, in this flight's chart."""

    procedure_uid: str
    course_rad: float               # the runway course, math-ENU (the direction of travel on final)
    glidepath_tan: float            # tan of the coded glidepath descent
    aim_altitude_m: float           # the chart origin's altitude (m MSL): every chart height is above it
    target_e: float                 # the threshold in the chart
    target_n: float
    final: tuple[ChartFix, ...]     # IF → FAF → (step-down fixes) → MAPt, in flying order
    transitions: tuple[tuple[ChartFix, ...], ...]   # each IAF → … → its merge fix on the final

    @property
    def faf(self) -> ChartFix:
        return next(fix for fix in self.final if fix.role == "FAF")

    def axes(self, e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return runway_axes(
            e, n, course_rad=self.course_rad, target_e=self.target_e, target_n=self.target_n
        )

    def published_legs(self) -> list[tuple[ChartFix, ChartFix]]:
        """Every coded pre-final leg: the transitions' legs and the final's legs up to the FAF."""
        legs: list[tuple[ChartFix, ChartFix]] = []
        for branch in self.transitions:
            legs.extend(zip(branch[:-1], branch[1:]))
        pre_final = [fix for fix in self.final if fix.d >= self.faf.d]
        legs.extend(zip(pre_final[:-1], pre_final[1:]))
        return legs

    def distance_to_published_m(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Horizontal distance from chart points ``[N]`` to the nearest published pre-final
        leg; ``inf`` where the procedure codes no pre-final leg at all."""
        e = np.asarray(e, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        best = np.full(e.shape, np.inf)
        for start, end in self.published_legs():
            best = np.minimum(best, _segment_distance(e, n, start, end))
        return best

    def floor_altitude_m(self, d: float) -> float | None:
        """The coded at-or-above altitude that still applies ``d`` metres out: the floor of
        the nearest final fix at or inside ``d``, or None where none is coded."""
        ahead = [fix for fix in self.final if fix.d <= d and fix.altitude_min_m is not None]
        if not ahead:
            return None
        return max(ahead, key=lambda fix: fix.d).altitude_min_m

    def ceiling_altitude_m(self, d: float) -> float | None:
        """The coded at-or-below altitude (an ``at`` or a block's top) at the next final fix
        ahead of ``d``, or None where none is coded there."""
        ahead = [fix for fix in self.final if fix.d <= d]
        if not ahead:
            return None
        return max(ahead, key=lambda fix: fix.d).altitude_max_m

    def speed_limit_ahead_mps(self, d: float) -> float | None:
        """The tightest coded speed limit at a FINAL fix at or inside ``d``, or None (no
        document on this machine codes one). A transition fix's limit applies on its own
        leg, which an along-course distance cannot place."""
        limits = [
            fix.speed_max_mps for fix in self.final
            if fix.d <= d and fix.speed_max_mps is not None
        ]
        return min(limits) if limits else None


def _segment_distance(e: np.ndarray, n: np.ndarray, a: ChartFix, b: ChartFix) -> np.ndarray:
    ab_e, ab_n = b.e - a.e, b.n - a.n
    length2 = ab_e * ab_e + ab_n * ab_n
    if length2 <= 0.0:
        return np.hypot(e - a.e, n - a.n)
    t = np.clip(((e - a.e) * ab_e + (n - a.n) * ab_n) / length2, 0.0, 1.0)
    return np.hypot(e - (a.e + t * ab_e), n - (a.n + t * ab_n))


def _chart_fix(
    fix: ProcedureFix, series: FlightSeries, *, course_rad: float, target_e: float, target_n: float
) -> ChartFix:
    frame = series.frame
    east = (fix.lon_deg - frame.lon0) * frame.m_per_deg_lon
    north = (fix.lat_deg - frame.lat0) * METRES_PER_DEG_LAT
    e, n = frame.from_world_horizontal(east, north)
    d, xt = runway_axes(
        np.array([e]), np.array([n]), course_rad=course_rad, target_e=target_e, target_n=target_n
    )
    return ChartFix(
        ident=fix.ident, role=fix.role, e=float(e), n=float(n),
        d=float(d[0]), xt=float(xt[0]),
        altitude_min_m=fix.altitude_min_m, altitude_max_m=fix.altitude_max_m,
        speed_max_mps=fix.speed_max_mps,
    )


def runway_skeleton(
    series: FlightSeries, *, root: str | Path = DEFAULT_PROCEDURE_ROOT
) -> RunwaySkeleton:
    """The coded approach onto ``series``' runway, in its chart.

    Raises when the runway has no RNAV(GPS) document, when the document's threshold is not
    the manifest's target (`THRESHOLD_TOLERANCE_M`), or when the final codes no FAF.
    """
    scenario = series.scenario
    runway = str(scenario.source.get("runway") or "")
    if not runway:
        raise ValueError(f"{series.flight_id}: the scenario names no runway")
    document = procedure_skeleton(series.airport, runway, root=root)
    target = scenario.target
    axes = dict(
        course_rad=float(target.psi),
        target_e=float(series.target_chart[0]),
        target_n=float(series.target_chart[1]),
    )
    final = tuple(_chart_fix(fix, series, **axes) for fix in document.final.fixes)
    mapt = final[-1]
    offset = math.hypot(mapt.e - axes["target_e"], mapt.n - axes["target_n"])
    if offset > THRESHOLD_TOLERANCE_M:
        raise ValueError(
            f"{series.flight_id}: {document.procedure_uid}'s threshold sits {offset:.0f} m from "
            f"the manifest's target for {series.airport} {runway}; not the same runway"
        )
    return RunwaySkeleton(
        procedure_uid=document.procedure_uid,
        # One source with `final_approach_arrays`: the target's gamma is the coded DESCENT.
        glidepath_tan=math.tan(-float(target.gamma)),
        aim_altitude_m=float(series.frame.alt0),
        final=final,
        transitions=tuple(
            tuple(_chart_fix(fix, series, **axes) for fix in branch.fixes)
            for branch in document.transitions
        ),
        **axes,
    )


class SkeletonCache:
    """One skeleton per (airport, runway), read on first use — beside `runway_skeleton`,
    which it caches (moved from `outputs/plan/strategy.py` 2026-09-16: the control path's
    plan token reads skeletons too, and a cache of this module's reading is this module's)."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], RunwaySkeleton] = {}

    def for_series(self, series: FlightSeries) -> RunwaySkeleton:
        key = (series.airport, str(series.scenario.source.get("runway")))
        if key not in self._by_key:
            self._by_key[key] = runway_skeleton(series)
        return self._by_key[key]
