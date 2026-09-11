"""The eight plan parameters read off an observed track (design §3).

Every parameter has a definition, a range set by the procedure and the aircraft, and this
extractor — the supervision of the plan head and the input of the oracle-ceiling test. The
track is the OBSERVED rows from the anchor on (`FlightSeries.values`, never the fitted
supervision tail, which is a construction); the remaining path is the package's one
definition (`approach_difficulty.remaining_path_profile_m`, which does read the tail to the
threshold) so a parameter binned on distance-to-go agrees with the anytime grid; the join
is the measurement's own gate (`final_approach_geometry.truth_final_gate`), not a second
notion of "on the final".

Operating parameters (predicted): ``T_s`` the time from the anchor to the threshold;
``V_mid_mps`` the ground speed held before deceleration; ``d_decel_m`` the remaining path at
which the speed first drops below ``V_final + DECEL_MARGIN_MPS``; ``V_final_mps`` the final
approach ground speed; ``h_capture_m`` the chart height (above the threshold aim point) at
which the final is captured LATERALLY — the glidepath is usually captured later.
Route parameters (assigned, or a distribution — never a point output): ``d_join_m`` the
remaining path at which the flight becomes established; ``side`` which side the base leg
comes from; ``L_pre_m`` the path flown before the join.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from aircraft.aero_params import stall_speed_ms
from ts_transformer.config import CONTROL_SPEED_FLOOR_MARGIN_DEFAULT
from ts_transformer.data.approach_difficulty import remaining_path_profile_m
from ts_transformer.data.channels import IDX, states_from_channels
from ts_transformer.data.dataset import truth_duration_s
from ts_transformer.geometry.final_approach_geometry import truth_final_gate
from ts_transformer.outputs.plan.skeleton import RNP_HALF_WIDTH_M, RunwaySkeleton

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

PLAN_PARAMETERS = (
    "T_s", "V_mid_mps", "d_decel_m", "V_final_mps", "h_capture_m", "d_join_m", "side", "L_pre_m",
)
OPERATING_PARAMETERS = PLAN_PARAMETERS[:5]
ROUTE_PARAMETERS = PLAN_PARAMETERS[5:]

#: `V_mid` is the MEAN ground speed over the path flown BEFORE the deceleration point —
#: its length over its time — "the speed held before deceleration" as the number the time
#: closure needs (a median over-reads a varying speed: the first readout, 2026-09-10, read a
#: 10–20 km band median and the schedule then missed the flown time by 61 s on the vectored
#: stratum). A flight already decelerating at the anchor (fewer than this many rows before
#: the point, or no point) holds what it has, its anchor ground speed.
V_MID_MIN_ROWS = 3
#: `d_decel` is where the ground speed first drops below the scenario TARGET speed (the
#: type's approach reference speed the threshold state carries) plus this — the straight-in
#: residual readout's ``target_plus_10`` threshold, `experiments.straight_in_residual_readout`.
DECEL_MARGIN_MPS = 10.0
#: The side is the sign of the widest cross-track over this window before the join (the
#: base leg; the widest row cannot cancel the way a mean over a converging intercept can),
#: and "straight-in" (0) when that width is under the base-leg floor.
SIDE_WINDOW_S = 60.0
SIDE_MIN_OFFSET_M = 500.0
#: A join is "on a published transition" when the track over this window before it stays
#: inside the RNP box of a coded pre-final leg.
JOIN_WINDOW_S = 60.0
#: The stall margin the ranges read the aircraft's speed floor at — the package's one
#: margin, at the sea-level stall speed.
STALL_MARGIN = CONTROL_SPEED_FLOOR_MARGIN_DEFAULT
#: The pre-final path as at most this many fly-by WAYPOINTS (design §3b / §8, the fixed-K
#: representation, 2026-09-11): each of the path's turns is the FIX where the two legs it
#: joins intersect — the RNAV way of coding a route, and what a radar vector "fly
#: heading X until Y" comes to. A fix is absolute, so a turn flown at another radius
#: than the truth's cuts or widens one corner and drifts nothing after it (a turn given
#: as a heading change did: the vectored route came out 4.4 km longer than the plan at
#: the median, 2026-09-11 smoke); a turn's END as the point was ambiguous about the leg
#: heading it starts. A turn over `FLY_BY_SPLIT_DEG` (a reversal: parallel legs meet at
#: infinity) is two fixes about its mid-tangent. A plan with fewer turns pads with none
#: (a fix on the line between its neighbours is no turn), so the head's output stays
#: fixed-size and continuous. The heading is read over ±`TURN_HEADING_HALF_WINDOW_ROWS`
#: rows (ADS-B position noise over one 2 s step is a few degrees of heading); a row
#: turns where the smoothed heading rate exceeds `TURN_RATE_MIN_DEG_S` (a 10° bank at
#: 100 m/s is 1.7°/s; a 5° bank at 130 m/s is 0.6°/s and is NOT read as a turn — a 0.5°/s
#: rule over ±3 rows was tried to catch such wide vectors and measured worse on the
#: 48-flight smoke, chamfer 85 → 106 m and twice the fixes dropped, 2026-09-11); a
#: one-row lull inside a turn is bridged; a heading change under
#: `TURN_MIN_DEG` is not a turn. Where more than `MAX_WAYPOINTS` fixes remain, those of
#: the largest turns are kept and the rest COUNTED (`waypoints_dropped`).
MAX_WAYPOINTS = 4
FLY_BY_SPLIT_DEG = 120.0
TURN_HEADING_HALF_WINDOW_ROWS = 2
TURN_RATE_MIN_DEG_S = 1.0
TURN_MIN_DEG = 5.0
#: A fix this close to the centreline is ON it: the fix of the turn onto the final.
ON_COURSE_FIX_M = 300.0
#: Two nearly parallel legs meet far away: a fix farther from its turn's midpoint than
#: three turn radii plus this is the turn's midpoint instead (a fly-over point).
FIX_GUARD_M = 500.0


@dataclass(frozen=True)
class PlanLabels:
    """The eight parameters of one flight from one anchor, and their ranges.

    A ``None`` is a parameter the track does not define (no join; never slower than the
    deceleration threshold; a join BEFORE the anchor, which censors the capture height, the
    side and the pre-final path — the window never saw them): never silently zero.
    """

    T_s: float
    V_mid_mps: float
    d_decel_m: float | None
    V_final_mps: float
    h_capture_m: float | None
    d_join_m: float | None
    side: int | None                # +1 right, −1 left, 0 straight-in; None = censored
    L_pre_m: float | None
    #: The truth gate is already open at the anchor: the flight joined before the window
    #: (59 % of KRDU val at L−1, 2026-09-10). `d_join_m` is then the anchor's remaining path
    #: and the three route/capture parameters are None.
    join_at_anchor: bool
    #: `V_mid_mps` is the anchor ground speed because the flight was already decelerating
    #: at the anchor (fewer than `V_MID_MIN_ROWS` rows before the deceleration point).
    v_mid_from_anchor: bool
    #: Whether the 60 s before the join lie inside a published pre-final leg's RNP box —
    #: a coded transition rather than a radar vector. None when the flight has no join
    #: or the procedure codes no pre-final leg.
    join_on_transition: bool | None
    #: ``(low, high)`` per parameter, from the procedure and the aircraft; None = unbounded
    #: on that side (the documents on this machine code no speed limit, so `V_mid`'s
    #: ceiling is None everywhere).
    ranges: dict[str, tuple[float | None, float | None]]
    #: Where the track sits against the skeleton at the anchor, for the readout.
    remaining_path_at_anchor_m: float
    ground_speed_at_anchor_mps: float
    #: The pre-final path's fly-by waypoints, ``((e, n), …)`` in the chart, at most
    #: `MAX_WAYPOINTS`, in path order; None where the pre-final path is censored (the
    #: join before the anchor, or no join). `waypoints_dropped` counts the fixes beyond
    #: the cap.
    waypoints: tuple[tuple[float, float], ...] | None = None
    waypoints_dropped: int = 0
    #: Per waypoint, ``(remaining path at the turn's middle, the median ground speed over
    #: the turn's rows)`` —
    #: a vector is a heading AND a speed instruction; the route's fly-by radius is this
    #: speed's and the schedule runs through these points under the waypoints route.
    waypoint_speeds: tuple[tuple[float, float], ...] | None = None
    #: Per waypoint, the time from the anchor to the turn's middle (s): the lead a single
    #: step must predict (design v5 §9 step 3a).
    waypoint_times_s: tuple[float, ...] | None = None
    #: Per waypoint, the chart height at the turn's middle (m above the aim point, as
    #: `h_capture_m`): the altitude an instruction carries.
    waypoint_heights_m: tuple[float, ...] | None = None

    def parameters(self) -> dict[str, float | int | None]:
        return {name: getattr(self, name) for name in PLAN_PARAMETERS}

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ranges"] = {name: list(bounds) for name, bounds in self.ranges.items()}
        out["waypoints"] = None if self.waypoints is None else [list(fix) for fix in self.waypoints]
        out["waypoint_speeds"] = None if self.waypoint_speeds is None else [list(p) for p in self.waypoint_speeds]
        out["waypoint_times_s"] = None if self.waypoint_times_s is None else list(self.waypoint_times_s)
        out["waypoint_heights_m"] = None if self.waypoint_heights_m is None else list(self.waypoint_heights_m)
        return out


def _ground_speeds(series: FlightSeries, rows: slice) -> np.ndarray:
    """Physical horizontal ground speed (m/s) of the observed rows, from the chart velocities
    through the same inverse the export uses (never the chart derivatives themselves)."""
    states = states_from_channels(
        series.times[rows], series.values[rows], series.frame,
        mass_kg=float(series.scenario.initial.m),
    )
    return np.array([state.V * np.cos(state.gamma) for _t, state in states], dtype=np.float64)


def _leg_line(e, n, rows, heading_fallback: float, row_fallback: int) -> tuple[np.ndarray, float]:
    """A point on a held leg and its heading: the mean of the leg's rows and the heading
    between its first and last; a leg of fewer than two rows (one turn straight into the
    next) is the smoothed heading at the boundary, through that row."""
    if len(rows) >= 2:
        a, b = int(rows[0]), int(rows[-1])
        return np.array([e[rows].mean(), n[rows].mean()]), math.atan2(n[b] - n[a], e[b] - e[a])
    return np.array([e[row_fallback], n[row_fallback]]), heading_fallback


def _intersection(pa: np.ndarray, ha: float, pb: np.ndarray, hb: float) -> np.ndarray | None:
    """Where the line through ``pa`` heading ``ha`` meets the line through ``pb`` heading
    ``hb``; None for parallel lines."""
    ua = np.array([math.cos(ha), math.sin(ha)])
    ub = np.array([math.cos(hb), math.sin(hb)])
    det = -ua[0] * ub[1] + ub[0] * ua[1]
    if abs(det) < 1e-9:
        return None
    rhs = pb - pa
    t = (-rhs[0] * ub[1] + ub[0] * rhs[1]) / det
    return pa + t * ua


def extract_waypoints(e: np.ndarray, n: np.ndarray, dt_s: float, *, max_waypoints: int = MAX_WAYPOINTS,
                      ) -> tuple[tuple[tuple[float, float], ...], int]:
    """The path ``(e, n)`` (rows ``dt_s`` apart) as at most ``max_waypoints`` fly-by fixes —
    where the two legs each turn joins intersect, in path order (a turn over
    `FLY_BY_SPLIT_DEG` as two fixes about its mid-tangent) — and how many fixes beyond the
    cap were dropped (those of the largest turns are kept)."""
    fixes, dropped = _turn_fixes(e, n, dt_s, max_waypoints=max_waypoints)
    return tuple(fix for fix, _rows in fixes), dropped


def _turn_fixes(e: np.ndarray, n: np.ndarray, dt_s: float, *, max_waypoints: int,
                ) -> tuple[list[tuple[tuple[float, float], int]], int]:
    """`extract_waypoints` with each fix's turn rows beside it: ``(fix, (first, middle, last))``."""
    w = TURN_HEADING_HALF_WINDOW_ROWS
    e, n = np.asarray(e, dtype=np.float64), np.asarray(n, dtype=np.float64)
    if len(e) < 2 * w + 3:
        return (), 0
    centre = np.arange(w, len(e) - w)
    s = _arc_length_m(e, n)
    psi = np.unwrap(np.arctan2(n[centre + w] - n[centre - w], e[centre + w] - e[centre - w]))
    turning = np.abs(np.diff(psi) / dt_s) > math.radians(TURN_RATE_MIN_DEG_S)
    for k in range(1, len(turning) - 1):
        if not turning[k] and turning[k - 1] and turning[k + 1]:
            turning[k] = True
    intervals: list[tuple[int, int]] = []
    k = 0
    while k < len(turning):
        if not turning[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(turning) and turning[j + 1]:
            j += 1
        if abs(float(psi[j + 1] - psi[k])) >= math.radians(TURN_MIN_DEG):
            intervals.append((k, j))
        k = j + 1
    fixes: list[tuple[int, int, float, np.ndarray, tuple[int, int, int]]] = []   # (turn, half, |turn|, fix, rows)
    prev_end = -1
    for m, (k, j) in enumerate(intervals):
        change = float(psi[j + 1] - psi[k])
        next_start = intervals[m + 1][0] if m + 1 < len(intervals) else len(psi) - 1
        before = centre[prev_end + 1 : k + 1]
        after = centre[j + 1 : next_start + 1]
        pa, ha = _leg_line(e, n, before, float(psi[k]), int(centre[k]))
        pb, hb = _leg_line(e, n, after, float(psi[j + 1]), int(centre[j + 1]))
        mid_row = int(centre[(k + j + 1) // 2])
        rows_of_turn = (int(centre[k]), mid_row, int(centre[j + 1]))
        pm = np.array([e[mid_row], n[mid_row]])
        # the turn's own radius (its arc length over its angle) bounds where its fix can be
        radius = (float(s[centre[j + 1]] - s[centre[k]])) / max(abs(change), 1e-6)
        reach = 3.0 * radius + FIX_GUARD_M

        def guarded(fix):
            return pm if fix is None or float(np.hypot(*(fix - pm))) > reach else fix

        if abs(change) > math.radians(FLY_BY_SPLIT_DEG):
            hm = ha + 0.5 * change
            halves = ((0, guarded(_intersection(pa, ha, pm, hm))), (1, guarded(_intersection(pm, hm, pb, hb))))
            fixes.extend((m, half, 0.5 * abs(change), fix, rows_of_turn) for half, fix in halves)
        else:
            fixes.append((m, 0, abs(change), guarded(_intersection(pa, ha, pb, hb)), rows_of_turn))
        prev_end = j
    by_size = sorted(fixes, key=lambda item: item[2], reverse=True)
    kept = sorted(by_size[:max_waypoints], key=lambda item: (item[0], item[1]))
    return [((float(fix[0]), float(fix[1])), rows) for _m, _half, _size, fix, rows in kept], len(by_size) - len(kept)


def _arc_length_m(e: np.ndarray, n: np.ndarray) -> np.ndarray:
    steps = np.hypot(np.diff(e), np.diff(n))
    return np.concatenate(([0.0], np.cumsum(steps)))


def extract_plan(series: FlightSeries, anchor: int, skeleton: RunwaySkeleton, *,
                 max_waypoints: int = MAX_WAYPOINTS) -> PlanLabels:
    """The eight parameters of ``series`` from ``anchor``, inside ``skeleton`` — and the
    pre-final path's fly-by waypoints (at most ``max_waypoints``), the fixed-K
    representation."""
    rows = slice(anchor, series.n_samples)
    times = np.asarray(series.times[rows], dtype=np.float64)
    values = np.asarray(series.values[rows], dtype=np.float64)
    e, n, u = values[:, IDX["e"]], values[:, IDX["n"]], values[:, IDX["u"]]
    d, xt = skeleton.axes(e, n)
    remaining = np.asarray(remaining_path_profile_m(series)[rows], dtype=np.float64)
    speed = _ground_speeds(series, rows)
    dt = float(series.times[1] - series.times[0]) if series.n_samples > 1 else 1.0

    # the join: the measurement's gate on the observed rows
    gate = truth_final_gate(
        torch.from_numpy(d)[None], torch.from_numpy(xt)[None],
        torch.ones((1, len(d)), dtype=torch.bool),
    )[0].numpy()
    join = int(np.flatnonzero(gate)[0]) if gate.any() else None

    V_final = float(speed[-1])
    T = truth_duration_s(series, anchor)
    scenario = series.scenario
    slow = np.flatnonzero(speed < float(scenario.target.V) + DECEL_MARGIN_MPS)
    d_decel = float(remaining[slow[0]]) if slow.size else None
    held_rows = int(slow[0]) if slow.size else len(speed)
    v_mid_from_anchor = held_rows < V_MID_MIN_ROWS
    if v_mid_from_anchor:
        V_mid = float(speed[0])
    else:
        held = slice(0, held_rows)
        V_mid = float(_arc_length_m(e[held], n[held])[-1] / (times[held_rows - 1] - times[0]))

    join_at_anchor = join == 0
    if join is None or join_at_anchor:
        h_capture = L_pre = side = waypoints = waypoint_speeds = waypoint_times = waypoint_heights = None
        waypoints_dropped = 0
        d_join = None if join is None else float(remaining[0])
        on_transition = None
    else:
        h_capture = float(u[join])
        d_join = float(remaining[join])
        L_pre = float(_arc_length_m(e, n)[join])
        # read past the join by the join window, so the turn ONTO the final — usually
        # still under way where the gate opens — is a complete turn with a leg after it
        stop = min(len(e), join + 1 + max(1, int(round(JOIN_WINDOW_S / dt))))
        fixes, waypoints_dropped = _turn_fixes(e[:stop], n[:stop], dt, max_waypoints=max_waypoints)
        waypoints = tuple(fix for fix, _rows in fixes)
        waypoint_speeds = tuple(
            (float(remaining[mid]), float(np.median(speed[first : last + 1])))
            for _fix, (first, mid, last) in fixes
        )
        waypoint_times = tuple(float(times[mid] - times[0]) for _fix, (_first, mid, _last) in fixes)
        waypoint_heights = tuple(float(u[mid]) for _fix, (_first, mid, _last) in fixes)
        # the fix of the turn ONTO the final lies where the last leg meets the course —
        # often past the join (the gate opens inside a wide corridor before the turn is
        # complete). It is kept: the route takes it as the join itself. A fix past the
        # join and OFF the course is a turn inside the final the route cannot pass
        # through and come back from (it looped: 38 % of vectored smoke flights left
        # the corridor after the join, 2026-09-11) — dropped, not counted.
        if waypoints:
            fix_e = np.array([fix[0] for fix in waypoints])
            fix_n = np.array([fix[1] for fix in waypoints])
            fix_d, fix_xt = skeleton.axes(fix_e, fix_n)
            keep = (fix_d > d_join) | (np.abs(fix_xt) <= ON_COURSE_FIX_M)
            waypoints = tuple(fix for fix, ok in zip(waypoints, keep, strict=True) if ok)
            waypoint_speeds = tuple(p for p, ok in zip(waypoint_speeds, keep, strict=True) if ok)
            waypoint_times = tuple(t for t, ok in zip(waypoint_times, keep, strict=True) if ok)
            waypoint_heights = tuple(h for h, ok in zip(waypoint_heights, keep, strict=True) if ok)
        window = max(1, int(round(SIDE_WINDOW_S / dt)))
        before = xt[max(0, join - window):join]
        widest = float(before[np.argmax(np.abs(before))])
        side = 0 if abs(widest) < SIDE_MIN_OFFSET_M else (1 if widest > 0.0 else -1)
        approach = slice(max(0, join - max(1, int(round(JOIN_WINDOW_S / dt)))), join + 1)
        distance = skeleton.distance_to_published_m(e[approach], n[approach])
        on_transition = None if np.isinf(distance).all() else bool(np.max(distance) <= RNP_HALF_WIDTH_M)

    # ranges: the procedure's where coded, the aircraft's otherwise
    stall = STALL_MARGIN * stall_speed_ms(
        float(scenario.initial.m), wing_area_m2=float(scenario.aero.S),
        cl_max=float(scenario.aero.Cl_max),
    )
    # the shortest legal route at the fastest legal speed: the beeline is its lower bound,
    # the fastest legal speed the coded limit, else the type's approach maximum
    beeline = float(np.hypot(e[0] - skeleton.target_e, n[0] - skeleton.target_n))
    limit = skeleton.speed_limit_ahead_mps(float(d[0]))
    fastest = limit if limit is not None else float(scenario.aircraft.approach.max_speed_ms)
    # the capture height is bounded where the capture HAPPENS, by the constraint coded at
    # the next fix ahead of the join (its floor, and its ceiling where one is coded). NOT
    # by the glidepath: the join is the LATERAL gate, and measured on KRDU val (2026-09-10)
    # most flights are still above the glidepath there and capture it later.
    captured = join is not None and not join_at_anchor
    floor_alt = skeleton.floor_altitude_m(float(d[join])) if captured else None
    ceiling_alt = skeleton.ceiling_altitude_m(float(d[join])) if captured else None
    ranges = {
        "T_s": (beeline / fastest, None),
        "V_mid_mps": (stall, limit),
        "d_decel_m": (0.0, float(remaining[0])),
        # the published V_ref is an AIRSPEED window the evaluation reads with the METAR
        # headwind (`evaluation/docs/THRESHOLD_SPEED_GATE.md`); a ground speed has no
        # floor there but the stall margin's
        "V_final_mps": (stall, None),
        "h_capture_m": (
            None if floor_alt is None else floor_alt - skeleton.aim_altitude_m,
            None if ceiling_alt is None else ceiling_alt - skeleton.aim_altitude_m,
        ),
        "d_join_m": (0.0, float(remaining[0])),
        "side": (-1.0, 1.0),
        "L_pre_m": (0.0, float(remaining[0])),
    }
    return PlanLabels(
        T_s=T, V_mid_mps=V_mid, d_decel_m=d_decel, V_final_mps=V_final,
        h_capture_m=h_capture, d_join_m=d_join, side=side, L_pre_m=L_pre,
        join_on_transition=on_transition, ranges=ranges,
        join_at_anchor=join_at_anchor, v_mid_from_anchor=v_mid_from_anchor,
        remaining_path_at_anchor_m=float(remaining[0]),
        ground_speed_at_anchor_mps=float(speed[0]),
        waypoints=waypoints, waypoints_dropped=waypoints_dropped, waypoint_speeds=waypoint_speeds,
        waypoint_times_s=waypoint_times, waypoint_heights_m=waypoint_heights,
    )
