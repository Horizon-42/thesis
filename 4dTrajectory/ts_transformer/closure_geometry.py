"""Closed-form approach geometry for the closure decoder (scene design §五 P1).

The closure decoder does not draw the path with a network: it predicts a few DECISION
quantities and the path follows from geometry — anchor pose → (turns and straights) →
the localizer at the join distance ``d_join`` → the final leg to the threshold — with a
speed profile giving the time (``closure_profile``). This module is the geometry half:
the path families, the vertical profile, the two timings the oracle studies use, and the
per-flight fits that turn a truth path into that family's parameters (the closure
decoder's labels).

Families, each a function of the anchor pose and the runway course ``psi`` (math-ENU,
pointing INBOUND toward the threshold; chart origin = threshold; runway axes ``d``
upstream, ``xt`` to the right, see ``final_approach_geometry.runway_axes``):

* ``rule_template`` (F0) — the Phase 0 template (moved here from
  ``docs/phase0_intent_diagnostics.py``): straight to the threshold when already aligned
  (the anchor is the join), a trombone (downwind → 90° base → 90° onto the final) when
  outbound and offset, otherwise the shortest Dubins CSC path to the join pose. One
  decision quantity: ``d_join`` (a trombone already past it joins at its own distance).
* ``dubins_join`` (F2) — hold the anchor heading for ``d_downwind`` metres (a straight
  along the heading, not runway distance), then the shortest Dubins CSC to the join pose.
  Two decision quantities ``(d_join, d_downwind)``; it contains the trombone when the
  downwind end is the trombone's base-turn point.
* ``via_dubins`` (F3) — Dubins CSC to a VIA pose, then Dubins CSC to the join pose. Four
  decision quantities ``(d_join, via_e, via_n, via_heading)``. A via pose is only
  informative off the shortest path (any pose on a single CSC path reproduces it), so
  the fitted label is CANONICALISED to the earliest pose along the fitted path that
  still reproduces it (``canonical_via``): the anchor itself for a pure CSC flight, the
  base-turn point for a trombone.

Two labels of every fitted family are canonicalised because the objective cannot see
them: ``d_join`` is the runway distance where the path FIRST settles on the localizer
(``localizer_entry``; a CSC whose straight runs along the localizer reproduces the
same path for every smaller ``d_join``), and F3's via as above.

Turn radius = ``turn_radius_m`` (a ``BANK_RAD`` bank at the anchor ground speed, capped
at ``TURN_SPEED_CAP_MPS``). Timing: ``truth_timed`` gives the path the truth's time at
the same fraction of arc length (the geometry-only oracle: a perfect speed profile),
``naive_timed`` a linear deceleration from the anchor speed to ``THRESHOLD_SPEED_MPS``.
Vertical: ``vertical_profile`` descends linearly to the glidepath height at the join,
then flies the glidepath (the chart origin is the threshold-crossing aim point: height
``d·tan GPA``, no TCH).

Fits (``fit_*``) minimise the ORDER-PRESERVING horizontal error at the same fraction of
arc length (``geometric_metrics.arc_aligned_ade_m`` on the horizontal plane — the
horizontal part of the truth-timed ADE, and immune to the detours an order-blind chamfer
lets through) over the family's parameters; chamfer and Fréchet are then read as
independent residuals. F3 is seeded with F1's and F2's solutions (their via poses) and
with the anchor pose itself, so its residual is never worse than theirs by more than what
re-expressing them as via-Dubins paths costs (a trombone starts with a kink onto the
runway-parallel downwind that a Dubins path rounds off). F2 does NOT contain F1: its
straight runs along the anchor's own heading, the trombone's along the runway axis, so
they coincide only for an exactly outbound anchor. The oracle prints the share of flights
on which each nesting fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import minimize

from anchor_eligibility import GRAVITY_MPS2
from geometric_metrics import arc_aligned_ade_m, cumulative_arc_m

BANK_RAD = math.radians(25.0)       # a standard-rate-ish approach bank; not an envelope limit
TURN_SPEED_CAP_MPS = 100.0          # the turn radius is sized at approach speed
THRESHOLD_SPEED_MPS = 70.0
PATH_STEP_M = 50.0

# Rule template thresholds (Phase 0 values, unchanged).
STRAIGHT_OFFSET_RADII = 2.5
STRAIGHT_COS_ALIGN = 0.7
STRAIGHT_NEAR_JOIN_M = 500.0
STRAIGHT_NEAR_OFFSET_M = 1000.0
TROMBONE_COS_ALIGN = -0.3
TROMBONE_OFFSET_RADII = 2.0

KIND_STRAIGHT = "straight"
KIND_TROMBONE = "trombone"
KIND_TROMBONE_PAST_JOIN = "trombone-past-join"
KIND_DUBINS = "dubins"
KIND_DOWNWIND_DUBINS = "downwind-dubins"
KIND_VIA_DUBINS = "via-dubins"

Pose = tuple[float, float, float]   # (e, n, heading)
POSE_TOLERANCE = (1e-6, 1e-6)       # metres, radians: two poses this close are one pose


def turn_radius_m(speed_mps: float) -> float:
    return min(float(speed_mps), TURN_SPEED_CAP_MPS) ** 2 / (GRAVITY_MPS2 * math.tan(BANK_RAD))


# Mirrors of final_approach_geometry.runway_axes / chart_from_axes (torch) for the numpy
# fitting loops; tests/test_closure_geometry.py pins them to the torch versions.
def runway_axes_np(e, n, psi: float) -> tuple[np.ndarray, np.ndarray]:
    e, n = np.asarray(e, dtype=np.float64), np.asarray(n, dtype=np.float64)
    return -(e * math.cos(psi) + n * math.sin(psi)), e * math.sin(psi) - n * math.cos(psi)


def chart_from_axes_np(d, xt, psi: float) -> tuple[np.ndarray, np.ndarray]:
    d, xt = np.asarray(d, dtype=np.float64), np.asarray(xt, dtype=np.float64)
    return -d * math.cos(psi) + xt * math.sin(psi), -d * math.sin(psi) - xt * math.cos(psi)


def _unit(heading: float) -> np.ndarray:
    return np.array([math.cos(heading), math.sin(heading)])


def _points_for(length: float, step: float) -> int:
    """Points that sample ``length`` at most ``step`` apart, endpoints included."""
    return max(2, math.ceil(length / step - 1e-9) + 1)


def _segment(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
    """``a`` to ``b`` at most ``step`` apart; a zero-length segment is its single point,
    so the ``[1:]`` concatenations never repeat a node."""
    length = float(np.hypot(*(b - a)))
    if length < 1e-9:
        return np.asarray(a, dtype=np.float64)[None, :].copy()
    return a + np.linspace(0.0, 1.0, _points_for(length, step))[:, None] * (b - a)


def arc_points(centre: np.ndarray, radius: float, start: float, sweep: float, step: float = PATH_STEP_M) -> np.ndarray:
    """An arc at most ``step`` apart; a zero-sweep arc is its single point."""
    if abs(radius * sweep) < 1e-9:
        return (centre + radius * _unit(start))[None, :]
    n = _points_for(abs(radius * sweep), step)
    angles = start + np.linspace(0.0, sweep, n)
    return centre + radius * np.stack([np.cos(angles), np.sin(angles)], 1)


def dubins_csc(p0, h0: float, p1, h1: float, radius: float, step: float = PATH_STEP_M) -> np.ndarray | None:
    """Shortest turn-straight-turn path between two poses (LSL / RSR / LSR / RSL).
    Identical poses (within ``POSE_TOLERANCE``) return the single point — LSR would
    otherwise fly a loop. A CSC path exists for every other pair in exact arithmetic
    (same-side circles coincide only for identical poses; a same-position pose with
    another heading is a loop, not an absence), so ``None`` is a floating-point corner
    between the two tolerances below; the fits treat it as an absent candidate."""
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    if (np.hypot(*(p1 - p0)) < POSE_TOLERANCE[0]
            and abs((h1 - h0 + math.pi) % (2 * math.pi) - math.pi) < POSE_TOLERANCE[1]):
        return p0[None, :].copy()
    best = None
    for s0, s1 in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        c0 = p0 + radius * np.array([-s0 * math.sin(h0), s0 * math.cos(h0)])
        c1 = p1 + radius * np.array([-s1 * math.sin(h1), s1 * math.cos(h1)])
        dc = c1 - c0
        distance = float(np.hypot(*dc))
        theta = math.atan2(dc[1], dc[0])
        if s0 == s1:
            if distance < 1e-6:
                continue
            psi = theta
        else:
            if distance < 2 * radius:
                continue
            psi = theta + s0 * math.asin(2 * radius / distance)
        a0, a1 = psi - s0 * math.pi / 2, psi - s1 * math.pi / 2
        t0 = c0 + radius * _unit(a0)
        t1 = c1 + radius * _unit(a1)
        f0, f1 = h0 - s0 * math.pi / 2, h1 - s1 * math.pi / 2
        d0 = (s0 * (a0 - f0)) % (2 * math.pi)
        d1 = (s1 * (f1 - a1)) % (2 * math.pi)
        length = radius * (d0 + d1) + float(np.hypot(*(t1 - t0)))
        if best is None or length < best[0]:
            best = (length, s0, s1, c0, c1, f0, a0, a1, d0, d1, t0, t1)
    if best is None:
        return None
    _length, s0, s1, c0, c1, f0, a0, a1, d0, d1, t0, t1 = best
    return np.concatenate([
        arc_points(c0, radius, f0, s0 * d0, step),
        _segment(t0, t1, step)[1:],
        arc_points(c1, radius, a1, s1 * d1, step)[1:],
    ])


def trombone(p0, psi: float, d0: float, xt0: float, d_join: float, radius: float,
             step: float = PATH_STEP_M) -> tuple[np.ndarray, float, Pose]:
    """Downwind → 90° base turn → base leg → 90° turn onto the final at ``d_join``.

    The downwind (along the upstream axis from ``p0``) continues to ``d = d_join``; the
    base leg lies at ``d_join + radius``; the turn onto the final ends on the localizer at
    ``d_join`` heading for the threshold. An anchor already past ``d_join`` turns at once
    and joins at its own distance. Returns the path, the join used, and the pose where
    the downwind ends (the base-turn point — the path's via)."""
    p0 = np.asarray(p0, dtype=np.float64)
    ud = -_unit(psi)                                  # upstream
    ux = np.array([math.sin(psi), -math.cos(psi)])    # right of the course
    side = 1.0 if xt0 > 0 else -1.0
    points = [p0.copy()]
    if d0 < d_join:
        points.extend(_segment(p0, p0 + (d_join - d0) * ud, step)[1:])
    d_turn = max(d0, d_join)
    start = points[-1]
    via = (float(start[0]), float(start[1]), math.atan2(ud[1], ud[0]))
    centre1 = start - side * radius * ux
    heading_base = -side * ux
    cross = ud[0] * heading_base[1] - ud[1] * heading_base[0]
    sweep = (math.pi / 2) * (1.0 if cross > 0 else -1.0)
    points.extend(arc_points(centre1, radius, math.atan2(*(start - centre1)[::-1]), sweep, step)[1:])
    base_start = points[-1]
    base_end = (d_turn + radius) * ud + side * radius * ux
    points.extend(_segment(base_start, base_end, step)[1:])
    centre2 = d_turn * ud + side * radius * ux
    heading_final = -ud
    cross = heading_base[0] * heading_final[1] - heading_base[1] * heading_final[0]
    sweep = (math.pi / 2) * (1.0 if cross > 0 else -1.0)
    points.extend(arc_points(centre2, radius, math.atan2(*(base_end - centre2)[::-1]), sweep, step)[1:])
    return np.array(points), d_turn, via


def final_leg(d_join: float, psi: float, step: float = PATH_STEP_M) -> np.ndarray:
    """The localizer from just inside ``d_join`` to the threshold (the chart origin)."""
    d = np.arange(d_join - step, 0.0, -step)
    e, n = chart_from_axes_np(d, np.zeros_like(d), psi)
    return np.concatenate([np.stack([e, n], 1), np.zeros((1, 2))])


@dataclass(frozen=True)
class AnchorPose:
    """The anchor's horizontal state in the threshold chart."""
    position: np.ndarray        # (e, n)
    heading: float              # math-ENU, from the chart velocity channels
    speed_mps: float            # ground speed
    d: float                    # runway distance
    xt: float                   # cross-track

    @classmethod
    def from_state(cls, state, psi: float) -> "AnchorPose":
        state = np.asarray(state, dtype=np.float64)      # channels (e, n, u, edot, ndot, udot)
        d, xt = runway_axes_np(state[0], state[1], psi)
        return cls(state[:2].copy(), math.atan2(state[4], state[3]), float(np.hypot(state[3], state[4])),
                   float(d), float(xt))

    @property
    def radius(self) -> float:
        return turn_radius_m(self.speed_mps)

    @property
    def pose(self) -> Pose:
        return (float(self.position[0]), float(self.position[1]), self.heading)


@dataclass(frozen=True)
class ClosurePath:
    """A horizontal path ``[N, 2]`` from the anchor to the threshold, the join it uses,
    the arc length at which it reaches the join (0 = the anchor is the join), the
    construction, the family parameters that produced it (the decoder's labels), and the
    pose where the construction changes leg (downwind end / via; ``None`` when it has none)."""
    horizontal: np.ndarray
    d_join: float
    s_join: float
    kind: str
    params: dict[str, float] = field(default_factory=dict)
    via: Pose | None = None

    @property
    def arc(self) -> np.ndarray:
        return cumulative_arc_m(self.horizontal)

    @property
    def length(self) -> float:
        return float(self.arc[-1])

    def pose_at(self, s: float) -> Pose:
        """The path's pose at arc length ``s`` (heading from the local step)."""
        arc = self.arc
        i = int(np.clip(np.searchsorted(arc, s), 1, len(arc) - 1))
        step = self.horizontal[i] - self.horizontal[i - 1]
        f = (s - arc[i - 1]) / max(arc[i] - arc[i - 1], 1e-9)
        p = self.horizontal[i - 1] + np.clip(f, 0.0, 1.0) * step
        return (float(p[0]), float(p[1]), math.atan2(step[1], step[0]))


def _close(horizontal: np.ndarray, d_join: float, psi: float, kind: str, params: dict, via: Pose | None) -> ClosurePath:
    """Append the final leg (without doubling a node the path already ends on) and
    measure the join's arc length."""
    leg = final_leg(d_join, psi)
    if np.hypot(*(horizontal[-1] - leg[0])) < 1e-6:
        leg = leg[1:]
    s_join = float(cumulative_arc_m(horizontal)[-1])
    return ClosurePath(np.concatenate([horizontal, leg]), float(d_join), s_join, kind, dict(params), via)


def join_pose(d_join: float, psi: float) -> np.ndarray:
    e, n = chart_from_axes_np(d_join, 0.0, psi)
    return np.array([float(e), float(n)])


def straight_path(anchor: AnchorPose) -> ClosurePath:
    """Straight to the threshold (the chart origin): the aligned, straight-in case. The
    anchor IS the join (``s_join = 0``), so the vertical profile is the glidepath from
    the first step on."""
    horizontal = _segment(anchor.position, np.zeros(2), PATH_STEP_M)
    return ClosurePath(horizontal, anchor.d, 0.0, KIND_STRAIGHT, {"d_join": anchor.d})


def is_straight_in(anchor: AnchorPose, psi: float, d_join: float) -> bool:
    cos_align = math.cos(anchor.heading - psi)
    return ((abs(anchor.xt) < STRAIGHT_OFFSET_RADII * anchor.radius and cos_align > STRAIGHT_COS_ALIGN)
            or (d_join >= anchor.d - STRAIGHT_NEAR_JOIN_M and abs(anchor.xt) < STRAIGHT_NEAR_OFFSET_M))


def rule_template(anchor: AnchorPose, psi: float, d_join: float) -> ClosurePath:
    """F0: the Phase 0 template — straight / trombone / Dubins chosen by rule from the
    anchor's alignment and offset, joining at ``d_join`` (the trombone may join later)."""
    if is_straight_in(anchor, psi, d_join):
        return straight_path(anchor)
    cos_align = math.cos(anchor.heading - psi)
    if cos_align < TROMBONE_COS_ALIGN and abs(anchor.xt) >= TROMBONE_OFFSET_RADII * anchor.radius:
        horizontal, d_used, via = trombone(anchor.position, psi, anchor.d, anchor.xt, d_join, anchor.radius)
        kind = KIND_TROMBONE if d_used == d_join else KIND_TROMBONE_PAST_JOIN
        return _close(horizontal, d_used, psi, kind, {"d_join": d_used}, via)
    horizontal = dubins_csc(anchor.position, anchor.heading, join_pose(d_join, psi), psi, anchor.radius)
    if horizontal is None:
        raise RuntimeError("the anchor sits on the join pose with another heading: no CSC path")
    return _close(horizontal, d_join, psi, KIND_DUBINS, {"d_join": d_join}, None)


def dubins_join(anchor: AnchorPose, psi: float, d_join: float, d_downwind: float = 0.0) -> ClosurePath | None:
    """F2: hold the anchor heading for ``d_downwind`` metres, then the shortest Dubins
    CSC to the join pose (``None`` when that CSC does not exist)."""
    d_downwind = max(float(d_downwind), 0.0)
    start = anchor.position + d_downwind * _unit(anchor.heading)
    rest = dubins_csc(start, anchor.heading, join_pose(d_join, psi), psi, anchor.radius)
    if rest is None:
        return None
    horizontal = np.concatenate([_segment(anchor.position, start, PATH_STEP_M)[:-1], rest]) if d_downwind > 0 else rest
    via = (float(start[0]), float(start[1]), anchor.heading)
    return _close(horizontal, d_join, psi, KIND_DOWNWIND_DUBINS, {"d_join": d_join, "d_downwind": d_downwind}, via)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def via_dubins(anchor: AnchorPose, psi: float, d_join: float, via_e: float, via_n: float,
               via_heading: float) -> ClosurePath | None:
    """F3: Dubins CSC to a via pose, then Dubins CSC to the join pose (``None`` when
    either CSC does not exist). The params carry the via both in the chart and in runway
    axes with its heading relative to the course, wrapped to [−π, π) — the label a decoder
    regresses (one distribution for every runway, no branch cut when stored as cos / sin)."""
    via = np.array([via_e, via_n])
    first = dubins_csc(anchor.position, anchor.heading, via, via_heading, anchor.radius)
    second = dubins_csc(via, via_heading, join_pose(d_join, psi), psi, anchor.radius)
    if first is None or second is None:
        return None
    via_d, via_xt = runway_axes_np(via_e, via_n, psi)
    heading = wrap_angle(float(via_heading))
    return _close(np.concatenate([first, second[1:]]), d_join, psi, KIND_VIA_DUBINS,
                  {"d_join": d_join, "via_e": via_e, "via_n": via_n, "via_heading": heading,
                   "via_d": float(via_d), "via_xt": float(via_xt), "via_heading_rel": wrap_angle(heading - psi)},
                  (float(via_e), float(via_n), heading))


# ── profiles ────────────────────────────────────────────────────────────────

def vertical_profile(path: ClosurePath, anchor_u: float, tan_gpa: float) -> np.ndarray:
    """Height at every path point: linear from the anchor height to the glidepath at the
    join, then the glidepath ``(length − s)·tan GPA`` down to 0 at the threshold. With
    ``s_join = 0`` (the anchor is the join) only the anchor keeps its own height."""
    s = path.arc
    return np.where(
        s <= path.s_join,
        anchor_u + (s / max(path.s_join, 1.0)) * (path.d_join * tan_gpa - anchor_u),
        (path.length - s) * tan_gpa,
    )


def strictly_increasing(t: np.ndarray) -> np.ndarray:
    """The record contract wants a strictly increasing clock; a repeated node (zero-length
    step) would otherwise repeat a time."""
    return np.maximum.accumulate(t + np.arange(len(t)) * 1e-6)


def truth_timed(path: ClosurePath, truth_xy: np.ndarray, truth_t: np.ndarray) -> np.ndarray:
    """Time at every path point = the truth's time at the same fraction of arc length
    (``truth_xy`` / ``truth_t`` start at the anchor, ``truth_t[0] == 0``)."""
    s_truth = cumulative_arc_m(truth_xy)
    return strictly_increasing(np.interp(path.arc / path.length * s_truth[-1], s_truth, truth_t))


def naive_times(s: np.ndarray, speed0_mps: float) -> np.ndarray:
    """Time along arc lengths ``s`` from a ground speed falling linearly from the anchor's
    to the threshold's over the whole length."""
    speed = speed0_mps + (THRESHOLD_SPEED_MPS - speed0_mps) * s / s[-1]
    return strictly_increasing(np.concatenate([[0.0], np.cumsum(np.diff(s) / (0.5 * (speed[1:] + speed[:-1])))]))


def naive_timed(path: ClosurePath, speed0_mps: float) -> np.ndarray:
    return naive_times(path.arc, speed0_mps)


def path_record(path: ClosurePath, times: np.ndarray, anchor_u: float, tan_gpa: float) -> tuple[np.ndarray, np.ndarray]:
    """``(offsets_s, values[N, 6])`` after the anchor in channel order (positions; the
    velocity channels zero — the geometry oracle scores positions only)."""
    values = np.zeros((len(times), 6))
    values[:, 0], values[:, 1] = path.horizontal[:, 0], path.horizontal[:, 1]
    values[:, 2] = vertical_profile(path, anchor_u, tan_gpa)
    return times[1:], values[1:]


# ── fits ────────────────────────────────────────────────────────────────────

D_JOIN_MIN_M = 500.0
FIT_GRID_STEP_M = 500.0
CANONICAL_VIA_TOLERANCE_M = 25.0
VIA_SEED_FRACTIONS = (0.25, 0.4, 0.6)
LOCALIZER_ENTRY_XT_M = 5.0
LOCALIZER_ENTRY_ALIGN_RAD = math.radians(1.0)


def localizer_entry(path: ClosurePath, psi: float) -> tuple[int, float] | None:
    """The node at which the path first settles on the localizer (within
    ``LOCALIZER_ENTRY_XT_M`` and heading inbound within ``LOCALIZER_ENTRY_ALIGN_RAD``, and
    stays there to the threshold): its index and its runway distance — the identifiable
    join of any fitted path. ``None`` when no step of the path qualifies (a straight
    path to the threshold from off the centreline never settles on it)."""
    xy = path.horizontal
    d, xt = runway_axes_np(xy[:, 0], xy[:, 1], psi)
    step = np.diff(xy, axis=0)
    heading = np.arctan2(step[:, 1], step[:, 0])
    aligned = np.abs((heading - psi + math.pi) % (2 * math.pi) - math.pi) < LOCALIZER_ENTRY_ALIGN_RAD
    on = (np.abs(xt[:-1]) < LOCALIZER_ENTRY_XT_M) & aligned
    if not on.any():
        return None
    stays = np.flip(np.logical_and.accumulate(np.flip(on)))
    index = int(np.argmax(stays))
    return index, float(d[index])


def _horizontal3(xy: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(xy, dtype=np.float64)[:, :2], np.zeros((len(xy), 1))], 1)


def path_error_m(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    """The fit objective: mean horizontal distance at the same fraction of each path's
    arc length (order-preserving, speed-free)."""
    return arc_aligned_ade_m(_horizontal3(a_xy), _horizontal3(b_xy))


def _cost(path: ClosurePath | None, truth_xy: np.ndarray) -> float:
    """``path_error_m`` of a candidate, infinite for a construction that does not exist."""
    return math.inf if path is None else path_error_m(path.horizontal, truth_xy)


def _refine(objective, x0: np.ndarray, *, maxfev: int) -> np.ndarray:
    result = minimize(objective, x0, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 10.0, "fatol": 1.0})
    return result.x if result.fun < objective(x0) else x0


def _d_join_grid(anchor: AnchorPose, d_join0: float) -> np.ndarray:
    high = max(anchor.d, d_join0) + 5_000.0
    return np.append(np.arange(D_JOIN_MIN_M, high + FIT_GRID_STEP_M, FIT_GRID_STEP_M), max(d_join0, D_JOIN_MIN_M))


def fit_rule_template(anchor: AnchorPose, psi: float, truth_xy: np.ndarray, d_join0: float) -> ClosurePath:
    """F1: the rule template with ``d_join`` chosen by ``path_error_m`` (a grid that
    includes the truth gate's ``d_join0``, then local refinement)."""
    def build(d_join):
        try:
            return rule_template(anchor, psi, max(float(d_join), D_JOIN_MIN_M))
        except RuntimeError:          # the anchor on the join pose with another heading
            return None
    grid = _d_join_grid(anchor, d_join0)
    costs = [_cost(build(d), truth_xy) for d in grid]
    x = _refine(lambda v: _cost(build(v[0]), truth_xy), np.array([grid[int(np.argmin(costs))]]), maxfev=60)
    fitted = build(x[0])
    if fitted is None:
        raise RuntimeError("no rule-template join exists for this anchor")
    return _relabelled_join(fitted, lambda d: build(d), psi, truth_xy)


def _relabelled_join(fitted: ClosurePath, rebuild, psi: float, truth_xy: np.ndarray) -> ClosurePath:
    """``fitted`` rebuilt with its join at the localizer entry (the identifiable label)
    when the path has one and the rebuild costs nothing; otherwise ``fitted`` itself."""
    entry = localizer_entry(fitted, psi)
    if entry is None:
        return fitted
    rebuilt = rebuild(entry[1])
    return rebuilt if rebuilt is not None and _cost(rebuilt, truth_xy) <= _cost(fitted, truth_xy) + 1e-9 else fitted


def fit_dubins_join(anchor: AnchorPose, psi: float, truth_xy: np.ndarray, d_join0: float,
                    seed: ClosurePath | None = None) -> ClosurePath:
    """F2: ``(d_join, d_downwind)`` by ``path_error_m`` — a grid scaled to the flight
    (downwind up to the longer of the anchor's runway distance and the truth's length)
    plus ``seed``'s base-turn point projected onto the anchor heading (F1's trombone,
    which F2 reproduces only when the anchor already points down the reciprocal course),
    then Nelder–Mead."""
    def build(v):
        return dubins_join(anchor, psi, max(float(v[0]), D_JOIN_MIN_M), float(v[1]))
    reach = max(anchor.d, float(cumulative_arc_m(truth_xy)[-1]))
    grid = [(d, w) for d in _d_join_grid(anchor, d_join0) for w in np.linspace(0.0, reach, 8)]
    if seed is not None and seed.via is not None:
        along = float((np.array(seed.via[:2]) - anchor.position) @ _unit(anchor.heading))
        grid.append((seed.d_join, max(along, 0.0)))
    costs = [_cost(build(v), truth_xy) for v in grid]
    x = _refine(lambda v: _cost(build(v), truth_xy), np.array(grid[int(np.argmin(costs))]), maxfev=200)
    fitted = build(x)
    return _relabelled_join(fitted, lambda d: build((d, fitted.params["d_downwind"])), psi, truth_xy)


def canonical_via(path: ClosurePath, anchor: AnchorPose, psi: float, d_join: float) -> tuple[Pose | None, float, float]:
    """The earliest pose along ``path`` — from the anchor up to where it settles on the
    localizer at ``d_join`` — from which ``via_dubins`` joining at ``d_join`` reproduces
    ``path`` within ``CANONICAL_VIA_TOLERANCE_M``: the identifiable label of a via-Dubins
    path. Returns the pose, its arc length, and the arc length of the localizer entry
    (the anchor, s = 0, for a path that is a single CSC; the base-turn point for a
    trombone). ``None`` as the pose when no candidate reproduces the path — a fit whose
    legs turn through more than a half circle, whose sub-paths are not shortest paths."""
    arc = path.arc
    entry = localizer_entry(path, psi)
    s_entry = float(arc[entry[0]]) if entry is not None else path.length

    def pose_at(s: float) -> Pose:
        # At the anchor the pose is the anchor's own (a node-step heading there would
        # differ by half a node's turn and force a full loop out of ``via_dubins``).
        return anchor.pose if s <= 0.0 else path.pose_at(s)

    # Reproduction is not monotone along the path (a pose just past the junction of two
    # CSC legs reproduces nothing, one further on may again), so every node is tried and
    # the earliest that reproduces is the label.
    for s in (float(s) for s in arc[arc <= s_entry + 1e-6]):
        if _cost(via_dubins(anchor, psi, d_join, *pose_at(s)), path.horizontal) < CANONICAL_VIA_TOLERANCE_M:
            return pose_at(s), s, s_entry
    return None, math.nan, s_entry


def fit_via_dubins(anchor: AnchorPose, psi: float, truth_xy: np.ndarray, d_join0: float,
                   seeds: tuple[ClosurePath, ...] = ()) -> tuple[ClosurePath, float]:
    """F3: ``(d_join, via_e, via_n, via_heading)`` by ``path_error_m``, Nelder–Mead from
    several starts — the anchor pose, each ``seeds`` path's own via (so the FITTED F3
    residual is never worse than F1's or F2's beyond re-expressing them as via-Dubins
    paths) and the truth path's poses at ``VIA_SEED_FRACTIONS`` — keeping the best; its
    labels are then canonicalised (``_canonical_label``), which may cost residual on a
    looping fit. Returns the path and the label spread: the distance between the
    canonical vias of the best and the runner-up start when the runner-up is within 10 %
    of the best cost (NaN when no runner-up is that close, or either label is not
    canonical) — how identifiable the label is on this flight."""
    def build(v):
        return via_dubins(anchor, psi, max(float(v[0]), D_JOIN_MIN_M), float(v[1]), float(v[2]), float(v[3]))

    starts = [np.array([d_join0, *anchor.pose])]        # the plain CSC from the anchor
    for p in seeds:
        if p.via is None:
            continue
        # A via at the anchor's own position with another heading would make the first
        # Dubins leg a full loop (a trombone past its join turns at once); seed the
        # anchor's exact pose instead.
        at_anchor = np.hypot(p.via[0] - anchor.position[0], p.via[1] - anchor.position[1]) < 1.0
        starts.append(np.array([p.d_join, *(anchor.pose if at_anchor else p.via)]))
    s_truth = cumulative_arc_m(truth_xy)
    for fraction in VIA_SEED_FRACTIONS:
        i = int(np.clip(np.searchsorted(s_truth, fraction * s_truth[-1]), 1, len(truth_xy) - 2))
        step = truth_xy[i + 1] - truth_xy[i - 1]
        starts.append(np.array([d_join0, truth_xy[i, 0], truth_xy[i, 1], math.atan2(step[1], step[0])]))
    objective = lambda v: _cost(build(v), truth_xy)  # noqa: E731
    solutions = sorted(((objective(x), x) for x in (_refine(objective, x0, maxfev=400) for x0 in starts)), key=lambda c: c[0])
    best_cost, best_x = solutions[0]
    if not math.isfinite(best_cost):
        raise RuntimeError("no via-Dubins start produced a path")
    labelled = _canonical_label(build(best_x), anchor, psi, best_cost)
    spread = math.nan
    if labelled.params["canonical"] and len(solutions) > 1 and solutions[1][0] <= 1.1 * best_cost:
        runner = _canonical_label(build(solutions[1][1]), anchor, psi, solutions[1][0])
        if runner.params["canonical"]:
            spread = float(np.hypot(runner.via[0] - labelled.via[0], runner.via[1] - labelled.via[1]))
    return labelled, spread


def _canonical_label(fitted: ClosurePath, anchor: AnchorPose, psi: float, fit_error_m: float) -> ClosurePath:
    """``fitted`` re-expressed by its canonical labels (join at the localizer entry, the
    earliest reproducing via) when they exist — ``params["canonical"]``; otherwise the
    fitted labels themselves (they reproduce the fit by definition, but are not
    identifiable). ``params`` also carry the via's arc fraction of the path up to its
    localizer entry (of the whole path when it has none) and the fit's own residual, so
    the labelled path's residual can be read against it."""
    entry = localizer_entry(fitted, psi)
    d_entry = entry[1] if entry is not None else fitted.d_join
    pose, s_via, s_entry = canonical_via(fitted, anchor, psi, d_entry)
    if pose is None:
        s_fitted = float(fitted.arc[int(np.argmin(np.hypot(*(fitted.horizontal - np.array(fitted.via[:2])).T)))])
        params = {**fitted.params, "via_fraction": s_fitted / s_entry if s_entry > 0 else 0.0,
                  "fit_error_m": fit_error_m, "canonical": False}
        return ClosurePath(fitted.horizontal, fitted.d_join, fitted.s_join, fitted.kind, params, fitted.via)
    labelled = via_dubins(anchor, psi, d_entry, *pose)     # reproduces: never None
    params = {**labelled.params, "via_fraction": s_via / s_entry if s_entry > 0 else 0.0,
              "fit_error_m": fit_error_m, "canonical": True}
    return ClosurePath(labelled.horizontal, labelled.d_join, labelled.s_join, labelled.kind, params, labelled.via)
