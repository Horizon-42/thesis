"""Turn-straight-turn path primitives on the horizontal chart (no torch).

The shortest path between two poses at a fixed turn radius (`dubins_csc`: the four CSC
families), arcs and straights sampled at most `PATH_STEP_M` apart, the turn radius a
`BANK_RAD` bank gives at approach speed, and the runway-axes mirrors the NumPy fitting loops
use. They live in the geometry plane because two consumers drew them; since the closure
decoder was archived (2026-09-18, `archive/closure_2026_09/closure/geometry.py`, whose
`test_closure_geometry.py` pinned the axes mirrors to that path's torch versions) the one
live drawer is the rule guidance's route builder (`outputs/guidance/route`), which uses
`chart_from_axes_np`, `dubins_csc`, `arc_points`, `segment_points` and `unit_vector` —
`runway_axes_np` and `turn_radius_m` / `BANK_RAD` are kept unread for the manoeuvre-token
plan's segment geometry.
"""

from __future__ import annotations

import math

import numpy as np

from ts_transformer.geometry.flyability import G as GRAVITY_MPS2

BANK_RAD = math.radians(25.0)       # a standard-rate-ish approach bank; not an envelope limit
TURN_SPEED_CAP_MPS = 100.0          # the turn radius is sized at approach speed
PATH_STEP_M = 50.0
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


def unit_vector(heading: float) -> np.ndarray:
    return np.array([math.cos(heading), math.sin(heading)])


def points_for(length: float, step: float) -> int:
    """Points that sample ``length`` at most ``step`` apart, endpoints included."""
    return max(2, math.ceil(length / step - 1e-9) + 1)


def segment_points(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
    """``a`` to ``b`` at most ``step`` apart; a zero-length segment is its single point,
    so the ``[1:]`` concatenations never repeat a node."""
    length = float(np.hypot(*(b - a)))
    if length < 1e-9:
        return np.asarray(a, dtype=np.float64)[None, :].copy()
    return a + np.linspace(0.0, 1.0, points_for(length, step))[:, None] * (b - a)


def arc_points(centre: np.ndarray, radius: float, start: float, sweep: float, step: float = PATH_STEP_M) -> np.ndarray:
    """An arc at most ``step`` apart; a zero-sweep arc is its single point."""
    if abs(radius * sweep) < 1e-9:
        return (centre + radius * unit_vector(start))[None, :]
    n = points_for(abs(radius * sweep), step)
    angles = start + np.linspace(0.0, sweep, n)
    return centre + radius * np.stack([np.cos(angles), np.sin(angles)], 1)


def dubins_csc(p0, h0: float, p1, h1: float, radius: float, step: float = PATH_STEP_M,
               *, end_radius: float | None = None, max_sweep_rad: float | None = None) -> np.ndarray | None:
    """Shortest turn-straight-turn path between two poses (LSL / RSR / LSR / RSL), the first
    turn at ``radius`` and the last at ``end_radius`` (``radius`` by default — the plan
    route's turns onto the join are flown at the speed the schedule has there, not at the
    anchor's; the closure decoder keeps one radius). With ``max_sweep_rad`` a family whose
    two arcs together sweep more than it is not a candidate (a loop is not a route, and
    neither is a teardrop of two half-circles).
    Identical poses (within ``POSE_TOLERANCE``) return the single point — LSR would
    otherwise fly a loop. A CSC path exists for every other pair in exact arithmetic
    (same-side circles coincide only for identical poses; a same-position pose with
    another heading is a loop, not an absence), so ``None`` is a floating-point corner
    between the two tolerances below; the fits treat it as an absent candidate."""
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    r0, r1 = float(radius), float(radius if end_radius is None else end_radius)
    if (np.hypot(*(p1 - p0)) < POSE_TOLERANCE[0]
            and abs((h1 - h0 + math.pi) % (2 * math.pi) - math.pi) < POSE_TOLERANCE[1]):
        return p0[None, :].copy()
    best = None
    for s0, s1 in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        c0 = p0 + r0 * np.array([-s0 * math.sin(h0), s0 * math.cos(h0)])
        c1 = p1 + r1 * np.array([-s1 * math.sin(h1), s1 * math.cos(h1)])
        dc = c1 - c0
        distance = float(np.hypot(*dc))
        theta = math.atan2(dc[1], dc[0])
        if s0 == s1:
            # the outer tangent of two circles turning the same way: it exists once one
            # circle is not inside the other
            if distance < abs(r0 - r1) + 1e-6:
                continue
            psi = theta + math.asin(s0 * (r0 - r1) / distance) if distance > 0.0 else theta
        else:
            # the inner tangent: the circles must not overlap
            if distance < r0 + r1:
                continue
            psi = theta + s0 * math.asin((r0 + r1) / distance)
        a0, a1 = psi - s0 * math.pi / 2, psi - s1 * math.pi / 2
        t0 = c0 + r0 * unit_vector(a0)
        t1 = c1 + r1 * unit_vector(a1)
        f0, f1 = h0 - s0 * math.pi / 2, h1 - s1 * math.pi / 2
        d0 = (s0 * (a0 - f0)) % (2 * math.pi)
        d1 = (s1 * (f1 - a1)) % (2 * math.pi)
        if max_sweep_rad is not None and d0 + d1 > max_sweep_rad:
            continue
        length = r0 * d0 + r1 * d1 + float(np.hypot(*(t1 - t0)))
        if best is None or length < best[0]:
            best = (length, s0, s1, c0, c1, f0, a0, a1, d0, d1, t0, t1)
    if best is None:
        return None
    _length, s0, s1, c0, c1, f0, a0, a1, d0, d1, t0, t1 = best
    return np.concatenate([
        arc_points(c0, r0, f0, s0 * d0, step),
        segment_points(t0, t1, step)[1:],
        arc_points(c1, r1, a1, s1 * d1, step)[1:],
    ])
