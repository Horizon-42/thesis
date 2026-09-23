"""The allowed region of each instruction, from its issue point — one implementation for the
labeller's checks, the executor's limits and the display (vocabulary design §2).

Pure functions over numpy arrays; every angle in degrees (compass for tracks, descending
positive for path angles), every length in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aerodynamic_model.common import GRAVITY_MPS2
from ts_transformer.instructions.words import wrap180


# ---- heading
def heading_band(track_deg, target_deg: float, tolerance_deg: float) -> np.ndarray:
    """Hold: the track within ±tolerance of the target (circular difference)."""
    return np.abs(wrap180(np.asarray(track_deg) - target_deg)) <= tolerance_deg


def turn_rate_ok(turn_deg: float, mean_rate_deg_s: float, max_rate_deg_s: float, max_bank_deg: float,
                 rate_min_deg_s: float, rate_max_deg_s: float, bank_max_deg: float, rate_min_from_deg: float) -> bool:
    """Turn: never faster than the highest turn rate nor steeper than the highest bank; a turn of at
    least ``rate_min_from_deg`` also averages at least the lowest rate (a smaller change of the
    ground track is mostly wind drift and has no lower bound)."""
    slow = abs(turn_deg) >= rate_min_from_deg and mean_rate_deg_s < rate_min_deg_s
    return max_rate_deg_s <= rate_max_deg_s and max_bank_deg <= bank_max_deg and not slow


def turn_progress_ok(track_unwrapped_deg, target_unwrapped_deg: float, tolerance_deg: float) -> bool:
    """Turn: from the first row the track moves monotonically toward the target — it never
    backs off by more than the tolerance and never passes the target by more than it."""
    track = np.asarray(track_unwrapped_deg, dtype=np.float64)
    if len(track) == 0:
        return True
    direction = 1.0 if target_unwrapped_deg >= track[0] else -1.0
    progress = (track - track[0]) * direction
    total = (target_unwrapped_deg - track[0]) * direction
    backing = np.maximum.accumulate(progress) - progress
    return bool(np.max(backing) <= tolerance_deg and np.max(progress) <= total + tolerance_deg)


def turn_path(from_track_deg: float, turn_deg: float, speeds_mps, step_s: float, rate_deg_s: float,
              bank_max_deg: float, within_deg: float) -> tuple[np.ndarray, bool]:
    """A turn at a constant turn RATE, flown at the speeds the flight flew (one per row from the
    issue row), from the issue point until its track comes within ``within_deg`` of the target:
    ``(path [k, 2] in (E, N) metres from the issue point, whether it got there)``. One point per
    row; the last one is where the track enters the band — inside the step where it does, since a
    turn enters it between rows — or the last row when the flight ends first. ``turn_deg`` > 0
    turns right (clockwise). Where the rate would need more bank than ``bank_max_deg`` at the speed
    flown, the bank-limited rate is flown instead. Each step flies the mean of its two ends' tracks
    (a second-order step). With ``within_deg`` = the heading tolerance the end is where a hold
    begins — the labeller's own definition (§3.2)."""
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    side, need = math.copysign(1.0, turn_deg), abs(turn_deg) - within_deg
    points = [np.zeros(2)]
    turned = 0.0
    for row in range(1, len(speeds)):
        if turned >= need:
            return np.array(points), True
        v = 0.5 * (speeds[row - 1] + speeds[row])
        rate = min(rate_deg_s, math.degrees(GRAVITY_MPS2 * math.tan(math.radians(bank_max_deg)) / v))
        fraction = min(1.0, (need - turned) / (rate * step_s))
        after = turned + fraction * rate * step_s
        track = math.radians(from_track_deg + side * 0.5 * (turned + after))
        points.append(points[-1] + fraction * v * step_s * np.array([math.sin(track), math.cos(track)]))
        turned = need if fraction < 1.0 else after
    return np.array(points), turned >= need


@dataclass(frozen=True)
class TurnEnds:
    """Where a heading word's turn may take the aircraft (§2.3), in metres from the issue point:
    the fastest and the slowest turn, each to where its track enters the target's band, and the
    latest start. A turn begun ``d`` seconds late is the same turn moved ``d`` seconds of straight
    flight along the issue track."""

    fast: np.ndarray        # [k, 2] the fastest turn (the highest rate, bank-limited)
    slow: np.ndarray        # [k, 2] the slowest turn (the lowest rate), or to the flight's end
    finished: bool          # the slowest turn enters the band before the flight ends
    late: np.ndarray        # [2] the latest start's shift

    @property
    def corners(self) -> np.ndarray:
        """``[4, 2]`` where the turn may end: the fastest and the slowest turn's ends, then the same
        two moved by the latest start — a parallelogram."""
        return np.array([self.fast[-1], self.slow[-1], self.slow[-1] + self.late, self.fast[-1] + self.late])

    @property
    def outline(self) -> np.ndarray:
        """``[k, 2]`` the region the turn may sweep, as a ring (first point not repeated): the fastest
        turn begun on time, the two on-time ends, the slowest turn begun as late as allowed back to
        where it began, and the straight flight before it back to the issue point. For drawing: a
        slowest turn cut short by the flight's end, on a turn of nearly 150° at a lowest rate under
        0.5°/s, can make the ring cross itself (its hold is not judged)."""
        return np.vstack((self.fast, self.slow[-1:], (self.slow + self.late)[::-1]))


def turn_ends(from_track_deg: float, turn_deg: float, speeds_mps, step_s: float, rate_min_deg_s: float,
              rate_max_deg_s: float, bank_max_deg: float, within_deg: float, delay_max_s: float) -> TurnEnds:
    """The two extreme turns of §2.3 from one issue point (`turn_path`), and the latest start at
    the issue speed."""
    fast, _ = turn_path(from_track_deg, turn_deg, speeds_mps, step_s, rate_max_deg_s, bank_max_deg, within_deg)
    slow, finished = turn_path(from_track_deg, turn_deg, speeds_mps, step_s, rate_min_deg_s, bank_max_deg, within_deg)
    track = math.radians(from_track_deg)
    late = float(np.asarray(speeds_mps)[0]) * delay_max_s * np.array([math.sin(track), math.cos(track)])
    return TurnEnds(fast=fast, slow=slow, finished=finished, late=late)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """``[k, 2]``: the convex hull of ``[n, 2]`` points, counter-clockwise in (E, N), without
    repeating the first point (Andrew's monotone chain)."""
    unique = sorted({(float(e), float(n)) for e, n in points})
    if len(unique) < 3:
        return np.array(unique)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def inside_convex(points_en, outline: np.ndarray, tolerance_m: float = 1e-6) -> np.ndarray:
    """Whether each point lies in (or within ``tolerance_m`` of) a counter-clockwise convex polygon."""
    points = np.atleast_2d(np.asarray(points_en, dtype=np.float64))
    edges = np.roll(outline, -1, axis=0) - outline
    offsets = points[:, None, :] - outline[None, :, :]
    left = (edges[None, :, 0] * offsets[..., 1] - edges[None, :, 1] * offsets[..., 0]) / np.hypot(*edges.T)[None, :]
    return (left >= -tolerance_m).all(axis=1)


@dataclass(frozen=True)
class HoldFunnel:
    """A hold's allowed positions up to some length along θ (§2.3)."""

    starts: np.ndarray              # [k, 2] where the turn may end (or the one issue point)
    length_m: float
    outline: np.ndarray             # [k, 2] counter-clockwise, first point not repeated
    start_half_width_m: float       # the turn end's half extent across θ
    end_half_width_m: float         # that, plus the widening after the length


def hold_funnel(start_points, target_deg: float, tolerance_deg: float, length_m: float) -> HoldFunnel:
    """Where the turn may end (``start_points``: `TurnEnds.corners`, or the one issue point of a
    word the flight was already holding at entry) swept ``length_m`` along θ, every point of it
    opening a ± ``tolerance_deg`` cone. The sum of a convex set and a cone is convex: the hull of
    the start points and of each moved ``length_m`` along θ and ``length_m`` · tan(tolerance) to
    either side."""
    starts = np.atleast_2d(np.asarray(start_points, dtype=np.float64))
    theta = math.radians(target_deg)
    along, right = np.array([math.sin(theta), math.cos(theta)]), np.array([math.cos(theta), -math.sin(theta)])
    spread = float(funnel_half_width_m(length_m, tolerance_deg))
    far = starts + length_m * along
    across = starts @ right
    start_width = float(across.max() - across.min()) / 2.0
    return HoldFunnel(starts=starts, length_m=float(length_m),
                      outline=convex_hull(np.vstack((starts, far - spread * right, far + spread * right))),
                      start_half_width_m=start_width, end_half_width_m=start_width + spread)


def bank_deg_from_turn_rate(turn_rate_deg_s, ground_speed_mps):
    """The coordinated bank that turns at this rate at this speed."""
    rate = np.radians(np.asarray(turn_rate_deg_s))
    return np.degrees(np.arctan(np.abs(rate) * np.asarray(ground_speed_mps) / GRAVITY_MPS2))


def funnel_half_width_m(distance_m, tolerance_deg: float):
    """Hold: how far the track may wander off the nominal line after ``distance_m``."""
    return np.asarray(distance_m) * math.tan(math.radians(tolerance_deg))


# ---- approach
def corridor_half_width_m(before_threshold_m, half_width_m: float, widening_deg: float):
    """The corridor's half width at each distance before the threshold (angular: it widens outward)."""
    return half_width_m + np.maximum(np.asarray(before_threshold_m), 0.0) * math.tan(math.radians(widening_deg))


def heading_converges(heading_deg: float, course_deg: float, right_of_course_m: float, before_threshold_m: float,
                      heading_tolerance_deg: float, half_width_m: float, widening_deg: float) -> bool:
    """Clearance: flying this heading brings the aircraft onto the final — some track within the
    heading tolerance of it, at no more than 90° to the course, is already within the corridor's
    width, or crosses the extended centreline ahead of the threshold. (A track at angle ``a``
    to the course moves right at ``sin a`` per metre flown, and toward the threshold at ``cos a``.)"""
    angle = float(wrap180(heading_deg - course_deg))
    if abs(angle) > 90.0 + heading_tolerance_deg:
        return False
    if abs(right_of_course_m) <= corridor_half_width_m(before_threshold_m, half_width_m, widening_deg):
        return True
    toward = -math.copysign(1.0, right_of_course_m)          # the side the track must point to
    best = min(max(angle + toward * heading_tolerance_deg, -90.0), 90.0)
    if best * toward <= 0.0:
        return False
    return before_threshold_m - abs(right_of_course_m) / math.tan(math.radians(abs(best))) > 0.0


def corridor(right_of_course_m, track_minus_course_deg, before_threshold_m, half_width_m: float, widening_deg: float,
             course_tolerance_deg: float) -> np.ndarray:
    """After capture: before the threshold, within the (widening) half width of the extended
    centreline, and aligned with the course within the tolerance."""
    width = corridor_half_width_m(before_threshold_m, half_width_m, widening_deg)
    return ((np.asarray(before_threshold_m) >= 0.0)
            & (np.abs(np.asarray(right_of_course_m)) <= width)
            & (np.abs(np.asarray(track_minus_course_deg)) <= course_tolerance_deg))


# ---- vertical
def vertical_tube(distance_m, start_altitude_m: float, target_m: float | None, angle_bounds_deg: tuple[float, float],
                  tolerance_m: float) -> tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` altitude along the path from the issue point.

    ``angle_bounds_deg`` is the class's ``(lowest, steepest)`` path angle (descending positive,
    a climb negative); ``target_m`` is the plane the move stops at, ``None`` for "descend to
    land". A move is bounded by the lines at the two angles, and by the target once reached —
    the class's direction says from which side (a target on the other side, within the
    tolerance, holds the tube at the target).
    """
    s = np.asarray(distance_m, dtype=np.float64)
    shallow, steep = angle_bounds_deg
    high = start_altitude_m - s * math.tan(math.radians(shallow))
    low = start_altitude_m - s * math.tan(math.radians(steep))
    if target_m is not None:
        if steep > 0.0:      # a descent class: the lines come down onto the target
            low, high = np.maximum(low, target_m), np.maximum(high, target_m)
        else:                # the climb class: the lines go up onto it
            low, high = np.minimum(low, target_m), np.minimum(high, target_m)
    return low - tolerance_m, high + tolerance_m


def level_band(altitude_m, target_m: float, tolerance_m: float) -> np.ndarray:
    return np.abs(np.asarray(altitude_m) - target_m) <= tolerance_m


# ---- speed
def speed_band(speed_mps, target_mps: float, tolerance_mps: float) -> np.ndarray:
    return np.abs(np.asarray(speed_mps) - target_mps) <= tolerance_mps


def speed_transition_ok(speed_mps, target_mps: float, tolerance_mps: float) -> bool:
    """Transition: the speed moves monotonically toward the target (backing off at most the
    tolerance) and never passes it by more than the tolerance."""
    speed = np.asarray(speed_mps, dtype=np.float64)
    if len(speed) == 0:
        return True
    direction = 1.0 if target_mps >= speed[0] else -1.0
    progress = (speed - speed[0]) * direction
    total = (target_mps - speed[0]) * direction
    backing = np.maximum.accumulate(progress) - progress
    return bool(np.max(backing) <= tolerance_mps and np.max(progress) <= total + tolerance_mps)
