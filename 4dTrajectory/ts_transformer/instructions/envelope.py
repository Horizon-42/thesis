"""The allowed region of each instruction, from its issue point — one implementation for the
labeller's checks, the executor's limits and the display (vocabulary design §2).

Pure functions over numpy arrays; every angle in degrees (compass for tracks, descending
positive for path angles), every length in metres.
"""

from __future__ import annotations

import math

import numpy as np

from aerodynamic_model.common import GRAVITY_MPS2
from ts_transformer.instructions.words import wrap180


# ---- heading
def heading_band(track_deg, target_deg: float, tolerance_deg: float) -> np.ndarray:
    """Hold: the track within ±tolerance of the target (circular difference)."""
    return np.abs(wrap180(np.asarray(track_deg) - target_deg)) <= tolerance_deg


def turn_bank_ok(turn_deg: float, mean_bank_deg: float, max_bank_deg: float, bank_min_deg: float,
                 bank_max_deg: float, bank_min_from_deg: float) -> bool:
    """Turn: never beyond the highest bank; a turn of at least ``bank_min_from_deg`` is also flown
    at least at the lowest bank on average (else it ends beyond the widest arc)."""
    slow = abs(turn_deg) >= bank_min_from_deg and mean_bank_deg < bank_min_deg
    return max_bank_deg <= bank_max_deg and not slow


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
