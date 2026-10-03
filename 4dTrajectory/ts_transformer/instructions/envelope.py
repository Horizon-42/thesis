"""The allowed region of each instruction — one implementation for the labeller's checks and the executor's judge
(design §3.3–§3.6).

Pure functions over numpy arrays; every angle in degrees (compass for tracks, descending
positive for path angles), every length in metres.
"""

from __future__ import annotations

import math

import numpy as np

from aerodynamic_model.common import GRAVITY_MPS2
from ts_transformer.instructions.words import wrap180


# ---- heading (§3.3)
def heading_word_rows(word_rows: list[int], lead_rows: int, end_row: int) -> list[tuple[int, int]]:
    """The rows each heading word is judged on, ``(first, stop)``: a word said at row ``r`` says where the track is
    ``lead_rows`` later, so it is judged from ``r + lead_rows`` to the next word's row plus the lead, never at or past
    ``end_row`` (the end of the sentence, or of the flight judged); empty when the lead passes it."""
    rows = []
    for row, following in zip(word_rows, [*word_rows[1:], None]):
        first = row + lead_rows
        stop = end_row if following is None else min(following + lead_rows, end_row)
        rows.append((first, max(first, stop)))
    return rows


def heading_words_inside(track_deg, words: list[tuple[int, float]], lead_rows: int, end_row: int,
                         tolerance_deg: float) -> list[dict[str, int]]:
    """Each heading word ``(row, target)`` against the track: over its rows (`heading_word_rows`), how many lie within
    ``tolerance_deg`` of its target (circular difference)."""
    track = np.asarray(track_deg)
    out = []
    for (row, target), (first, stop) in zip(words, heading_word_rows([r for r, _ in words], lead_rows, end_row)):
        inside = np.abs(wrap180(track[first:stop] - target)) <= tolerance_deg
        out.append({"row": row, "rows": int(stop - first), "inside": int(inside.sum())})
    return out


# ---- turns
def bank_deg_from_turn_rate(turn_rate_deg_s, ground_speed_mps):
    """The coordinated bank that turns at this rate at this speed."""
    rate = np.radians(np.asarray(turn_rate_deg_s))
    return np.degrees(np.arctan(np.abs(rate) * np.asarray(ground_speed_mps) / GRAVITY_MPS2))


# ---- approach
def corridor_half_width_m(before_threshold_m, half_width_m: float, widening_deg: float):
    """The corridor's half width at each distance before the threshold (angular: it widens outward)."""
    return half_width_m + np.maximum(np.asarray(before_threshold_m), 0.0) * math.tan(math.radians(widening_deg))


def corridor(right_of_course_m, track_minus_course_deg, before_threshold_m, half_width_m: float, widening_deg: float,
             course_tolerance_deg: float) -> np.ndarray:
    """The capture corridor (the labeller's capture row): before the threshold, within the (widening) half width of the extended
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
    a climb negative); ``target_m`` is the plane the move stops at, ``None`` for "no
    level-off". A move is bounded by the lines at the two angles, and by the target once reached —
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
