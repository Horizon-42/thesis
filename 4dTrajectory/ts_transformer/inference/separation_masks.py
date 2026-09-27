"""The closed loop's two separation masks (multi-aircraft design §3.4 layer 2): which speed words and whether the approach
clearance may be said to an aircraft now, given the aircraft around it.

Where they run (user 2026-09-28, design §9 item 18): the scene closed loop knows every aircraft's state and runway, computes
the masks here and hands the speaker the words they leave; the prior reads only the instruction language. This package
does not reach the instruction language either, so the caller reads the words: it hands in each aircraft's speed along its
course and the speed target in force, and the speed each candidate word would set.

Both masks look only at the aircraft next ahead on the approach clock on the same runway or a pair separated as one
(`Separation.one_runway`; `ahead`):

- **speed words** (`speed_check`) — only when the aircraft and the one next ahead are both established on their finals
  (before that, how far it still has to fly depends on the heading words to come, so "bound to break the rule" cannot be
  told), and only while the aircraft is more than ``no_speed_within_m`` from its threshold (7110.65BB 5-7-1 b4: no speed
  adjustment inside 5 NM). Both are predicted to the moment the one ahead crosses its threshold; a word whose predicted
  gap on the approach clock is then under the distance the rules require there (`Separation.distance_nm`: the one-runway
  radar minimum or TBL 5-5-2, the distance design §2.5 gives the model) is masked. When every word falls short (a
  **fallback**), the words with the largest gap stay — every word slower than the present speed ties while the leader
  crosses inside their ramps, and a slower one never leaves less, so the slowest word is always among them (there is
  always a word to say; the check after the step does the rest). "Unchanged" is masked when the word in force is (a new
  word must be said) — so it stays allowed when the word in force is one of those best. **The prediction is an
  approximation, stated** (design §3.4): each aircraft moves along its own course from its present speed toward its
  target at the constant pace ``accel_mps2`` — the executor's speed law without its final exponential approach, and
  without the harder braking with which its "unspecified" law reaches the pilot's own speed by the threshold (so it is
  optimistic for such a leader) — with no turn, descent or wind acting on its speed.
- **approach clearance** (`clearance_check`) — "cleared" is masked while the nearest CLEARED aircraft ahead is closer than
  the in-trail minimum on the approach clock (`separation.in_trail_m`): vectoring continues.

Distances are metres, speeds metres per second along the course (ground speed × cos(track − course)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from geokit import NM_M

from ts_transformer.inference.runway_schedule import Separation
from ts_transformer.inference.separation import Traffic, in_trail_m


def ramp_distance_m(t_s: float, speed_mps: float, target_mps: np.ndarray, accel_mps2: float) -> np.ndarray:
    """How far an aircraft moves in ``t_s`` from ``speed_mps`` toward each ``target_mps`` at the constant pace
    ``accel_mps2``, holding the target once reached."""
    target = np.asarray(target_mps, dtype=np.float64)
    ramp_s = np.abs(target - speed_mps) / accel_mps2
    during = np.minimum(t_s, ramp_s)
    return (speed_mps * during + 0.5 * np.sign(target - speed_mps) * accel_mps2 * during ** 2
            + target * np.maximum(t_s - ramp_s, 0.0))


def ramp_time_s(distance_m: float, speed_mps: float, target_mps: float, accel_mps2: float) -> float:
    """How long the same motion takes to cover ``distance_m``."""
    ramp_s = abs(target_mps - speed_mps) / accel_mps2
    ramp_m = 0.5 * (speed_mps + target_mps) * ramp_s
    if distance_m > ramp_m:
        return ramp_s + (distance_m - ramp_m) / target_mps
    sign = math.copysign(1.0, target_mps - speed_mps)
    return (math.sqrt(speed_mps ** 2 + 2.0 * sign * accel_mps2 * distance_m) - speed_mps) / (sign * accel_mps2)


def ahead(traffic: Traffic, k: int, separation: Separation, eligible: np.ndarray) -> int | None:
    """Of the ``eligible`` aircraft, the one next ahead of ``k`` on the approach clock on its runway or one separated as
    one; None without a runway in force or such an aircraft."""
    runway = traffic.runway[k]
    if runway is None:
        return None
    candidates = [j for j in range(len(traffic.e_m))
                  if j != k and eligible[j] and traffic.runway[j] is not None
                  and separation.one_runway(runway, traffic.runway[j]) and traffic.along_m[j] > traffic.along_m[k]]
    return min(candidates, key=lambda j: traffic.along_m[j]) if candidates else None


def before_threshold_m(traffic: Traffic, k: int, separation: Separation) -> float:
    """How far aircraft ``k`` still is, along its course, from its runway's threshold (its approach clock position is its
    threshold's less this)."""
    return separation.along_nm[traffic.runway[k]] * NM_M - float(traffic.along_m[k])


@dataclass(frozen=True)
class SpeedCheck:
    """The speed-word mask on one aircraft: the one next ahead, each candidate word's predicted gap to it on the approach
    clock when it crosses its threshold, the distance required there, the words left, whether "unchanged" is, and
    whether every word fell short (``fallback``: the best ones were left)."""

    leader: int
    gaps_m: np.ndarray
    required_m: float
    allowed: np.ndarray
    unchanged_allowed: bool
    fallback: bool


def speed_check(traffic: Traffic, k: int, separation: Separation, along_speed_mps: np.ndarray,
                target_mps: np.ndarray, candidates_mps: np.ndarray, in_force: int, accel_mps2: float,
                no_speed_within_m: float) -> SpeedCheck | None:
    """The speed-word mask on aircraft ``k`` (module docstring): ``along_speed_mps`` and ``target_mps`` hold every
    aircraft's speed along its course and its speed target in force; ``candidates_mps`` the target each of ``k``'s words
    would set, ``in_force`` the one in force. None where the mask does not apply: ``k`` not established, within
    ``no_speed_within_m`` of its threshold, or no established aircraft next ahead of it."""
    if not traffic.established[k] or before_threshold_m(traffic, k, separation) <= no_speed_within_m:
        return None
    leader = ahead(traffic, k, separation, np.asarray(traffic.established, dtype=bool))
    if leader is None:
        return None
    crossing_s = ramp_time_s(before_threshold_m(traffic, leader, separation), float(along_speed_mps[leader]),
                             float(target_mps[leader]), accel_mps2)
    threshold_m = separation.along_nm[traffic.runway[leader]] * NM_M
    gaps = threshold_m - (float(traffic.along_m[k])
                          + ramp_distance_m(crossing_s, float(along_speed_mps[k]), candidates_mps, accel_mps2))
    required = separation.distance_nm(traffic.runway[leader], traffic.category[leader], traffic.runway[k],
                                      traffic.category[k]) * NM_M
    fallback = not bool((gaps >= required).any())
    allowed = gaps == gaps.max() if fallback else gaps >= required     # ties are exact: the same ramp, bit for bit
    return SpeedCheck(leader, gaps, required, allowed, bool(allowed[in_force]), fallback)


@dataclass(frozen=True)
class ClearanceCheck:
    """The clearance mask on one aircraft: the nearest cleared aircraft ahead, the gap to it on the approach clock, the
    in-trail minimum, and whether "cleared" may be said."""

    leader: int
    gap_m: float
    required_m: float
    allowed: bool


def clearance_check(traffic: Traffic, k: int, separation: Separation, cleared: np.ndarray) -> ClearanceCheck | None:
    """The clearance mask on aircraft ``k`` (module docstring); ``cleared`` holds which aircraft are cleared for their
    approach now. None without a cleared aircraft ahead of it."""
    leader = ahead(traffic, k, separation, np.asarray(cleared, dtype=bool))
    if leader is None:
        return None
    gap = float(traffic.along_m[leader] - traffic.along_m[k])
    required, _known = in_trail_m(separation, traffic.category[leader], traffic.category[k])
    return ClearanceCheck(leader, gap, required, gap >= required)
