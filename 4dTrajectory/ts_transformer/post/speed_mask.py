"""The speed-word mask (post-training §3, §8 C3; D92): a mask of a caller on the speed column of the commanded aircraft
(prior §7 item 3), from the aircraft next ahead of it on the approach clock.

It applies only when the commanded aircraft and the aircraft next ahead of it on the approach clock (on the same
runway, or a pair separated as one: `Separation.one_runway`) are both established on their finals (D92), and while the
commanded aircraft is more than `ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M` (9,260 m, 5 NM: 7110.65BB 5-7-1 b.4) from its
threshold. Both aircraft are predicted to the time when the aircraft ahead crosses its threshold, each along its
course from its present speed along the course toward its speed target at the executor's rate
(`VocabularySpec.speed_accel_max_mps2`, vocabulary §6 item 1), holding the target once reached. A speed word whose
predicted gap on the approach clock is then less than the distance the rules require there (`Separation.distance_nm`)
is masked. When every word falls short, nothing is masked.

**The prediction is an approximation, stated**: a constant rate, no turn, descent or wind acting on the speed. The
speeds are ground speeds along the course (the ground speed of the 2 s displacement × cos(track − course)), as the
speed words are ground-speed targets (vocabulary §3.6). A recorded aircraft has no speed word: its target is its present
speed along its course. "Unchanged" and "unspecified" (the pilot's own speed, which the mask cannot predict) are never
masked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from geokit import NM_M

from ts_transformer.inference.runway_schedule import Separation
from ts_transformer.inference.separation import Traffic
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.spec import ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M
from ts_transformer.instructions.words import SPEED, UNCHANGED, Words
from ts_transformer.prior.inputs import motion
from ts_transformer.post.scene import AircraftAt


def ramp_distance_m(t_s: float, speed_mps: float, target_mps: np.ndarray, accel_mps2: float) -> np.ndarray:
    """How far an aircraft moves in ``t_s`` from ``speed_mps`` toward each ``target_mps`` at the constant rate
    ``accel_mps2``, holding the target once reached."""
    target = np.asarray(target_mps, dtype=np.float64)
    ramp_s = np.abs(target - speed_mps) / accel_mps2
    during = np.minimum(t_s, ramp_s)
    return (speed_mps * during + 0.5 * np.sign(target - speed_mps) * accel_mps2 * during ** 2
            + target * np.maximum(t_s - ramp_s, 0.0))


def ramp_time_s(distance_m: float, speed_mps: float, target_mps: float, accel_mps2: float) -> float:
    """How long the same motion takes to cover ``distance_m`` (both speeds positive)."""
    if not (speed_mps > 0.0 and target_mps > 0.0):
        raise ValueError(f"a speed of {speed_mps:.1f} m/s toward {target_mps:.1f} m/s along the course does not reach "
                         "the threshold")
    ramp_s = abs(target_mps - speed_mps) / accel_mps2
    ramp_m = 0.5 * (speed_mps + target_mps) * ramp_s
    if distance_m > ramp_m:
        return ramp_s + (distance_m - ramp_m) / target_mps
    sign = math.copysign(1.0, target_mps - speed_mps)
    return (math.sqrt(speed_mps ** 2 + 2.0 * sign * accel_mps2 * distance_m) - speed_mps) / (sign * accel_mps2)


@dataclass(frozen=True)
class SpeedCheck:
    """The mask on the commanded aircraft (module docstring): the words of the speed column it permits
    (`grammar.column_words` order) and, where it applies, the aircraft ahead, each word's predicted gap (NaN for the
    words it does not predict), the distance required, and whether every word fell short (then nothing is masked)."""

    permitted: np.ndarray
    applies: bool
    leader: int | None = None
    gaps_m: np.ndarray | None = None
    required_m: float = math.nan
    fallback: bool = False


def along_course_speeds(aircraft: AircraftAt, scene: Traffic, step_s: float) -> np.ndarray:
    """``[N]``: each aircraft's ground speed along the course of its R (module docstring), NaN without R in force;
    ``scene`` the `Traffic` of the same aircraft (`post.traffic.traffic`)."""
    moving = motion(aircraft.at, aircraft.before, aircraft.known, step_s)
    return moving.ground_speed_mps * np.cos(np.radians(scene.track_minus_course_deg))


def next_ahead(scene: Traffic, k: int, separation: Separation) -> int | None:
    """The aircraft next ahead of ``k`` on the approach clock, on its runway or one separated as one (a runway in force),
    established or not."""
    runway = scene.runway[k]
    ahead = [j for j in range(len(scene.e_m))
             if j != k and scene.runway[j] is not None and separation.one_runway(runway, scene.runway[j])
             and scene.along_m[j] > scene.along_m[k]]
    return min(ahead, key=lambda j: scene.along_m[j]) if ahead else None


def speed_check(scene: Traffic, along_speed_mps: np.ndarray, separation: Separation, words: Words,
                n_candidates: int) -> SpeedCheck:
    """The speed-word mask on the commanded aircraft (aircraft 0 of ``scene``, module docstring); ``along_speed_mps``
    each aircraft's ground speed along the course of its R."""
    speed_words = column_words(SPEED, words, n_candidates)
    every = np.ones(len(speed_words), dtype=bool)
    if not scene.established[0]:
        return SpeedCheck(every, applies=False)
    before_m = separation.along_nm[scene.runway[0]] * NM_M - float(scene.along_m[0])
    if before_m <= ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M:
        return SpeedCheck(every, applies=False)
    leader = next_ahead(scene, 0, separation)
    if leader is None or not scene.established[leader]:
        return SpeedCheck(every, applies=False)
    accel = words.spec.speed_accel_max_mps2
    leader_speed = float(along_speed_mps[leader])
    leader_threshold_m = separation.along_nm[scene.runway[leader]] * NM_M
    crossing_s = ramp_time_s(leader_threshold_m - float(scene.along_m[leader]), leader_speed, leader_speed, accel)
    predicted = np.array([w != UNCHANGED and words.speed_mps(int(w)) is not None for w in speed_words])
    targets = np.array([words.speed_mps(int(w)) if p else np.nan for w, p in zip(speed_words, predicted)])
    gaps = np.full(len(speed_words), np.nan)
    gaps[predicted] = leader_threshold_m - (float(scene.along_m[0]) + ramp_distance_m(
        crossing_s, float(along_speed_mps[0]), targets[predicted], accel))
    required = separation.distance_nm(scene.runway[leader], scene.category[leader], scene.runway[0],
                                      scene.category[0]) * NM_M
    short = predicted & ~(gaps >= required)
    fallback = bool(short[predicted].all())
    return SpeedCheck(every if fallback else ~short, applies=True, leader=leader, gaps_m=gaps, required_m=required,
                      fallback=fallback)
