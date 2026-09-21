"""Fly a sentence: the simplest kinematics that can turn, descend and slow towards the words.

A sentence says only WHERE to go — turn to this angle off the course, come down to this height
above the threshold, slow to this ground speed — and never how fast to get there; that is the
executor's business and the aircraft's (two-tier plan v3, D50). Drawing "what the words alone
say" therefore needs a rule for the how, and this is the smallest one that is still an aircraft:
turn at the route's bank angle, close a height error on the controller's time constant inside its
flight-path-angle limits, and accelerate at its limit.

IT IS A DIAGNOSTIC AND A BASELINE, NEVER THE MODEL'S ANSWER (design §5.5, and the user's standing
rule): the line it draws is what a rule-follower would fly given the words, which is exactly what
makes the distance between it and the observed track readable as "what the words did not say".

EVERY CONSTANT IS IMPORTED. The bank angle and the turn radius come from the route builder, the
height time constant, the flight-path-angle limits and the acceleration limit from the guidance
controller, gravity from the flyability geometry. Retyping any of them here would be a second
definition of a number the executor already flies by, and the two would drift apart silently.

WHAT IS NOT MODELLED, because it cannot be read out of a sentence (design §5.4): no wind (the
words are GROUND speeds, so the wind is already inside the number), no aircraft type (one bank
angle for a CRJ9 and an A333 alike), and no controller lead — a real instruction is spoken a few
seconds before the track moves, so a track flown from the words turns systematically LATE. That
is a property of the reading, not a defect of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ts_transformer.data.approach_difficulty import (
    ESTABLISHED_CROSS_TRACK_M, ESTABLISHED_TRACK_TOLERANCE_DEG,
)
from ts_transformer.data.runway_context import wrap_deg
from ts_transformer.geometry.flyability import G
from ts_transformer.manoeuvre.instructions import INSTRUCTION_KINDS, Reading, Vocabulary
from ts_transformer.outputs.guidance.controller import (
    ACCEL_MAX_MPS2, CLIMB_MAX_RAD, DESCENT_MAX_RAD,
)
from ts_transformer.outputs.guidance.route import ROUTE_BANK_RAD

#: The reading rule's sibling: a track drawn under other rules is another track, so the name
#: travels with the artefact and the view states it.
METHOD = "instruction-kinematics-v1"
STEP_S = 1.0
#: How far past the observed track the flown words are allowed to run before the integration is
#: cut. Without a cap a sentence that never reaches the threshold integrates forever; with one,
#: `end_reason` says which happened. 120 s at a 70 m/s approach speed is ~8 km of extra path.
OVERRUN_S = 120.0

END_CROSSED = "crossed-threshold"
END_TIME_CAP = "time-cap"


@dataclass(frozen=True)
class Start:
    """Where the flown words begin: the observed track's FIRST ROW (design §5.4-1).

    The sentence has no starting point in it — every word is a target, not a place — so the
    start is borrowed from the observation. The consequence belongs on the screen: the two
    tracks share a first point BY CONSTRUCTION, so the distance between them afterwards is what
    the words left unsaid, not any model's error.
    """

    to_go_m: float
    cross_m: float
    height_m: float
    ground_speed_mps: float
    relative_course_deg: float

    @classmethod
    def from_course_frame(cls, frame: dict[str, np.ndarray]) -> "Start":
        return cls(
            to_go_m=float(frame["to_go_m"][0]),
            cross_m=float(frame["cross_m"][0]),
            height_m=float(frame["height_m"][0]),
            ground_speed_mps=float(frame["ground_speed_mps"][0]),
            relative_course_deg=float(frame["relative_course_deg"][0]),
        )


@dataclass(frozen=True)
class GeometricTrack:
    """The flown sentence, in the same runway frame the words were read in.

    A word is a BAND, not a point: the vertical and speed words each carry a tolerance and an
    executor inside it has obeyed. So a sentence names a family of tracks, and this carries the
    family's edges as well as its centre — `height_lo_m` / `height_hi_m` for the vertical one
    (see below), and, for the speed one, two whole tracks of this type flown at
    `speed_scale = 1 ∓ tolerance` (a fraction of each centre IS that word's tolerance, so the
    scale reproduces every speed word at its own band edge exactly).
    """

    t_s: np.ndarray
    to_go_m: np.ndarray
    cross_m: np.ndarray
    height_m: np.ndarray
    #: The same track flown at the SHALLOWEST and STEEPEST angle each vertical word allows. They
    #: are two height columns rather than two tracks because the commanded angle enters only the
    #: height step: `step_m` is the ground speed, the turn rate reads the speed, and the stopping
    #: test reads `to_go` / `cross` / the course — so the edges share every horizontal column and
    #: the stopping time with the centre, row for row. `lo` is the SHALLOWER descent, which loses
    #: less height and therefore stays ABOVE (the climb mode included: its `lo` climbs steeper).
    height_lo_m: np.ndarray
    height_hi_m: np.ndarray
    ground_speed_mps: np.ndarray
    relative_course_deg: np.ndarray
    end_reason: str
    #: How far from the threshold it stopped, horizontally — `hypot(to_go, cross)`, NOT the
    #: along-course distance alone. A track that crosses the threshold plane two kilometres to
    #: the left has not reached the runway, and an along-course number would report 0 for it.
    #: The words are never extended to reach the runway: a sentence that does not get there
    #: says so here.
    final_gap_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tS": [round(float(v), 1) for v in self.t_s],
            "toGoM": [round(float(v), 1) for v in self.to_go_m],
            "crossM": [round(float(v), 1) for v in self.cross_m],
            "heightM": [round(float(v), 1) for v in self.height_m],
            "groundSpeedMps": [round(float(v), 2) for v in self.ground_speed_mps],
            "relCourseDeg": [round(float(v), 2) for v in self.relative_course_deg],
            "endReason": self.end_reason,
            "finalGapM": round(self.final_gap_m, 1),
        }

    def edge_dict(self) -> dict[str, Any]:
        """One EDGE of the speed band, as the frontend reads it: the plan view draws it and
        nothing else does, so it carries neither the geodetic columns (it runs within ~100 m of
        the centre, and a third line in the 3D scene is a thicker line, not a fact) nor a height
        (the height chart draws the VERTICAL band). `endS` is what the arrival window is made of
        and is checked against it."""
        return {
            "tS": [round(float(v), 1) for v in self.t_s],
            "toGoM": [round(float(v), 1) for v in self.to_go_m],
            "crossM": [round(float(v), 1) for v in self.cross_m],
            "endReason": self.end_reason,
            "endS": round(float(self.t_s[-1]), 1),
        }


def on_final(cross_m: float, relative_course_deg: float) -> bool:
    """Whether this state is on the final approach, by the SAME rule `course_frame_rows`
    applies to every observed row: inside the centreline corridor and tracking the course.

    At the threshold the on-final gate's cone is floored to the same 500 m, so this is the
    crossing contract's lateral test as well — read here from the established rule because that
    is the rule the vocabulary itself is read under (`established_from_start`).
    """
    return (
        abs(cross_m) < ESTABLISHED_CROSS_TRACK_M
        and abs(wrap_deg(relative_course_deg)) <= ESTABLISHED_TRACK_TOLERANCE_DEG
    )


def turn_rate_rad_s(speed_mps: float) -> float:
    """The most a coordinated turn at the route's bank angle gives at this speed: ``g·tan φ / V``.

    NOT ``V / route_turn_radius_m(V)``. That function floors the RADIUS below 40 m/s so a route
    arc is never drawn tighter than a slow aircraft can fly; read as a rate the same floor
    inverts the physics — it sends the turn rate to zero as V falls, where a real aircraft at a
    fixed bank turns FASTER. The two agree above 40 m/s, which is everywhere the vocabulary can
    reach (its slowest word is 100 kt), so this is about saying the right thing, not a different
    number.
    """
    return G * math.tan(ROUTE_BANK_RAD) / speed_mps


def fly(reading: Reading, vocabulary: Vocabulary, start: Start, observed_s: float,
        speed_scale: float = 1.0) -> GeometricTrack:
    """Integrate the sentence from ``start`` at `STEP_S`, until it lands or runs out of budget.

    The budget is ``observed_s + OVERRUN_S`` and is applied HERE, because `assumptions` states
    that rule as part of the artefact: a caller free to pass its own cap would make that
    statement false for anyone who read it.

    The words in force at each step come from `Reading.words_at` — the one query the executor
    uses too, so the flown words and the fed words cannot come apart. The runway and duration
    words take no part: the runway names the frame this is already in, and the duration word says
    how far apart two events were, which the real clock here already knows.

    THREE HEIGHTS COME OUT OF ONE PASS: the commanded angle, and the shallowest and steepest each
    vertical word's tolerance allows. They can share a pass because the angle enters only the
    height step — every horizontal column, and therefore the stopping test and the stopping time,
    is the same for all three. Integrating the edges separately would produce the same numbers at
    three times the cost, and would let them drift apart if this loop ever changed.

    ``speed_scale`` multiplies every speed TARGET, which is how the speed word's own band is
    flown: its tolerance is a fraction of each centre, so ``1 ± fraction`` puts every word of the
    sentence on the same edge of its band at once. It does NOT touch the start speed — that is
    borrowed from the observation (`Start`), so the centre and both edges share a first point by
    construction, exactly as the centre and the observed track do.
    """
    heading_column = INSTRUCTION_KINDS.index("heading")
    vertical_column = INSTRUCTION_KINDS.index("vertical")
    speed_column = INSTRUCTION_KINDS.index("speed")

    t = float(reading.event_times_s[0])
    if t != 0.0:
        raise ValueError(f"{reading.flight_id}: the first event is at {t:g} s, not at the track's own 0")
    limit_s = observed_s + OVERRUN_S

    to_go, cross = start.to_go_m, start.cross_m
    height, speed, course = start.height_m, start.ground_speed_mps, start.relative_course_deg
    # The band's two edges start where the centre does: the sentence says no starting point, so
    # all three borrow the observation's first row and the corridor opens from zero width.
    height_lo = height_hi = height
    rows = [(t, to_go, cross, height, height_lo, height_hi, speed, course)]
    end_reason = END_TIME_CAP

    while t < limit_s:
        was_ahead = to_go > 0.0
        words = reading.words_at(np.array([t]))[0]
        target_course = vocabulary.heading_centre_deg(int(words[heading_column]))
        vertical_word = int(words[vertical_column])
        target_gamma_deg = vocabulary.vertical_centre_deg(vertical_word)
        # The word's own band. `lo` is the SHALLOWER descent (the smaller angle, descent being
        # positive), which loses less height and so stays above; for the climb mode it is the
        # steeper climb, and stays above for the same reason.
        tolerance_deg = vocabulary.vertical_tolerance_deg(vertical_word)
        target_speed = vocabulary.speed_centre_mps(int(words[speed_column])) * speed_scale

        # turn: towards the target angle, at most as fast as the bank angle allows
        error_deg = wrap_deg(target_course - course)
        most_deg = math.degrees(turn_rate_rad_s(speed)) * STEP_S
        course = wrap_deg(course + math.copysign(min(abs(error_deg), most_deg), error_deg))

        # descend or climb: the word IS the flight path angle, so there is no error to close —
        # only the airframe's own limits, and a floor. The word counts descent POSITIVE, gamma
        # counts climb positive, hence the sign.
        gamma = min(max(math.radians(-target_gamma_deg), -DESCENT_MAX_RAD), CLIMB_MAX_RAD)
        gamma_lo = min(max(math.radians(-(target_gamma_deg - tolerance_deg)), -DESCENT_MAX_RAD), CLIMB_MAX_RAD)
        gamma_hi = min(max(math.radians(-(target_gamma_deg + tolerance_deg)), -DESCENT_MAX_RAD), CLIMB_MAX_RAD)

        # The state speed is a GROUND speed at both ends — `course_frame`'s
        # `ground_speed_mps` is the horizontal rate (measured: |Δposition| / (V·Δt) has median
        # 0.99966 over the artefact's 8,438 rows), and a speed word is defined as ground speed.
        # So the horizontal step is V·dt outright and the climb is V·tan γ·dt. Treating V as a
        # path speed (V·cos γ, V·sin γ) under-advances by 1/cos γ, up to 0.55 % at the 6° cap.
        step_m = speed * STEP_S
        # The floor the height word used to provide implicitly: an angle command does not stop on
        # its own, so without this the aircraft flies through the runway and keeps descending.
        # It levels at the threshold instead, and the sentence has to say "landed" to end.
        # Each of the three levels at the threshold on its OWN height, so the corridor closes
        # onto the floor rather than being clipped against the centre's.
        def floored(angle: float, above: float) -> float:
            return max(angle, math.atan2(-above, step_m) if above > 0.0 else 0.0)

        height += step_m * math.tan(floored(gamma, height))
        height_lo += step_m * math.tan(floored(gamma_lo, height_lo))
        height_hi += step_m * math.tan(floored(gamma_hi, height_hi))

        # advance along the NEW course. The signs are the course frame's own algebra
        # (`approach_difficulty.course_frame_rows`): with `to_go = -(e·cosψ + n·sinψ)` and
        # `cross = e·sinψ - n·cosψ`, moving at V along ψ + χ gives d(to_go)/dt = -V·cos χ and
        # d(cross)/dt = -V·sin χ exactly. A positive χ is anticlockwise in math-ENU, which is
        # LEFT of the course, so it eats `cross`.
        to_go -= step_m * math.cos(math.radians(course))
        cross -= step_m * math.sin(math.radians(course))

        # accelerate LAST, so one step is flown at one speed rather than at the old speed
        # vertically and the new one horizontally.
        speed += min(max(target_speed - speed, -ACCEL_MAX_MPS2 * STEP_S), ACCEL_MAX_MPS2 * STEP_S)
        t += STEP_S
        rows.append((t, to_go, cross, height, height_lo, height_hi, speed, course))

        # "It crossed the threshold" is `to_go ≤ 0` AND ON THE FINAL — never the plane alone.
        # A vectored flight's downwind runs parallel to the course and several km abeam, and
        # passes the plane out there; the repo cut records on that mistake once already
        # (2026-09-09, median |cross| 8.7 km). Here it stopped 14 of 40 flown sentences off the
        # centreline and 10 of them before their last heading word was ever flown.
        if was_ahead and to_go <= 0.0 and on_final(cross, course):
            end_reason = END_CROSSED
            break

    columns = np.asarray(rows, dtype=np.float64)
    return GeometricTrack(
        t_s=columns[:, 0], to_go_m=columns[:, 1], cross_m=columns[:, 2], height_m=columns[:, 3],
        height_lo_m=columns[:, 4], height_hi_m=columns[:, 5],
        ground_speed_mps=columns[:, 6], relative_course_deg=columns[:, 7],
        end_reason=end_reason, final_gap_m=float(math.hypot(columns[-1, 1], columns[-1, 2])),
    )


def speed_band(reading: Reading, vocabulary: Vocabulary, start: Start,
               observed_s: float) -> dict[str, Any]:
    """The speed word's tolerance, flown: the same sentence with every speed word at the bottom
    of its band and again at the top, and the arrival window that opens between them.

    This one cannot be a pair of columns the way the vertical band is. A speed change moves the
    horizontal step, the turn radius (``g·tanφ/V``) and therefore the moment the track crosses
    the threshold — so each edge is a track of its own, with its own clock and its own ending.

    ``arrivalWindowS`` is ``[the fast edge's crossing, the slow edge's crossing]`` and is ``None``
    unless BOTH edges reached the runway: a window whose far end is the integration budget is not
    an arrival time, it is the stopping rule, and printing it as one would measure this module
    instead of the vocabulary.
    """
    fraction = vocabulary.speed_tolerance_fraction
    low = fly(reading, vocabulary, start, observed_s, speed_scale=1.0 - fraction)
    high = fly(reading, vocabulary, start, observed_s, speed_scale=1.0 + fraction)
    both_crossed = low.end_reason == END_CROSSED and high.end_reason == END_CROSSED
    return {
        "low": low.edge_dict(),
        "high": high.edge_dict(),
        "arrivalWindowS": ([round(float(high.t_s[-1]), 1), round(float(low.t_s[-1]), 1)]
                           if both_crossed else None),
    }


def gap_to_observed(track: GeometricTrack, frame: dict[str, np.ndarray]) -> dict[str, float]:
    """How far the flown words ran from the aircraft, over the time both were flying.

    The horizontal distance in the runway frame, sampled on the observed rows inside the flown
    track's span — NOT over the whole observed track, because after the flown words stop there is
    nothing to compare against and averaging in that stretch would measure the stopping rule.
    The covered fraction is reported so the mean is never read as if it spanned the approach.
    """
    observed_t = np.asarray(frame["t"], dtype=np.float64)
    observed_t = observed_t - observed_t[0]
    inside = observed_t <= track.t_s[-1]
    times = observed_t[inside]
    to_go = np.interp(times, track.t_s, track.to_go_m)
    cross = np.interp(times, track.t_s, track.cross_m)
    gap = np.hypot(to_go - np.asarray(frame["to_go_m"])[inside], cross - np.asarray(frame["cross_m"])[inside])
    return {
        "meanGapM": round(float(gap.mean()), 1),
        "gapP95M": round(float(np.percentile(gap, 95)), 1),
        "comparedS": round(float(times[-1] - times[0]), 1),
        "comparedFraction": round(float(times[-1] - times[0]) / float(observed_t[-1] - observed_t[0]), 3),
    }


def assumptions() -> dict[str, Any]:
    """What the flown words assume, as the file states it. An approximation nobody can see
    stated is worse than none (design §5.4), so this block travels with every export and the
    view shows it."""
    return {
        "method": METHOD,
        "dtS": STEP_S,
        "bankDeg": round(math.degrees(ROUTE_BANK_RAD), 1),
        "gravityMps2": G,
        "verticalIsCommandedAngle": True,   # the word IS gamma; there is no height error to close
        "heightFloorM": 0.0,                # it levels at the threshold rather than flying through it
        "descentMaxDeg": round(math.degrees(DESCENT_MAX_RAD), 1),
        "climbMaxDeg": round(math.degrees(CLIMB_MAX_RAD), 1),
        "accelMaxMps2": ACCEL_MAX_MPS2,
        "startsAt": "observed-first-row",
        "stopRule": f"{END_CROSSED} or {END_TIME_CAP} at the observed duration + {OVERRUN_S:g} s",
        "windModelled": False,
        "aircraftTypeModelled": False,
        # Where the corridor on screen comes from, and the one thing a reader would otherwise
        # assume: the two bands are flown ONE KIND AT A TIME, the other word at its centre. The
        # joint 2×2 envelope would be wider and would answer a different question — how much of
        # the corridor is THIS kind's slack is the one worth attributing.
        "verticalBandFrom": "vocabulary.verticalTolerance",
        "speedBandFrom": "vocabulary.speedTolerance",
        "bandsAreJoint": False,
        "constantsFrom": [
            "outputs/guidance/route.py", "outputs/guidance/controller.py", "geometry/flyability.py",
        ],
    }
