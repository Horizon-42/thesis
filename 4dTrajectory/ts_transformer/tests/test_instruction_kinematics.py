"""The flown sentence, against hand-computed cases (design §7, T5).

Each case fixes ONE of the three rules — turn, descend, accelerate — at a number that can be
worked out on paper from the constants the module imports, so a drift in any of those constants
fails here rather than quietly redrawing every track.
"""

import math

import numpy as np
import pytest

from ts_transformer.manoeuvre import instruction_kinematics as kinematics
from ts_transformer.manoeuvre.instructions import (
    INSTRUCTION_KINDS, Reading, TERMINAL_CONTINUE, TERMINAL_LANDED, Vocabulary,
)
from ts_transformer.data.approach_difficulty import ESTABLISHED_CROSS_TRACK_M
from ts_transformer.outputs.guidance.controller import (
    ACCEL_MAX_MPS2, CLIMB_MAX_RAD, DESCENT_MAX_RAD, HEIGHT_GAIN_S,
)
from ts_transformer.outputs.guidance.route import route_turn_radius_m

VOCABULARY = Vocabulary()
COLUMN = {kind: index for index, kind in enumerate(INSTRUCTION_KINDS)}
#: A speed word's own centre, so a case that is not about acceleration has none.
STEADY_WORD = 4
STEADY_MPS = VOCABULARY.speed_centre_mps(STEADY_WORD)


def sentence(heading: int = 0, altitude: int = 0, speed: int = STEADY_WORD,
             runway: int = 0, duration: int = 0) -> Reading:
    """A one-event sentence: the words below, in force from the start to the end."""
    words = np.zeros((1, len(INSTRUCTION_KINDS)), dtype=np.int64)
    words[0, COLUMN["heading"]] = heading
    words[0, COLUMN["altitude"]] = altitude
    words[0, COLUMN["speed"]] = speed
    words[0, COLUMN["runway"]] = runway
    words[0, COLUMN["duration"]] = duration
    words[0, COLUMN["terminal"]] = TERMINAL_LANDED
    return Reading(
        dataset_id="KRDU:TEST", flight_id="TEST", instructions=(),
        event_times_s=np.array([0.0]), words=words, runway="05L",
        established_from_start=True, duration_s=600.0,
    )


def start(**overrides) -> kinematics.Start:
    values = {
        "to_go_m": 20000.0, "cross_m": 0.0, "height_m": 0.0,
        "ground_speed_mps": STEADY_MPS, "relative_course_deg": 0.0,
    }
    values.update(overrides)
    return kinematics.Start(**values)


def test_holding_the_course_flies_exactly_as_far_as_the_speed_says():
    """Nothing to turn, descend or accelerate towards: 100 s at the word's own speed eats
    exactly 100 s of ground, and nothing moves sideways or vertically."""
    # The budget is the observed duration plus the overrun, applied inside `fly` because the
    # `geometry` block states that rule as part of the artefact.
    track = kinematics.fly(sentence(), VOCABULARY, start(), observed_s=80.0)
    budget_s = 80.0 + kinematics.OVERRUN_S
    assert track.end_reason == kinematics.END_TIME_CAP
    assert track.t_s[-1] == pytest.approx(budget_s)
    assert track.to_go_m[-1] == pytest.approx(20000.0 - STEADY_MPS * budget_s, abs=1e-6)
    assert np.allclose(track.cross_m, 0.0, atol=1e-9)
    assert np.allclose(track.height_m, 0.0, atol=1e-9)
    assert np.allclose(track.ground_speed_mps, STEADY_MPS)


def test_a_turn_onto_the_course_flies_the_route_builders_radius():
    """From 90° off, turning onto the course: the arc it flies divided by the angle it turns
    through IS the radius, and it must be the route builder's own — `V² / (g·tan 20°)`."""
    track = kinematics.fly(
        sentence(heading=0), VOCABULARY, start(relative_course_deg=90.0), observed_s=280,
    )
    arrived = int(np.flatnonzero(np.abs(track.relative_course_deg) > 1e-9)[-1]) + 1
    assert track.relative_course_deg[arrived] == pytest.approx(0.0, abs=1e-9)   # on it, not past it

    radius_m = STEADY_MPS * float(track.t_s[arrived]) / math.radians(90.0)
    assert radius_m == pytest.approx(route_turn_radius_m(STEADY_MPS), rel=0.02)

    # The quarter turn also displaces it by one radius along each axis. The two come out
    # +4 % and −2.5 % because a 1 s Euler step flies its whole length on the NEW course —
    # the arc is walked as a 32-sided polygon, not integrated exactly.
    assert abs(track.to_go_m[0] - track.to_go_m[arrived]) == pytest.approx(
        route_turn_radius_m(STEADY_MPS), rel=0.05,
    )
    assert abs(track.cross_m[0] - track.cross_m[arrived]) == pytest.approx(
        route_turn_radius_m(STEADY_MPS), rel=0.05,
    )


def test_the_descent_is_held_at_the_limit_then_converges_on_the_time_constant():
    """From 1000 m with the threshold as the target: far out the height error is huge, so the
    flight-path angle saturates at `DESCENT_MAX_RAD`; inside `V·HEIGHT_GAIN_S·tan(6°)` of the
    target it comes off the limit and closes exponentially with `HEIGHT_GAIN_S`."""
    track = kinematics.fly(sentence(altitude=0), VOCABULARY, start(height_m=1000.0), observed_s=280)
    rate = np.diff(track.height_m) / kinematics.STEP_S
    assert rate[0] == pytest.approx(-STEADY_MPS * math.tan(DESCENT_MAX_RAD), abs=1e-9)

    knee_m = STEADY_MPS * HEIGHT_GAIN_S * math.tan(DESCENT_MAX_RAD)
    knee = int(np.flatnonzero(track.height_m < knee_m)[0])
    after = int(knee + HEIGHT_GAIN_S / kinematics.STEP_S)
    assert track.height_m[after] / track.height_m[knee] == pytest.approx(1 / math.e, rel=0.08)


def test_acceleration_is_capped_at_the_controllers_limit():
    top = VOCABULARY.speed_words - 1
    target = VOCABULARY.speed_centre_mps(top)
    track = kinematics.fly(sentence(speed=top), VOCABULARY, start(), observed_s=0.0)
    rate = np.diff(track.ground_speed_mps)
    reached = int(np.flatnonzero(track.ground_speed_mps >= target - 1e-9)[0])

    # imported from the controller, NOT from `kinematics.ACCEL_MAX_MPS2`: asserting against
    # the module's own re-export would pass even if the module retyped the number.
    assert np.allclose(rate[: reached - 1], ACCEL_MAX_MPS2 * kinematics.STEP_S)
    # and it settles AT the word rather than overshooting it
    assert track.ground_speed_mps[reached] == pytest.approx(target)
    assert np.allclose(rate[reached:], 0.0)


def test_it_stops_at_the_threshold_plane_and_says_so():
    track = kinematics.fly(sentence(), VOCABULARY, start(to_go_m=500.0), observed_s=480)
    assert track.end_reason == kinematics.END_CROSSED
    assert track.to_go_m[-1] <= 0.0
    assert track.final_gap_m < STEADY_MPS      # within one step of the threshold, on the centreline


def test_passing_the_plane_off_the_centreline_is_not_a_crossing():
    """THE crossing contract: `to_go ≤ 0` AND on the final, never the plane alone. A vectored
    downwind runs parallel to the course several kilometres abeam and passes the plane out
    there; the repo cut records on that mistake once already (2026-09-09)."""
    track = kinematics.fly(
        sentence(), VOCABULARY, start(to_go_m=500.0, cross_m=4 * ESTABLISHED_CROSS_TRACK_M),
        observed_s=480,
    )
    assert track.end_reason == kinematics.END_TIME_CAP     # it flew past the plane and kept going
    assert track.to_go_m[-1] < 0.0
    assert track.final_gap_m > ESTABLISHED_CROSS_TRACK_M

    # and the same flight ON the centreline is a crossing
    close = kinematics.fly(sentence(), VOCABULARY, start(to_go_m=500.0, cross_m=100.0), observed_s=480)
    assert close.end_reason == kinematics.END_CROSSED


def test_a_plane_crossing_on_the_wrong_heading_is_not_a_crossing_either():
    """The other half of the gate: on the centreline but tracking across it."""
    track = kinematics.fly(
        sentence(heading=9), VOCABULARY,                    # +90°: straight across the course
        start(to_go_m=200.0, cross_m=0.0, relative_course_deg=90.0), observed_s=480,
    )
    assert track.end_reason == kinematics.END_TIME_CAP


def test_the_cross_track_sign_is_the_course_frames_own():
    """A positive relative course is anticlockwise in math-ENU, which is LEFT of the inbound
    course — so it must take `cross` NEGATIVE. Nothing else in this file can tell the two signs
    apart: every other lateral case either starts on the centreline with course 0, or asserts
    an absolute value."""
    track = kinematics.fly(
        sentence(heading=3), VOCABULARY,                    # +30°, held
        start(to_go_m=20000.0, cross_m=0.0, relative_course_deg=30.0), observed_s=60,
    )
    assert track.cross_m[-1] < -1000.0
    assert track.to_go_m[-1] < 20000.0                      # and it still makes ground


def test_a_flight_that_starts_past_the_threshold_plane_is_not_stopped_on_step_one():
    """17 of KRDU's 40 drawn flights enter the slice 14–26 km PAST the threshold plane, out on a
    downwind. The crossing is a TRANSITION from ahead of the plane to past it, so a flight that
    begins past it is not stopped there."""
    track = kinematics.fly(
        sentence(heading=0), VOCABULARY,
        start(to_go_m=-15000.0, cross_m=18000.0, relative_course_deg=-122.0), observed_s=480,
    )
    assert len(track.t_s) == 601
    assert track.end_reason == kinematics.END_TIME_CAP


def test_being_past_the_plane_already_is_not_crossing_it():
    """The crossing is a TRANSITION — from ahead of the threshold to past it — not a state. A
    flight that begins beyond the threshold, on the centreline and tracking the course, passes
    both halves of the on-final gate on step one; only the transition test keeps it flying.
    This is the shape 17 of KRDU's 40 drawn flights arrive in."""
    track = kinematics.fly(
        sentence(heading=0), VOCABULARY,
        start(to_go_m=-1000.0, cross_m=0.0, relative_course_deg=0.0), observed_s=180,
    )
    assert track.end_reason == kinematics.END_TIME_CAP
    assert len(track.t_s) == 301


def test_leaving_through_the_plane_outbound_is_not_a_crossing():
    """Flying the other way through the plane is not an arrival either."""
    track = kinematics.fly(
        sentence(heading=18), VOCABULARY,                  # 180°: straight back out
        start(to_go_m=-2000.0, cross_m=0.0, relative_course_deg=180.0), observed_s=180,
    )
    assert max(track.to_go_m) > 0.0                        # it did pass the plane
    assert track.end_reason == kinematics.END_TIME_CAP


def test_a_sentence_that_does_not_reach_the_runway_says_how_far_short_it_stopped():
    """The words are never extended to reach the threshold: flying 90° off the course for the
    whole budget gets nowhere, and the gap is the readout that says so."""
    track = kinematics.fly(
        sentence(heading=9), VOCABULARY, start(to_go_m=20000.0), observed_s=60,
    )
    assert track.end_reason == kinematics.END_TIME_CAP
    assert track.final_gap_m > 15000.0


def test_the_climb_limit_binds_when_the_word_is_above_the_aircraft():
    """9 of KRDU's 40 flown sentences climb at the start, because the first altitude word is
    read off the first plateau and back-dated to t = 0. The climb cap is what they climb at."""
    track = kinematics.fly(sentence(altitude=4), VOCABULARY, start(height_m=0.0), observed_s=60)
    rate = np.diff(track.height_m) / kinematics.STEP_S
    assert rate[0] == pytest.approx(STEADY_MPS * math.tan(CLIMB_MAX_RAD), abs=1e-9)


def test_the_runway_duration_and_terminal_words_take_no_part_in_the_flying():
    """The runway names the frame this is already in, the duration word says how far apart two
    events were, and the terminal word says how the sentence ENDS — not where to fly. The
    terminal one is named here because the design was written as "stop at the terminal word"
    and then reversed (§5.4-2); without a test the reversal can be silently re-flipped."""
    plain = kinematics.fly(sentence(), VOCABULARY, start(), observed_s=80.0)
    other = kinematics.fly(sentence(runway=1, duration=90), VOCABULARY, start(), observed_s=80.0)
    # neither stopped at the terminal word: both ran the whole budget
    assert plain.t_s[-1] == other.t_s[-1] == 80.0 + kinematics.OVERRUN_S
    assert np.allclose(plain.to_go_m, other.to_go_m)
    assert np.allclose(plain.cross_m, other.cross_m)
    assert np.allclose(plain.height_m, other.height_m)


def test_the_words_change_at_their_event_and_not_before():
    """Two events: the turn only begins when the second event's word comes into force."""
    words = np.zeros((2, len(INSTRUCTION_KINDS)), dtype=np.int64)
    words[:, COLUMN["speed"]] = STEADY_WORD
    words[1, COLUMN["heading"]] = 9                       # +90° at t = 60 s
    words[:, COLUMN["terminal"]] = [TERMINAL_CONTINUE, TERMINAL_LANDED]
    reading = Reading(
        dataset_id="KRDU:TEST", flight_id="TEST", instructions=(),
        event_times_s=np.array([0.0, 60.0]), words=words, runway="05L",
        established_from_start=True, duration_s=600.0,
    )
    track = kinematics.fly(reading, VOCABULARY, start(), observed_s=0.0)
    assert np.allclose(track.relative_course_deg[: int(60 / kinematics.STEP_S) + 1], 0.0)
    assert track.relative_course_deg[-1] > 45.0


def test_the_first_event_must_be_the_tracks_own_zero():
    reading = sentence()
    moved = Reading(
        dataset_id=reading.dataset_id, flight_id=reading.flight_id, instructions=(),
        event_times_s=np.array([12.0]), words=reading.words, runway=reading.runway,
        established_from_start=True, duration_s=600.0,
    )
    with pytest.raises(ValueError, match="not at the track's own 0"):
        kinematics.fly(moved, VOCABULARY, start(), observed_s=80)


def test_the_gap_is_measured_only_where_both_tracks_were_flying():
    """A flown sentence that stops early must not be averaged against the rest of the
    observation — that would measure the stopping rule, not the words."""
    track = kinematics.fly(sentence(), VOCABULARY, start(), observed_s=80.0)
    flown_s = float(track.t_s[-1])                      # 80 + the overrun
    rows = np.arange(0.0, 2 * flown_s + 2.0, 2.0)       # the observation runs twice as long
    observed = {
        "t": rows,
        "to_go_m": 20000.0 - STEADY_MPS * rows,
        "cross_m": np.zeros(len(rows)),
    }
    gap = kinematics.gap_to_observed(track, observed)
    assert gap["meanGapM"] == pytest.approx(0.0, abs=1e-6)   # same rule, same line
    assert gap["comparedS"] == pytest.approx(flown_s)
    assert gap["comparedFraction"] == pytest.approx(0.5, abs=0.01)
