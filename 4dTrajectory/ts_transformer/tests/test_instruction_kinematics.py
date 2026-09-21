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
from ts_transformer.geometry.flyability import G
from ts_transformer.outputs.guidance.controller import CLIMB_MAX_RAD, DESCENT_MAX_RAD

VOCABULARY = Vocabulary()
COLUMN = {kind: index for index, kind in enumerate(INSTRUCTION_KINDS)}
#: A speed word's own centre, so a case that is not about acceleration has none.
STEADY_WORD = 4
STEADY_MPS = VOCABULARY.speed_centre_mps(STEADY_WORD)
#: The LEVEL mode's word. It is not 0 — word 0 is the climb the go-around uses — so a case that
#: is not about the vertical has to name it, or every flight in this file would climb away.
LEVEL_WORD = VOCABULARY.vertical_modes_deg.index(0.0)
GLIDEPATH_WORD = VOCABULARY.vertical_modes_deg.index(3.1)


def sentence(heading: int = 0, vertical: int = LEVEL_WORD, speed: int = STEADY_WORD,
             runway: int = 0, duration: int = 0) -> Reading:
    """A one-event sentence: the words below, in force from the start to the end."""
    words = np.zeros((1, len(INSTRUCTION_KINDS)), dtype=np.int64)
    words[0, COLUMN["heading"]] = heading
    words[0, COLUMN["vertical"]] = vertical
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


def test_a_turn_flies_the_radius_the_MEASURED_bank_gives():
    """From 90° off to a 30° word: the arc flown divided by the angle turned through IS the
    radius, and it must be `V² / (g·tan φ)` at this module's own bank.

    That bank is MEASURED on the arrival fleet (`TURN_BANK_RAD`), not borrowed from the route
    builder's 20°, which answers a different question — how tight an arc may be DRAWN. Plateau to
    plateau the fleet turns at 0.67 of what 20° gives, and a preview that turns half again too
    fast finishes each turn early and then flies straight while the aircraft is still turning.
    """
    word = VOCABULARY.heading_bin(30.0)
    target_deg = VOCABULARY.heading_centre_deg(word)
    track = kinematics.fly(
        sentence(heading=word), VOCABULARY, start(relative_course_deg=90.0), observed_s=280,
    )
    arrived = int(np.flatnonzero(np.abs(track.relative_course_deg - target_deg) > 1e-9)[-1]) + 1
    assert track.relative_course_deg[arrived] == pytest.approx(target_deg, abs=1e-9)
    radius_m = STEADY_MPS * float(track.t_s[arrived]) / math.radians(90.0 - target_deg)
    expected = STEADY_MPS ** 2 / (G * math.tan(kinematics.TURN_BANK_RAD))
    assert radius_m == pytest.approx(expected, rel=0.02)


def test_only_the_ESTABLISHED_word_tracks_the_centreline_and_a_direction_word_flies_straight():
    """The vocabulary's one POSITION word against its 72 direction words.

    Measured on 150 KRDU arrivals: a sentence whose lateral behaviour is all direction words
    reaches the threshold aligned but a median 2,464 m to the side — the real tracks are 13 m off
    at that moment — and only 36.7 % of sentences cross ON the final. That is not a decoder
    failing; it is a vocabulary of velocity targets having nothing that constrains a POSITION.
    Word 0 keeps meaning what it says (fly the course's DIRECTION); the established word joins
    the line.
    """
    established = VOCABULARY.heading_established_word
    off = kinematics.fly(sentence(heading=established), VOCABULARY, start(cross_m=3000.0), observed_s=400)
    assert abs(off.cross_m[-1]) < abs(off.cross_m[0]) / 10.0
    assert off.end_reason == kinematics.END_CROSSED
    assert 0.0 < off.relative_course_deg.max() <= kinematics.INTERCEPT_MAX_DEG + 1e-9

    # the SAME start under direction word 0 flies a parallel line and never joins it: the two
    # differ only in what they constrain, and only when the aircraft is off the line
    parallel = kinematics.fly(sentence(heading=0), VOCABULARY, start(cross_m=3000.0), observed_s=400)
    assert parallel.cross_m[-1] == pytest.approx(3000.0)
    assert parallel.end_reason == kinematics.END_TIME_CAP

    # on the centreline the two are indistinguishable, which is why 92 % of the v13 rows carrying
    # word 0 sat inside the corridor and the decoder's patch went unnoticed
    for word in (0, established):
        on = kinematics.fly(sentence(heading=word), VOCABULARY, start(), observed_s=80.0)
        assert np.allclose(on.cross_m, 0.0, atol=1e-9)


def test_the_commanded_angle_is_flown_outright_with_no_error_to_close():
    """The vertical word IS the flight path angle, so the descent is that angle from the first
    step — there is no height error and no time constant. It holds until the floor."""
    word = GLIDEPATH_WORD
    track = kinematics.fly(sentence(vertical=word), VOCABULARY, start(height_m=1000.0), observed_s=280)
    rate = np.diff(track.height_m) / kinematics.STEP_S
    expected = -STEADY_MPS * math.tan(math.radians(VOCABULARY.vertical_centre_deg(word)))
    assert rate[0] == pytest.approx(expected, abs=1e-9)
    assert rate[5] == pytest.approx(expected, abs=1e-9)         # constant, not converging


def test_the_descent_levels_at_the_threshold_instead_of_flying_through_it():
    """An angle command does not stop on its own. Without a floor the aircraft would keep
    descending below the runway; it levels at the threshold and waits for the sentence to end."""
    track = kinematics.fly(sentence(vertical=GLIDEPATH_WORD), VOCABULARY,
                           start(height_m=60.0, to_go_m=20000.0), observed_s=600)
    assert track.height_m.min() >= -1e-9
    assert track.height_m[-1] == pytest.approx(0.0, abs=1e-9)


def test_acceleration_is_capped_at_the_measured_limit():
    top = VOCABULARY.speed_words - 1
    target = VOCABULARY.speed_centre_mps(top)
    # far enough out, and long enough, that it reaches the word before it reaches the threshold:
    # the measured cap is 0.5 m/s², so 74 → 157 m/s takes 166 s
    track = kinematics.fly(sentence(speed=top), VOCABULARY, start(to_go_m=60000.0), observed_s=200.0)
    rate = np.diff(track.ground_speed_mps)
    reached = int(np.flatnonzero(track.ground_speed_mps >= target - 1e-9)[0])

    # the cap is this module's own, MEASURED on the fleet (|dV/dt| where the speed is changing is
    # p50 0.19 / p90 0.54 / p99 0.99 m/s² over 62,383 samples, so the old 1.0 was the p99)
    assert np.allclose(rate[: reached - 1], kinematics.ACCEL_MAX_MPS2 * kinematics.STEP_S)
    # and it settles AT the word rather than overshooting it
    assert track.ground_speed_mps[reached] == pytest.approx(target)
    assert np.allclose(rate[reached:], 0.0)


def test_a_start_the_sentence_could_never_reach_is_REFUSED_not_flown():
    """The start is borrowed from the observation, and an arrival's first ADS-B row is
    occasionally corrupt — one KMSY track reports 2017 m/s, 4,000 kt. Flying from it produces a
    787 km track and a 566 km "gap" that swamps every distribution it lands in, while measuring
    nothing about the vocabulary. The criterion is the failure's own physics: a start whose speed
    the acceleration cap cannot bring to the sentence's FIRST speed word inside the budget.
    """
    budget = 100.0 + kinematics.OVERRUN_S
    wanted = VOCABULARY.speed_centre_mps(STEADY_WORD)
    closable = kinematics.ACCEL_MAX_MPS2 * budget
    with pytest.raises(kinematics.UnreachableStart, match="cannot close"):
        kinematics.fly(sentence(), VOCABULARY, start(ground_speed_mps=wanted + closable + 1.0), observed_s=100.0)
    # and a start it CAN close is flown, right up to the boundary
    ok = kinematics.fly(sentence(), VOCABULARY, start(ground_speed_mps=wanted + closable - 1.0), observed_s=100.0)
    assert ok.ground_speed_mps[0] == pytest.approx(wanted + closable - 1.0)


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
        sentence(heading=36), VOCABULARY,                  # 180°: straight back out
        start(to_go_m=-2000.0, cross_m=0.0, relative_course_deg=180.0), observed_s=180,
    )
    assert max(track.to_go_m) > 0.0                        # it did pass the plane
    assert track.end_reason == kinematics.END_TIME_CAP


def test_a_sentence_that_does_not_reach_the_runway_says_how_far_short_it_stopped():
    """The words are never extended to reach the threshold: flying 90° off the course for the
    whole budget gets nowhere, and the gap is the readout that says so."""
    track = kinematics.fly(
        sentence(heading=18), VOCABULARY, start(to_go_m=20000.0), observed_s=60,
    )
    assert track.end_reason == kinematics.END_TIME_CAP
    assert track.final_gap_m > 15000.0


def test_every_vertical_word_is_inside_the_executors_limits():
    """The invariant that keeps a legal sentence flyable: the controller's limits must cover
    every angle the vocabulary can command, tolerance band included. The climb cap was 2° while
    the vertical instruction was a height; the go-around mode is a 3° climb whose band reaches
    3.21°, so the cap moved to 4°. Without this test the two drift apart silently and a flown
    go-around climbs shallower than the sentence says."""
    for word in range(VOCABULARY.vertical_words):
        commanded = math.radians(-VOCABULARY.vertical_centre_deg(word))        # climb positive
        band = math.radians(VOCABULARY.vertical_tolerance_deg(word))
        assert -DESCENT_MAX_RAD <= commanded - band, VOCABULARY.vertical_centre_deg(word)
        assert commanded + band <= CLIMB_MAX_RAD, VOCABULARY.vertical_centre_deg(word)


def test_the_go_around_climb_word_climbs_at_its_own_angle():
    """And now it is actually flown: word 0 is the 3° climb, inside the cap."""
    climb = VOCABULARY.vertical_modes_deg.index(-3.0)
    track = kinematics.fly(sentence(vertical=climb), VOCABULARY, start(height_m=0.0), observed_s=60)
    rate = np.diff(track.height_m) / kinematics.STEP_S
    assert rate[0] == pytest.approx(STEADY_MPS * math.tan(math.radians(3.0)), abs=1e-9)


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
    words[:, COLUMN["vertical"]] = LEVEL_WORD
    words[1, COLUMN["heading"]] = 18                      # +90° at t = 60 s
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


# ── the corridor a sentence allows (T10) ─────────────────────────────────────


def test_the_vertical_edges_share_every_horizontal_column_with_the_centre():
    """The claim that lets the band be two HEIGHT columns instead of two tracks: the commanded
    angle enters only the height step, so the edges are the same flight at a different height.
    If this ever stops holding, the frontend draws the fan against the wrong x values — a
    corridor somewhere else entirely — and nothing on screen would look wrong."""
    track = kinematics.fly(sentence(vertical=GLIDEPATH_WORD), VOCABULARY,
                           start(height_m=1500.0), observed_s=300.0)
    for name in ("to_go_m", "cross_m", "ground_speed_mps", "relative_course_deg"):
        assert getattr(track, name).shape == track.height_lo_m.shape
    assert track.height_lo_m.shape == track.height_m.shape == track.height_hi_m.shape


def test_the_shallower_edge_stays_above_including_the_climb_mode():
    """`lo` is the smaller descent angle, which loses less height. For the CLIMB mode the same
    rule reads as the steeper climb, and it still stays above — a reader that took the names for
    heights would turn the corridor inside out while leaving it exactly as thick."""
    for word in range(len(VOCABULARY.vertical_modes_deg)):
        track = kinematics.fly(sentence(vertical=word), VOCABULARY,
                               start(height_m=1500.0), observed_s=200.0)
        assert np.all(track.height_lo_m >= track.height_hi_m - 1e-9), VOCABULARY.vertical_centre_deg(word)


def test_the_fan_opens_with_distance_and_closes_on_the_floor():
    """Its widest point is BEFORE the threshold, not at it: the executor levels at the floor, so
    both edges arrive at 0. That is the executor's rule, not the vocabulary's slack shrinking —
    the legend has to say so, which is only true if the numbers do this."""
    track = kinematics.fly(sentence(vertical=GLIDEPATH_WORD), VOCABULARY,
                           start(height_m=900.0), observed_s=600.0)
    width = track.height_lo_m - track.height_hi_m
    assert width[0] == 0.0                                   # all three share a first point
    assert width[20] > width[5] > 0.0                        # and it opens with the ground covered
    assert track.height_m[-1] == 0.0 and width[-1] == 0.0     # then closes on the floor


def test_the_band_is_the_words_own_tolerance_not_a_fixed_angle():
    """The level mode's tolerance is ABSOLUTE and every other mode's is a fraction of itself, so
    the corridor a level word opens is not the one a 4.4° word opens."""
    level = kinematics.fly(sentence(vertical=LEVEL_WORD), VOCABULARY, start(height_m=900.0), observed_s=200.0)
    steep = kinematics.fly(sentence(vertical=VOCABULARY.vertical_modes_deg.index(4.4)), VOCABULARY,
                           start(height_m=900.0), observed_s=200.0)
    level_width = level.height_lo_m[100] - level.height_hi_m[100]
    steep_width = steep.height_lo_m[100] - steep.height_hi_m[100]
    assert steep_width > level_width > 0.0
    # and each matches the angle its own word allows, over the ground it covered
    run = 100 * STEADY_MPS * kinematics.STEP_S
    tolerance = VOCABULARY.vertical_tolerance_deg(LEVEL_WORD)
    assert level_width == pytest.approx(2 * run * math.tan(math.radians(tolerance)), rel=1e-6)


def test_the_speed_scale_moves_the_target_not_the_start():
    """The three runs share a first point by construction — the start is the observation's own
    first row (`Start`), the same rule that makes the centre and the observed track share one."""
    slow = kinematics.fly(sentence(), VOCABULARY, start(), observed_s=200.0, speed_scale=0.97)
    assert slow.ground_speed_mps[0] == STEADY_MPS
    # …and it converges on the scaled word instead of the word
    assert slow.ground_speed_mps[-1] == pytest.approx(STEADY_MPS * 0.97, rel=1e-9)


def test_the_arrival_window_is_the_two_edges_own_crossings():
    """The speed tolerance's cost is arrival TIME, so the window is what it is for. The FAST
    edge covers the same ground sooner, so it lands first and the window reads [fast, slow]."""
    reading = sentence(vertical=GLIDEPATH_WORD)
    band = kinematics.speed_band(reading, VOCABULARY, start(to_go_m=8000.0, height_m=400.0), observed_s=300.0)
    assert band["low"]["endReason"] == band["high"]["endReason"] == kinematics.END_CROSSED
    assert band["high"]["endS"] < band["low"]["endS"]
    assert band["arrivalWindowS"] == [band["high"]["endS"], band["low"]["endS"]]


def test_no_window_when_an_edge_never_reached_the_runway():
    """A window whose far end is the integration budget is not an arrival time, it is the
    stopping rule — and printing it as one would measure this module, not the vocabulary."""
    # heading 90° off the course: neither edge ever crosses the threshold on the final
    band = kinematics.speed_band(sentence(heading=18), VOCABULARY, start(), observed_s=60.0)
    assert band["low"]["endReason"] == kinematics.END_TIME_CAP
    assert band["arrivalWindowS"] is None


def test_the_assumptions_say_where_each_band_came_from():
    """A corridor on screen that cannot be traced back to the number that drew it is an
    approximation nobody can see stated — and `bandsAreJoint` is the one a reader would
    otherwise assume wrongly."""
    assumptions = kinematics.assumptions()
    assert "vertical" in assumptions["verticalBandFrom"]
    assert "speed" in assumptions["speedBandFrom"]
    assert assumptions["bandsAreJoint"] is False
