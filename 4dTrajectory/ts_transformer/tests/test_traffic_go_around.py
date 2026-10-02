"""The go-around's reward (`experiments/traffic_go_around`, multi-aircraft design §6.6 step 8 item 9) on hand-built
sentences: what each of S, H, Q and the landing reads, and the ends it is not given on."""

import math

import numpy as np
import pytest

from ts_transformer.experiments.traffic_go_around import (
    CLIMB_FULL_SHARE, CLIMB_ZERO_SHARE, GO_AROUND_EXTRA_S, LANDED_WEIGHT, PART_WEIGHT, RETURN_FULL_S, RETURN_ZERO_S,
    after_go_around, linear, runway_at,
)
from ts_transformer.autopilot.vertical import GO_AROUND_CLIMB_GRADIENT
from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, APPROACH_GO_AROUND, RUNWAY, UNCHANGED

STEP_S = 2.0
STEPS = 400
AROUND = 10                       # the go-around's own step
SPEED = 70.0
ENTRY_M = 500.0


def _said(runway_at_go_around=1):
    said = np.full((STEPS, 6), UNCHANGED)
    said[0] = [0, APPROACH_CLEARED, 0, 0, 0, 0]
    said[5, RUNWAY] = runway_at_go_around
    said[AROUND, APPROACH] = APPROACH_GO_AROUND
    said[AROUND + 80, APPROACH] = APPROACH_CLEARED
    return said


def _flight(*, reach_step=AROUND + 40, capture_step=AROUND + 200, margin=1.5):
    """300 m up at the go-around, the approach altitude reached at ``reach_step``, captured again at ``capture_step``
    (None: never), ``margin`` the tightest from 30 s after the go-around on, looser before and after."""
    heights = np.where(np.arange(STEPS) >= reach_step, ENTRY_M + 10.0, 300.0)
    captured = np.zeros(STEPS, dtype=bool)
    captured[:AROUND] = True
    if capture_step is not None:
        captured[capture_step:] = True
    margins = np.full(STEPS, np.inf)
    margins[AROUND: AROUND + 15] = 0.8 + 0.4          # tight at the go-around itself (before τ): not read
    margins[AROUND + 15: AROUND + 60] = margin
    return heights, captured, margins


def _score(outcome="landed", landed_here=True, judged_to=STEPS - 1, **flight):
    heights, captured, margins = _flight(**flight)
    return after_go_around(_said(), AROUND, outcome, landed_here, margins, heights, np.full(STEPS, SPEED), captured,
                           judged_to, ENTRY_M, STEP_S, GO_AROUND_EXTRA_S, GO_AROUND_EXTRA_S)


def test_the_runway_in_force_at_the_go_around_is_read_from_the_words():
    said = _said(runway_at_go_around=3)
    assert runway_at(said, AROUND) == 3 and runway_at(said, 4) == 0


def test_a_landed_go_around_scores_the_landing_and_the_three_parts():
    scored = _score()
    need = (ENTRY_M - 300.0) / (SPEED * GO_AROUND_CLIMB_GRADIENT)
    assert scored.need_s == pytest.approx(need) and scored.climbed_s == 80.0 and 80.0 <= CLIMB_FULL_SHARE * need
    assert (scored.separation, scored.climb, scored.back, scored.landed) == (pytest.approx(0.5), 1.0, 1.0, 1.0)
    assert scored.margin_min == pytest.approx(1.5) and scored.back_s == 400.0
    assert scored.reward == pytest.approx(LANDED_WEIGHT + PART_WEIGHT * 2.5)
    # the most a go-around's sentence gets: under a clean landing's 1
    best = _score(margin=3.0)
    assert best.reward == pytest.approx(0.9) and best.reward < 1.0


def test_a_safe_timeout_scores_the_three_parts_and_any_other_end_nothing():
    assert _score(outcome="timeout").reward == pytest.approx(PART_WEIGHT * 2.5)
    for outcome, here in (("lost_separation", True), ("crossed_off_runway", True), ("below_glidepath", True),
                          ("landed", False)):                     # landed, but not in the airport's landing direction
        scored = _score(outcome=outcome, landed_here=here)
        assert scored.reward == 0.0 and scored.separation == scored.climb == scored.back == 0.0


def test_the_climb_and_the_return_fall_off_linearly():
    need = (ENTRY_M - 300.0) / (SPEED * GO_AROUND_CLIMB_GRADIENT)
    late = AROUND + int(round(1.6 * need / STEP_S))
    scored = _score(reach_step=late)
    assert scored.climb == pytest.approx(linear((late - AROUND) * STEP_S, CLIMB_FULL_SHARE * need,
                                                CLIMB_ZERO_SHARE * need))
    assert 0.0 < scored.climb < 1.0
    assert _score(reach_step=STEPS).climb == 0.0                  # never up there
    back = _score(capture_step=AROUND + 250)                       # 500 s
    assert back.back == pytest.approx((RETURN_ZERO_S - 500.0) / (RETURN_ZERO_S - RETURN_FULL_S))
    assert _score(capture_step=None).back == 0.0 and _score(capture_step=None).back_s is None


def test_the_return_is_given_only_when_separation_and_the_climb_are():
    assert _score(margin=1.0).separation == 0.0 and _score(margin=1.0).back == 0.0
    assert _score(reach_step=STEPS).back == 0.0
    assert _score(margin=1.0, outcome="timeout").reward == pytest.approx(PART_WEIGHT * 1.0)


def test_a_capture_before_tau_reads_the_one_step_at_tau():
    scored = _score(capture_step=AROUND + 5, margin=1.25)
    assert scored.margin_min == pytest.approx(1.25) and scored.separation == pytest.approx(0.25)
    # ended before τ: nothing to read
    assert _score(outcome="timeout", judged_to=AROUND + 10).separation == 0.0


def test_already_up_there_at_the_go_around_is_a_full_climb():
    heights, captured, margins = _flight()
    heights[:] = ENTRY_M + 50.0
    scored = after_go_around(_said(), AROUND, "landed", True, margins, heights, np.full(STEPS, SPEED), captured,
                             STEPS - 1, ENTRY_M, STEP_S, GO_AROUND_EXTRA_S, GO_AROUND_EXTRA_S)
    assert scored.need_s == 0.0 and scored.climb == 1.0 and scored.climbed_s == 0.0
    assert scored.margin_min == pytest.approx(1.5)


def test_nothing_read_is_none_never_a_nan():
    """The row is JSON: a margin never read (judged alone throughout, or ended before τ) and an end not scored are
    None, never inf or NaN."""
    heights, captured, margins = _flight()
    margins[:] = np.inf
    alone = after_go_around(_said(), AROUND, "landed", True, margins, heights, np.full(STEPS, SPEED), captured,
                            STEPS - 1, ENTRY_M, STEP_S, GO_AROUND_EXTRA_S, GO_AROUND_EXTRA_S)
    assert alone.separation == 1.0 and alone.margin_min is None
    assert _score(outcome="timeout", judged_to=AROUND + 10).margin_min is None
    lost = _score(outcome="lost_separation")
    assert lost.margin_min is None and lost.need_s is None
    assert all(not (isinstance(v, float) and not math.isfinite(v)) for v in lost.fields().values())
