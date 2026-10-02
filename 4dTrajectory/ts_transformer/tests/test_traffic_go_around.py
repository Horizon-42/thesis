"""The go-around's reward (`experiments/traffic_go_around`, multi-aircraft design §6.6 step 8 item 9) on hand-built
flights: what each of S, H, Q and the landing reads, and the ends it is not given on."""

import math

import numpy as np
import pytest

from ts_transformer.experiments.traffic_go_around import (
    CLIMB_FULL_SHARE, CLIMB_ZERO_SHARE, GO_AROUND_EXTRA_S, LANDED_WEIGHT, PART_WEIGHT, RETURN_FULL_S, RETURN_ZERO_S,
    UNSCORED, after_go_around, linear,
)

STEP_S = 2.0
STEPS = 400
AROUND = 10                       # the go-around's own step
ENTRY_M = 500.0
NEED_S = 70.0                     # the executor's own climb from 300 m to the approach altitude (given here)


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


def _score(outcome="landed", landed_here=True, judged_to=STEPS - 1, need_s=NEED_S, **flight):
    heights, captured, margins = _flight(**flight)
    return after_go_around(AROUND, outcome, landed_here, margins, heights, captured, judged_to, ENTRY_M, need_s, STEP_S,
                           GO_AROUND_EXTRA_S, GO_AROUND_EXTRA_S)


def test_a_landed_go_around_scores_the_landing_and_the_three_parts():
    scored = _score()
    assert scored.need_s == NEED_S and scored.climbed_s == 80.0 and 80.0 <= CLIMB_FULL_SHARE * NEED_S
    assert (scored.separation, scored.climb, scored.back, scored.landed) == (pytest.approx(0.5), 1.0, 1.0, 1.0)
    assert scored.margin_min == pytest.approx(1.5) and scored.back_s == 400.0
    assert scored.reward == pytest.approx(LANDED_WEIGHT + PART_WEIGHT * 2.5)
    # the most a go-around's sentence gets: under a clean landing's 1
    best = _score(margin=3.0)
    assert best.reward == pytest.approx(0.9) and best.reward < 1.0


def test_a_go_around_that_did_not_land_scores_the_three_parts_but_a_loss_or_a_crash_nothing():
    """The user, 2026-10-02: a go-around that did not land still helped the others land — S, H and Q as computed; a loss
    of separation (the user's earlier rule) and a crash (my reading) get nothing."""
    for outcome, here in (("timeout", True), ("crossed_too_high", True), ("crossed_off_runway", True),
                          ("crossed_other_runway", True), ("crossed_without_capture", True), ("below_glidepath", True),
                          ("landed", False)):                     # landed, but not in the airport's landing direction
        scored = _score(outcome=outcome, landed_here=here)
        assert scored.landed == 0.0 and scored.reward == pytest.approx(PART_WEIGHT * 2.5), outcome
    # every end the loop gives is either scored or not: the executor's, the judge's loss and the glidepath stop
    from ts_transformer.autopilot.judge import OUTCOMES
    from ts_transformer.experiments.prior_free_generation import BELOW_GLIDEPATH
    from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
    scored = {"landed", "timeout", "crossed_too_high", "crossed_off_runway", "crossed_other_runway",
              "crossed_without_capture", BELOW_GLIDEPATH}
    assert scored | set(UNSCORED) == set(OUTCOMES) | {LOST_SEPARATION, BELOW_GLIDEPATH}
    assert not scored & set(UNSCORED) and set(UNSCORED) - {LOST_SEPARATION} <= set(OUTCOMES)
    for outcome in UNSCORED:
        scored = _score(outcome=outcome)
        assert scored.reward == 0.0 and scored.separation == scored.climb == scored.back == 0.0


def test_the_climb_and_the_return_fall_off_linearly():
    late = AROUND + int(round(1.6 * NEED_S / STEP_S))
    scored = _score(reach_step=late)
    assert scored.climb == pytest.approx(linear((late - AROUND) * STEP_S, CLIMB_FULL_SHARE * NEED_S,
                                                CLIMB_ZERO_SHARE * NEED_S))
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


def test_a_climb_as_fast_as_the_executor_s_is_full_on_the_step_grid():
    """The climb is read on the 2 s step grid (the first step at the approach altitude); the executor's own time is put
    on it too, so a climb exactly as fast as the executor's own scores 1 however few seconds it needs (review of the
    floor, 2026-10-02: a 1 s need read as 2 s scored 0)."""
    for need_s in (1.0, 3.0, 5.0):
        reach = AROUND + int(math.ceil(need_s / STEP_S))
        assert _score(need_s=need_s, reach_step=reach).climb == 1.0
    assert _score(need_s=1.0, reach_step=AROUND + 3).climb == 0.0                 # 6 s against 2 s on the grid


def test_already_up_there_at_the_go_around_is_a_full_climb():
    heights, captured, margins = _flight()
    heights[:] = ENTRY_M + 50.0
    scored = after_go_around(AROUND, "landed", True, margins, heights, captured, STEPS - 1, ENTRY_M, 0.0, STEP_S,
                             GO_AROUND_EXTRA_S, GO_AROUND_EXTRA_S)
    assert scored.need_s == 0.0 and scored.climb == 1.0 and scored.climbed_s == 0.0
    assert scored.margin_min == pytest.approx(1.5)


def test_nothing_read_is_none_never_a_nan():
    """The row is JSON: a margin never read (judged alone throughout, or ended before τ) and an end not scored are
    None, never inf or NaN."""
    heights, captured, margins = _flight()
    margins[:] = np.inf
    alone = after_go_around(AROUND, "landed", True, margins, heights, captured, STEPS - 1, ENTRY_M, NEED_S, STEP_S,
                            GO_AROUND_EXTRA_S, GO_AROUND_EXTRA_S)
    assert alone.separation == 1.0 and alone.margin_min is None
    assert _score(outcome="timeout", judged_to=AROUND + 10).margin_min is None
    lost = _score(outcome="lost_separation")
    assert lost.margin_min is None and lost.need_s is None
    assert all(not (isinstance(v, float) and not math.isfinite(v)) for v in lost.fields().values())
