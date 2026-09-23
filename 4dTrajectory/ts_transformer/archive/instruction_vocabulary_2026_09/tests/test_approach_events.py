"""B0′′'s measurement: the parallel-partner rule and the two event definitions."""

from __future__ import annotations

import pytest

from ts_transformer.experiments.approach_events import (
    CLIMB_M, ESTABLISHED_CROSS_M, FINAL_HALF_M, INNER_M, LOW_M, MIN_RUN_M, NEAR_M, SEPARABLE_M,
    SIDE_STEP_MAX_M, TRACK_TOLERANCE_DEG, parallel_partner,
)
from final_approach.frame import RunwayFrame, TrackPoint


KRDU = ["05L", "23R", "05R", "23L", "14", "32"]


def test_a_parallel_threshold_finds_its_partner_and_a_single_one_has_none():
    assert parallel_partner("05L", KRDU) == "05R"
    assert parallel_partner("05R", KRDU) == "05L"
    assert parallel_partner("23L", KRDU) == "23R"
    assert parallel_partner("14", KRDU) is None            # no side letter: no partner
    assert parallel_partner("32", KRDU) is None


def test_a_partner_is_never_the_threshold_itself_nor_a_different_number():
    assert parallel_partner("12L", ["12L"]) is None        # the only one of its number
    assert parallel_partner("12L", ["12L", "30R"]) is None  # 30R is the reciprocal, not the parallel


def test_the_centre_runway_pairs_with_a_side_one():
    assert parallel_partner("17C", ["17L", "17C", "17R"]) in {"17L", "17R"}


def test_the_go_around_point_must_be_low_AND_near():
    # the first reading called 460 m "low" with no distance bound, and matched ordinary approaches
    # 11-21 km out, where a 3 degree path is higher than that. Low alone is not a go-around.
    assert 0.0 < LOW_M < LOW_M + CLIMB_M
    assert NEAR_M == pytest.approx(5_000.0) and LOW_M == pytest.approx(300.0)


def test_the_runway_choice_is_read_outside_the_crossing_zone_and_must_be_sustained():
    # the first reading put every switch at 0.11-0.18 km: the thresholds and the crossing runways
    # converge there, so "the nearest centreline" flickers for every flight.
    assert INNER_M > 0.0 and MIN_RUN_M > 0.0


def test_two_parallels_are_separable_only_when_their_established_corridors_do_not_overlap():
    assert SEPARABLE_M == 2 * ESTABLISHED_CROSS_M
    assert 213.3 < SEPARABLE_M and 392.0 < SEPARABLE_M      # KSJC, KSTL: NOT separable this way
    assert 1067.9 > SEPARABLE_M and 1826.4 > SEPARABLE_M    # KRDU, KSMF: separable


def test_side_step_eligibility_follows_the_AIM_not_our_corridor():
    # AIM 5-4-19 a: parallels "separated by 1,200 feet or less"
    assert SIDE_STEP_MAX_M == pytest.approx(365.76)
    assert 213.3 <= SIDE_STEP_MAX_M                          # KSJC: a side-step is authorised
    assert 392.0 > SIDE_STEP_MAX_M                           # KSTL: it is not


def test_a_point_short_of_the_threshold_projects_NEGATIVE_along():
    """The sign that invalidated two whole readings of this runner.

    `RunwayFrame`'s along axis points ALONG THE LANDING DIRECTION, so an aircraft still on
    approach has along_m < 0 and its distance to go is -along_m. Reading `along_m > 0` as
    "ahead of the threshold" instead selects the rows PAST it, and the reciprocal end of the same
    pavement, whose along is large and positive — which is exactly what happened.
    """
    frame = RunwayFrame(ident="36", lat=36.0, lon=-78.0, elevation_m=100.0, course_deg=0.0)
    # 3 km due SOUTH of a north-facing threshold: still to fly, so along is negative
    south = TrackPoint(lat=36.0 - 3000.0 / 111_320.0, lon=-78.0, alt_m=400.0)
    assert frame.project(south).along_m < 0.0
    assert -frame.project(south).along_m == pytest.approx(3000.0, rel=1e-2)
    # 1 km PAST it, over the runway: positive
    north = TrackPoint(lat=36.0 + 1000.0 / 111_320.0, lon=-78.0, alt_m=100.0)
    assert frame.project(north).along_m > 0.0


def test_the_published_final_course_is_narrow_enough_to_separate_every_parallel_pair():
    # AIM 1-1-18 d 4: total width usually 700 ft at the threshold, so +/-350 ft
    assert FINAL_HALF_M == pytest.approx(106.68)
    for separation in (213.3, 392.0, 1067.9, 1826.4):      # KSJC, KSTL, KRDU, KSMF
        assert FINAL_HALF_M <= separation / 2 + 0.1        # the corridors touch at worst
    assert FINAL_HALF_M < ESTABLISHED_CROSS_M              # our 500 m is the wide, far-field one


def test_the_track_tolerance_is_the_established_rule_s_own():
    assert TRACK_TOLERANCE_DEG == pytest.approx(30.0)
