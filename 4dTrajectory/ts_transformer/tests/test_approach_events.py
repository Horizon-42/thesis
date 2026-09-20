"""B0′′'s measurement: the parallel-partner rule and the two event definitions."""

from __future__ import annotations

import pytest

from ts_transformer.experiments.approach_events import (
    CLIMB_M, ESTABLISHED_CROSS_M, INNER_M, LOW_M, MIN_RUN_M, NEAR_M, SEPARABLE_M, SIDE_STEP_MAX_M,
    parallel_partner,
)


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
