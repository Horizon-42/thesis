"""Runway assignment: one runway out, and the failure modes that used to slip through."""

from __future__ import annotations

import math

import pytest
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from final_approach import Assignment, LandingScreen, RunwayFrame, TrackPoint, assign_runway
from final_approach.tests.test_fit import synthetic_approach


def parallel_frame(base: RunwayFrame, ident: str, separation_m: float) -> RunwayFrame:
    """A runway parallel to ``base``, offset ``separation_m`` to its right.

    Derived from the frame's own cross-axis rather than hand-written lat/lon, so the
    fixture cannot drift out of agreement with the projection it is testing.
    """
    course = math.radians(base.course_deg)
    east_hat, north_hat = math.sin(course), math.cos(course)
    # The +cross direction in (east, north): see RunwayFrame.project.
    east, north = separation_m * north_hat, -separation_m * east_hat
    return RunwayFrame(
        ident=ident,
        lat=base.lat + north / METRES_PER_DEG_LAT,
        lon=base.lon + east / metres_per_deg_lon(base.lat),
        elevation_m=base.elevation_m,
        course_deg=base.course_deg,
    )


# KSJC 30L/30R: 228.4 m apart, the tightest parallel pair in this project's fleet and
# the one whose shipped artifacts had 169 landings written into BOTH files.
KSJC_SEPARATION_M = 228.4
KSJC_30L = RunwayFrame(ident="30L", lat=37.362, lon=-121.929, elevation_m=18.6, course_deg=302.0)
KSJC_30R = parallel_frame(KSJC_30L, "30R", KSJC_SEPARATION_M)
# The opposite end of the same physical runway -- indistinguishable by cross-track.
KSJC_12R = RunwayFrame(ident="12R", lat=KSJC_30L.lat, lon=KSJC_30L.lon, elevation_m=18.6, course_deg=122.0)

PARALLELS = [KSJC_30L, KSJC_30R, KSJC_12R]


def test_the_fixture_really_is_one_separation_apart():
    """Guards the fixture itself: a wrong offset would make every test below vacuous."""
    assert KSJC_30L.distance_m(TrackPoint(KSJC_30R.lat, KSJC_30R.lon, 0.0)) == pytest.approx(
        KSJC_SEPARATION_M, abs=1.0
    )
    assert abs(KSJC_30L.project(TrackPoint(KSJC_30R.lat, KSJC_30R.lon, 0.0)).along_m) < 1.0


def test_assigns_the_runway_actually_flown():
    track = synthetic_approach(frame=KSJC_30L)
    result = assign_runway(track, PARALLELS)
    assert result.outcome == "assigned"
    assert result.runway == "30L"


def test_a_parallel_neighbour_never_also_claims_it():
    """The regression that matters: one track, one runway, structurally."""
    track = synthetic_approach(frame=KSJC_30L)
    result = assign_runway(track, PARALLELS)
    assert result.runway == "30L"
    # The loser was measured, not ignored -- and it scores about the runway separation.
    assert result.scores["30R"] == pytest.approx(228.4, abs=15.0)
    assert result.scores["30L"] < 5.0


def test_the_opposite_end_is_excluded_by_direction_not_offset():
    """12R shares 30L's centreline, so cross-track cannot separate them at all."""
    track = synthetic_approach(frame=KSJC_30L)
    result = assign_runway(track, PARALLELS)
    assert result.runway == "30L"
    assert "12R" not in result.scores  # dropped by ``approaching``, before scoring


def test_flying_the_other_direction_assigns_the_other_end():
    track = synthetic_approach(frame=KSJC_12R)
    result = assign_runway(track, PARALLELS)
    assert result.outcome == "assigned"
    assert result.runway == "12R"


def test_a_genuine_tie_is_reported_not_guessed():
    """Flown down the middle of two parallels: refusing to choose is the correct answer."""
    midpoint = RunwayFrame(
        ident="mid",
        lat=(KSJC_30L.lat + KSJC_30R.lat) / 2,
        lon=(KSJC_30L.lon + KSJC_30R.lon) / 2,
        elevation_m=18.6,
        course_deg=302.0,
    )
    result = assign_runway(synthetic_approach(frame=midpoint), [KSJC_30L, KSJC_30R])
    assert result.outcome == "ambiguous"
    assert result.runway is None
    assert result.margin_m is not None and result.margin_m < 50.0
    assert set(result.scores) == {"30L", "30R"}  # the evidence survives the refusal


def test_an_overflight_is_not_a_landing():
    high = synthetic_approach(frame=KSJC_30L, tch_m=3000.0, glidepath_deg=0.0)
    result = assign_runway(high, PARALLELS)
    assert result.outcome == "not_landing"
    assert result.reason is not None


def test_sparse_coverage_is_unassignable_not_misjudged():
    """A coverage failure must never be reported as a badly flown approach.

    This track DID reach the threshold area and DID descend -- it is unarguably a
    landing -- but reception was sparse enough that the window holds too few samples
    to fit. That is a statement about the receiver, so it gets its own outcome rather
    than being scored as a poor approach or silently dropped.
    """
    sparse = synthetic_approach(frame=KSJC_30L, end_along_m=-320.0, step_m=700.0)
    result = assign_runway(sparse, PARALLELS)
    assert result.outcome == "unassignable"
    assert result.runway is None
    assert result.reason is not None


def test_a_track_that_never_reaches_the_airport_is_not_a_landing():
    """Distinct from sparse coverage: no evidence it landed here at all."""
    distant = synthetic_approach(frame=KSJC_30L, end_along_m=-6000.0)
    assert assign_runway(distant, PARALLELS).outcome == "not_landing"


def test_assignment_never_rejects_a_badly_flown_approach():
    """The load-bearing property: assignment answers WHICH runway, never HOW GOOD.

    A 4.2 deg descent misses the vertical gate badly downstream -- but it must still be
    ASSIGNED, or the established/pass rates get selected into existence. The offset is
    kept away from the neighbour so this tests quality-blindness, not tie-breaking.
    """
    sloppy = synthetic_approach(frame=KSJC_30L, cross_m=-60.0, glidepath_deg=4.2)
    result = assign_runway(sloppy, PARALLELS)
    assert result.outcome == "assigned"
    assert result.runway == "30L"


def test_every_outcome_carries_its_evidence():
    result = assign_runway(synthetic_approach(frame=KSJC_30L), PARALLELS)
    assert isinstance(result, Assignment)
    assert result.fit is not None and result.fit.runway == "30L"
    assert result.margin_m is not None and result.margin_m > 0


def test_a_short_track_is_not_a_landing():
    single = [TrackPoint(KSJC_30L.lat, KSJC_30L.lon, 20.0)]
    assert assign_runway(single, PARALLELS).outcome == "not_landing"


def test_screen_thresholds_are_caller_supplied():
    """The screen is data, not hard-coded policy."""
    track = synthetic_approach(frame=KSJC_30L)
    strict = assign_runway(track, PARALLELS, screen=LandingScreen(descent_margin_m=100_000.0))
    assert strict.outcome == "not_landing"
