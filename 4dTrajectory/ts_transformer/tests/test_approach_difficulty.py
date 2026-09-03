"""Route-difficulty covariates: the mix an ADE has to be read against.

Two properties carry the weight. The covariates must describe the geometry a human would
read off a plan view (a straight-in is 1.0, a downwind-and-base is well above it), and they
must be INDEPENDENT of ``coordinate_frame`` — the chart's horizontal axes are east/north
under one setting and along/cross-runway under the other, so a covariate computed in chart
axes would silently mean two different things across an ablation.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (REPO_ROOT, TS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aerodynamic_model.common import GeodeticState  # noqa: E402
from approach_difficulty import (  # noqa: E402
    ESTABLISHED_CROSS_TRACK_M,
    approach_difficulty,
    difficulty_block,
)
from channels import target_chart_position  # noqa: E402
from coordinate_frames import AirportENUFrame, ENUFrame, RunwayAlignedFrame  # noqa: E402
from dataset import FlightSeries  # noqa: E402

THRESHOLD_LAT, THRESHOLD_LON, THRESHOLD_ALT = 35.8745, -78.802, 132.0
APPROACH_COURSE_RAD = math.radians(30.0)   # math-ENU, deliberately not axis-aligned


def _target(course_rad: float = APPROACH_COURSE_RAD) -> GeodeticState:
    return GeodeticState(
        latitude=THRESHOLD_LAT, longitude=THRESHOLD_LON, altitude=THRESHOLD_ALT,
        V=70.0, psi=course_rad, gamma=0.0, m=60_000.0,
    )


def _series_in_frame(
    world_en: list[tuple[float, float]],
    *,
    frame,
    track_rad: float,
    speed: float = 70.0,
) -> FlightSeries:
    """One flight whose horizontal path is ``world_en``, expressed in ``frame``'s axes.

    The caller gives world east/north metres from the threshold; every series is built by
    projecting THOSE through the frame, so the three frames describe one physical flight.
    Under the airport-anchored frame the threshold itself sits away from the origin, so
    the projected path is shifted by the target's chart position.
    """
    target = _target(frame_course(frame))
    shift = target_chart_position(target, frame)
    values = []
    for east, north in world_en:
        first, second = frame.from_world_horizontal(east, north)
        vel_first, vel_second = frame.from_world_horizontal(
            speed * math.cos(track_rad), speed * math.sin(track_rad)
        )
        values.append([
            first + shift[0], second + shift[1], 400.0 + shift[2],
            vel_first, vel_second, -3.0,
        ])
    times = np.arange(len(world_en), dtype=np.float64) * 2.0
    return FlightSeries(
        flight_id="TEST1_05L_abc123_20260101T000000Z",
        scenario=SimpleNamespace(target=_target(frame_course(frame))),
        frame=frame,
        times=times,
        values=np.array(values, dtype=np.float64),
    )


def frame_course(frame) -> float:
    """The approach course the frame was built on (both frames share one threshold)."""
    return getattr(frame, "heading_rad", APPROACH_COURSE_RAD)


def _frames():
    common = dict(lat0=THRESHOLD_LAT, lon0=THRESHOLD_LON, alt0=THRESHOLD_ALT)
    return [
        ENUFrame(**common),
        RunwayAlignedFrame(**common, heading_rad=APPROACH_COURSE_RAD),
        # Anchored ~2 km from the threshold: the covariates must not notice.
        AirportENUFrame(
            lat0=THRESHOLD_LAT + 0.015, lon0=THRESHOLD_LON - 0.012,
            alt0=THRESHOLD_ALT - 20.0, code="KTST",
        ),
    ]


def _inbound(distance_m: float, cross_m: float) -> tuple[float, float]:
    """World EN of a point ``distance_m`` before the threshold, ``cross_m`` to its right."""
    cosine, sine = math.cos(APPROACH_COURSE_RAD), math.sin(APPROACH_COURSE_RAD)
    return (
        -distance_m * cosine + cross_m * sine,
        -distance_m * sine - cross_m * cosine,
    )


@pytest.mark.parametrize("frame", _frames(), ids=["enu", "runway-aligned", "airport-enu"])
def test_a_straight_in_scores_unit_tortuosity_in_either_frame(frame) -> None:
    path = [_inbound(d, 0.0) for d in (12_000.0, 8_000.0, 4_000.0, 0.0)]
    series = _series_in_frame(path, frame=frame, track_rad=APPROACH_COURSE_RAD)

    difficulty = approach_difficulty(series, anchor=0)

    assert difficulty.anchor_range_m == pytest.approx(12_000.0, rel=1e-9)
    assert difficulty.remaining_path_m == pytest.approx(12_000.0, rel=1e-9)
    assert difficulty.route_tortuosity == pytest.approx(1.0, rel=1e-9)
    assert difficulty.anchor_cross_track_m == pytest.approx(0.0, abs=1e-6)
    assert difficulty.established_at_anchor


@pytest.mark.parametrize("frame", _frames(), ids=["enu", "runway-aligned", "airport-enu"])
def test_a_downwind_and_base_scores_the_ratio_of_the_two_legs(frame) -> None:
    # 12 km to go in a straight line, flown as a 10 km + 10 km dogleg: exactly 5/3.
    path = [_inbound(12_000.0, 0.0), _inbound(6_000.0, -8_000.0), _inbound(0.0, 0.0)]
    series = _series_in_frame(path, frame=frame, track_rad=APPROACH_COURSE_RAD)

    difficulty = approach_difficulty(series, anchor=0)

    assert difficulty.remaining_path_m == pytest.approx(20_000.0, rel=1e-9)
    assert difficulty.route_tortuosity == pytest.approx(5.0 / 3.0, rel=1e-9)


@pytest.mark.parametrize("frame", _frames(), ids=["enu", "runway-aligned", "airport-enu"])
def test_cross_track_is_signed_right_of_the_inbound_course(frame) -> None:
    right = _series_in_frame(
        [_inbound(12_000.0, 900.0), _inbound(0.0, 0.0)],
        frame=frame, track_rad=APPROACH_COURSE_RAD,
    )
    left = _series_in_frame(
        [_inbound(12_000.0, -900.0), _inbound(0.0, 0.0)],
        frame=frame, track_rad=APPROACH_COURSE_RAD,
    )

    assert approach_difficulty(right, 0).anchor_cross_track_m == pytest.approx(900.0, abs=1e-6)
    assert approach_difficulty(left, 0).anchor_cross_track_m == pytest.approx(-900.0, abs=1e-6)


def test_a_downwind_anchor_is_not_established() -> None:
    frame = _frames()[0]
    series = _series_in_frame(
        [_inbound(12_000.0, 9_000.0), _inbound(6_000.0, 0.0), _inbound(0.0, 0.0)],
        frame=frame, track_rad=APPROACH_COURSE_RAD,
    )
    difficulty = approach_difficulty(series, anchor=0)
    assert abs(difficulty.anchor_cross_track_m) > ESTABLISHED_CROSS_TRACK_M
    assert not difficulty.established_at_anchor


def test_an_outbound_anchor_on_the_centreline_is_not_established() -> None:
    # A teardrop flies the extended centreline in the WRONG direction: cross-track alone
    # would call this established, which is why the flag also tests the track angle.
    frame = _frames()[0]
    series = _series_in_frame(
        [_inbound(12_000.0, 0.0), _inbound(0.0, 0.0)],
        frame=frame, track_rad=APPROACH_COURSE_RAD + math.pi,
    )
    difficulty = approach_difficulty(series, anchor=0)
    assert difficulty.anchor_cross_track_m == pytest.approx(0.0, abs=1e-6)
    assert not difficulty.established_at_anchor


def test_an_anchor_past_the_threshold_is_not_established() -> None:
    frame = _frames()[0]
    series = _series_in_frame(
        [_inbound(-2_000.0, 0.0), _inbound(-4_000.0, 0.0)],
        frame=frame, track_rad=APPROACH_COURSE_RAD,
    )
    assert not approach_difficulty(series, 0).established_at_anchor


def test_every_frame_agrees_on_every_covariate() -> None:
    path = [_inbound(14_000.0, 400.0), _inbound(7_000.0, -6_000.0), _inbound(0.0, 0.0)]
    enu, *others = _frames()
    left = approach_difficulty(
        _series_in_frame(path, frame=enu, track_rad=APPROACH_COURSE_RAD), 0
    )
    for other in others:
        right = approach_difficulty(
            _series_in_frame(path, frame=other, track_rad=APPROACH_COURSE_RAD), 0
        )
        assert left.anchor_range_m == pytest.approx(right.anchor_range_m, rel=1e-9)
        assert left.remaining_path_m == pytest.approx(right.remaining_path_m, rel=1e-9)
        assert left.route_tortuosity == pytest.approx(right.route_tortuosity, rel=1e-9)
        assert left.anchor_cross_track_m == pytest.approx(
            right.anchor_cross_track_m, abs=1e-6
        )
        assert left.established_at_anchor == right.established_at_anchor


def test_the_covariates_describe_the_route_after_the_anchor_not_the_whole_track() -> None:
    # A full pattern: abeam the field on a 9 km downwind, out to 16 km, base, then final.
    # From anchor 0 the aircraft still has 37 km to fly for 9.8 km of progress; by anchor 2
    # the turns are behind it and the same flight is a plain 16 km straight-in.
    frame = _frames()[0]
    series = _series_in_frame(
        [
            _inbound(4_000.0, 9_000.0),      # abeam the threshold, on downwind
            _inbound(16_000.0, 9_000.0),     # end of downwind
            _inbound(16_000.0, 0.0),         # rolled out on the centreline
            _inbound(0.0, 0.0),              # threshold
        ],
        frame=frame, track_rad=APPROACH_COURSE_RAD,
    )
    from_downwind = approach_difficulty(series, anchor=0)
    assert from_downwind.remaining_path_m == pytest.approx(37_000.0, rel=1e-9)
    assert from_downwind.route_tortuosity > 3.5

    on_final = approach_difficulty(series, anchor=2)
    assert on_final.remaining_path_m == pytest.approx(16_000.0, rel=1e-9)
    assert on_final.route_tortuosity == pytest.approx(1.0, rel=1e-9)


def test_an_anchor_on_the_threshold_fails_loudly() -> None:
    frame = _frames()[0]
    series = _series_in_frame(
        [_inbound(0.0, 0.0), _inbound(-1_000.0, 0.0)],
        frame=frame, track_rad=APPROACH_COURSE_RAD,
    )
    with pytest.raises(ValueError, match="no approach left"):
        approach_difficulty(series, anchor=0)


def test_difficulty_block_publishes_the_thresholds_it_used() -> None:
    rows = [
        {"anchor_range_m": 12_000.0, "remaining_path_m": 12_000.0,
         "route_tortuosity": 1.0, "anchor_cross_track_m": 10.0,
         "established_at_anchor": True},
        {"anchor_range_m": 12_000.0, "remaining_path_m": 30_000.0,
         "route_tortuosity": 2.5, "anchor_cross_track_m": 9_000.0,
         "established_at_anchor": False},
    ]
    block = difficulty_block(rows)
    assert block["flights"] == 2
    assert block["established_at_anchor_fraction"] == pytest.approx(0.5)
    assert block["route_tortuosity"]["median"] == pytest.approx(1.75)
    # The definition travels with the number it produced.
    assert block["established_cross_track_m"] == ESTABLISHED_CROSS_TRACK_M
    assert "established_track_tolerance_deg" in block
