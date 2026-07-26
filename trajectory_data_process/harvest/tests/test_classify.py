"""Track classification keeps identity and arrival endpoint on the assigned final pass."""

from __future__ import annotations

import math

import pytest

from final_approach import TrackPoint
from geokit import METRES_PER_DEG_LAT
from trajectory_data_process.harvest.airports import Airport, Runway
from trajectory_data_process.harvest.classify import classify_track
from trajectory_data_process.harvest.tracks import Sample, Track


LAT = 35.0
LON = -78.0
ELEVATION_M = 100.0


def _airport() -> Airport:
    runway = Runway(
        airport="KFIT",
        ident="36",
        lat=LAT,
        lon=LON,
        elevation_hae_m=ELEVATION_M,
        elevation_msl_m=ELEVATION_M,
        course_deg=0.0,
        hae_minus_msl_m=0.0,
        threshold_crossing_height_m=15.0,
        published_glidepath_deg=3.0,
        position_source="faa_cifp_path_point",
        vertical_source="faa_cifp_path_point",
    )
    return Airport("KFIT", LAT, LON, ELEVATION_M, (runway,))


def _sample(time_s: float, along_m: float, height_m: float) -> Sample:
    return Sample(
        time_s=time_s,
        lat=LAT + along_m / METRES_PER_DEG_LAT,
        lon=LON,
        alt_hae_m=ELEVATION_M + height_m,
        on_ground=False,
    )


def test_landing_anchor_belongs_to_last_inbound_pass_not_earlier_high_overflight():
    # The first pass wins a whole-track horizontal argmin because it samples the threshold
    # exactly, but it is 600 m high and travelling in the reciprocal direction.
    early = [
        _sample(float(i), along_m, 1000.0 - i * 80.0)
        for i, along_m in enumerate((5000.0, 4000.0, 3000.0, 2000.0, 1000.0, 0.0))
    ]

    # The final pass is a normal 3-degree inbound approach. Its last observation is 100 m
    # short of the threshold, so only final-pass selection can keep identity on this landing.
    slope = math.tan(math.radians(3.0))
    final = [
        _sample(
            float(len(early) + i),
            along_m,
            15.0 - slope * along_m,
        )
        for i, along_m in enumerate(range(-5000, 0, 100))
    ]
    track = Track("abc123", "FIT123", tuple([*early, *final]))

    classified = classify_track(track, _airport())

    assert classified.outcome == "assigned"
    assert classified.runway == "36"
    assert classified.landing_sample_index == len(track.samples) - 1
    assert classified.landing_sample_index > len(early)
    runway_frame = _airport().runway("36").frame("hae")
    endpoint = track.samples[classified.landing_sample_index]
    assert runway_frame.project(
        TrackPoint(endpoint.lat, endpoint.lon, endpoint.alt_hae_m)
    ).along_m == pytest.approx(-100.0)
