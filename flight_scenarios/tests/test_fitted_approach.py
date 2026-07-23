"""The fitted ADS-B crossing used by the optimizer target and TS tail."""

import math

import pytest

from flight_scenarios.build import build_scenario, build_scenarios_from_arrivals
from flight_scenarios.fitted_approach import fit_flight_final_approach
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon


LAT, LON, ELEVATION_M = 35.0, -78.0, 100.0
CROSS_M, CROSSING_HEIGHT_M = 25.0, 17.5


def fitted_flight(*, cross_slope: float = 0.0) -> dict:
    """A 100 m/s due-north 3-degree approach ending 500 m before threshold."""
    waypoints = []
    for along_m in range(-5000, 0, 500):
        time_s = (along_m + 5000) / 100.0
        lat = LAT + along_m / METRES_PER_DEG_LAT
        cross_m = CROSS_M + cross_slope * along_m
        lon = LON + cross_m / metres_per_deg_lon(LAT)
        height = CROSSING_HEIGHT_M - math.tan(math.radians(3.0)) * along_m
        waypoints.append([time_s, lon, lat, ELEVATION_M + height])
    return {
        "id": "FIT001",
        "callsign": "FIT001",
        "type": "A320",
        "icao24": "abc001",
        "arr_airport": "KFIT",
        "runway": "36",
        "landing_time_utc": "2026-01-01T00:00:00Z",
        "altitude_source": "opensky_history_geoaltitude_m",
        "runway_target": {
            "lat": LAT,
            "lon": LON,
            "elevation_msl_m": ELEVATION_M,
            "elevation_hae_m": ELEVATION_M,
            "hae_minus_msl_m": 0.0,
            "course_deg": 0.0,
            "threshold_crossing_height_m": 15.0,
            "published_glidepath_deg": 3.0,
            "position_source": "faa_cifp_path_point",
            "vertical_source": "faa_cifp_path_point",
        },
        "waypoints": waypoints,
    }


def test_fit_recovers_crossing_and_uniform_terminal_tail():
    fitted = fit_flight_final_approach(fitted_flight())
    assert fitted is not None
    assert fitted.crossing.lat == pytest.approx(LAT, abs=1e-9)
    assert fitted.crossing.lon == pytest.approx(
        LON + CROSS_M / metres_per_deg_lon(LAT), abs=1e-9
    )
    assert fitted.crossing.alt_m == pytest.approx(ELEVATION_M + CROSSING_HEIGHT_M)
    assert fitted.crossing_time_s == pytest.approx(50.0)

    tail = fitted.uniform_tail(after_time_s=44.0, dt_s=2.0)
    assert [row.time_s for row in tail] == pytest.approx([46.0, 48.0, 50.0])
    assert [fitted.frame.project(row.point).along_m for row in tail] == pytest.approx(
        [-400.0, -200.0, 0.0], abs=1e-6
    )
    assert [row.terminal for row in tail] == [False, False, True]


def test_fitted_target_uses_fitted_threshold_kinematics_not_rollout():
    cross_slope = 0.1
    flight = fitted_flight(cross_slope=cross_slope)
    threshold = [
        LON + CROSS_M / metres_per_deg_lon(LAT),
        LAT,
        ELEVATION_M,
    ]
    last_approach_position = flight["waypoints"][-1][1:]
    flight["waypoints"].extend(
        [[time_s, *last_approach_position] for time_s in (50.0, 60.0)]
        + [[time_s, *threshold] for time_s in (70.0, 80.0, 90.0)]
    )
    measured = build_scenario(flight)
    fitted = build_scenario(flight, target_from_fitted_adsb=True)

    assert measured.target.V == pytest.approx(0.0)
    assert fitted.source["target_source"] == "fitted_adsb_crossing"
    assert fitted.target.latitude == pytest.approx(LAT, abs=1e-9)
    assert fitted.target.altitude == pytest.approx(ELEVATION_M + CROSSING_HEIGHT_M)
    vertical_rate = -100.0 * math.tan(math.radians(3.0))
    ground_speed = math.hypot(100.0, cross_slope * 100.0)
    assert fitted.target.V == pytest.approx(math.hypot(ground_speed, vertical_rate))
    assert fitted.target.psi == pytest.approx(math.atan2(100.0, cross_slope * 100.0))
    assert fitted.target.gamma == pytest.approx(math.atan2(vertical_rate, ground_speed))


def test_fit_uses_hae_before_local_msl_conversion():
    fitted = fit_flight_final_approach(fitted_flight())
    assert fitted is not None
    assert fitted.frame.elevation_m == ELEVATION_M


def test_fitted_target_does_not_convert_an_already_msl_fit_twice():
    flight = fitted_flight()
    flight["runway_target"].update(
        elevation_msl_m=133.5,
        hae_minus_msl_m=-33.5,
    )

    # This is the standard CLI seam: load_model_arrivals converts the flight first,
    # then build_scenario receives the already-MSL dict.
    [scenario] = build_scenarios_from_arrivals(
        [flight],
        aircraft_type="A320",
        target_from_fitted_adsb=True,
    )

    assert scenario.target.altitude == pytest.approx(133.5 + CROSSING_HEIGHT_M)
