"""The fitted ADS-B crossing used by the optimizer target and TS tail."""

import math

import pytest

from flight_scenarios.build import build_scenario
from flight_scenarios.fitted_approach import fit_flight_final_approach
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon


LAT, LON, ELEVATION_M = 35.0, -78.0, 100.0
CROSS_M, CROSSING_HEIGHT_M = 25.0, 17.5


def fitted_flight() -> dict:
    """A 100 m/s due-north 3-degree approach ending 500 m before threshold."""
    waypoints = []
    for along_m in range(-5000, 0, 500):
        time_s = (along_m + 5000) / 100.0
        lat = LAT + along_m / METRES_PER_DEG_LAT
        lon = LON + CROSS_M / metres_per_deg_lon(LAT)
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
        "altitude_source": "opensky_history_geoaltitude_m_to_msl_egm96",
        "runway_target": {
            "lat": LAT,
            "lon": LON,
            "elevation_msl_m": ELEVATION_M,
            "course_deg": 0.0,
            "threshold_crossing_height_m": 15.0,
            "published_glidepath_deg": 3.0,
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


def test_fitted_target_changes_only_position_from_measured_terminal_state():
    flight = fitted_flight()
    measured = build_scenario(flight)
    fitted = build_scenario(flight, target_from_fitted_adsb=True)

    assert fitted.source["target_source"] == "fitted_adsb_crossing"
    assert fitted.target.latitude == pytest.approx(LAT, abs=1e-9)
    assert fitted.target.altitude == pytest.approx(ELEVATION_M + CROSSING_HEIGHT_M)
    assert fitted.target.V == pytest.approx(measured.target.V)
    assert fitted.target.psi == pytest.approx(measured.target.psi)
    assert fitted.target.gamma == pytest.approx(measured.target.gamma)


def test_fit_refuses_unconverted_hae_waypoints():
    flight = {**fitted_flight(), "altitude_source": "opensky_history_geoaltitude_m"}
    with pytest.raises(ValueError, match="requires MSL"):
        fit_flight_final_approach(flight)
