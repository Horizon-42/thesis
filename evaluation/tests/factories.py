"""Small, shared record factories for evaluation tests.

Tests should share the input contract, not import implementation helpers from
other ``test_*.py`` modules.  Keeping the defaults physically consistent also
makes datum and threshold-height failures intentional rather than fixture drift.
"""

from __future__ import annotations

from typing import Any

from evaluation import AssessmentContext
from evaluation.thresholds import Benchmark
from final_approach.event_contract import CENSORED_EVENT_METHOD, EVENT_SCHEMA_VERSION


TARGET = {
    "lat": 35.0,
    "lon": -78.0,
    "alt": 130.0,
    "V": 70.0,
    "psi": 0.0,
    "gamma": -0.05,
    "m": 60_000.0,
}


def assessment_context(
    *, benchmark: Benchmark = "lpv", baro_vnav_approved: bool | None = None
) -> AssessmentContext:
    is_lpv = benchmark == "lpv"
    return AssessmentContext(
        benchmark=benchmark,
        airport="KRDU",
        runway="05L",
        runway_course_deg=0.0,
        runway_width_m=45.72,
        runway_source="faa_nasr_apt_rwy",
        runway_source_cycle="2026-08-06",
        procedure_source=(
            "faa_cifp_path_point" if is_lpv else "faa_terminal_procedure"
        ),
        procedure_source_cycle="2026-08-06",
        threshold_elevation_hae_m=130.0,
        threshold_elevation_msl_m=100.0,
        threshold_crossing_height_m=30.0,
        lpv_lateral_fsd_m=106.75 if is_lpv else None,
        baro_vnav_approved=(
            not is_lpv if baro_vnav_approved is None else baro_vnav_approved
        ),
    )


def observed_event(
    *, cross_m: float = 0.0, vertical_m: float = 0.0
) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "status": "estimated",
        "method": CENSORED_EVENT_METHOD,
        "observability": "right_censored",
        "runway": "05L",
        "threshold_crossing_lat": TARGET["lat"],
        "threshold_crossing_lon": TARGET["lon"],
        # HAE = desired MSL crossing altitude + 30 m geoid offset.
        "threshold_crossing_altitude_m": TARGET["alt"] + 30.0 + vertical_m,
        "altitude_datum": "hae",
        "signed_cross_track_m": cross_m,
        "source_sample_range": [0, 1],
        "event_time_s": None,
        "interpolation_fraction": None,
        "extrapolation_distance_m": 325.0,
        "uncertainty": {"status": "uncalibrated"},
    }


def trajectory_payload(
    *,
    subject: str = "optimized",
    event: dict[str, Any] | None = None,
    final_lat: float = 35.0,
    final_lon: float = -78.0,
    final_alt: float = 130.0,
    final_t: float = 100.0,
) -> dict[str, Any]:
    target = dict(TARGET)
    first = {"t": 0.0, **target, "lat": 34.9, "alt": 1_000.0}
    last = {
        "t": final_t,
        **target,
        "lat": final_lat,
        "lon": final_lon,
        "alt": final_alt,
    }
    source: dict[str, Any] = {
        "id": "TEST1",
        "subject": subject,
        "arr_airport": "KRDU",
        "runway": "05L",
        "icao24": "abc123",
        "landing_time_utc": "2026-08-12T00:00:00Z",
        "flight_key": "TEST1_05L_abc123_20260812T000000Z",
    }
    if subject == "observed":
        source["hae_minus_msl_m"] = 30.0
    if event is not None:
        source["observed_threshold_event"] = event
    return {
        "source": source,
        "initial_state": {key: value for key, value in first.items() if key != "t"},
        "target_state": target,
        "final_time_s": final_t,
        "states": [first, last],
        "controls": (
            [] if subject == "observed" else [{"thrust": 1.0}, {"thrust": 1.0}]
        ),
    }


def observed_payload(
    *, cross_m: float = 0.0, vertical_m: float = 0.0
) -> dict[str, Any]:
    return trajectory_payload(
        subject="observed",
        event=observed_event(cross_m=cross_m, vertical_m=vertical_m),
    )
