"""Small, shared record factories for evaluation tests.

Tests should share the input contract, not import implementation helpers from
other ``test_*.py`` modules.  Keeping the defaults physically consistent also
makes datum and threshold-height failures intentional rather than fixture drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation import AssessmentContext
from evaluation.thresholds import Benchmark
from final_approach.event_contract import CENSORED_EVENT_METHOD, EVENT_SCHEMA_VERSION
from flight_scenarios.crossing_span import CROSSING_SPAN_KEY, crossing_span_from_event
from geokit import metres_per_deg_lon
from trajectory_data_process.harvest.airports import APPROACH_LEG_TCH_SOURCE


TARGET = {
    "lat": 35.0,
    "lon": -78.0,
    "alt": 130.0,
    "V": 70.0,
    "psi": 0.0,
    "gamma": -0.05,
    "m": 60_000.0,
}

# Narrow-body stall facts: an A320 wing (S = 122.6 m²) with the 30–100 t bucket's landing
# Cl_max of 2.7 (aero_params_for_aircraft; the A320 family itself is calibrated to 3.0,
# which the fixture deliberately does not track -- the block is producer-supplied and
# the tests derive their window from it). At the TARGET mass of 60 t the 1-g window is
# [66.3, 76.6] m/s, so the default crossing V of 70.0 m/s passes with margin on both sides.
LANDING_AERO = {"wing_area_m2": 122.6, "cl_max_landing": 2.7}

# One row of a computed record's controls (the contract's three columns: thrust
# fraction, bank, load factor). Straight and level: the speed gate reads load_factor.
CONTROL = {"thrust": 1.0, "bank_rad": 0.0, "load_factor": 1.0}


def assessment_context(
    *, benchmark: Benchmark = "lpv", baro_vnav_approved: bool | None = None
) -> AssessmentContext:
    is_lpv = benchmark == "lpv"
    return AssessmentContext(
        benchmark=benchmark,
        airport="KRDU",
        runway="05L",
        threshold_lat=TARGET["lat"],
        threshold_lon=TARGET["lon"],
        runway_course_deg=0.0,
        runway_width_m=45.72,
        runway_source="faa_nasr_apt_rwy",
        runway_source_cycle="2026-08-06",
        # The non-LPV context has a TCH and approved Baro-VNAV: the runway-leg source.
        procedure_source=(
            "faa_cifp_path_point" if is_lpv else APPROACH_LEG_TCH_SOURCE
        ),
        procedure_source_cycle="2026-08-06",
        threshold_elevation_hae_m=130.0,
        threshold_elevation_msl_m=100.0,
        threshold_crossing_height_m=30.0,
        lpv_course_width_m=106.75 if is_lpv else None,
        baro_vnav_approved=(
            not is_lpv if baro_vnav_approved is None else baro_vnav_approved
        ),
    )


def observed_event(
    *,
    cross_m: float = 0.0,
    vertical_m: float = 0.0,
    # In the window at the default 60 t A320-class mass ([66.3, 76.6] m/s), so a
    # default observed fixture passes all three gates; pass None to model an event
    # that fitted no speed (pre-field or too few speed-bearing samples).
    ground_speed_m_s: float | None = 70.0,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "status": "estimated",
        "method": CENSORED_EVENT_METHOD,
        "observability": "right_censored",
        "runway": "05L",
        # The crossing position ENCODES the cross-track offset (runway course 0°:
        # cross-track = east), so grading the position recovers ``cross_m`` — the
        # payload is physically self-consistent, not two disagreeing claims.
        "threshold_crossing_lat": TARGET["lat"],
        "threshold_crossing_lon": (
            TARGET["lon"] + cross_m / metres_per_deg_lon(TARGET["lat"])
        ),
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
    if ground_speed_m_s is not None:
        event["crossing_ground_speed_m_s"] = ground_speed_m_s
    return event


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
    states = [first, last]
    if subject == "observed":
        source["hae_minus_msl_m"] = 30.0
        # The resolved-airframe stall facts the baseline speed gate anchors on —
        # the same block the producer writes for a resolvable icao24.
        source["landing_aero"] = dict(LANDING_AERO)
        if event is not None and event.get("status") == "estimated":
            # The span the real producer serializes: marker + (for censored
            # events) the appended inferred crossing row. final_time_s below
            # stays anchored to the last MEASURED row, per the record contract.
            span, appended = crossing_span_from_event(
                event, states, hae_minus_msl_m=30.0
            )
            source[CROSSING_SPAN_KEY] = span
            states = states + appended
    else:
        # Computed records carry the producer-written stall facts the speed gate
        # anchors on; without them the composite verdict is indeterminate by design.
        source["landing_aero"] = dict(LANDING_AERO)
    if event is not None:
        source["observed_threshold_event"] = event
    return {
        "source": source,
        "initial_state": {key: value for key, value in first.items() if key != "t"},
        "target_state": target,
        "final_time_s": final_t,
        "states": states,
        "controls": (
            [] if subject == "observed" else [dict(CONTROL), dict(CONTROL)]
        ),
    }


def observed_payload(
    *, cross_m: float = 0.0, vertical_m: float = 0.0
) -> dict[str, Any]:
    return trajectory_payload(
        subject="observed",
        event=observed_event(cross_m=cross_m, vertical_m=vertical_m),
    )


def write_batch(root: Path, payloads: list[dict[str, Any]]) -> Path:
    """A batch directory the way the modeling producers write one: records plus a
    ``summary.json`` roster naming each row's airport and runway."""
    rows = []
    for index, payload in enumerate(payloads):
        name = f"record_{index}_eval.json"
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
        rows.append({
            "eval_file": name,
            "arr_airport": payload["source"]["arr_airport"],
            "runway": payload["source"]["runway"],
        })
    (root / "summary.json").write_text(json.dumps({"results": rows}), encoding="utf-8")
    return root
