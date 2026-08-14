"""Arrival-event dispatch uses stored geometry and never refits ADS-B."""

from __future__ import annotations

import inspect

import pytest

import evaluation.arrival as arrival_module
from evaluation import AssessmentContext, arrival_deviation, record_from_dict


def context(*, benchmark="lpv") -> AssessmentContext:
    return AssessmentContext(
        benchmark=benchmark,
        airport="KRDU",
        runway="05L",
        runway_course_deg=0.0,
        runway_width_m=45.72,
        runway_source="faa_nasr_apt_rwy",
        runway_source_cycle="2026-08-06",
        procedure_source="faa_cifp_path_point",
        procedure_source_cycle="2026-08-06",
        threshold_elevation_hae_m=130.0,
        threshold_elevation_msl_m=100.0,
        threshold_crossing_height_m=30.0,
        lpv_lateral_fsd_m=106.75 if benchmark == "lpv" else None,
        baro_vnav_approved=benchmark != "lpv",
    )


def payload(*, subject="optimized", event=None, final_lat=35.0, final_lon=-78.0):
    target = {"lat": 35.0, "lon": -78.0, "alt": 130.0, "V": 70.0,
              "psi": 0.0, "gamma": -0.05, "m": 60_000.0}
    first = {"t": 0.0, **target, "lat": 34.95, "alt": 500.0}
    last = {"t": 100.0, **target, "lat": final_lat, "lon": final_lon}
    source = {
        "id": "TEST1", "subject": subject, "arr_airport": "KRDU",
        "runway": "05L", "hae_minus_msl_m": 30.0,
    }
    if event is not None:
        source["observed_threshold_event"] = event
    return {
        "source": source,
        "initial_state": {key: value for key, value in first.items() if key != "t"},
        "target_state": target,
        "final_time_s": 100.0,
        "states": [first, last],
        "controls": [] if subject == "observed" else [{"thrust": 1.0}] * 2,
    }


def event(*, cross=4.0, vertical=5.0):
    return {
        "schema_version": "observed-threshold-event-v4",
        "status": "estimated", "method": "final_segment_window_ensemble",
        "method_version": 4, "runway": "05L",
        "component_methods": {
            "lateral": "final_segment_window_ensemble",
            "vertical": "final_segment_window_ensemble",
        },
        "threshold_crossing_lat": 35.0, "threshold_crossing_lon": -78.0,
        "threshold_crossing_altitude_m": 160.0 + vertical,
        "altitude_datum": "hae", "signed_cross_track_m": cross,
        "cross_track_sigma_m": 0.5, "altitude_sigma_m": 0.75,
        "component_source_sample_ranges": {
            "lateral": [0, 1], "vertical": [0, 1],
        },
        "fit_window_m": [-5000.0, -300.0],
        "sample_count": 8, "along_track_span_m": 3000.0,
        "extrapolation_m": 300.0, "glidepath_deg": 3.0,
    }


def test_computed_terminal_state_reports_along_and_cross_separately():
    outcome = arrival_deviation(
        record_from_dict(payload(final_lon=-77.9999)), context=context()
    )
    assert outcome.event_status == "terminal_state"
    assert outcome.deviation.along_track_m == pytest.approx(0.0, abs=0.1)
    assert outcome.deviation.cross_track_m > 0.0


def test_computed_final_segment_is_interpolated_when_it_brackets_the_threshold():
    value = payload(final_lat=35.0001)
    value["states"][-2]["lat"] = 34.9999
    value["states"][-2]["alt"] = 140.0
    outcome = arrival_deviation(record_from_dict(value), context=context())
    assert outcome.event_status == "interpolated_threshold"
    assert outcome.deviation.along_track_m == pytest.approx(0.0, abs=0.01)
    assert outcome.deviation.vertical_m == pytest.approx(5.0, abs=0.1)


def test_computed_trajectory_ending_before_threshold_is_not_reached():
    outcome = arrival_deviation(
        record_from_dict(payload(final_lat=34.999)), context=context()
    )
    assert outcome.deviation is None
    assert outcome.event_status == "not_reached"


def test_computed_target_altitude_must_match_authoritative_tch_context():
    value = payload()
    value["target_state"]["alt"] = 131.0

    with pytest.raises(ValueError, match="target_state.alt.*authoritative"):
        arrival_deviation(record_from_dict(value), context=context())


def test_observed_record_consumes_serialized_event_and_converts_hae_to_msl():
    outcome = arrival_deviation(
        record_from_dict(payload(subject="observed", event=event(vertical=5.0))),
        context=context(),
    )
    deviation = outcome.deviation
    assert outcome.event_status == "estimated"
    assert deviation.cross_track_m == pytest.approx(4.0)
    assert deviation.vertical_m == pytest.approx(5.0)
    assert deviation.extrapolated is True
    assert deviation.extrapolation_m == pytest.approx(300.0)


def test_direct_lateral_observation_still_reports_fitted_vertical_extrapolation():
    direct = event(vertical=0.0)
    direct.update(
        method="direct_lateral_fitted_vertical",
        component_methods={
            "lateral": "threshold_plane_interpolation",
            "vertical": "final_segment_window_ensemble",
        },
        lateral_extrapolation_m=0.0,
    )

    outcome = arrival_deviation(
        record_from_dict(payload(subject="observed", event=direct)),
        context=context(),
    )

    assert outcome.deviation is not None
    assert outcome.deviation.extrapolated is True


def test_observed_record_datum_offset_must_match_authoritative_context():
    value = payload(subject="observed", event=event())
    value["source"]["hae_minus_msl_m"] = 31.0

    with pytest.raises(ValueError, match="hae_minus_msl_m.*authoritative"):
        arrival_deviation(record_from_dict(value), context=context())


def test_unavailable_observed_event_is_indeterminate_input_not_a_refit_request():
    record = record_from_dict(payload(subject="observed", event={
        "status": "unavailable", "unavailable_reason": "no assignment fit",
    }))
    outcome = arrival_deviation(record, context=context())
    assert outcome.deviation is None
    assert outcome.event_status == "unavailable"
    assert outcome.reason == "no assignment fit"


def test_observed_evaluation_has_no_fitter_dependency():
    # Protect the architectural boundary even if a future local import would evade
    # the runtime monkeypatch used by the pipeline integration test.
    assert "fit_final_segment" not in inspect.getsource(arrival_module)


def test_version_two_observed_event_requires_local_reclassification():
    legacy = event()
    legacy["schema_version"] = "observed-threshold-event-v2"
    legacy["method_version"] = 2

    with pytest.raises(ValueError, match="reclassify-existing"):
        arrival_deviation(
            record_from_dict(payload(subject="observed", event=legacy)),
            context=context(),
        )


def test_observed_event_must_match_the_assessed_runway():
    bad = event()
    bad["runway"] = "23R"
    with pytest.raises(ValueError, match="disagrees"):
        arrival_deviation(
            record_from_dict(payload(subject="observed", event=bad)), context=context()
        )
