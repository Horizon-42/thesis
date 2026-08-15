"""Arrival-event dispatch uses stored geometry and never refits ADS-B."""

from __future__ import annotations

import pytest

from evaluation import arrival_deviation, record_from_dict
from evaluation.tests.factories import (
    assessment_context,
    observed_event,
    trajectory_payload,
)
from final_approach.event_contract import EVENT_SCHEMA_VERSION


def test_computed_terminal_state_reports_along_and_cross_separately():
    outcome = arrival_deviation(
        record_from_dict(trajectory_payload(final_lon=-77.9999)),
        context=assessment_context(),
    )
    assert outcome.event_status == "terminal_state"
    assert outcome.deviation.along_track_m == pytest.approx(0.0, abs=0.1)
    assert outcome.deviation.cross_track_m > 0.0


def test_computed_final_segment_is_interpolated_when_it_brackets_the_threshold():
    value = trajectory_payload(final_lat=35.0001)
    value["states"][-2]["lat"] = 34.9999
    value["states"][-2]["alt"] = 140.0
    outcome = arrival_deviation(
        record_from_dict(value), context=assessment_context()
    )
    assert outcome.event_status == "interpolated_threshold"
    assert outcome.deviation.along_track_m == pytest.approx(0.0, abs=0.01)
    assert outcome.deviation.vertical_m == pytest.approx(5.0, abs=0.1)


def test_computed_trajectory_ending_before_threshold_is_not_reached():
    outcome = arrival_deviation(
        record_from_dict(trajectory_payload(final_lat=34.999)),
        context=assessment_context(),
    )
    assert outcome.deviation is None
    assert outcome.event_status == "not_reached"


def test_computed_target_altitude_must_match_authoritative_tch_context():
    value = trajectory_payload()
    value["target_state"]["alt"] = 131.0

    with pytest.raises(ValueError, match="target_state.alt.*authoritative"):
        arrival_deviation(record_from_dict(value), context=assessment_context())


def test_observed_record_consumes_serialized_event_and_converts_hae_to_msl():
    outcome = arrival_deviation(
        record_from_dict(
            trajectory_payload(
                subject="observed",
                event=observed_event(cross_m=4.0, vertical_m=5.0),
            )
        ),
        context=assessment_context(),
    )
    deviation = outcome.deviation
    assert outcome.event_status == "estimated"
    assert deviation.cross_track_m == pytest.approx(4.0)
    assert deviation.vertical_m == pytest.approx(5.0)
    assert deviation.extrapolation_m == pytest.approx(325.0)


def test_obsolete_hybrid_observed_method_is_rejected():
    hybrid = observed_event()
    hybrid["method"] = "direct_lateral_fitted_vertical"

    with pytest.raises(ValueError, match="unsupported"):
        arrival_deviation(
            record_from_dict(trajectory_payload(subject="observed", event=hybrid)),
            context=assessment_context(),
        )


def test_observed_record_datum_offset_must_match_authoritative_context():
    value = trajectory_payload(subject="observed", event=observed_event())
    value["source"]["hae_minus_msl_m"] = 31.0

    with pytest.raises(ValueError, match="hae_minus_msl_m.*authoritative"):
        arrival_deviation(record_from_dict(value), context=assessment_context())


def test_unavailable_observed_event_is_indeterminate_input_not_a_refit_request():
    record = record_from_dict(trajectory_payload(subject="observed", event={
        "schema_version": EVENT_SCHEMA_VERSION,
        "status": "unavailable",
        "method": "none",
        "observability": "unavailable",
        "runway": "05L",
        "unavailable_reason": "no assignment fit",
    }))
    outcome = arrival_deviation(record, context=assessment_context())
    assert outcome.deviation is None
    assert outcome.event_status == "unavailable"
    assert outcome.reason == "no assignment fit"


@pytest.mark.parametrize("schema", ["observed-threshold-event-v7", None])
def test_obsolete_or_missing_event_schema_requires_local_reclassification(schema):
    legacy = observed_event()
    if schema is None:
        del legacy["schema_version"]
    else:
        legacy["schema_version"] = schema

    with pytest.raises(ValueError, match="reclassify-existing"):
        arrival_deviation(
            record_from_dict(trajectory_payload(subject="observed", event=legacy)),
            context=assessment_context(),
        )


def test_observed_event_must_match_the_assessed_runway():
    bad = observed_event()
    bad["runway"] = "23R"
    with pytest.raises(ValueError, match="disagrees"):
        arrival_deviation(
            record_from_dict(trajectory_payload(subject="observed", event=bad)),
            context=assessment_context(),
        )
