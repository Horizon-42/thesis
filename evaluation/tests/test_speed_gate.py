"""The stall-anchored threshold-crossing speed gate (see docs/THRESHOLD_SPEED_GATE.md)."""

from __future__ import annotations

import math

import pytest

from evaluation import evaluate_batch, evaluate_record, record_from_dict, speed_gate_bounds
from evaluation.speed_gate import (
    SPEED_GATE_UPPER_ADDITIVE_MS,
    VREF_STALL_MULTIPLIER,
)
from evaluation.tests.factories import (
    LANDING_AERO,
    assessment_context,
    observed_event,
    observed_payload,
    trajectory_payload,
)

# The factory record crosses at m = 60 t with A320-class stall facts; derive the exact
# window once, the same way the gate does (single formula, aircraft.aero_params).
_VS = math.sqrt(
    2.0 * 60_000.0 * 9.81
    / (1.225 * LANDING_AERO["wing_area_m2"] * LANDING_AERO["cl_max_landing"])
)
_LOWER = VREF_STALL_MULTIPLIER * _VS
_UPPER = _LOWER + SPEED_GATE_UPPER_ADDITIVE_MS


def _payload_with_crossing_speed(v_ms: float) -> dict:
    value = trajectory_payload()
    value["states"][-1]["V"] = v_ms
    return value


def test_bounds_come_from_the_record_own_mass_and_stall_facts():
    bounds = speed_gate_bounds(60_000.0, LANDING_AERO)
    assert bounds.stall_speed_ms == pytest.approx(_VS)
    assert bounds.lower_ms == pytest.approx(_LOWER)
    assert bounds.upper_ms == pytest.approx(_UPPER)
    # A heavier crossing raises the whole window (stall speed grows with sqrt(m)).
    heavier = speed_gate_bounds(70_000.0, LANDING_AERO)
    assert heavier.lower_ms > bounds.lower_ms


def test_computed_crossing_inside_the_window_passes():
    result = evaluate_record(
        record_from_dict(trajectory_payload()), context=assessment_context()
    )
    assert result.speed_result == "pass"
    assert result.verdict == "pass"
    assert result.speed_bounds is not None
    assert result.speed_bounds.lower_ms == pytest.approx(_LOWER)


@pytest.mark.parametrize("v_ms", [_LOWER, _UPPER])
def test_the_window_is_inclusive_at_both_edges(v_ms):
    result = evaluate_record(
        record_from_dict(_payload_with_crossing_speed(v_ms)),
        context=assessment_context(),
    )
    assert result.speed_result == "pass"


def test_a_fast_crossing_fails_the_composite_with_a_speed_violation():
    result = evaluate_record(
        record_from_dict(_payload_with_crossing_speed(_UPPER + 0.5)),
        context=assessment_context(),
    )
    assert result.speed_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("speed",)


def test_a_slow_crossing_below_vref_fails():
    result = evaluate_record(
        record_from_dict(_payload_with_crossing_speed(_LOWER - 0.5)),
        context=assessment_context(),
    )
    assert result.speed_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("speed",)


def test_a_computed_record_without_landing_aero_is_indeterminate_not_bypassed():
    """Absent stall facts must be loud — a gate that silently never binds reads as
    'applied' while deciding nothing."""
    value = trajectory_payload()
    del value["source"]["landing_aero"]

    result = evaluate_record(record_from_dict(value), context=assessment_context())

    assert result.speed_result == "indeterminate"
    assert result.verdict == "indeterminate"
    assert result.reason is not None and "landing_aero" in result.reason


def test_an_explicit_null_landing_aero_reads_as_unspecified():
    value = trajectory_payload()
    value["source"]["landing_aero"] = None

    result = evaluate_record(record_from_dict(value), context=assessment_context())

    assert result.speed_result == "indeterminate"
    assert result.verdict == "indeterminate"


@pytest.mark.parametrize(
    "block",
    [
        {"wing_area_m2": 122.6},                          # missing cl_max_landing
        {"wing_area_m2": -1.0, "cl_max_landing": 2.7},    # non-physical
        {"wing_area_m2": 122.6, "cl_max_landing": True},  # bool is not a number
        "A320",                                           # not an object at all
    ],
)
def test_a_present_but_malformed_landing_aero_raises(block):
    value = trajectory_payload()
    value["source"]["landing_aero"] = block

    with pytest.raises(ValueError, match="landing_aero"):
        evaluate_record(record_from_dict(value), context=assessment_context())


def test_observed_records_are_speed_graded_on_the_ground_speed_proxy():
    """The baseline runs the SAME three gates as its modeled twins (owner decision
    2026-08-24): the fitted crossing GROUND speed is judged against the stall window
    at the resolved airframe's mass, as a stated proxy — the wind caveat lives in the
    criterion id and the methodology, not in a refusal to grade."""
    result = evaluate_record(
        record_from_dict(observed_payload()), context=assessment_context()
    )
    # Default fixture: 70 m/s ground speed inside [66.3, 76.6] at 60 t.
    assert result.speed_result == "pass"
    assert result.speed_bounds is not None
    assert result.verdict == "pass"
    assert result.reason is None

    slow = trajectory_payload(
        subject="observed", event=observed_event(ground_speed_m_s=60.0)
    )
    result = evaluate_record(
        record_from_dict(slow), context=assessment_context()
    )
    assert result.speed_result == "fail"
    assert result.verdict == "fail"
    assert "speed" in result.violations


def test_observed_speed_grades_indeterminate_without_resolved_airframe_or_speed():
    """The two honest gaps stay loud: no stall window without an airframe, nothing to
    judge without a fitted crossing speed — and either one composes indeterminate."""
    unresolved = observed_payload()
    del unresolved["source"]["landing_aero"]
    result = evaluate_record(
        record_from_dict(unresolved), context=assessment_context()
    )
    assert result.speed_result == "indeterminate"
    assert result.verdict == "indeterminate"
    assert "airframe" in (result.reason or "")

    speedless = trajectory_payload(
        subject="observed", event=observed_event(ground_speed_m_s=None)
    )
    result = evaluate_record(
        record_from_dict(speedless), context=assessment_context()
    )
    assert result.speed_result == "indeterminate"
    assert result.verdict == "indeterminate"
    assert "no crossing ground speed" in (result.reason or "")


def test_report_serializes_the_speed_window_and_counts():
    report = evaluate_batch(
        [record_from_dict(trajectory_payload())],
        contexts={
            ("KRDU", "05L"): assessment_context(benchmark="rnp_apch_lnav_vnav_baro")
        },
    )
    row = report["trajectories"][0]
    assert row["speed_result"] == "pass"
    assert row["bounds"]["speed_criterion"] == "vref_1p23_vs1g_to_vref_plus_20kt"
    assert row["bounds"]["speed_lower_ms"] == pytest.approx(_LOWER)
    assert row["bounds"]["speed_upper_ms"] == pytest.approx(_UPPER)
    assert row["deviation"]["crossing_speed_ms"] == pytest.approx(70.0)
    assert report["speed_result_counts"] == {"pass": 1, "fail": 0, "indeterminate": 0}
    assert report["crossing_speed_ms"]["mean"] == pytest.approx(70.0)
    methodology = report["methodology"]["terminal_speed"]
    assert methodology["vref_stall_multiplier"] == 1.23
    assert any("25.125" in source["document"] for source in methodology["sources"])
