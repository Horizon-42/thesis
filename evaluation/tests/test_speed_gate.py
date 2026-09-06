"""The published-V_ref threshold-crossing speed gate (see docs/THRESHOLD_SPEED_GATE.md)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from aircraft.reference_speeds import reference_speed
from evaluation import evaluate_batch, evaluate_record, record_from_dict, speed_gate_bounds
from evaluation.metrics import METHODOLOGY, REPORT_SCHEMA_VERSION
from evaluation.speed_gate import (
    LOAD_FACTOR_ASSUMED_1G,
    LOAD_FACTOR_FROM_CONTROLS,
    MASS_BASIS_CROSSING,
    MASS_BASIS_TYPE_RANGE,
    NO_CROSSING_REASON,
    NO_REFERENCE_SPEED_REASON,
    NO_TYPECODE_REASON,
    OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID,
    OBSERVED_SPEED_CRITERION_ID,
    SPEED_CRITERION_ID,
    SPEED_GATE_UPPER_ADDITIVE_MS,
    TYPECODE_KEYS,
    record_typecode,
)
from evaluation.tests.factories import (
    AIRCRAFT_TYPE,
    TARGET,
    assessment_context,
    observed_event,
    observed_payload,
    trajectory_payload,
)
from geokit import kt_to_ms

# The factory record crosses at TARGET["m"] as an A320. The window is derived through
# the SAME function the gate calls, from the SAME published table -- a test that
# restated 136 kt would be a second copy of the table that cannot notice it changing.
_REF = reference_speed(AIRCRAFT_TYPE)
_BOUNDS = speed_gate_bounds(_REF, load_factor=1.0, crossing_mass_kg=TARGET["m"])
_LOWER, _UPPER = _BOUNDS.lower_ms, _BOUNDS.upper_ms
_OBSERVED = speed_gate_bounds(_REF, load_factor=1.0, crossing_mass_kg=None)


def _payload(*, v_ms: float = 70.0, load_factor: float = 1.0) -> dict:
    value = trajectory_payload()
    value["states"][-1]["V"] = v_ms
    for control in value["controls"]:
        control["load_factor"] = load_factor
    return value


def test_the_window_is_the_published_speed_scaled_to_the_crossing_mass():
    assert _BOUNDS.reference_typecode == AIRCRAFT_TYPE
    assert _BOUNDS.mass_basis == MASS_BASIS_CROSSING
    assert _BOUNDS.vref_low_ms == pytest.approx(kt_to_ms(_REF.vref_kt(TARGET["m"], edge="low")))
    assert _BOUNDS.vref_high_ms == pytest.approx(kt_to_ms(_REF.vref_kt(TARGET["m"], edge="high")))
    assert _BOUNDS.reference_sources == {
        "approach_speed": _REF.approach_speed_source, "malw": _REF.malw_source,
        "min_mass": _REF.min_mass_source,
    }
    assert _LOWER == pytest.approx(_BOUNDS.vref_low_ms)
    assert _UPPER == pytest.approx(_BOUNDS.vref_high_ms + SPEED_GATE_UPPER_ADDITIVE_MS)
    # At the quoted landing weight the window is exactly the published pair.
    at_malw = speed_gate_bounds(_REF, load_factor=1.0, crossing_mass_kg=_REF.malw_kg)
    assert at_malw.vref_low_ms == pytest.approx(kt_to_ms(_REF.approach_speed_min_kt))
    assert at_malw.vref_high_ms == pytest.approx(kt_to_ms(_REF.approach_speed_max_kt))
    # A heavier crossing raises the whole window (V_ref grows with sqrt(m)).
    heavier = speed_gate_bounds(_REF, load_factor=1.0, crossing_mass_kg=TARGET["m"] + 10_000.0)
    assert heavier.lower_ms > _LOWER and heavier.upper_ms > _UPPER
    assert heavier.lower_ms == pytest.approx(_LOWER * math.sqrt((TARGET["m"] + 10_000.0) / TARGET["m"]))


def test_the_observed_window_spans_the_type_published_mass_range():
    """An ADS-B track carries no mass: the lower edge is V_ref at the type's published
    minimum operating mass, the upper edge V_ref at its MALW plus the additive."""
    assert _OBSERVED.mass_basis == MASS_BASIS_TYPE_RANGE
    assert _OBSERVED.vref_low_ms == pytest.approx(kt_to_ms(_REF.vref_kt(_REF.min_mass_kg, edge="low")))
    assert _OBSERVED.vref_high_ms == pytest.approx(kt_to_ms(_REF.approach_speed_max_kt))
    assert _OBSERVED.lower_ms < _LOWER and _OBSERVED.upper_ms > _UPPER
    # A dual flap-configuration type keeps both published values in play.
    dual = reference_speed("B738")
    window = speed_gate_bounds(dual, load_factor=1.0, crossing_mass_kg=dual.malw_kg)
    assert window.vref_low_ms == pytest.approx(kt_to_ms(dual.approach_speed_min_kt))
    assert window.vref_high_ms == pytest.approx(kt_to_ms(dual.approach_speed_max_kt))
    assert dual.approach_speed_min_kt < dual.approach_speed_max_kt


def test_a_type_range_window_needs_a_published_minimum_mass():
    no_min = replace(_REF, min_mass_kg=None, min_mass_kind=None, min_mass_source=None)
    with pytest.raises(ValueError, match="minimum mass"):
        speed_gate_bounds(no_min, load_factor=1.0, crossing_mass_kg=None)
    # The crossing-mass window does not need it.
    assert speed_gate_bounds(no_min, load_factor=1.0, crossing_mass_kg=TARGET["m"]).lower_ms == pytest.approx(_LOWER)


def test_the_lower_bound_scales_with_the_square_root_of_the_load_factor():
    """Lift is n·m·g, so V_ref under the manoeuvre is V_ref·sqrt(n); the +20 kt edge is an
    energy criterion defined at 1 g and does not move."""
    banked = speed_gate_bounds(_REF, load_factor=1.21, crossing_mass_kg=TARGET["m"])
    assert banked.vref_low_ms == pytest.approx(_BOUNDS.vref_low_ms)
    assert banked.lower_ms == pytest.approx(_LOWER * 1.1)
    assert banked.upper_ms == pytest.approx(_UPPER)
    assert not banked.empty


def test_a_push_over_does_not_relax_the_lower_bound():
    pushed = speed_gate_bounds(_REF, load_factor=0.8, crossing_mass_kg=TARGET["m"])
    assert pushed.lower_ms == pytest.approx(_LOWER)
    assert pushed.vref_low_ms == pytest.approx(_BOUNDS.vref_low_ms)


def test_a_crossing_pulled_hard_enough_has_no_admissible_speed():
    """Past n ~ 1.3 the load-factor V_ref exceeds the 1-g energy limit: the window is
    empty, both bounds say so on the row, and every speed fails -- the manoeuvre, not
    the speed, is what disqualifies the crossing (docs section 6)."""
    bounds = speed_gate_bounds(_REF, load_factor=2.0, crossing_mass_kg=TARGET["m"])
    assert bounds.empty and bounds.lower_ms > bounds.upper_ms
    value = _payload(v_ms=(bounds.lower_ms + bounds.upper_ms) / 2.0, load_factor=2.0)
    result = evaluate_record(record_from_dict(value), context=assessment_context())
    assert result.speed_result == "fail"
    assert result.speed_bounds.empty


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_an_impossible_load_factor_raises(bad):
    with pytest.raises(ValueError, match="load factor"):
        speed_gate_bounds(_REF, load_factor=bad, crossing_mass_kg=TARGET["m"])


def test_record_typecode_reads_the_producer_keys():
    assert TYPECODE_KEYS == ("dynamics_typecode", "aircraft_type")
    assert record_typecode({"dynamics_typecode": "b738"}) == "B738"
    assert record_typecode({"aircraft_type": " a320 "}) == "A320"
    assert record_typecode({"dynamics_typecode": "B738", "aircraft_type": "A320"}) == "B738"
    assert record_typecode({}) is None
    assert record_typecode({"aircraft_type": ""}) is None
    assert record_typecode({"aircraft_type": None}) is None


def test_computed_crossing_inside_the_window_passes():
    result = evaluate_record(
        record_from_dict(trajectory_payload()), context=assessment_context()
    )
    assert result.speed_result == "pass"
    assert result.verdict == "pass"
    assert result.speed_bounds is not None
    assert result.speed_bounds.lower_ms == pytest.approx(_LOWER)
    assert result.speed_bounds.reference_typecode == AIRCRAFT_TYPE
    assert result.speed_criterion == SPEED_CRITERION_ID
    assert result.speed_reason is None
    assert result.deviation.crossing_load_factor == 1.0
    assert result.deviation.crossing_load_factor_source == LOAD_FACTOR_FROM_CONTROLS
    assert result.speed_margin_ms == pytest.approx(min(70.0 - _LOWER, _UPPER - 70.0))


@pytest.mark.parametrize("v_ms", [_LOWER, _UPPER])
def test_the_window_is_inclusive_at_both_edges(v_ms):
    result = evaluate_record(
        record_from_dict(_payload(v_ms=v_ms)), context=assessment_context()
    )
    assert result.speed_result == "pass"
    assert result.speed_margin_ms == pytest.approx(0.0, abs=1e-9)


def test_a_fast_crossing_fails_the_composite_with_a_speed_violation():
    result = evaluate_record(
        record_from_dict(_payload(v_ms=_UPPER + 0.5)), context=assessment_context()
    )
    assert result.speed_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("speed",)
    assert result.speed_margin_ms == pytest.approx(-0.5)


def test_a_slow_crossing_below_vref_fails():
    result = evaluate_record(
        record_from_dict(_payload(v_ms=_LOWER - 0.5)), context=assessment_context()
    )
    assert result.speed_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("speed",)


def test_the_crossing_load_factor_comes_from_the_last_control():
    """The same crossing speed passes at 1 g and fails at 1.21 g: the lower bound is
    V_ref under the lift the manoeuvre demands, read from the control active over the
    final rollout step."""
    v_ms = _LOWER * 1.05
    level = evaluate_record(
        record_from_dict(_payload(v_ms=v_ms, load_factor=1.0)),
        context=assessment_context(),
    )
    assert level.speed_result == "pass"

    banked = evaluate_record(
        record_from_dict(_payload(v_ms=v_ms, load_factor=1.21)),
        context=assessment_context(),
    )
    assert banked.speed_result == "fail"
    assert banked.violations == ("speed",)
    assert banked.deviation.crossing_load_factor == 1.21
    assert banked.deviation.crossing_load_factor_source == LOAD_FACTOR_FROM_CONTROLS
    assert banked.speed_bounds.lower_ms == pytest.approx(_LOWER * 1.1)
    assert banked.speed_bounds.upper_ms == pytest.approx(_UPPER)


def test_an_interpolated_crossing_anchors_on_the_blended_mass_and_speed():
    """A bracket crossing blends the two states; the window must use THAT mass, not
    the final sample's, and judge THAT speed."""
    value = trajectory_payload(final_lat=35.0001)
    value["states"][-2].update(lat=34.9999, m=61_000.0, V=68.0)
    value["states"][-1].update(m=59_000.0, V=72.0)
    result = evaluate_record(record_from_dict(value), context=assessment_context())
    assert result.event_status == "interpolated_threshold"
    assert result.deviation.crossing_mass_kg == pytest.approx(60_000.0)
    assert result.deviation.crossing_speed_ms == pytest.approx(70.0)
    assert result.speed_bounds.vref_low_ms == pytest.approx(
        kt_to_ms(_REF.vref_kt(60_000.0, edge="low"))
    )
    assert result.speed_result == "pass"


def test_only_the_final_step_control_anchors_the_bound():
    value = _payload(v_ms=_LOWER * 1.05, load_factor=1.0)
    value["controls"][0]["load_factor"] = 2.0   # an earlier pull-up is not the crossing's
    result = evaluate_record(record_from_dict(value), context=assessment_context())
    assert result.speed_result == "pass"
    assert result.deviation.crossing_load_factor == 1.0


def test_controls_without_a_load_factor_column_are_malformed():
    value = trajectory_payload()
    for control in value["controls"]:
        del control["load_factor"]
    with pytest.raises(ValueError, match="load_factor"):
        evaluate_record(record_from_dict(value), context=assessment_context())


def test_a_record_without_controls_is_judged_at_a_declared_1g():
    """State-output predictions carry no controls: the row says 1 g was assumed."""
    value = trajectory_payload(subject="predicted")
    value["controls"] = []
    result = evaluate_record(record_from_dict(value), context=assessment_context())
    assert result.speed_result == "pass"
    assert result.deviation.crossing_load_factor == 1.0
    assert result.deviation.crossing_load_factor_source == LOAD_FACTOR_ASSUMED_1G


def test_a_computed_record_without_a_type_is_indeterminate_not_bypassed():
    """A record naming no aircraft must be loud — a gate that silently never binds
    reads as 'applied' while deciding nothing."""
    value = trajectory_payload()
    del value["source"]["dynamics_typecode"]

    result = evaluate_record(record_from_dict(value), context=assessment_context())

    assert result.speed_result == "indeterminate"
    assert result.verdict == "indeterminate"
    assert result.speed_reason == NO_TYPECODE_REASON
    assert result.reason is not None and "aircraft type" in result.reason
    assert result.speed_margin_ms is None


def test_a_type_without_a_published_entry_is_indeterminate_and_named():
    value = trajectory_payload()
    value["source"]["dynamics_typecode"] = "ZZZZ"

    result = evaluate_record(record_from_dict(value), context=assessment_context())

    assert result.speed_result == "indeterminate"
    assert result.verdict == "indeterminate"
    assert result.speed_reason == NO_REFERENCE_SPEED_REASON.format(typecode="ZZZZ")
    assert "ZZZZ" in result.reason
    # On a row whose composite is a lateral FAIL, ``reason`` is not written but the
    # speed reason still is -- readable per row, not only through the composite.
    wide = trajectory_payload(final_lon=-78.0 + 200.0 / 91_000.0)
    wide["source"]["dynamics_typecode"] = "ZZZZ"
    report = evaluate_batch(
        [record_from_dict(wide)], contexts={("KRDU", "05L"): assessment_context()}
    )
    [row] = report["trajectories"]
    assert row["verdict"] == "fail" and row["lateral_result"] == "fail"
    assert "reason" not in row
    assert row["speed_reason"] == NO_REFERENCE_SPEED_REASON.format(typecode="ZZZZ")


def test_observed_records_are_speed_graded_over_the_type_mass_range():
    """The baseline runs the SAME three gates as its modeled twins (owner decision
    2026-08-24): the fitted crossing GROUND speed is judged against the type's
    published window over its published mass range — the mass of an ADS-B track is
    not measured — as a stated proxy when no wind report is usable."""
    result = evaluate_record(
        record_from_dict(observed_payload()), context=assessment_context()
    )
    assert result.speed_result == "pass"
    assert result.speed_bounds is not None
    assert result.speed_bounds.mass_basis == MASS_BASIS_TYPE_RANGE
    assert result.speed_bounds.lower_ms == pytest.approx(_OBSERVED.lower_ms)
    assert result.speed_bounds.upper_ms == pytest.approx(_OBSERVED.upper_ms)
    assert result.speed_criterion == OBSERVED_SPEED_CRITERION_ID
    assert result.deviation.crossing_speed_ms is None
    assert result.deviation.crossing_load_factor == 1.0
    assert result.deviation.crossing_load_factor_source == LOAD_FACTOR_ASSUMED_1G
    assert result.verdict == "pass"
    assert result.reason is None

    slow = trajectory_payload(
        subject="observed", event=observed_event(ground_speed_m_s=_OBSERVED.lower_ms - 1.0)
    )
    result = evaluate_record(record_from_dict(slow), context=assessment_context())
    assert result.speed_result == "fail"
    assert result.verdict == "fail"
    assert "speed" in result.violations
    assert result.speed_margin_ms == pytest.approx(-1.0)


def test_observed_speed_grades_indeterminate_without_resolved_airframe_or_speed():
    """The honest gaps stay loud: no window without an airframe, none for a type with
    no published entry or no published minimum mass, nothing to judge without a
    fitted crossing speed — and each composes indeterminate with its reason."""
    unresolved = observed_payload()
    del unresolved["source"]["aircraft_type"]
    result = evaluate_record(record_from_dict(unresolved), context=assessment_context())
    assert result.speed_result == "indeterminate"
    assert result.verdict == "indeterminate"
    assert "airframe" in (result.reason or "")

    unknown_type = observed_payload()
    unknown_type["source"]["aircraft_type"] = "ZZZZ"
    result = evaluate_record(record_from_dict(unknown_type), context=assessment_context())
    assert result.speed_result == "indeterminate"
    assert "ZZZZ" in result.reason

    speedless = trajectory_payload(
        subject="observed", event=observed_event(ground_speed_m_s=None)
    )
    result = evaluate_record(record_from_dict(speedless), context=assessment_context())
    assert result.speed_result == "indeterminate"
    assert result.verdict == "indeterminate"
    assert "no crossing ground speed" in (result.reason or "")


def test_an_observed_type_without_a_published_minimum_mass_is_indeterminate(monkeypatch):
    import evaluation.metrics as metrics
    no_min = replace(_REF, min_mass_kg=None, min_mass_kind=None, min_mass_source=None)
    monkeypatch.setattr(metrics, "reference_speed", lambda code: no_min)
    result = evaluate_record(record_from_dict(observed_payload()), context=assessment_context())
    assert result.speed_result == "indeterminate"
    assert "minimum operating mass" in result.reason
    # The same type still frames a computed record's window at its crossing mass.
    computed = evaluate_record(record_from_dict(trajectory_payload()), context=assessment_context())
    assert computed.speed_result == "pass"


def test_report_serializes_the_speed_window_load_factor_and_counts():
    # 1.1 g lifts the lower edge by sqrt(1.1): the 70 m/s crossing still passes.
    unsolved = {**_payload(), "states": [], "controls": [], "final_time_s": None, "target_state": None}
    report = evaluate_batch(
        [
            record_from_dict(_payload(load_factor=1.1)),
            record_from_dict({**_payload(), "source": {**_payload()["source"], "dynamics_typecode": "ZZZZ"}}),
            record_from_dict(unsolved),
        ],
        contexts={
            ("KRDU", "05L"): assessment_context(benchmark="rnp_apch_lnav_vnav_baro")
        },
    )
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    row = report["trajectories"][0]
    assert row["speed_result"] == "pass"
    assert row["bounds"]["speed_criterion"] == SPEED_CRITERION_ID
    assert row["bounds"]["reference_typecode"] == AIRCRAFT_TYPE
    assert row["bounds"]["mass_basis"] == MASS_BASIS_CROSSING
    assert row["bounds"]["vref_low_ms"] == pytest.approx(_BOUNDS.vref_low_ms)
    assert row["bounds"]["vref_high_ms"] == pytest.approx(_BOUNDS.vref_high_ms)
    assert row["bounds"]["reference_sources"]["approach_speed"] == _REF.approach_speed_source
    assert row["speed_reason"] is None
    assert row["bounds"]["speed_lower_ms"] == pytest.approx(_LOWER * math.sqrt(1.1))
    assert row["bounds"]["speed_upper_ms"] == pytest.approx(_UPPER)
    assert row["deviation"]["crossing_speed_ms"] == pytest.approx(70.0)
    assert row["deviation"]["crossing_load_factor"] == 1.1
    assert row["deviation"]["crossing_load_factor_source"] == LOAD_FACTOR_FROM_CONTROLS
    assert row["crossing_load_factor"] == 1.1      # flat too, like the two speeds
    assert "speed_uncertainty_ms" not in row
    unknown = report["trajectories"][1]
    assert unknown["bounds"]["reference_typecode"] is None and unknown["bounds"]["speed_lower_ms"] is None
    assert unknown["bounds"]["reference_sources"] is None
    assert unknown["speed_reason"] == NO_REFERENCE_SPEED_REASON.format(typecode="ZZZZ")
    # An unsolved row is speed-indeterminate too, with its own reason: the reasons sum
    # to the indeterminate tally, over the same rows.
    assert report["trajectories"][2]["speed_reason"] == NO_CROSSING_REASON
    assert report["speed_result_counts"] == {"pass": 1, "fail": 0, "indeterminate": 2}
    assert report["speed_indeterminate_reasons"] == {
        NO_CROSSING_REASON: 1,
        NO_REFERENCE_SPEED_REASON.format(typecode="ZZZZ"): 1,
    }
    assert sum(report["speed_indeterminate_reasons"].values()) == report["speed_result_counts"]["indeterminate"]
    assert "speed_marginal" not in report and "speed_uncertainty_unknown" not in report
    assert report["crossing_speed_ms"]["mean"] == pytest.approx(70.0)
    assert report["crossing_load_factor"]["max"] == 1.1 and report["crossing_load_factor"]["below_1g"] == 0
    methodology = report["methodology"]["terminal_speed"]
    assert "reference_speeds.json" in methodology["reference_speeds"]["table"]
    assert methodology["reference_speeds"]["record_type_keys"] == list(TYPECODE_KEYS)
    assert any("Aircraft Characteristics Database" in source["document"] for source in methodology["sources"])
    assert "uncertainty_ms" not in methodology["observed_wind_correction"]


def test_the_methodology_agrees_with_itself_about_the_observed_proxy():
    """Every block that mentions the observed ground speed says it is GRADED as the
    proxy (owner decision 2026-08-24)."""
    speed = METHODOLOGY["terminal_speed"]
    ground = METHODOLOGY["observed_crossing_ground_speed"]
    assert speed["observed_proxy_criterion"] == OBSERVED_SPEED_CRITERION_ID
    assert speed["observed_wind_correction"]["criterion"] == OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID
    # The ground-speed block names BOTH ways it is used: corrected into the estimate
    # that is judged when a report exists, judged as the proxy otherwise.
    assert OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID in ground["use"]
    assert OBSERVED_SPEED_CRITERION_ID in ground["use"]
    assert "metar_airspeed_estimate" in speed["subjects"] and "proxy" in speed["subjects"]
    assert MASS_BASIS_TYPE_RANGE in speed["reference_speeds"]["mass_bases"]
