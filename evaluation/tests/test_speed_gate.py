"""The stall-anchored threshold-crossing speed gate (see docs/THRESHOLD_SPEED_GATE.md)."""

from __future__ import annotations

import math

import pytest

from aircraft.aero_params import stall_speed_ms
from evaluation import evaluate_batch, evaluate_record, record_from_dict, speed_gate_bounds
from evaluation.metrics import METHODOLOGY, REPORT_SCHEMA_VERSION
from evaluation.speed_gate import (
    LOAD_FACTOR_ASSUMED_1G,
    LOAD_FACTOR_FROM_CONTROLS,
    OBSERVED_SPEED_CRITERION_ID,
    SPEED_CRITERION_ID,
    SPEED_GATE_UPPER_ADDITIVE_MS,
    VREF_STALL_MULTIPLIER,
)
from evaluation.tests.factories import (
    LANDING_AERO,
    TARGET,
    assessment_context,
    observed_event,
    observed_payload,
    trajectory_payload,
)

# The factory record crosses at TARGET["m"] with the factory stall facts. The window is
# derived through the SAME function the gate calls -- a test that restated g and rho0
# would be a second copy of the model that cannot notice the first one changing.
_VS = stall_speed_ms(
    TARGET["m"],
    wing_area_m2=LANDING_AERO["wing_area_m2"],
    cl_max=LANDING_AERO["cl_max_landing"],
)
_LOWER = VREF_STALL_MULTIPLIER * _VS
_UPPER = _LOWER + SPEED_GATE_UPPER_ADDITIVE_MS


def _payload(*, v_ms: float = 70.0, load_factor: float = 1.0) -> dict:
    value = trajectory_payload()
    value["states"][-1]["V"] = v_ms
    for control in value["controls"]:
        control["load_factor"] = load_factor
    return value


def test_bounds_come_from_the_record_own_mass_load_factor_and_stall_facts():
    bounds = speed_gate_bounds(60_000.0, LANDING_AERO, load_factor=1.0)
    assert bounds.stall_speed_ms == pytest.approx(_VS)
    assert bounds.stall_speed_at_n_ms == pytest.approx(_VS)
    assert bounds.lower_ms == pytest.approx(_LOWER)
    assert bounds.upper_ms == pytest.approx(_UPPER)
    # A heavier crossing raises the whole window (stall speed grows with sqrt(m)).
    heavier = speed_gate_bounds(70_000.0, LANDING_AERO, load_factor=1.0)
    assert heavier.lower_ms > bounds.lower_ms


def test_the_lower_bound_scales_with_the_square_root_of_the_load_factor():
    """Lift is n·m·g, so the wing reaches Cl_max at Vs1g·sqrt(n); the +20 kt edge is an
    energy criterion defined at 1 g and does not move."""
    banked = speed_gate_bounds(60_000.0, LANDING_AERO, load_factor=1.21)
    assert banked.stall_speed_ms == pytest.approx(_VS)
    assert banked.stall_speed_at_n_ms == pytest.approx(_VS * 1.1)
    assert banked.lower_ms == pytest.approx(_LOWER * 1.1)
    assert banked.upper_ms == pytest.approx(_UPPER)
    assert not banked.empty


def test_a_push_over_does_not_relax_the_lower_bound():
    pushed = speed_gate_bounds(60_000.0, LANDING_AERO, load_factor=0.8)
    assert pushed.lower_ms == pytest.approx(_LOWER)
    assert pushed.stall_speed_at_n_ms == pytest.approx(_VS)


def test_a_crossing_pulled_hard_enough_has_no_admissible_speed():
    """Past n ~ 1.3 the load-factor V_ref exceeds the 1-g energy limit: the window is
    empty, both bounds say so on the row, and every speed fails -- the manoeuvre, not
    the speed, is what disqualifies the crossing (docs section 6)."""
    bounds = speed_gate_bounds(60_000.0, LANDING_AERO, load_factor=2.0)
    assert bounds.empty and bounds.lower_ms > bounds.upper_ms
    value = _payload(v_ms=(bounds.lower_ms + bounds.upper_ms) / 2.0, load_factor=2.0)
    result = evaluate_record(record_from_dict(value), context=assessment_context())
    assert result.speed_result == "fail"
    assert result.speed_bounds.empty


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_an_impossible_load_factor_raises(bad):
    with pytest.raises(ValueError, match="load factor"):
        speed_gate_bounds(60_000.0, LANDING_AERO, load_factor=bad)


def test_computed_crossing_inside_the_window_passes():
    result = evaluate_record(
        record_from_dict(trajectory_payload()), context=assessment_context()
    )
    assert result.speed_result == "pass"
    assert result.verdict == "pass"
    assert result.speed_bounds is not None
    assert result.speed_bounds.lower_ms == pytest.approx(_LOWER)
    assert result.speed_criterion == SPEED_CRITERION_ID
    assert result.deviation.crossing_load_factor == 1.0
    assert result.deviation.crossing_load_factor_source == LOAD_FACTOR_FROM_CONTROLS


@pytest.mark.parametrize("v_ms", [_LOWER, _UPPER])
def test_the_window_is_inclusive_at_both_edges(v_ms):
    result = evaluate_record(
        record_from_dict(_payload(v_ms=v_ms)), context=assessment_context()
    )
    assert result.speed_result == "pass"


def test_a_fast_crossing_fails_the_composite_with_a_speed_violation():
    result = evaluate_record(
        record_from_dict(_payload(v_ms=_UPPER + 0.5)), context=assessment_context()
    )
    assert result.speed_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("speed",)


def test_a_slow_crossing_below_vref_fails():
    result = evaluate_record(
        record_from_dict(_payload(v_ms=_LOWER - 0.5)), context=assessment_context()
    )
    assert result.speed_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("speed",)


def test_the_crossing_load_factor_comes_from_the_last_control():
    """The same crossing speed passes at 1 g and fails at 1.21 g: the lower bound is the
    stall speed under the lift the manoeuvre demands, read from the control active
    over the final rollout step."""
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
    assert result.speed_bounds.stall_speed_ms == pytest.approx(
        stall_speed_ms(60_000.0, wing_area_m2=LANDING_AERO["wing_area_m2"],
                       cl_max=LANDING_AERO["cl_max_landing"])
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
    at the resolved airframe's mass and a declared 1 g, as a stated proxy — the wind
    caveat lives in the criterion id and the methodology, not in a refusal to grade."""
    result = evaluate_record(
        record_from_dict(observed_payload()), context=assessment_context()
    )
    # Default fixture: 70 m/s ground speed inside [66.3, 76.6] at 60 t.
    assert result.speed_result == "pass"
    assert result.speed_bounds is not None
    assert result.speed_criterion == OBSERVED_SPEED_CRITERION_ID
    assert result.deviation.crossing_speed_ms is None
    assert result.deviation.crossing_load_factor == 1.0
    assert result.deviation.crossing_load_factor_source == LOAD_FACTOR_ASSUMED_1G
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


def test_report_serializes_the_speed_window_load_factor_and_counts():
    # 1.1 g lifts the lower edge to 66.3 x sqrt(1.1) = 69.5 m/s: the 70 m/s crossing
    # still passes, now by half a metre per second instead of four.
    report = evaluate_batch(
        [record_from_dict(_payload(load_factor=1.1))],
        contexts={
            ("KRDU", "05L"): assessment_context(benchmark="rnp_apch_lnav_vnav_baro")
        },
    )
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    row = report["trajectories"][0]
    assert row["speed_result"] == "pass"
    assert row["bounds"]["speed_criterion"] == SPEED_CRITERION_ID
    assert row["bounds"]["stall_speed_ms"] == pytest.approx(_VS)
    assert row["bounds"]["stall_speed_at_n_ms"] == pytest.approx(_VS * math.sqrt(1.1))
    assert row["bounds"]["speed_lower_ms"] == pytest.approx(_LOWER * math.sqrt(1.1))
    assert row["bounds"]["speed_upper_ms"] == pytest.approx(_UPPER)
    assert row["deviation"]["crossing_speed_ms"] == pytest.approx(70.0)
    assert row["deviation"]["crossing_load_factor"] == 1.1
    assert row["deviation"]["crossing_load_factor_source"] == LOAD_FACTOR_FROM_CONTROLS
    assert row["crossing_load_factor"] == 1.1      # flat too, like the two speeds
    assert report["speed_result_counts"] == {"pass": 1, "fail": 0, "indeterminate": 0}
    assert report["crossing_speed_ms"]["mean"] == pytest.approx(70.0)
    assert report["crossing_load_factor"] == {
        "mean": 1.1, "min": 1.1, "p95": 1.1, "max": 1.1, "below_1g": 0, "assumed_1g": 0,
    }
    methodology = report["methodology"]["terminal_speed"]
    assert methodology["vref_stall_multiplier"] == 1.23
    assert set(methodology["load_factor"]["sources"]) == {
        LOAD_FACTOR_FROM_CONTROLS, LOAD_FACTOR_ASSUMED_1G,
    }
    assert any("25.125" in source["document"] for source in methodology["sources"])


def test_the_methodology_agrees_with_itself_about_the_observed_proxy():
    """Every block that mentions the observed ground speed says it is GRADED as the
    proxy (owner decision 2026-08-24). Until v7 the ground-speed block still carried
    the pre-decision 'audit only, never composed' wording, inside the same report whose
    terminal_speed block said the opposite."""
    speed = METHODOLOGY["terminal_speed"]
    ground = METHODOLOGY["observed_crossing_ground_speed"]
    assert speed["observed_proxy_criterion"] == OBSERVED_SPEED_CRITERION_ID
    assert OBSERVED_SPEED_CRITERION_ID in ground["use"]
    assert "the quantity their speed gate judges" in ground["use"]
    assert "proxy" in speed["subjects"]
