"""The observed baseline's load factor is MEASURED from its own ADS-B kinematics.

A windowed fit of ψ and γ over the final 20 s of measured track, inverted through the
point-mass rotational equations, anchors the speed gate's lower bound the same way the
last control does for a computed record; a window too short to fit is declared 1 g.
"""

from __future__ import annotations

import math

import pytest

from aircraft.aero_params import GRAVITY_M_S2
from aircraft.kinematics import load_factor_from_rates
from aircraft.reference_speeds import reference_speed
from evaluation import evaluate_batch, evaluate_record, record_from_dict, speed_gate_bounds
from evaluation.arrival import LOAD_FACTOR_MIN_SAMPLES, LOAD_FACTOR_WINDOW_S
from evaluation.speed_gate import (
    LOAD_FACTOR_ASSUMED_1G,
    LOAD_FACTOR_FROM_ADSB,
    OBSERVED_SPEED_CRITERION_ID,
    OBSERVED_SPEED_CRITERION_STEM,
)
from evaluation.tests.factories import (
    AIRCRAFT_TYPE,
    assessment_context,
    observed_track_payload,
)

# The observed window at 1 g, derived through the gate's own function from the
# published table (the type's mass range, since an ADS-B track carries no mass).
_LOWER_1G = speed_gate_bounds(
    reference_speed(AIRCRAFT_TYPE), load_factor=1.0, crossing_mass_kg=None
).lower_ms


def _grade(payload):
    return evaluate_record(record_from_dict(payload), context=assessment_context())


def test_the_inversion_is_the_point_mass_rotational_equations():
    assert load_factor_from_rates(70.0, -0.05, 0.0, 0.0) == pytest.approx(math.cos(0.05))
    # A level 25 deg coordinated turn: n = 1 / cos(25 deg) = 1.103.
    bank = math.radians(25.0)
    psi_rate = GRAVITY_M_S2 * math.tan(bank) / 70.0
    assert load_factor_from_rates(70.0, 0.0, 0.0, psi_rate) == pytest.approx(1.0 / math.cos(bank))
    # A pull-up at 0.5 m/s^2 of normal acceleration adds 0.051 g.
    assert load_factor_from_rates(70.0, 0.0, 0.5 / 70.0, 0.0) == pytest.approx(1.0 + 0.5 / GRAVITY_M_S2)


def test_a_straight_stabilized_final_measures_one_g_and_keeps_the_1g_window():
    result = _grade(observed_track_payload())
    deviation = result.deviation
    assert deviation.crossing_load_factor_source == LOAD_FACTOR_FROM_ADSB
    assert deviation.crossing_load_factor == pytest.approx(math.cos(0.05), abs=1e-6)
    window = deviation.crossing_load_factor_window
    assert window["samples"] >= LOAD_FACTOR_MIN_SAMPLES
    assert window["duration_s"] == pytest.approx(LOAD_FACTOR_WINDOW_S)
    assert window["ends_m_before_threshold"] == 0.0
    assert window["psi_rate_rad_s"] == pytest.approx(0.0, abs=1e-12)
    # Below 1 g clamps to the 1-g floor: the bound is the same as before, measured.
    assert result.speed_bounds.lower_ms == pytest.approx(_LOWER_1G)
    assert result.speed_criterion == OBSERVED_SPEED_CRITERION_ID
    assert OBSERVED_SPEED_CRITERION_ID.startswith(OBSERVED_SPEED_CRITERION_STEM)


def test_a_turning_final_lifts_the_lower_bound_by_the_square_root_of_n():
    psi_rate = 0.02   # rad/s: a 7.7 deg bank at 70 m/s
    result = _grade(observed_track_payload(psi_rate_rad_s=psi_rate))
    expected = load_factor_from_rates(70.0, -0.05 , 0.0, psi_rate)
    assert result.deviation.crossing_load_factor == pytest.approx(expected, abs=1e-6)
    assert result.deviation.crossing_load_factor_window["psi_rate_rad_s"] == pytest.approx(psi_rate)
    assert result.speed_bounds.lower_ms == pytest.approx(_LOWER_1G * math.sqrt(expected))


def test_a_pull_up_can_fail_a_ground_speed_that_clears_the_1g_floor():
    gamma_rate = 0.01   # rad/s over the window: +0.07 g of normal acceleration
    v_ms = _LOWER_1G * 1.02
    level = _grade(observed_track_payload(ground_speed_m_s=v_ms))
    assert level.speed_result == "pass"
    pulled = _grade(observed_track_payload(gamma_rate_rad_s=gamma_rate, ground_speed_m_s=v_ms))
    assert pulled.deviation.crossing_load_factor > 1.05
    assert pulled.speed_result == "fail"
    assert pulled.violations == ("speed",)


def test_a_heading_that_turns_through_the_branch_cut_reads_as_one_turn():
    psi_rate = 0.02
    result = _grade(observed_track_payload(psi_rate_rad_s=psi_rate, psi0_rad=math.pi - 0.2))
    expected = load_factor_from_rates(70.0, -0.05, 0.0, psi_rate)
    assert result.deviation.crossing_load_factor == pytest.approx(expected, abs=1e-6)


def test_a_censored_track_measures_its_last_window_and_says_how_far_out_it_ended():
    result = _grade(observed_track_payload(censored=True, psi_rate_rad_s=0.02))
    assert result.event_status == "estimated"
    assert result.deviation.crossing_load_factor_source == LOAD_FACTOR_FROM_ADSB
    assert result.deviation.crossing_load_factor_window["ends_m_before_threshold"] == 325.0
    assert result.deviation.crossing_load_factor > 1.005


def test_the_window_ends_at_the_bracket_not_at_the_track_end():
    """Real tracks can keep a few samples after the crossing pair (12 of 2,000 KRDU
    records): the window must end at the bracket's right sample, so a turn that
    starts AFTER the crossing must not enter n."""
    steady = _grade(observed_track_payload(psi_rate_rad_s=0.0, trailing_samples=4))
    assert steady.deviation.crossing_load_factor_window["ends_m_before_threshold"] == 0.0
    # Bend the heading only on the trailing samples: n at the crossing stays at cos γ.
    payload = observed_track_payload(trailing_samples=4)
    for sample in payload["states"][-4:]:
        sample["psi"] += 0.5
    bent = _grade(payload)
    assert bent.deviation.crossing_load_factor == pytest.approx(math.cos(0.05), abs=1e-6)
    assert bent.deviation.crossing_load_factor_window["psi_rate_rad_s"] == pytest.approx(0.0, abs=1e-12)


def test_too_short_a_window_is_declared_not_guessed():
    result = _grade(observed_track_payload(samples=LOAD_FACTOR_MIN_SAMPLES - 1))
    assert result.deviation.crossing_load_factor == 1.0
    assert result.deviation.crossing_load_factor_source == LOAD_FACTOR_ASSUMED_1G
    assert "declared" in result.deviation.crossing_load_factor_window["reason"]
    assert result.speed_bounds.lower_ms == pytest.approx(_LOWER_1G)


def test_the_report_counts_measured_and_declared_load_factors_apart():
    report = evaluate_batch(
        [
            record_from_dict(observed_track_payload(psi_rate_rad_s=0.02)),
            record_from_dict(observed_track_payload(samples=2)),
        ],
        contexts={("KRDU", "05L"): assessment_context()},
    )
    aggregate = report["crossing_load_factor"]
    assert aggregate["adsb_kinematics"] == 1 and aggregate["assumed_1g"] == 1
    assert aggregate["max"] > 1.005 and aggregate["min"] == 1.0
    rows = report["trajectories"]
    assert rows[0]["deviation"]["crossing_load_factor_window"]["samples"] >= LOAD_FACTOR_MIN_SAMPLES
    assert rows[1]["deviation"]["crossing_load_factor_window"]["samples"] == 2
    assert LOAD_FACTOR_FROM_ADSB in report["methodology"]["terminal_speed"]["load_factor"]["sources"]
