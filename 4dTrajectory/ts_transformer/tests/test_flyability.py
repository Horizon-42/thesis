"""The flyability inversion: does it recover controls we know the answer to?

The inversion is closed-form, so it can be tested against analytically constructed
trajectories rather than against a solver's output. A coordinated level turn at a known
bank is the strongest case — it pins load factor AND bank simultaneously, and getting
either the sign convention or the transport terms wrong breaks it visibly.

What is deliberately NOT asserted: that any particular real trajectory is flyable. The
check carries a known systematic bias (one clean-configuration drag polar, one Cl_max,
against approaches actually flown dirty), which is why the reported metric is the delta
against the observed tracks rather than an absolute rate.
"""

import math
import sys
from pathlib import Path

import pytest

_TS_DIR = Path(__file__).resolve().parents[1]
if str(_TS_DIR) not in sys.path:
    sys.path.insert(0, str(_TS_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aircraft.aero_params import aero_params_for_aircraft  # noqa: E402
from aircraft.aircraft_sets import AIRCRAFT_PRESETS  # noqa: E402
from flyability import (  # noqa: E402
    G, HARD_VIOLATIONS, SOFT_VIOLATIONS, Envelope, calibrated_report, flyability_batch,
    flyability_summary, isa_density, report_for_records, required_controls,
)

A320 = AIRCRAFT_PRESETS["A320"]
AERO = aero_params_for_aircraft(A320)
MASS = 62_000.0


def _state(t, *, lat=35.9, lon=-78.8, alt=1000.0, V=100.0, psi=0.0, gamma=0.0, m=MASS):
    return {"t": t, "lat": lat, "lon": lon, "alt": alt, "V": V, "psi": psi,
            "gamma": gamma, "m": m, }


def _level_turn(bank_deg: float, *, V=100.0, n_samples=9, dt=2.0, lat=0.0):
    """A coordinated level turn at a known bank, sampled on a uniform grid.

    Coordinated + level ⇒ n = 1/cos(bank) and psi_dot = g tan(bank) / V. `lat=0` keeps the
    meridian-convergence transport term at zero so the test isolates the inversion itself
    (the transport terms get their own test).
    """
    psi_rate = G * math.tan(math.radians(bank_deg)) / V
    return [_state(i * dt, V=V, psi=i * dt * psi_rate, gamma=0.0, lat=lat)
            for i in range(n_samples)]


# ── The inversion ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bank_deg", [0.0, 10.0, 25.0, -30.0])
def test_coordinated_level_turn_inverts_to_its_own_bank_and_load_factor(bank_deg):
    # The analytic pair: n = 1/cos(mu) for a level turn. Interior samples only — the ends
    # use a one-sided difference, which is first-order and not expected to hit this.
    states = _level_turn(bank_deg)
    controls = required_controls(states, A320, aero=AERO, transport="none")

    for control in controls[1:-1]:
        assert math.degrees(control.bank_rad) == pytest.approx(bank_deg, abs=0.5)
        assert control.load_factor == pytest.approx(1.0 / math.cos(math.radians(bank_deg)),
                                                    rel=1e-3)


def test_straight_level_flight_needs_one_g_and_no_bank():
    controls = required_controls([_state(i * 2.0) for i in range(5)], A320, aero=AERO,
                                 transport="none")
    for control in controls[1:-1]:
        assert control.load_factor == pytest.approx(1.0, rel=1e-6)
        assert control.bank_rad == pytest.approx(0.0, abs=1e-9)
        assert control.flyable


def test_steady_descent_needs_thrust_below_level_flight():
    # On a 3-degree descent gravity does part of the work, so required thrust drops by
    # ~m*g*sin(gamma) relative to level. This is the sign check on the along-track equation:
    # get it backwards and every descent reads as needing MORE thrust.
    level = required_controls([_state(i * 2.0) for i in range(5)], A320, aero=AERO,
                              transport="none")[2]
    descent = required_controls([_state(i * 2.0, gamma=math.radians(-3.0)) for i in range(5)],
                                A320, aero=AERO, transport="none")[2]
    expected_drop = MASS * G * math.sin(math.radians(3.0))
    assert descent.thrust_n == pytest.approx(level.thrust_n - expected_drop, rel=0.02)


def test_heading_difference_takes_the_short_way_around_the_branch_cut():
    # psi wrapping from +179deg to -179deg is a 2deg turn, not a 358deg one. A plain
    # subtraction would infer an enormous psi_dot and report an impossible bank — exactly
    # where an aircraft turns onto final.
    dt, V = 2.0, 100.0
    states = [_state(i * dt, V=V, psi=math.radians(a))
              for i, a in enumerate((177.0, 179.0, -179.0, -177.0, -175.0))]
    controls = required_controls(states, A320, aero=AERO, transport="none")

    # 2 deg per 2 s at 100 m/s is a gentle turn: bank = atan(psi_dot * V / g).
    expected = math.degrees(math.atan(math.radians(1.0) * V / G))
    for control in controls[1:-1]:
        assert math.degrees(control.bank_rad) == pytest.approx(expected, abs=0.5)
        assert control.flyable


def test_transport_terms_change_the_answer_and_default_to_approx():
    # Straight-and-level at high latitude: the rotating tangent plane contributes a real
    # psi_dot/gamma_dot. Not subtracting it bills the aircraft for a bank it never flew.
    states = [_state(i * 2.0, lat=60.0, psi=math.radians(30.0)) for i in range(5)]
    without = required_controls(states, A320, aero=AERO, transport="none")[2]
    with_approx = required_controls(states, A320, aero=AERO, transport="approx")[2]
    assert with_approx.bank_rad != pytest.approx(without.bank_rad, abs=1e-6)

    default = required_controls(states, A320, aero=AERO)[2]
    assert default.bank_rad == pytest.approx(with_approx.bank_rad, abs=1e-12)


# ── Envelope + classification ────────────────────────────────────────────────

def test_stall_is_flagged_when_lift_demand_exceeds_cl_max():
    # Slow and heavy: Cl_required = n m g / (0.5 rho S V^2) blows past Cl_max as V drops.
    slow = [_state(i * 2.0, V=45.0, alt=0.0) for i in range(5)]
    controls = required_controls(slow, A320, aero=AERO, transport="none")
    assert "stall" in controls[2].violations and not controls[2].flyable

    fast = required_controls([_state(i * 2.0, V=140.0, alt=0.0) for i in range(5)],
                             A320, aero=AERO, transport="none")[2]
    assert "stall" not in fast.violations


def test_negative_required_thrust_is_reported_but_not_counted_as_unflyable():
    # A steep descent needs less thrust than idle — i.e. drag augmentation (speedbrake,
    # gear, flaps). Real flown approaches do this constantly, so counting it as unflyable
    # would score the observed truth as infeasible. It must appear in `violations` (so the
    # demand stays visible) yet leave `flyable` true.
    steep = [_state(i * 2.0, gamma=math.radians(-8.0), V=110.0) for i in range(5)]
    control = required_controls(steep, A320, aero=AERO, transport="none")[2]

    assert control.thrust_n < 0.0
    assert "thrust_negative" in control.violations
    assert "thrust_negative" in SOFT_VIOLATIONS and "thrust_negative" not in HARD_VIOLATIONS
    assert control.flyable
    assert control.hard_violations == ()

    summary = flyability_summary(required_controls(steep, A320, aero=AERO, transport="none"),
                                 aircraft_code="A320")
    assert summary["soft"]["thrust_negative"] > 0
    assert "thrust_negative" not in summary["violations"]


def test_envelope_reads_its_limits_from_the_aircraft_and_aero_params():
    envelope = Envelope.for_aircraft(A320, AERO)
    assert envelope.max_thrust_n == A320.engine.max_thrust_total_n == 240_000.0
    # Cl_max comes from aero_params_for_aircraft (2.7 for an A320), NOT from
    # LoadFactorSimulator's hardcoded 1.5 — the two disagree by 80% and aero_params.py
    # documents itself as the source of truth for the stall model.
    assert envelope.cl_max == AERO.Cl_max == 2.7


def test_isa_density_matches_the_repo_atmosphere():
    from aerodynamic_model.common import Atmosphere

    for altitude in (0.0, 500.0, 2000.0, 8000.0):
        assert isa_density(altitude) == pytest.approx(
            Atmosphere().get_ISA_density(altitude), rel=1e-12)


def test_a_sample_too_slow_to_invert_is_flagged_rather_than_dividing_by_zero():
    stopped = [_state(i * 2.0, V=0.2) for i in range(5)]
    control = required_controls(stopped, A320, aero=AERO)[2]
    assert control.violations == ("speed_below_1ms",) and not control.flyable
    assert math.isnan(control.load_factor)


def test_required_controls_rejects_a_trajectory_too_short_to_difference():
    with pytest.raises(ValueError, match="at least two"):
        required_controls([_state(0.0)], A320, aero=AERO)


# ── Reporting ────────────────────────────────────────────────────────────────

def test_calibrated_report_states_the_observed_baseline_and_the_delta():
    # The headline is the DELTA, because both sides carry the same clean-polar bias.
    envelopes = {"A320": Envelope.for_aircraft(A320, AERO)}
    flyable = [flyability_summary(required_controls(_level_turn(10.0), A320, aero=AERO,
                                                    transport="none"), aircraft_code="A320")]
    stalled = [flyability_summary(required_controls(
        [_state(i * 2.0, V=45.0, alt=0.0) for i in range(5)], A320, aero=AERO,
        transport="none"), aircraft_code="A320")]

    report = calibrated_report(stalled, flyable, envelopes=envelopes)
    assert report["observed_baseline"]["fully_flyable_rate"] == 1.0
    assert report["predicted"]["fully_flyable_rate"] == 0.0
    assert report["delta"]["fully_flyable_rate"] == pytest.approx(-1.0)
    assert "floor, not 100%" in report["note"]


def test_batch_roll_up_records_the_envelope_it_judged_against():
    envelopes = {"A320": Envelope.for_aircraft(A320, AERO)}
    summaries = [flyability_summary(required_controls(_level_turn(15.0), A320, aero=AERO,
                                                      transport="none"),
                                    aircraft_code="A320")]
    batch = flyability_batch(summaries, envelopes=envelopes)

    assert batch["trajectories"] == 1 and batch["fully_flyable"] == 1
    assert batch["fleet"] == {"A320": 1}
    assert batch["envelopes"]["A320"]["cl_max"] == 2.7
    # The bounds that are the project's working numbers rather than certified limits say so.
    assert "working values" in batch["envelopes"]["A320"]["note"]


# ── Mixed fleet ──────────────────────────────────────────────────────────────

def test_each_flight_is_judged_against_its_own_airframe():
    # The KRDU harvest spans 14 types. A shared envelope grades a regional jet by an A320's
    # Cl_max and max thrust, which silently mis-scores stall and thrust for most of a batch.
    # Same trajectory, two airframes -> judged against two different envelopes.
    heavy = AIRCRAFT_PRESETS["B77W"]
    states = [_state(i * 2.0, V=70.0, alt=300.0) for i in range(5)]

    report = report_for_records([states, states], [states, states], [A320, heavy],
                                transport="none")
    assert report["predicted"]["fleet"] == {"A320": 1, "B77W": 1}
    assert set(report["predicted"]["envelopes"]) == {"A320", "B77W"}

    # The two envelopes really do differ — otherwise this test would pass vacuously.
    envelopes = report["predicted"]["envelopes"]
    assert envelopes["A320"]["max_thrust_n"] != envelopes["B77W"]["max_thrust_n"]


def test_report_for_records_rejects_a_fleet_that_does_not_line_up():
    states = [_state(i * 2.0) for i in range(5)]
    with pytest.raises(ValueError, match="one aircraft per flight"):
        report_for_records([states, states], [states, states], [A320])
