"""The executor (`autopilot/`, executor design): conventions, words in force, the one-cycle plant and
the exact inverse, on synthetic states of a real airframe."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aircraft.aero_params import aero_params_for_aircraft
from flight_scenarios.scenario import aircraft_for_code
from ts_transformer.autopilot import inverse
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import AirportCharts, compass_deg, read_state, wrap180
from ts_transformer.autopilot.plant import Plant
from ts_transformer.autopilot.sentence import Delays, words_in_force
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, APPROACH, APPROACH_CLEARED, APPROACH_NOT_CLEARED, HEADING, SPEED, UNCHANGED, Words,
    compass_from_math_rad, wrap180 as np_wrap180,
)
from ts_transformer.tests.support import instruction_airport, instruction_spec as spec

F64 = torch.float64
CPU = torch.device("cpu")


# ---- conventions
def test_the_state_reads_as_the_words_read_a_flight():
    psi = torch.linspace(-7.0, 7.0, 101, dtype=F64)
    assert compass_deg(psi).numpy() == pytest.approx(compass_from_math_rad(psi.numpy()), abs=1e-9)
    angles = torch.linspace(-720.0, 720.0, 97, dtype=F64)
    assert wrap180(angles).numpy() == pytest.approx(np_wrap180(angles.numpy()), abs=1e-9)
    geometry = instruction_airport()
    charts = AirportCharts.of([geometry, geometry], dtype=F64, device=CPU)
    lat, lon = torch.tensor([35.1, 34.8], dtype=F64), torch.tensor([-78.3, -77.6], dtype=F64)
    e, n = charts.horizontal(lat, lon)
    want_e, want_n = geometry.frame.horizontal_from_latlon(lat.numpy(), lon.numpy())
    assert (e.numpy(), n.numpy()) == (pytest.approx(want_e, abs=1e-6), pytest.approx(want_n, abs=1e-6))
    # a state flying north-east, descending 3° at 100 m/s
    state = torch.tensor([[35.1, -78.3, 900.0, 100.0, math.radians(45.0), math.radians(-3.0), 60000.0]], dtype=F64)
    k = read_state(state, AirportCharts.of([geometry], dtype=F64, device=CPU))
    assert float(k.track_deg) == pytest.approx(45.0) and float(k.height_m) == 900.0
    assert float(k.ground_speed_mps) == pytest.approx(100.0 * math.cos(math.radians(3.0)))


# ---- words in force
def test_each_word_takes_effect_its_columns_delay_after_its_step_and_stays_in_force():
    one, words = spec(), Words(spec())
    grid = np.full((5, 6), UNCHANGED, dtype=np.int64)
    grid[0] = [0, APPROACH_NOT_CLEARED, words.heading_index(270.0), words.altitude_index(1200.0), 0,
               words.speed_index(100.0)]
    grid[2, HEADING] = words.heading_index(180.0)           # issued at t = 4 s
    grid[2, APPROACH] = APPROACH_CLEARED
    grid[3, ALTITUDE] = words.altitude_land                 # issued at t = 6 s
    grid[3, ANGLE] = 3
    grid[4, SPEED] = words.speed_unspecified                # issued at t = 8 s
    delays = Delays(heading_s=3.0, vertical_s=1.0, speed_s=0.0)
    force = words_in_force([grid], words, delays, cycle_s=1.0, cycles=20, device=CPU)
    heading = force.heading_deg[0].numpy()
    assert (heading[:7] == 270.0).all() and (heading[7:] == 180.0).all()        # 4 s + 3 s
    assert force.approach[0, 6] == APPROACH_NOT_CLEARED and force.approach[0, 7] == APPROACH_CLEARED
    land = force.land[0].numpy()
    assert not land[:7].any() and land[7:].all()                                 # 6 s + 1 s
    assert float(force.altitude_m[0, 0]) == 1200.0 and float(force.angle_deg[0, 7]) == one.descent_angle_centres_deg[2]
    assert not force.unspecified[0, :8].any() and force.unspecified[0, 8:].all()
    assert (force.runway[0] == 0).all()
    assert force.issued_step[0, 6, HEADING] == 0 and force.issued_step[0, 7, HEADING] == 2
    assert force.issued_step[0, 19, SPEED] == 4                                  # held after the last step


def test_a_sentence_must_write_every_column_at_step_0():
    words = Words(spec())
    grid = np.full((3, 6), UNCHANGED, dtype=np.int64)
    grid[0, :5] = [0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="step 0 must write every column"):
        words_in_force([grid], words, Delays(0.0, 0.0, 0.0), cycle_s=1.0, cycles=4, device=CPU)


# ---- the plant and the inverse
def _a320(states: list[list[float]]) -> tuple[FlightInputs, AirportCharts]:
    aircraft = aircraft_for_code("A320")
    aero = aero_params_for_aircraft(aircraft)
    rows = len(states)
    inputs = FlightInputs(
        initial_state=torch.tensor(states, dtype=F64),
        aero_params=torch.tensor([[aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall]] * rows,
                                 dtype=F64),
        frame_params=torch.tensor([[35.0, -78.0, 100.0, 0.0]] * rows, dtype=F64),
        max_thrust_n=torch.full((rows,), float(aircraft.engine.max_thrust_total_n), dtype=F64),
    )
    geometry = instruction_airport()
    return inputs, AirportCharts.of([geometry] * rows, dtype=F64, device=CPU)


def _state(track_deg: float, gamma_deg: float, speed: float = 100.0, height: float = 1500.0) -> list[float]:
    return [35.05, -78.05, height, speed, math.radians(90.0 - track_deg), math.radians(gamma_deg), 62000.0]


def _fly(inputs, charts, track_rate, gamma_rate, accel, previous_bank=None, *, bank_cap_deg=30.0,
         bank_rate_deg_s=1000.0):
    k0 = read_state(inputs.initial_state, charts)
    rows = len(inputs.initial_state)
    previous = torch.zeros(rows, dtype=F64) if previous_bank is None else previous_bank

    def column(value):
        return torch.as_tensor(value, dtype=F64).expand(rows).clone()

    att = inverse.attitude(k0, column(track_rate), column(gamma_rate), previous,
                           bank_cap_rad=math.radians(bank_cap_deg), bank_rate_rad_s=math.radians(bank_rate_deg_s),
                           cycle_s=1.0)
    thr = inverse.thrust(k0, column(accel), att.load_factor, inputs.aero_params, inputs.max_thrust_n)
    command = torch.stack((thr.fraction, att.bank_rad, att.load_factor), dim=1)
    k1 = read_state(Plant(inputs).step(inputs.initial_state, command, 1.0), charts)
    rates = (wrap180(k1.track_deg - k0.track_deg), k1.gamma_rad - k0.gamma_rad, k1.speed_mps - k0.speed_mps)
    return att, thr, rates


def test_the_inverse_flies_the_rates_it_was_asked_for():
    """§13 E2: one cycle forward through the dynamics gives the rates the inverse solved for, to
    within what the zero-order hold costs (V and γ move inside the cycle while the controls do not)."""
    inputs, charts = _a320([_state(45.0, 0.0), _state(45.0, -3.0), _state(300.0, -3.0, 80.0, 900.0),
                            _state(10.0, 2.0, 120.0)])
    asked_track = torch.tensor([2.0, -2.5, 1.5, 0.0], dtype=F64)             # compass deg/s, + = right
    asked_gamma = torch.tensor([0.0, math.radians(0.3), math.radians(-0.2), 0.0], dtype=F64)
    asked_accel = torch.tensor([-0.3, -0.2, 0.0, 0.4], dtype=F64)
    att, thr, (track, gamma, speed) = _fly(inputs, charts, asked_track, asked_gamma, asked_accel)
    assert not any(bool(b.any()) for b in (*att.binds.values(), *thr.binds.values()))
    assert track.numpy() == pytest.approx(asked_track.numpy(), rel=0.02, abs=0.01)
    assert gamma.numpy() == pytest.approx(asked_gamma.numpy(), rel=0.05, abs=math.radians(0.01))
    # the speed rate moves with the gravity term while γ turns inside the cycle: g · |γ̇| · Δt / 2
    hold = 9.81 * np.abs(asked_gamma.numpy()) * 1.0 / 2.0
    assert np.all(np.abs(speed.numpy() - asked_accel.numpy()) <= hold + 0.005)
    # a right (clockwise) turn is a NEGATIVE bank in the dynamics' convention, a left turn positive
    assert float(att.bank_rad[0]) < 0.0 < float(att.bank_rad[1])


def test_each_limit_binds_on_the_quantity_it_bounds():
    inputs, charts = _a320([_state(90.0, 0.0)] * 4)
    # a bank cap costs turn rate and keeps the path: 10°/s at 100 m/s wants ~60° of bank
    att, _, (track, gamma, _) = _fly(inputs, charts, 10.0, 0.0, 0.0, bank_cap_deg=25.0)
    assert att.binds["bank_cap"].all() and float(att.bank_rad[0]) == pytest.approx(-math.radians(25.0))
    capped_rate = math.degrees(9.81 * math.tan(math.radians(25.0)) / 100.0)
    assert float(track[0]) == pytest.approx(capped_rate, rel=0.02) and abs(float(gamma[0])) < math.radians(0.02)
    # the bank moves at most its rate per cycle
    att, _, _ = _fly(inputs, charts, 3.0, 0.0, 0.0, torch.zeros(4, dtype=F64), bank_rate_deg_s=2.0)
    assert att.binds["bank_rate"].all() and float(att.bank_rad[0]) == pytest.approx(-math.radians(2.0))
    # a pull the envelope does not allow is clamped to its load factor
    att, _, _ = _fly(inputs, charts, 0.0, math.radians(20.0), 0.0)
    assert att.binds["load_factor"].all() and float(att.load_factor[0]) == inverse.LOAD_FACTOR_MAX
    # more acceleration than the engines give, and more deceleration than idle and drag give
    _, thr, _ = _fly(inputs, charts, 0.0, 0.0, 6.0)
    assert thr.binds["thrust_max"].all() and float(thr.fraction[0]) == pytest.approx(1.0)
    _, thr, _ = _fly(inputs, charts, 0.0, 0.0, -8.0)
    assert thr.binds["thrust_min"].all() and not thr.binds["stall"].any()


def test_the_bank_cap_stays_inside_the_graders_envelope():
    inputs, charts = _a320([_state(0.0, 0.0)])
    k = read_state(inputs.initial_state, charts)
    with pytest.raises(ValueError, match="bank cap"):
        inverse.attitude(k, torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64), torch.zeros(1, dtype=F64),
                         bank_cap_rad=math.radians(50.0), bank_rate_rad_s=1.0, cycle_s=1.0)
