"""The attitude the Training views draw an aircraft in (`experiments/training_attitude.py`, doc 36 §4.12): one reading of a
state row with its bank and load factor, fed by the executor's own commands or by the teacher's inversion of an observed
flight."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aerodynamic_model.common import GeodeticState
from aerodynamic_model.simulator import Simulator
from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from aircraft.aero_params import aero_params_for_aircraft
from aircraft.aircraft_sets import A320
from ts_transformer.data.channels import channels_from_states
from ts_transformer.data.coordinate_frames import ENUFrame
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.experiments import training_attitude
from ts_transformer.experiments.training_attitude import ATTITUDE_FIELDS, attitude, attitude_payload, executor_attitude

AERO = aero_params_for_aircraft(A320)
AERO_ROW = np.array([AERO.S, AERO.Cl_max, AERO.Cd0, AERO.k, AERO.stall_threshold, AERO.k_stall])


def _row(speed=80.0, psi=0.0, gamma=0.0, mass=64000.0, altitude=0.0):
    return [35.8, -78.7, altitude, speed, psi, gamma, mass]


def test_the_heading_is_compass_the_bank_is_to_the_right_and_the_path_angle_is_gamma():
    states = np.array([_row(psi=0.0, gamma=math.radians(-3.0)), _row(psi=math.pi / 2)])
    got = attitude(states, np.radians([20.0, -15.0]), np.ones(2), AERO_ROW)
    assert got["headingDeg"] == pytest.approx([90.0, 0.0])            # math east → compass 090; math north → 000
    assert got["pathAngleDeg"] == pytest.approx([-3.0, 0.0])
    assert got["bankRightDeg"] == pytest.approx([-20.0, 15.0])         # the dynamics' positive bank turns left


def test_the_attack_is_the_lift_the_dynamics_asks_for_read_back_through_the_lift_curve():
    speed, mass, load = 70.0, 64000.0, 1.1
    got = attitude(np.array([_row(speed=speed, mass=mass)]), np.zeros(1), np.array([load]), AERO_ROW)
    lift = load * mass * GRAVITY_MPS2 / (0.5 * 1.225 * AERO.S * speed ** 2)   # sea level: ISA ρ0
    assert got["attackDeg"][0] == pytest.approx(math.degrees((lift - Simulator.CL0) / Simulator.CL_alpha))
    # past the stall the dynamics flies Cl_max, and so does the reading
    slow = attitude(np.array([_row(speed=30.0, mass=mass)]), np.zeros(1), np.ones(1), AERO_ROW)
    assert slow["attackDeg"][0] == pytest.approx(math.degrees((AERO.Cl_max - Simulator.CL0) / Simulator.CL_alpha))


def test_a_payload_carries_every_field_at_its_points():
    got = attitude_payload(attitude(np.array([_row(), _row()]), np.zeros(2), np.ones(2), AERO_ROW))
    assert tuple(got) == ATTITUDE_FIELDS and all(len(values) == 2 for values in got.values())
    assert got["bankRightDeg"] == [0.0, 0.0] and str(got["bankRightDeg"][0]) == "0.0"   # never a "-0.0"


def test_an_executor_row_reads_the_commands_of_the_cycle_that_starts_there_and_the_last_row_its_last_cycle():
    states = torch.tensor([[_row(psi=0.1 * k) for k in range(4)]], dtype=torch.float64)       # 3 cycles: 4 rows
    commands = torch.tensor([[[0.3, math.radians(b), 1.0] for b in (5.0, 10.0, 15.0)]], dtype=torch.float64)
    flown = SimpleNamespace(states=states, commands=commands)
    got = executor_attitude(flown, 0, [0, 2, 3], AERO_ROW)
    assert got["bankRightDeg"] == pytest.approx([-5.0, -15.0, -15.0])
    assert got["headingDeg"] == pytest.approx([(90.0 - math.degrees(0.1 * k)) % 360.0 for k in (0, 2, 3)])


def _turning_series(bank_left_deg: float, speed=80.0, gamma_deg=-2.0, step_s=2.0, rows=40) -> FlightSeries:
    """An observed flight in a steady coordinated turn — its rows as the data plane stores them (channels)."""
    frame = ENUFrame(lat0=35.8, lon0=-78.7, alt0=0.0)
    gamma = math.radians(gamma_deg)
    rate = GRAVITY_MPS2 * math.tan(math.radians(bank_left_deg)) / (speed * math.cos(gamma))   # math ψ̇: left is positive
    e = n = 0.0
    height, samples = 900.0, []
    for row in range(rows):
        psi = 0.3 + rate * row * step_s
        lat, lon = frame.latlon_from_horizontal(e, n)
        samples.append((row * step_s, GeodeticState(lat, lon, height, speed, psi, gamma, 64000.0)))
        # exact arc to the next row
        e += speed * math.cos(gamma) * (math.sin(psi + rate * step_s) - math.sin(psi)) / rate
        n += -speed * math.cos(gamma) * (math.cos(psi + rate * step_s) - math.cos(psi)) / rate
        height += speed * math.sin(gamma) * step_s
    times, values = channels_from_states(samples, frame)
    scenario = SimpleNamespace(initial=SimpleNamespace(m=64000.0), has_dynamics=True,
                               dynamics=lambda _purpose: (A320, AERO))
    return FlightSeries(flight_id="TURN", scenario=scenario, frame=frame, times=times, values=values)


def test_an_observed_flight_reads_the_bank_it_turned_at_through_the_teachers_inversion():
    got = training_attitude.observed_attitude(_turning_series(bank_left_deg=20.0))
    inner = slice(2, -2)                                                   # the gradient's one-sided edges aside
    assert got["bankRightDeg"][inner] == pytest.approx(-20.0, abs=0.3)
    assert got["pathAngleDeg"] == pytest.approx(-2.0, abs=1e-6)
    right = training_attitude.observed_attitude(_turning_series(bank_left_deg=-10.0))
    assert right["bankRightDeg"][inner] == pytest.approx(10.0, abs=0.3)


def test_a_flight_without_an_airframe_has_its_heading_and_path_angle_and_no_bank_or_attack():
    """C31: the dynamics has no airframe for it — the inversion reads one (its drag), and none is guessed."""
    series = _turning_series(bank_left_deg=20.0)
    series.scenario.has_dynamics = False
    series.scenario.dynamics = None                                    # never asked
    got = training_attitude.observed_attitude(series)
    assert got["bankRightDeg"] is None and got["attackDeg"] is None
    assert got["pathAngleDeg"] == pytest.approx(-2.0, abs=1e-6) and len(got["headingDeg"]) == 40
    written = attitude_payload(got, slice(0, 3))
    assert written["bankRightDeg"] is None and len(written["headingDeg"]) == 3
