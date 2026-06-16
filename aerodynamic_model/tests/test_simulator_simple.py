import math
from dataclasses import fields
from pathlib import Path
import sys
import unittest

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from aircraft_sets import A320  # noqa: E402
from common import Atmosphere, State  # noqa: E402
from simulator_simple import LoadFactorControl, LoadFactorSimulator  # noqa: E402


class TestLoadFactorSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = LoadFactorSimulator()
        self.atmosphere = Atmosphere()

    def test_control_interface_uses_load_factor(self):
        control_fields = {field.name for field in fields(LoadFactorControl)}

        self.assertIn("load_factor", control_fields)
        self.assertNotIn("attack_rad", control_fields)

    def test_aerodynamic_coefficients_use_current_state(self):
        speed = 140.0
        altitude = 1000.0
        mass = A320.mass_kg
        load_factor = 1.2

        Cl, Cd, stalled = self.simulator.get_aerodynamic_coefficients(
            load_factor,
            speed,
            self.atmosphere,
            altitude,
            mass,
        )

        rho = self.atmosphere.get_ISA_density(altitude)
        expected_Cl = (
            load_factor
            * mass
            * self.simulator.g
            / (0.5 * rho * speed**2 * A320.wing_area_m2)
        )
        self.assertAlmostEqual(Cl, expected_Cl)
        self.assertAlmostEqual(Cd, self.simulator.Cd0 + self.simulator.k * Cl**2)
        self.assertFalse(stalled)

    def test_level_one_g_zero_bank_has_no_flight_path_rate(self):
        state_vec = np.array([0.0, 0.0, 1000.0, 150.0, 0.0, 0.0, A320.mass_kg])
        control = LoadFactorControl(
            thrust=A320.approach_thrust_guess_n,
            bank_rad=0.0,
            load_factor=1.0,
        )

        derivatives = self.simulator.dynamics(
            0.0,
            state_vec,
            control,
            self.atmosphere,
        )

        self.assertAlmostEqual(derivatives[5], 0.0)

    def test_stall_limits_actual_load_factor(self):
        speed = 80.0
        altitude = 1000.0
        mass = A320.mass_kg
        state_vec = np.array([0.0, 0.0, altitude, speed, 0.0, 0.0, mass])
        control = LoadFactorControl(
            thrust=A320.approach_thrust_guess_n,
            bank_rad=0.0,
            load_factor=5.0,
        )

        derivatives = self.simulator.dynamics(
            0.0,
            state_vec,
            control,
            self.atmosphere,
        )

        rho = self.atmosphere.get_ISA_density(altitude)
        actual_load_factor = (
            0.5
            * rho
            * speed**2
            * self.simulator.Cl_max
            * A320.wing_area_m2
            / (mass * self.simulator.g)
        )
        expected_gamma_rate = self.simulator.g * (actual_load_factor - 1.0) / speed
        self.assertLess(actual_load_factor, control.load_factor)
        self.assertAlmostEqual(derivatives[5], expected_gamma_rate)

    def test_simulate_advances_one_rk4_step_for_load_factor_control(self):
        initial_state = State(
            x=0.0,
            y=0.0,
            h=1000.0,
            V=150.0,
            psi=0.0,
            gamma=0.0,
            m=A320.mass_kg,
        )
        control = LoadFactorControl(
            thrust=A320.approach_thrust_guess_n,
            bank_rad=math.radians(10.0),
            load_factor=1.0,
        )

        next_state = self.simulator.simulate(
            initial_state,
            control,
            self.atmosphere,
            t=0.0,
            dt=0.2,
        )

        self.assertIsInstance(next_state, State)
        self.assertGreater(next_state.x, initial_state.x)
        self.assertEqual(next_state.m, initial_state.m)


if __name__ == "__main__":
    unittest.main()
