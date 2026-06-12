import math
from dataclasses import fields
from pathlib import Path
import sys
import unittest


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from aircraft_sets import A320  # noqa: E402
from simulator import Atmosphere, Control, Simulator, State  # noqa: E402


class TestAlphaInputSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = Simulator()
        self.atmosphere = Atmosphere()

    def test_control_interface_uses_attack_angle(self):
        # This protects the alpha-input simulator boundary.
        control_fields = {field.name for field in fields(Control)}

        self.assertIn("attack_rad", control_fields)
        self.assertNotIn("load_factor", control_fields)

    def test_aerodynamic_coefficients_are_derived_from_alpha(self):
        alpha = math.radians(5.0)

        Cl, Cd = self.simulator.get_aerodynamic_coefficients(alpha)

        expected_Cl = self.simulator.CL0 + self.simulator.CL_alpha * alpha
        expected_Cd = self.simulator.Cd0 + self.simulator.k * expected_Cl**2
        self.assertAlmostEqual(Cl, expected_Cl)
        self.assertAlmostEqual(Cd, expected_Cd)

    def test_aerodynamic_forces_and_load_factor_follow_alpha_model(self):
        V = 120.0
        h = 1000.0
        m = 10000.0
        alpha = math.radians(3.0)

        lift, drag = self.simulator.get_aerodynamic_forces(
            V, alpha, self.atmosphere, h
        )
        load_factor = self.simulator.get_load_factor(lift, m)

        rho = self.atmosphere.get_ISA_density(h)
        Cl, Cd = self.simulator.get_aerodynamic_coefficients(alpha)
        dynamic_pressure_area = 0.5 * rho * V**2 * A320.wing_area_m2
        self.assertAlmostEqual(lift, dynamic_pressure_area * Cl)
        self.assertAlmostEqual(drag, dynamic_pressure_area * Cd)
        self.assertAlmostEqual(load_factor, lift / (m * self.simulator.g))

    def test_dynamics_uses_bank_to_split_lift_and_alpha_thrust_normal_force(self):
        state = State(
            x=100.0,
            y=200.0,
            h=1000.0,
            V=120.0,
            psi=math.radians(30.0),
            gamma=math.radians(2.0),
            m=10000.0,
        )
        control = Control(
            thrust=12000.0,
            bank_rad=math.radians(25.0),
            attack_rad=math.radians(4.0),
        )
        state_vec = [state.x, state.y, state.h, state.V, state.psi, state.gamma, state.m]

        derivatives = self.simulator.dynamics(
            0.0, state_vec, control, self.atmosphere
        )

        lift, drag = self.simulator.get_aerodynamic_forces(
            state.V, control.attack_rad, self.atmosphere, state.h
        )
        normal_force = lift + control.thrust * math.sin(control.attack_rad)
        self.assertAlmostEqual(
            derivatives[0], state.V * math.cos(state.gamma) * math.cos(state.psi)
        )
        self.assertAlmostEqual(
            derivatives[1], state.V * math.cos(state.gamma) * math.sin(state.psi)
        )
        self.assertAlmostEqual(derivatives[2], state.V * math.sin(state.gamma))
        self.assertAlmostEqual(
            derivatives[3],
            (control.thrust * math.cos(control.attack_rad) - drag) / state.m
            - self.simulator.g * math.sin(state.gamma),
        )
        self.assertAlmostEqual(
            derivatives[4],
            normal_force
            * math.sin(control.bank_rad)
            / (state.m * state.V * math.cos(state.gamma)),
        )
        self.assertAlmostEqual(
            derivatives[5],
            (
                normal_force * math.cos(control.bank_rad)
                - state.m * self.simulator.g * math.cos(state.gamma)
            )
            / (state.m * state.V),
        )
        self.assertEqual(derivatives[6], 0)

    def test_zero_bank_has_no_heading_rate(self):
        state_vec = [0.0, 0.0, 1000.0, 120.0, 0.0, 0.0, 10000.0]
        control = Control(
            thrust=12000.0,
            bank_rad=0.0,
            attack_rad=math.radians(4.0),
        )

        derivatives = self.simulator.dynamics(
            0.0, state_vec, control, self.atmosphere
        )

        self.assertAlmostEqual(derivatives[4], 0.0)

    def test_simulate_returns_requested_samples_for_alpha_control(self):
        initial_state = State(
            x=0.0,
            y=0.0,
            h=1000.0,
            V=120.0,
            psi=0.0,
            gamma=0.0,
            m=10000.0,
        )
        control = Control(
            thrust=12000.0,
            bank_rad=math.radians(10.0),
            attack_rad=math.radians(2.0),
        )
        t_eval = [0.0, 0.5, 1.0]

        solution = self.simulator.simulate(
            initial_state,
            control,
            self.atmosphere,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
        )

        self.assertTrue(solution.success)
        self.assertEqual(list(solution.t), t_eval)
        self.assertEqual(solution.y.shape, (7, len(t_eval)))


if __name__ == "__main__":
    unittest.main()
