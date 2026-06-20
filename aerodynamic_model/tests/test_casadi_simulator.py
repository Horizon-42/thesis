import math
import unittest

import casadi as ca
import numpy as np


from aerodynamic_model.aircraft_sets import A320  # noqa: E402
from aerodynamic_model.casadi_simulator import make_dynamics_model, make_integrator  # noqa: E402
from aerodynamic_model.common import Atmosphere  # noqa: E402


class TestCasadiSimulator(unittest.TestCase):
    def setUp(self):
        self.model = make_dynamics_model()
        self.atmosphere = Atmosphere()
        # Keep this order aligned with casadi_simulator.make_dynamics_model().
        self.aero_params = np.array(
            [
                A320.wing_area_m2,
                1.5,  # Cl_max
                0.02,  # Cd0
                0.04,  # induced drag factor
                0.9,  # stall threshold
                0.1,  # stall drag factor
            ],
            dtype=float,
        )

    def _state_vector(
        self,
        *,
        speed: float = 150.0,
        altitude: float = 1000.0,
        psi: float = 0.0,
        gamma: float = 0.0,
        mass: float = A320.mass_kg,
    ) -> np.ndarray:
        # State order: x, y, h, V, psi, gamma, m.
        return np.array([0.0, 0.0, altitude, speed, psi, gamma, mass], dtype=float)

    def _control_vector(
        self,
        *,
        thrust: float = A320.approach_thrust_guess_n,
        bank_rad: float = 0.0,
        load_factor: float = 1.0,
    ) -> np.ndarray:
        # Control order: thrust, bank angle, commanded load factor.
        return np.array([thrust, bank_rad, load_factor], dtype=float)

    def _rhs(self, state_vec: np.ndarray, control_vec: np.ndarray) -> np.ndarray:
        # rhs_func evaluates the continuous-time derivative xdot = f(x, u, p).
        result = self.model["rhs_func"](
            x=state_vec,
            u=control_vec,
            aero_params=self.aero_params,
        )
        return np.array(result["xdot"], dtype=float).reshape(-1)

    def test_model_exposes_symbolic_contract(self):
        # The outer model dict is our local interface for tests and callers.
        self.assertEqual(
            set(self.model.keys()),
            {"x", "u", "aero_params", "xdot", "dae", "rhs_func"},
        )
        self.assertEqual(self.model["x"].shape, (7, 1))
        self.assertEqual(self.model["u"].shape, (3, 1))
        self.assertEqual(self.model["aero_params"].shape, (6, 1))
        self.assertEqual(self.model["xdot"].shape, (7, 1))

        dae = self.model["dae"]
        # The inner DAE dict is the part CasADi integrator consumes.
        self.assertEqual(set(dae.keys()), {"x", "p", "ode"})
        self.assertEqual(dae["x"].shape, (7, 1))
        self.assertEqual(dae["p"].shape, (9, 1))
        self.assertEqual(dae["ode"].shape, (7, 1))

    def test_dae_can_be_wrapped_as_casadi_function_without_free_variables(self):
        dae = self.model["dae"]

        # If any symbols are missing from x or p, CasADi reports free variables.
        dae_rhs = ca.Function("dae_rhs", [dae["x"], dae["p"]], [dae["ode"]])

        state_vec = self._state_vector()
        control_vec = self._control_vector()
        params = np.concatenate([control_vec, self.aero_params])
        derivatives = np.array(dae_rhs(state_vec, params), dtype=float).reshape(-1)

        self.assertEqual(derivatives.shape, (7,))
        self.assertTrue(np.all(np.isfinite(derivatives)))

    def test_rhs_matches_one_g_level_flight_kinematics(self):
        state_vec = self._state_vector(speed=150.0, gamma=0.0)
        control_vec = self._control_vector(bank_rad=0.0, load_factor=1.0)

        derivatives = self._rhs(state_vec, control_vec)

        # With zero heading and zero flight path angle, motion is along +x.
        self.assertAlmostEqual(derivatives[0], 150.0)
        self.assertAlmostEqual(derivatives[1], 0.0)
        self.assertAlmostEqual(derivatives[2], 0.0)
        # One-g, wings-level flight should not turn or change gamma.
        self.assertAlmostEqual(derivatives[4], 0.0)
        self.assertAlmostEqual(derivatives[5], 0.0)
        self.assertAlmostEqual(derivatives[6], 0.0)

    def test_rhs_limits_actual_load_factor_when_stalled(self):
        speed = 80.0
        altitude = 1000.0
        state_vec = self._state_vector(speed=speed, altitude=altitude)
        control_vec = self._control_vector(load_factor=5.0)

        derivatives = self._rhs(state_vec, control_vec)

        # Once Cl is capped at Cl_max, the actual load factor is physics-limited.
        rho = self.atmosphere.get_ISA_density(altitude)
        wing_area_m2, cl_max = self.aero_params[:2]
        actual_load_factor = 0.5 * rho * speed**2 * cl_max * wing_area_m2 / (
            A320.mass_kg * 9.81
        )
        expected_gamma_rate = 9.81 * (actual_load_factor - 1.0) / speed

        self.assertLess(actual_load_factor, control_vec[2])
        self.assertAlmostEqual(derivatives[5], expected_gamma_rate)

    def test_symbolic_jacobians_can_be_built_and_evaluated(self):
        # Continuous-time linearization: A=dxdot/dx, B=dxdot/du, P=dxdot/dparams.
        jacobian_func = ca.Function(
            "rhs_jacobian",
            [self.model["x"], self.model["u"], self.model["aero_params"]],
            [
                ca.jacobian(self.model["xdot"], self.model["x"]),
                ca.jacobian(self.model["xdot"], self.model["u"]),
                ca.jacobian(self.model["xdot"], self.model["aero_params"]),
            ],
            ["x", "u", "aero_params"],
            ["A", "B", "P"],
        )

        result = jacobian_func(
            x=self._state_vector(),
            u=self._control_vector(bank_rad=math.radians(5.0)),
            aero_params=self.aero_params,
        )

        self.assertEqual(result["A"].shape, (7, 7))
        self.assertEqual(result["B"].shape, (7, 3))
        self.assertEqual(result["P"].shape, (7, 6))
        self.assertTrue(np.all(np.isfinite(np.array(result["A"], dtype=float))))
        self.assertTrue(np.all(np.isfinite(np.array(result["B"], dtype=float))))
        self.assertTrue(np.all(np.isfinite(np.array(result["P"], dtype=float))))

    def test_integrator_advances_one_fixed_time_step(self):
        # dt is the one-step integration horizon, not part of the RHS itself.
        step = make_integrator(self.model, dt=0.2)
        state_vec = self._state_vector()
        control_vec = self._control_vector(bank_rad=math.radians(10.0))
        params = np.concatenate([control_vec, self.aero_params])

        result = step(x0=state_vec, p=params)
        next_state = np.array(result["xf"], dtype=float).reshape(-1)

        self.assertEqual(next_state.shape, (7,))
        self.assertGreater(next_state[0], state_vec[0])
        self.assertAlmostEqual(next_state[6], state_vec[6])


if __name__ == "__main__":
    unittest.main()
