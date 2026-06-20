import math
import unittest

import casadi as ca
import numpy as np


from aerodynamic_model.aircraft_sets import A320  # noqa: E402
from aerodynamic_model.casadi_exprs import (  # noqa: E402
    aerodynamic_coefficients_expr,
    isa_density_expr,
)
from aerodynamic_model.casadi_simulator import (  # noqa: E402
    CasadiSimulator,
    GeoControl,
    GeoState,
    make_dynamics_model,
    make_integrator,
)


class TestCasadiSimulator(unittest.TestCase):
    def setUp(self):
        self.model = make_dynamics_model()
        altitude = ca.SX.sym("altitude")
        self.density_func = ca.Function(
            "density_from_expr",
            [altitude],
            [isa_density_expr(altitude)],
        )
        n_cmd = ca.SX.sym("n_cmd")
        speed = ca.SX.sym("speed")
        mass = ca.SX.sym("mass")
        aero_params = ca.SX.sym("aero_params", 6)
        rho = isa_density_expr(altitude)
        coeffs = aerodynamic_coefficients_expr(
            n_cmd,
            speed,
            mass,
            rho,
            aero_params,
        )
        self.coefficients_func = ca.Function(
            "aero_coefficients_from_expr",
            [n_cmd, speed, altitude, mass, aero_params],
            [
                coeffs["Cl_req"],
                coeffs["Cl"],
                coeffs["Cd"],
                coeffs["r"],
                coeffs["stalled"],
            ],
            ["n_cmd", "speed", "altitude", "mass", "aero_params"],
            ["cl_req", "cl", "cd", "r", "stalled"],
        )
        aux = self.model["aux"]
        self.aux_func = ca.Function(
            "dynamics_aux_from_expr",
            [self.model["x"], self.model["u"], self.model["aero_params"]],
            [aux["rho"], aux["Cl"], aux["Cd"], aux["stalled"], aux["n"], aux["D"]],
            ["x", "u", "aero_params"],
            ["rho", "cl", "cd", "stalled", "n", "drag"],
        )
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

    def _density(self, altitude: float) -> float:
        return float(self.density_func(altitude))

    def _coefficients(
        self,
        *,
        load_factor: float,
        speed: float,
        altitude: float,
        mass: float = A320.mass_kg,
    ) -> dict[str, float]:
        result = self.coefficients_func(
            n_cmd=load_factor,
            speed=speed,
            altitude=altitude,
            mass=mass,
            aero_params=self.aero_params,
        )
        return {name: float(value) for name, value in result.items()}

    def _aux(self, state_vec: np.ndarray, control_vec: np.ndarray) -> dict[str, float]:
        result = self.aux_func(
            x=state_vec,
            u=control_vec,
            aero_params=self.aero_params,
        )
        return {name: float(value) for name, value in result.items()}

    def test_model_exposes_symbolic_contract(self):
        # The outer model dict is our local interface for tests and callers.
        self.assertEqual(
            set(self.model.keys()),
            {"x", "u", "aero_params", "xdot", "dae", "rhs_func", "aux"},
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

        aux = self.model["aux"]
        self.assertEqual(set(aux.keys()), {"rho", "Cl", "Cd", "stalled", "n", "D"})
        for expression in aux.values():
            self.assertEqual(expression.shape, (1, 1))

    def test_aerodynamic_coefficients_expr_matches_unstalled_drag_polar(self):
        speed = 150.0
        altitude = 1000.0
        load_factor = 1.0

        coeffs = self._coefficients(
            load_factor=load_factor,
            speed=speed,
            altitude=altitude,
        )

        rho = self._density(altitude)
        wing_area_m2, cl_max, cd0, induced_drag_factor = self.aero_params[:4]
        expected_cl_req = load_factor * A320.mass_kg * 9.81 / (
            0.5 * rho * wing_area_m2 * speed**2
        )
        expected_cd = cd0 + induced_drag_factor * expected_cl_req**2

        self.assertLess(expected_cl_req, cl_max)
        self.assertAlmostEqual(coeffs["cl_req"], expected_cl_req)
        self.assertAlmostEqual(coeffs["cl"], expected_cl_req)
        self.assertAlmostEqual(coeffs["r"], expected_cl_req / cl_max)
        self.assertAlmostEqual(coeffs["cd"], expected_cd)
        self.assertEqual(coeffs["stalled"], 0.0)

    def test_aerodynamic_coefficients_expr_caps_lift_when_stalled(self):
        speed = 80.0
        altitude = 1000.0
        load_factor = 5.0

        coeffs = self._coefficients(
            load_factor=load_factor,
            speed=speed,
            altitude=altitude,
        )

        _, cl_max, cd0, induced_drag_factor, _, stall_drag_factor = self.aero_params
        expected_cd = cd0 + induced_drag_factor * cl_max**2 + stall_drag_factor

        self.assertGreater(coeffs["cl_req"], cl_max)
        self.assertAlmostEqual(coeffs["cl"], cl_max)
        self.assertAlmostEqual(coeffs["cd"], expected_cd)
        self.assertEqual(coeffs["stalled"], 1.0)

    def test_aux_outputs_are_evaluable_from_dynamics_symbols(self):
        state_vec = self._state_vector(speed=150.0, altitude=1000.0)
        control_vec = self._control_vector(load_factor=1.0)

        aux = self._aux(state_vec, control_vec)
        coeffs = self._coefficients(
            load_factor=control_vec[2],
            speed=state_vec[3],
            altitude=state_vec[2],
            mass=state_vec[6],
        )

        for value in aux.values():
            self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(aux["rho"], self._density(state_vec[2]))
        self.assertAlmostEqual(aux["cl"], coeffs["cl"])
        self.assertAlmostEqual(aux["cd"], coeffs["cd"])
        self.assertAlmostEqual(aux["stalled"], coeffs["stalled"])
        self.assertAlmostEqual(aux["n"], control_vec[2])
        self.assertGreater(aux["drag"], 0.0)

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
        rho = self._density(altitude)
        wing_area_m2, cl_max = self.aero_params[:2]
        actual_load_factor = 0.5 * rho * speed**2 * cl_max * wing_area_m2 / (
            A320.mass_kg * 9.81
        )
        expected_gamma_rate = 9.81 * (actual_load_factor - 1.0) / speed

        self.assertLess(actual_load_factor, control_vec[2])
        self.assertAlmostEqual(derivatives[5], expected_gamma_rate)

    def test_casadi_simulator_step_returns_geo_state_from_named_output(self):
        simulator = CasadiSimulator(A320, dt=0.2)
        state = GeoState(
            latitude=51.1139,
            longitude=-114.0203,
            altitude=1000.0,
            V=150.0,
            psi=0.0,
            gamma=0.0,
            m=A320.mass_kg,
        )
        control = GeoControl(
            thrust=A320.approach_thrust_guess_n,
            bank_rad=0.0,
            load_factor=1.0,
        )

        next_state = simulator.step(state, control, dt=0.2)

        self.assertIsInstance(next_state, GeoState)
        self.assertTrue(
            np.all(
                np.isfinite(
                    [
                        next_state.latitude,
                        next_state.longitude,
                        next_state.altitude,
                        next_state.V,
                        next_state.psi,
                        next_state.gamma,
                        next_state.m,
                    ]
                )
            )
        )
        self.assertAlmostEqual(next_state.latitude, state.latitude, places=5)
        self.assertGreater(next_state.longitude, state.longitude)
        self.assertAlmostEqual(next_state.m, state.m)

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
