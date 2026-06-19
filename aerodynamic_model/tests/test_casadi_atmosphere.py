from pathlib import Path
import sys
import unittest

import casadi as ca


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casadi_atmosphere import make_atmosphere_model  # noqa: E402


class TestCasadiAtmosphere(unittest.TestCase):
    def setUp(self):
        self.model = make_atmosphere_model()

    def test_model_exposes_single_altitude_input(self):
        self.assertEqual(self.model.name(), "atmosphere_model")
        self.assertEqual(self.model.n_in(), 1)
        self.assertEqual(self.model.n_out(), 1)

    def test_sea_level_density_matches_isa(self):
        self.assertAlmostEqual(float(self.model(0.0)), 1.225, places=12)

    def test_troposphere_density_matches_isa_reference_points(self):
        # ISA troposphere density values for h <= 11 km, kg/m^3.
        reference_density_by_altitude_m = {
            1000.0: 1.111642,
            5000.0: 0.736116,
            10000.0: 0.412706,
            11000.0: 0.363918,
        }

        for altitude_m, expected_density in reference_density_by_altitude_m.items():
            with self.subTest(altitude_m=altitude_m):
                actual_density = float(self.model(altitude_m))

                self.assertAlmostEqual(actual_density, expected_density, places=6)

    def test_density_decreases_with_altitude(self):
        densities = [
            float(self.model(altitude_m))
            for altitude_m in (0.0, 1000.0, 5000.0, 10000.0, 11000.0)
        ]

        for lower_density, higher_density in zip(densities, densities[1:]):
            self.assertGreater(lower_density, higher_density)

    def test_symbolic_density_gradient_matches_isa_reference(self):
        h = ca.SX.sym("h")
        rho = self.model(h)
        gradient_model = ca.Function("density_gradient", [h], [ca.jacobian(rho, h)])

        # d/dh rho0 * ((T0 - Lh) / T0)^4.25588 at h = 5000 m.
        self.assertAlmostEqual(
            float(gradient_model(5000.0)),
            -0.000079653,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
