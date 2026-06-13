import math
from pathlib import Path
import sys
import unittest


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from coordinates_convertor import GeodeticCoordinate  # noqa: E402
from geodetic_simulator import GeodeticSimulator, GeodeticState  # noqa: E402
from simulator import Control  # noqa: E402


class TestGeodeticSimulator(unittest.TestCase):
    def test_enu_velocity_components_follow_simulator_angle_convention(self):
        # This protects the simulator convention: psi=0 is East, psi=pi/2 is North.
        east_velocity = GeodeticSimulator.get_enu_velocity_components(
            V=120.0,
            gamma=0.0,
            psi=0.0,
        )
        north_velocity = GeodeticSimulator.get_enu_velocity_components(
            V=120.0,
            gamma=0.0,
            psi=math.pi / 2.0,
        )

        self.assertAlmostEqual(east_velocity[0], 120.0)
        self.assertAlmostEqual(east_velocity[1], 0.0, places=12)
        self.assertAlmostEqual(east_velocity[2], 0.0)
        self.assertAlmostEqual(north_velocity[0], 0.0, places=12)
        self.assertAlmostEqual(north_velocity[1], 120.0)
        self.assertAlmostEqual(north_velocity[2], 0.0)

    def test_enu_velocity_to_ecef_uses_enu_axes_expressed_in_ecef(self):
        # This protects the basis expansion: local East at lat=0/lon=0 is ECEF +Y.
        ecef_velocity = GeodeticSimulator.enu_velocity_to_ecef_velocity(
            enu_velocity=(10.0, 0.0, 0.0),
            geo_S=GeodeticCoordinate(0.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(ecef_velocity[0], 0.0)
        self.assertAlmostEqual(ecef_velocity[1], 10.0)
        self.assertAlmostEqual(ecef_velocity[2], 0.0)

    def test_ecef_velocity_to_enu_round_trip_keeps_same_local_components(self):
        # This protects the intended pipeline: old ENU -> ECEF -> ENU at the same frame.
        geo = GeodeticCoordinate(51.1139, -114.0203, 1084.0)
        original_enu = (80.0, 30.0, -5.0)

        ecef_velocity = GeodeticSimulator.enu_velocity_to_ecef_velocity(
            original_enu,
            geo,
        )
        restored_enu = GeodeticSimulator.ecef_velocity_to_enu_velocity(
            ecef_velocity,
            geo,
        )

        self.assertAlmostEqual(restored_enu[0], original_enu[0])
        self.assertAlmostEqual(restored_enu[1], original_enu[1])
        self.assertAlmostEqual(restored_enu[2], original_enu[2])

    def test_step_integrates_local_east_motion_into_longitude(self):
        # This protects the original step core: psi=0 should move east in geodetic output.
        simulator = GeodeticSimulator()
        state = GeodeticState(
            latitude=0.0,
            longitude=0.0,
            altitude=1000.0,
            V=120.0,
            psi=0.0,
            gamma=0.0,
            m=10000.0,
        )

        next_state = simulator.step(state, Control(12000.0, 0.0, 0.0), 0.2)

        self.assertGreater(next_state.longitude, state.longitude)
        self.assertAlmostEqual(next_state.latitude, state.latitude, places=6)
        self.assertGreater(next_state.V, 0.0)

    def test_step_stops_when_altitude_reaches_ground(self):
        simulator = GeodeticSimulator()
        state = GeodeticState(
            latitude=0.0,
            longitude=0.0,
            altitude=5.0,
            V=120.0,
            psi=0.0,
            gamma=math.radians(-20.0),
            m=10000.0,
        )

        with self.assertRaisesRegex(ValueError, "altitude below 0"):
            simulator.step(state, Control(0.0, 0.0, math.radians(-10.0)), 1.0)


if __name__ == "__main__":
    unittest.main()
