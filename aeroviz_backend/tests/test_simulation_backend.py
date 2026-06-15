import math
import unittest

from aeroviz_backend.simulation_backend import SimulationBackend, aircraft_catalog
from aircraft_sets import B77W


class TestSimulationBackend(unittest.TestCase):
    def test_reset_reads_frontend_payload_names(self):
        backend = SimulationBackend()

        snapshot = backend.reset({
            "state": {
                "lon": -114.0203,
                "lat": 51.1139,
                "altM": 1084.0,
                "speedMps": 135.0,
                "headingDeg": 12.0,
                "flightPathDeg": -3.0,
                "massKg": 12000.0,
                "aircraftType": "B77W",
            },
            "control": {
                "thrustN": 15000.0,
                "bankDeg": 5.0,
                "attackDeg": 4.0,
            },
        })

        self.assertTrue(snapshot["ok"])
        self.assertEqual(snapshot["elapsedS"], 0.0)
        self.assertAlmostEqual(snapshot["state"]["lon"], -114.0203)
        self.assertAlmostEqual(snapshot["state"]["lat"], 51.1139)
        self.assertAlmostEqual(snapshot["state"]["headingDeg"], 12.0)
        self.assertEqual(snapshot["state"]["aircraftType"], "B77W")
        self.assertAlmostEqual(snapshot["state"]["massKg"], B77W.mass_kg)
        self.assertEqual(backend.geodetic_simulator.simulator.aircraft, B77W)
        self.assertAlmostEqual(snapshot["control"]["bankDeg"], 5.0)
        self.assertAlmostEqual(snapshot["control"]["attackDeg"], 4.0)
        self.assertAlmostEqual(
            snapshot["aero"]["liftCoefficient"],
            backend.geodetic_simulator.simulator.CL0
            + backend.geodetic_simulator.simulator.CL_alpha * math.radians(4.0),
        )

    def test_reset_clamps_thrust_with_selected_aircraft_spec(self):
        backend = SimulationBackend()

        snapshot = backend.reset({
            "state": {"aircraftType": "C172"},
            "control": {"thrustN": 1000000.0},
        })

        self.assertAlmostEqual(snapshot["control"]["thrustN"], 3200.0)

    def test_reset_uses_selected_aircraft_default_control(self):
        backend = SimulationBackend()

        snapshot = backend.reset({"state": {"aircraftType": "C172"}})

        self.assertAlmostEqual(snapshot["control"]["thrustN"], 800.0)

    def test_step_advances_elapsed_time_and_position(self):
        backend = SimulationBackend()
        backend.reset({
            "state": {
                "lon": 0.0,
                "lat": 0.0,
                "altM": 1000.0,
                "speedMps": 120.0,
                "headingDeg": 0.0,
                "flightPathDeg": 0.0,
                "massKg": 78000.0,
                "aircraftType": "A320",
            }
        })

        snapshot = backend.step({"dtS": 0.2})

        self.assertAlmostEqual(snapshot["elapsedS"], 0.2)
        self.assertGreater(snapshot["state"]["lon"], 0.0)
        self.assertTrue(math.isfinite(snapshot["aero"]["liftCoefficient"]))

    def test_aircraft_catalog_exposes_performance_defaults(self):
        payload = aircraft_catalog()
        a320 = next(item for item in payload["aircraft"] if item["code"] == "A320")
        b77w = next(item for item in payload["aircraft"] if item["code"] == "B77W")

        self.assertEqual(a320["terminalSpeedKt"], 145.0)
        self.assertEqual(a320["maxThrustN"], 240000.0)
        self.assertEqual(a320["finalApproachMinNm"], 5.0)
        self.assertEqual(b77w["maxThrustN"], 1026000.0)


if __name__ == "__main__":
    unittest.main()
