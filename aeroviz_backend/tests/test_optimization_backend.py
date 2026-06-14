import unittest

import numpy as np

from aeroviz_backend import optimization_backend


class TestOptimizationBackend(unittest.TestCase):
    def test_optimize_formats_optimizer_result_for_frontend(self):
        calls = []

        class FakeTranscriptionOptimizor:
            def __init__(self, geodetic_simulator, n_segments, dt, max_iterations):
                calls.append({
                    "aircraft": geodetic_simulator.simulator.aircraft.code,
                    "n_segments": n_segments,
                    "dt": dt,
                    "max_iterations": max_iterations,
                })

            def optimize_trajectory(self, initial_state, target_state):
                calls.append({
                    "initial": initial_state,
                    "target": target_state,
                })
                return (
                    42.0,
                    np.array([[15000.0, 0.1, 0.2]]),
                    np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]),
                )

        original_optimizer = optimization_backend.TranscriptionOptimizor
        optimization_backend.TranscriptionOptimizor = FakeTranscriptionOptimizor
        try:
            result = optimization_backend.OptimizationBackend().optimize({
                "nSegments": 1,
                "dtS": 0.25,
                "maxIterations": 25,
                "initialState": {
                    "lon": -114.0203,
                    "lat": 51.1139,
                    "altM": 1084.0,
                    "speedMps": 135.0,
                    "headingDeg": 12.0,
                    "flightPathDeg": -3.0,
                    "aircraftType": "A320",
                },
                "targetState": {
                    "lon": -114.1,
                    "lat": 51.2,
                    "altM": 900.0,
                    "speedMps": 125.0,
                    "headingDeg": 18.0,
                    "flightPathDeg": -2.0,
                },
            })
        finally:
            optimization_backend.TranscriptionOptimizor = original_optimizer

        self.assertEqual(
            calls[0],
            {"aircraft": "A320", "n_segments": 1, "dt": 0.25, "max_iterations": 25},
        )
        self.assertAlmostEqual(calls[1]["initial"].longitude, -114.0203)
        self.assertAlmostEqual(calls[1]["target"].latitude, 51.2)
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["finalTimeS"], 42.0)
        self.assertEqual(result["nSegments"], 1)
        self.assertEqual(result["dtS"], 0.25)
        self.assertEqual(result["optimizer"], "transcription")
        self.assertEqual(result["controls"][0]["thrustN"], 15000.0)
        self.assertAlmostEqual(result["controls"][0]["bankDeg"], np.degrees(0.1))
        self.assertAlmostEqual(result["states"][0]["lat"], 51.0)
        self.assertAlmostEqual(result["states"][0]["headingDeg"], np.degrees(0.3))
        self.assertEqual(result["states"][0]["massKg"], 78000.0)
        self.assertEqual(result["states"][0]["aircraftType"], "A320")

    def test_optimize_can_select_single_shooting_optimizer(self):
        calls = []

        class FakeSingleShootingOptimizor:
            def __init__(
                self,
                geodetic_simulator,
                n_control_segments,
                dt,
                max_iterations,
            ):
                calls.append({
                    "aircraft": geodetic_simulator.simulator.aircraft.code,
                    "n_control_segments": n_control_segments,
                    "dt": dt,
                    "max_iterations": max_iterations,
                })

            def optimize_trajectory(self, initial_state, target_state):
                calls.append({
                    "initial": initial_state,
                    "target": target_state,
                })
                return (
                    37.0,
                    np.array([[14000.0, -0.1, 0.15], [13000.0, 0.0, 0.1]]),
                    None,
                )

        original_optimizer = optimization_backend.SingleShootingOptimizor
        optimization_backend.SingleShootingOptimizor = FakeSingleShootingOptimizor
        try:
            result = optimization_backend.OptimizationBackend().optimize({
                "optimizer": "singleShooting",
                "nSegments": 2,
                "dtS": 0.25,
                "maxIterations": 25,
                "initialState": {
                    "lon": -114.0203,
                    "lat": 51.1139,
                    "altM": 1084.0,
                    "speedMps": 135.0,
                    "headingDeg": 12.0,
                    "flightPathDeg": -3.0,
                    "aircraftType": "A320",
                },
                "targetState": {
                    "lon": -114.1,
                    "lat": 51.2,
                    "altM": 900.0,
                    "speedMps": 125.0,
                    "headingDeg": 18.0,
                    "flightPathDeg": -2.0,
                },
            })
        finally:
            optimization_backend.SingleShootingOptimizor = original_optimizer

        self.assertEqual(
            calls[0],
            {
                "aircraft": "A320",
                "n_control_segments": 2,
                "dt": 0.25,
                "max_iterations": 25,
            },
        )
        self.assertAlmostEqual(calls[1]["initial"].longitude, -114.0203)
        self.assertAlmostEqual(calls[1]["target"].latitude, 51.2)
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["optimizer"], "singleShooting")
        self.assertEqual(result["nSegments"], 2)
        self.assertEqual(result["controls"][0]["thrustN"], 14000.0)
        self.assertAlmostEqual(result["controls"][0]["bankDeg"], np.degrees(-0.1))
        self.assertEqual(result["states"], [])

    def test_optimize_rejects_unknown_optimizer(self):
        with self.assertRaisesRegex(ValueError, "optimizer must be one of"):
            optimization_backend.OptimizationBackend().optimize({
                "optimizer": "notReal",
                "initialState": {"aircraftType": "A320"},
                "targetState": {},
            })


if __name__ == "__main__":
    unittest.main()
