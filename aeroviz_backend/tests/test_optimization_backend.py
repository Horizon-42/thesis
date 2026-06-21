import unittest

import numpy as np

from aeroviz_backend import optimization_backend


class TestOptimizationBackend(unittest.TestCase):
    def test_optimize_formats_optimizer_result_for_frontend(self):
        calls = []

        class FakeTranscriptionOptimizor:
            def __init__(
                self,
                geodetic_simulator,
                n_segments,
                dt,
                max_iterations,
                arrival_time_s,
            ):
                calls.append({
                    "aircraft": geodetic_simulator.simulator.aircraft.code,
                    "n_segments": n_segments,
                    "dt": dt,
                    "arrival_time_s": arrival_time_s,
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
                "optimizer": "transcription",
                "nSegments": 1,
                "arrivalTimeS": 84.0,
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
            {
                "aircraft": "A320",
                "n_segments": 1,
                "dt": 0.25,
                "arrival_time_s": 84.0,
                "max_iterations": 25,
            },
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

    def test_optimize_defaults_to_casadi_optimizer_with_load_factor_controls(self):
        calls = []

        class FakeCasadiOptimizer:
            def __init__(self, n_segments, dt, max_duration, aircraft):
                calls.append({
                    "aircraft": aircraft.code,
                    "n_segments": n_segments,
                    "dt": dt,
                    "max_duration": max_duration,
                })

            def optimize_time_to_target(self, initial_state, target_state, max_duration):
                calls.append({
                    "initial": initial_state,
                    "target": target_state,
                    "max_duration": max_duration,
                })
                return (
                    51.0,
                    np.array([[15000.0, 0.1, 1.2], [14000.0, 0.0, 1.0]]),
                    np.array([
                        [51.0, -114.0, 1000.0, 130.0, 0.3, -0.05],
                        [51.2, -114.1, 900.0, 125.0, 0.31, -0.04],
                    ]),
                )

        original_optimizer = optimization_backend.CasadiOptimizer
        optimization_backend.CasadiOptimizer = FakeCasadiOptimizer
        try:
            result = optimization_backend.OptimizationBackend().optimize({
                "nSegments": 2,
                "arrivalTimeS": 84.0,
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
            optimization_backend.CasadiOptimizer = original_optimizer

        self.assertEqual(
            calls[0],
            {
                "aircraft": "A320",
                "n_segments": 2,
                "dt": 0.25,
                "max_duration": 84.0,
            },
        )
        self.assertAlmostEqual(calls[1]["initial"].longitude, -114.0203)
        self.assertAlmostEqual(calls[1]["target"].latitude, 51.2)
        self.assertEqual(calls[1]["max_duration"], 84.0)
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["optimizer"], "casadiIpopt")
        self.assertEqual(result["finalTimeS"], 51.0)
        self.assertEqual(result["nSegments"], 2)
        self.assertEqual(result["controls"][0]["thrustN"], 15000.0)
        self.assertAlmostEqual(result["controls"][0]["bankDeg"], np.degrees(0.1))
        self.assertEqual(result["controls"][0]["loadFactor"], 1.2)
        self.assertNotIn("attackDeg", result["controls"][0])
        self.assertAlmostEqual(result["states"][1]["lat"], 51.2)
        self.assertAlmostEqual(result["states"][1]["headingDeg"], np.degrees(0.31))

    def test_optimize_reuses_casadi_optimizer_for_same_solver_key(self):
        constructions = []
        solves = []

        class FakeCasadiOptimizer:
            def __init__(self, n_segments, dt, max_duration, aircraft):
                self.instance_id = len(constructions) + 1
                constructions.append({
                    "aircraft": aircraft.code,
                    "n_segments": n_segments,
                    "dt": dt,
                    "max_duration": max_duration,
                })

            def optimize_time_to_target(self, initial_state, target_state, max_duration):
                solves.append({
                    "instance_id": self.instance_id,
                    "initial_lon": initial_state.longitude,
                    "target_lon": target_state.longitude,
                    "max_duration": max_duration,
                })
                return (
                    51.0,
                    np.array([[15000.0, 0.1, 1.2], [14000.0, 0.0, 1.0]]),
                    np.array([
                        [51.0, -114.0, 1000.0, 130.0, 0.3, -0.05],
                        [51.2, -114.1, 900.0, 125.0, 0.31, -0.04],
                    ]),
                )

        def make_payload(initial_lon, target_lon, dt=0.25):
            return {
                "nSegments": 2,
                "arrivalTimeS": 84.0,
                "dtS": dt,
                "maxIterations": 25,
                "initialState": {
                    "lon": initial_lon,
                    "lat": 51.1139,
                    "altM": 1084.0,
                    "speedMps": 135.0,
                    "headingDeg": 12.0,
                    "flightPathDeg": -3.0,
                    "aircraftType": "A320",
                },
                "targetState": {
                    "lon": target_lon,
                    "lat": 51.2,
                    "altM": 900.0,
                    "speedMps": 125.0,
                    "headingDeg": 18.0,
                    "flightPathDeg": -2.0,
                },
            }

        original_optimizer = optimization_backend.CasadiOptimizer
        optimization_backend.CasadiOptimizer = FakeCasadiOptimizer
        try:
            backend = optimization_backend.OptimizationBackend()
            backend.optimize(make_payload(-114.0203, -114.1))
            backend.optimize(make_payload(-114.0300, -114.2))
            backend.optimize(make_payload(-114.0300, -114.2, dt=0.2))
        finally:
            optimization_backend.CasadiOptimizer = original_optimizer

        self.assertEqual(
            constructions,
            [
                {
                    "aircraft": "A320",
                    "n_segments": 2,
                    "dt": 0.25,
                    "max_duration": 84.0,
                },
                {
                    "aircraft": "A320",
                    "n_segments": 2,
                    "dt": 0.2,
                    "max_duration": 84.0,
                },
            ],
        )
        self.assertEqual([solve["instance_id"] for solve in solves], [1, 1, 2])
        self.assertAlmostEqual(solves[1]["initial_lon"], -114.0300)
        self.assertAlmostEqual(solves[1]["target_lon"], -114.2)
        self.assertEqual([solve["max_duration"] for solve in solves], [84.0, 84.0, 84.0])

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
                "arrivalTimeS": 84.0,
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

    def test_optimize_can_select_least_squares_transcription_optimizer(self):
        calls = []

        class FakeLeastSquaresTranscriptionOptimizor:
            def __init__(
                self,
                geodetic_simulator,
                n_segments,
                dt,
                arrival_time_s,
                max_iterations,
            ):
                calls.append({
                    "aircraft": geodetic_simulator.simulator.aircraft.code,
                    "n_segments": n_segments,
                    "dt": dt,
                    "arrival_time_s": arrival_time_s,
                    "max_iterations": max_iterations,
                })

            def optimize_trajectory(self, initial_state, target_state):
                calls.append({
                    "initial": initial_state,
                    "target": target_state,
                })
                return (
                    84.0,
                    np.array([[15000.0, 0.1, 0.2]]),
                    np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]),
                )

        original_optimizer = optimization_backend.LeastSquaresTranscriptionOptimizor
        optimization_backend.LeastSquaresTranscriptionOptimizor = (
            FakeLeastSquaresTranscriptionOptimizor
        )
        try:
            result = optimization_backend.OptimizationBackend().optimize({
                "optimizer": "leastSquaresTranscription",
                "nSegments": 1,
                "arrivalTimeS": 84.0,
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
            optimization_backend.LeastSquaresTranscriptionOptimizor = original_optimizer

        self.assertEqual(
            calls[0],
            {
                "aircraft": "A320",
                "n_segments": 1,
                "dt": 0.25,
                "arrival_time_s": 84.0,
                "max_iterations": 25,
            },
        )
        self.assertEqual(result["optimizer"], "leastSquaresTranscription")
        self.assertEqual(result["finalTimeS"], 84.0)
        self.assertEqual(result["nSegments"], 1)

    def test_optimize_can_select_warm_start_transcription_optimizer(self):
        calls = []

        class FakeWarmStartTranscriptionOptimizor:
            def __init__(
                self,
                geodetic_simulator,
                n_segments,
                dt,
                arrival_time_s,
                max_iterations,
            ):
                calls.append({
                    "aircraft": geodetic_simulator.simulator.aircraft.code,
                    "n_segments": n_segments,
                    "dt": dt,
                    "arrival_time_s": arrival_time_s,
                    "max_iterations": max_iterations,
                })

            def optimize_trajectory(self, initial_state, target_state):
                calls.append({
                    "initial": initial_state,
                    "target": target_state,
                })
                return (
                    79.0,
                    np.array([[15000.0, 0.1, 0.2]]),
                    np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]),
                )

        original_optimizer = optimization_backend.WarmStartTranscriptionOptimizor
        optimization_backend.WarmStartTranscriptionOptimizor = (
            FakeWarmStartTranscriptionOptimizor
        )
        try:
            result = optimization_backend.OptimizationBackend().optimize({
                "optimizer": "warmStartTranscription",
                "nSegments": 1,
                "arrivalTimeS": 84.0,
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
            optimization_backend.WarmStartTranscriptionOptimizor = original_optimizer

        self.assertEqual(
            calls[0],
            {
                "aircraft": "A320",
                "n_segments": 1,
                "dt": 0.25,
                "arrival_time_s": 84.0,
                "max_iterations": 25,
            },
        )
        self.assertEqual(result["optimizer"], "warmStartTranscription")
        self.assertEqual(result["finalTimeS"], 79.0)
        self.assertEqual(result["nSegments"], 1)

    def test_optimize_can_select_variable_time_warm_start_optimizer(self):
        calls = []

        class FakeVariableTimeWarmStartTranscriptionOptimizor:
            def __init__(
                self,
                geodetic_simulator,
                n_segments,
                dt,
                arrival_time_s,
                max_iterations,
            ):
                calls.append({
                    "aircraft": geodetic_simulator.simulator.aircraft.code,
                    "n_segments": n_segments,
                    "dt": dt,
                    "arrival_time_s": arrival_time_s,
                    "max_iterations": max_iterations,
                })

            def optimize_trajectory(self, initial_state, target_state):
                calls.append({
                    "initial": initial_state,
                    "target": target_state,
                })
                return (
                    91.0,
                    np.array([[15000.0, 0.1, 0.2]]),
                    np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]),
                )

        original_optimizer = (
            optimization_backend.VariableTimeWarmStartTranscriptionOptimizor
        )
        optimization_backend.VariableTimeWarmStartTranscriptionOptimizor = (
            FakeVariableTimeWarmStartTranscriptionOptimizor
        )
        try:
            result = optimization_backend.OptimizationBackend().optimize({
                "optimizer": "variableTimeWarmStartTranscription",
                "nSegments": 1,
                "arrivalTimeS": 84.0,
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
            optimization_backend.VariableTimeWarmStartTranscriptionOptimizor = (
                original_optimizer
            )

        self.assertEqual(
            calls[0],
            {
                "aircraft": "A320",
                "n_segments": 1,
                "dt": 0.25,
                "arrival_time_s": 84.0,
                "max_iterations": 25,
            },
        )
        self.assertEqual(result["optimizer"], "variableTimeWarmStartTranscription")
        self.assertEqual(result["finalTimeS"], 91.0)
        self.assertEqual(result["nSegments"], 1)

    def test_optimize_rejects_unknown_optimizer(self):
        with self.assertRaisesRegex(ValueError, "optimizer must be one of"):
            optimization_backend.OptimizationBackend().optimize({
                "optimizer": "notReal",
                "initialState": {"aircraftType": "A320"},
                "targetState": {},
            })

    def test_optimize_rejects_invalid_arrival_time(self):
        with self.assertRaisesRegex(ValueError, "arrivalTimeS must be between"):
            optimization_backend.OptimizationBackend().optimize({
                "arrivalTimeS": 0,
                "initialState": {"aircraftType": "A320"},
                "targetState": {},
            })

    def test_optimize_requires_arrival_time(self):
        with self.assertRaisesRegex(ValueError, "arrivalTimeS must be a number"):
            optimization_backend.OptimizationBackend().optimize({
                "initialState": {"aircraftType": "A320"},
                "targetState": {},
            })


if __name__ == "__main__":
    unittest.main()
