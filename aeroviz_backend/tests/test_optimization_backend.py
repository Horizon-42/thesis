import io
import unittest
from contextlib import redirect_stderr

import numpy as np

from aeroviz_backend import optimization_backend
from aeroviz_backend.simulation_backend import GeodeticSimulator
from aerodynamic_model.aircraft_sets import A320


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

    def test_optimize_echoes_procedure_constraint_summary(self):
        # Plumbing check: the canonical procedure constraint the frontend ships
        # is parsed by the backend and a validation summary is echoed back. The
        # NLP does not yet enforce the waypoint windows.
        class FakeTranscriptionOptimizor:
            def __init__(self, *args, **kwargs):
                pass

            def optimize_trajectory(self, initial_state, target_state):
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
                    "lon": -114.0203, "lat": 51.1139, "altM": 1084.0,
                    "speedMps": 135.0, "headingDeg": 12.0, "flightPathDeg": -3.0,
                    "aircraftType": "A320",
                },
                "targetState": {
                    "lon": -114.1, "lat": 51.2, "altM": 900.0,
                    "speedMps": 125.0, "headingDeg": 18.0, "flightPathDeg": -2.0,
                },
                "procedureConstraint": {
                    "procedureUid": "KRDU-R05LY-RW05L",
                    "branchId": "branch:R",
                    "waypoints": [
                        {"fixId": "fix:SCHOO", "ident": "SCHOO", "role": "IF",
                         "legType": "IF", "lonDeg": -78.9, "latDeg": 35.77,
                         "altitudeRefFt": 3000, "distanceFromStartM": 0},
                        {"fixId": "fix:WEPAS", "ident": "WEPAS", "role": "FAF",
                         "legType": "TF", "lonDeg": -78.88, "latDeg": 35.80,
                         "altitudeRefFt": 2200, "distanceFromStartM": 5500},
                        {"fixId": "fix:RW05L", "ident": "RW05L", "role": "MAPt",
                         "legType": "TF", "lonDeg": -78.80, "latDeg": 35.87,
                         "altitudeRefFt": 424, "distanceFromStartM": 15900},
                    ],
                },
            })
        finally:
            optimization_backend.TranscriptionOptimizor = original_optimizer

        self.assertEqual(
            result["procedureConstraintSummary"],
            {
                "waypointCount": 3,
                "monotonicDescent": True,
                "firstFixIdent": "SCHOO",
                "lastFixIdent": "RW05L",
            },
        )

    def test_optimize_omits_summary_without_procedure_constraint(self):
        class FakeTranscriptionOptimizor:
            def __init__(self, *args, **kwargs):
                pass

            def optimize_trajectory(self, initial_state, target_state):
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
                    "lon": -114.0203, "lat": 51.1139, "altM": 1084.0,
                    "speedMps": 135.0, "headingDeg": 12.0, "flightPathDeg": -3.0,
                    "aircraftType": "A320",
                },
                "targetState": {
                    "lon": -114.1, "lat": 51.2, "altM": 900.0,
                    "speedMps": 125.0, "headingDeg": 18.0, "flightPathDeg": -2.0,
                },
            })
        finally:
            optimization_backend.TranscriptionOptimizor = original_optimizer

        self.assertNotIn("procedureConstraintSummary", result)

    def test_optimize_defaults_to_direct_collocation_optimizer_with_load_factor_controls(self):
        # The default optimiser is the fixed-ENU direct-collocation
        # solver.  It shares the LoadFactorControl I/O shape with the
        # casadiIpopt multiple-shooting optimiser, so the format of
        # ``result["controls"][k]`` is unchanged.
        calls = []

        class FakeCasadiDirectCollocationOptimizer:
            def __init__(self, n_segments, dt, max_duration, aircraft,
                         collocation_scheme="hermiteSimpson"):
                calls.append({
                    "aircraft": aircraft.code,
                    "n_segments": n_segments,
                    "dt": dt,
                    "max_duration": max_duration,
                    "collocation_scheme": collocation_scheme,
                })

            def optimize_free_time(self, initial_state, target_state, max_duration):
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

        original_optimizer = optimization_backend.CasadiDirectCollocationOptimizer
        optimization_backend.CasadiDirectCollocationOptimizer = (
            FakeCasadiDirectCollocationOptimizer
        )
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
            optimization_backend.CasadiDirectCollocationOptimizer = original_optimizer

        self.assertEqual(
            calls[0],
            {
                "aircraft": "A320",
                "n_segments": 2,
                "dt": 0.25,
                "max_duration": 84.0,
                "collocation_scheme": "hermiteSimpson",
            },
        )
        self.assertAlmostEqual(calls[1]["initial"].longitude, -114.0203)
        self.assertAlmostEqual(calls[1]["target"].latitude, 51.2)
        self.assertEqual(calls[1]["max_duration"], 84.0)
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["optimizer"], "casadiDirectCollocation")
        self.assertEqual(result["finalTimeS"], 51.0)
        self.assertEqual(result["nSegments"], 2)
        self.assertEqual(result["controls"][0]["thrustN"], 15000.0)
        self.assertAlmostEqual(result["controls"][0]["bankDeg"], np.degrees(0.1))
        self.assertEqual(result["controls"][0]["loadFactor"], 1.2)
        self.assertNotIn("attackDeg", result["controls"][0])
        self.assertAlmostEqual(result["states"][1]["lat"], 51.2)
        self.assertAlmostEqual(result["states"][1]["headingDeg"], np.degrees(0.31))

    def test_direct_collocation_variant_names_select_their_defect_scheme(self):
        # Each scheme-suffixed optimizer name must reach
        # CasadiDirectCollocationOptimizer with the matching collocation_scheme.
        seen = {}

        class RecordingOptimizer:
            def __init__(self, n_segments, dt, max_duration, aircraft,
                         collocation_scheme="hermiteSimpson"):
                self.collocation_scheme = collocation_scheme

        original = optimization_backend.CasadiDirectCollocationOptimizer
        optimization_backend.CasadiDirectCollocationOptimizer = RecordingOptimizer
        try:
            for name, scheme in optimization_backend.DIRECT_COLLOCATION_SCHEMES.items():
                opt = optimization_backend.make_optimizer(
                    name, GeodeticSimulator(A320), 10, 0.2, 300, arrival_time_s=120.0,
                )
                seen[name] = opt.collocation_scheme
        finally:
            optimization_backend.CasadiDirectCollocationOptimizer = original

        self.assertEqual(seen, dict(optimization_backend.DIRECT_COLLOCATION_SCHEMES))
        self.assertEqual(seen["casadiDirectCollocation"], "hermiteSimpson")
        self.assertEqual(seen["casadiDirectCollocationRk4"], "rk4")
        # The normalized geodetic names select the metric-position *Normalized
        # schemes (same geodetic RHS, well-conditioned decision state).
        self.assertEqual(seen["casadiDirectCollocationNormalized"], "hermiteSimpsonNormalized")
        self.assertEqual(seen["casadiDirectCollocationNormalizedTrapezoidal"], "trapezoidalNormalized")
        self.assertEqual(seen["casadiDirectCollocationNormalizedRk4"], "rk4Normalized")

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
                # The default optimiser changed to direct collocation,
                # but the multiple-shooting casadiIpopt optimiser is
                # still cache-able; this test pins the optimiser to
                # exercise that path.
                "optimizer": "casadiIpopt",
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

    def test_optimize_reuses_direct_collocation_optimizer_for_same_solver_key(self):
        # The two CasADi optimisers share the cache slot but are keyed
        # separately by optimiser name, so switching between them must
        # rebuild the solver.  This test only exercises the
        # direct-collocation cache; the casadiIpopt cache is covered
        # above.
        constructions = []
        solves = []

        class FakeCasadiDirectCollocationOptimizer:
            def __init__(self, n_segments, dt, max_duration, aircraft,
                         collocation_scheme="hermiteSimpson"):
                self.instance_id = len(constructions) + 1
                constructions.append({
                    "aircraft": aircraft.code,
                    "n_segments": n_segments,
                    "dt": dt,
                    "max_duration": max_duration,
                    "collocation_scheme": collocation_scheme,
                })

            def optimize_free_time(self, initial_state, target_state, max_duration):
                solves.append({"instance_id": self.instance_id})
                return (
                    51.0,
                    np.array([[15000.0, 0.0, 1.0]]),
                    np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]),
                )

        def make_payload(dt=0.25):
            return {
                "nSegments": 2,
                "arrivalTimeS": 84.0,
                "dtS": dt,
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
            }

        original_optimizer = optimization_backend.CasadiDirectCollocationOptimizer
        optimization_backend.CasadiDirectCollocationOptimizer = (
            FakeCasadiDirectCollocationOptimizer
        )
        try:
            backend = optimization_backend.OptimizationBackend()
            backend.optimize(make_payload())
            backend.optimize(make_payload())  # same key -> reuse
            backend.optimize(make_payload(dt=0.2))  # new key -> rebuild
        finally:
            optimization_backend.CasadiDirectCollocationOptimizer = original_optimizer

        self.assertEqual([c["dt"] for c in constructions], [0.25, 0.2])
        self.assertEqual([s["instance_id"] for s in solves], [1, 1, 2])

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

    def test_optimize_logs_whole_flow_timing_to_server_log(self):
        # The backend times the ENTIRE flow (build + solve + playback) and
        # writes a breakdown to the server log (stderr).  When the optimiser
        # exposes a cold-start / free-time split, both appear in the line.
        class FakeCasadiDirectCollocationOptimizer:
            def __init__(self, n_segments, dt, max_duration, aircraft,
                         collocation_scheme="hermiteSimpson"):
                self.last_solve_timings = {
                    "coldStartS": 0.4,
                    "freeTimeSolveS": 0.6,
                    "solveTotalS": 1.0,
                }

            def optimize_free_time(self, initial_state, target_state, max_duration):
                return (
                    51.0,
                    np.array([[15000.0, 0.0, 1.0]]),
                    np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]),
                )

        original = optimization_backend.CasadiDirectCollocationOptimizer
        optimization_backend.CasadiDirectCollocationOptimizer = (
            FakeCasadiDirectCollocationOptimizer
        )
        buffer = io.StringIO()
        try:
            with redirect_stderr(buffer):
                optimization_backend.OptimizationBackend().optimize({
                    "optimizer": "casadiDirectCollocationNormalized",
                    "nSegments": 2,
                    "arrivalTimeS": 84.0,
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
            optimization_backend.CasadiDirectCollocationOptimizer = original

        log = buffer.getvalue()
        self.assertIn("optimization timing", log)
        self.assertIn("optimizer=casadiDirectCollocationNormalized", log)
        self.assertIn("build=", log)
        self.assertIn("coldStart=0.400s", log)
        self.assertIn("freeTime=0.600s", log)
        self.assertIn("playback=", log)
        self.assertIn("total=", log)

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
