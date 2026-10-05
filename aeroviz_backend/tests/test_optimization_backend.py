import io
import unittest
from contextlib import redirect_stderr

import numpy as np

from aeroviz_backend import optimization_backend
from aircraft.aircraft_sets import A320

#: What every CollocationOptimizer solve records (the backend reads its build split).
SOLVE_TIMINGS = {"buildS": 0.01, "solverSetupS": 0.02, "coldStartS": 0.0, "freeTimeSolveS": 0.03, "solveTotalS": 0.06}


class TestOptimizationBackend(unittest.TestCase):
    def test_optimize_formats_optimizer_result_for_frontend(self):
        calls = []

        class FakeCollocationOptimizer:
            def __init__(self, aircraft, *, scheme, n_segments, max_duration, state_substeps, max_iterations):
                self.last_solve_timings = SOLVE_TIMINGS
                calls.append({
                    "aircraft": aircraft.code,
                    "scheme": scheme,
                    "n_segments": n_segments,
                    "max_duration": max_duration,
                    "max_iterations": max_iterations,
                })

            def optimize_free_time(self, initial_state, target_state, max_duration):
                calls.append({
                    "initial": initial_state,
                    "target": target_state,
                })
                return (
                    42.0,
                    np.array([[15000.0, 0.1, 0.2]]),
                    np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]),
                )

        original_optimizer = optimization_backend.CollocationOptimizer
        optimization_backend.CollocationOptimizer = FakeCollocationOptimizer
        try:
            result = optimization_backend.OptimizationBackend().optimize({
                "optimizer": "casadiDirectCollocation",
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
            optimization_backend.CollocationOptimizer = original_optimizer

        self.assertEqual(
            calls[0],
            {
                "aircraft": "A320",
                "scheme": "hermiteSimpson",
                "n_segments": 1,
                "max_duration": 84.0,
                "max_iterations": 25,
            },
        )
        self.assertAlmostEqual(calls[1]["initial"].longitude, -114.0203)
        self.assertAlmostEqual(calls[1]["target"].latitude, 51.2)
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["finalTimeS"], 42.0)
        self.assertEqual(result["nSegments"], 1)
        # the timing breakdown is surfaced in the response (was log-only)
        self.assertIn("timings", result)
        self.assertGreaterEqual(result["timings"]["totalS"], 0.0)
        self.assertEqual(
            set(result["timings"]), {"buildS", "solveS", "playbackS", "totalS"}
        )
        self.assertEqual(result["optimizer"], "casadiDirectCollocation")
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
        class FakeCollocationOptimizer:
            def __init__(self, *args, **kwargs):
                self.last_solve_timings = SOLVE_TIMINGS

            def optimize_free_time(self, initial_state, target_state, max_duration):
                return (
                    42.0,
                    np.array([[15000.0, 0.1, 0.2]]),
                    np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]),
                )

        original_optimizer = optimization_backend.CollocationOptimizer
        optimization_backend.CollocationOptimizer = FakeCollocationOptimizer
        try:
            result = optimization_backend.OptimizationBackend().optimize({
                "optimizer": "casadiDirectCollocation",
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
            optimization_backend.CollocationOptimizer = original_optimizer

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
        class FakeCollocationOptimizer:
            def __init__(self, *args, **kwargs):
                self.last_solve_timings = SOLVE_TIMINGS

            def optimize_free_time(self, initial_state, target_state, max_duration):
                return (
                    42.0,
                    np.array([[15000.0, 0.1, 0.2]]),
                    np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]),
                )

        original_optimizer = optimization_backend.CollocationOptimizer
        optimization_backend.CollocationOptimizer = FakeCollocationOptimizer
        try:
            result = optimization_backend.OptimizationBackend().optimize({
                "optimizer": "casadiDirectCollocation",
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
            optimization_backend.CollocationOptimizer = original_optimizer

        self.assertNotIn("procedureConstraintSummary", result)

    def test_optimize_defaults_to_direct_collocation_optimizer_with_load_factor_controls(self):
        # The default optimiser is the fixed-ENU direct-collocation
        # solver.  It shares the LoadFactorControl I/O shape with the
        # former casadiIpopt multiple-shooting optimiser, so the format of
        # ``result["controls"][k]`` is unchanged.
        calls = []

        class FakeCasadiDirectCollocationOptimizer:
            def __init__(self, aircraft, *, scheme="hermiteSimpson", segments=None,
                         n_segments=None, max_duration=None, **_kwargs):
                self.last_solve_timings = SOLVE_TIMINGS
                calls.append({
                    "aircraft": aircraft.code,
                    "n_segments": n_segments,
                    "max_duration": max_duration,
                    "scheme": scheme,
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

        original_optimizer = optimization_backend.CollocationOptimizer
        optimization_backend.CollocationOptimizer = (
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
            optimization_backend.CollocationOptimizer = original_optimizer

        self.assertEqual(
            calls[0],
            {
                "aircraft": "A320",
                "n_segments": 2,
                "max_duration": 84.0,
                "scheme": "hermiteSimpson",
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
        # CollocationOptimizer with the matching scheme.
        seen = {}

        class RecordingOptimizer:
            def __init__(self, aircraft, *, scheme="hermiteSimpson", segments=None,
                         n_segments=None, max_duration=None, **_kwargs):
                self.scheme = scheme

        original = optimization_backend.CollocationOptimizer
        optimization_backend.CollocationOptimizer = RecordingOptimizer
        try:
            for name, scheme in optimization_backend.DIRECT_COLLOCATION_SCHEMES.items():
                opt = optimization_backend.make_optimizer(
                    name, A320, 10, 300, arrival_time_s=120.0,
                )
                seen[name] = opt.scheme
        finally:
            optimization_backend.CollocationOptimizer = original

        self.assertEqual(seen, dict(optimization_backend.DIRECT_COLLOCATION_SCHEMES))
        self.assertEqual(seen["casadiDirectCollocation"], "hermiteSimpson")
        self.assertEqual(seen["casadiDirectCollocationRk4"], "rk4")
        # The normalized geodetic names select the metric-position *Normalized
        # schemes (same geodetic RHS, well-conditioned decision state).
        self.assertEqual(seen["casadiDirectCollocationNormalized"], "hermiteSimpsonNormalized")
        self.assertEqual(seen["casadiDirectCollocationNormalizedTrapezoidal"], "trapezoidalNormalized")
        self.assertEqual(seen["casadiDirectCollocationNormalizedRk4"], "rk4Normalized")
        # The full-transport names select the EXACT-transport geodetic schemes
        # (same geodetic RHS plus the psi cross term the default schemes drop).
        self.assertEqual(seen["casadiDirectCollocationFullTransport"], "hermiteSimpsonFullTransport")
        self.assertEqual(seen["casadiDirectCollocationFullTransportTrapezoidal"], "trapezoidalFullTransport")
        self.assertEqual(seen["casadiDirectCollocationFullTransportRk4"], "rk4FullTransport")
        # Normalized + full transport (the two compose into one scheme).
        self.assertEqual(seen["casadiDirectCollocationNormalizedFullTransport"], "hermiteSimpsonNormalizedFullTransport")
        self.assertEqual(seen["casadiDirectCollocationNormalizedFullTransportTrapezoidal"], "trapezoidalNormalizedFullTransport")
        self.assertEqual(seen["casadiDirectCollocationNormalizedFullTransportRk4"], "rk4NormalizedFullTransport")

    def test_make_optimizer_passes_state_substeps_and_max_iterations(self):
        # SEAM: the request's optional stateSubsteps must reach
        # CollocationOptimizer(state_substeps=...) — absent = None = auto density —
        # and the request's maxIterations must cap IPOPT (the termination guarantee;
        # this arg used to be accepted and silently IGNORED by the collocation branch).
        seen = {}

        class RecordingOptimizer:
            def __init__(self, aircraft, *, state_substeps=None, max_iterations=3000, **_kwargs):
                seen["state_substeps"] = state_substeps
                seen["max_iterations"] = max_iterations

        original = optimization_backend.CollocationOptimizer
        optimization_backend.CollocationOptimizer = RecordingOptimizer
        try:
            optimization_backend.make_optimizer(
                "casadiDirectCollocation", A320, 10, 300,
                arrival_time_s=120.0, state_substeps=12,
            )
            self.assertEqual(seen["state_substeps"], 12)
            self.assertEqual(seen["max_iterations"], 300)
            optimization_backend.make_optimizer(
                "casadiDirectCollocation", A320, 10, 300,
                arrival_time_s=120.0,
            )
            self.assertIsNone(seen["state_substeps"])
        finally:
            optimization_backend.CollocationOptimizer = original

    def test_procedure_constraint_enforced_via_multiphase_scheme_only(self):
        """A procedureConstraint is enforced via the MULTIPHASE optimiser only when an explicit
        casadiMultiphase* scheme is selected; other schemes parse/echo it but do not enforce it."""
        # The unified CollocationOptimizer distinguishes the two paths by its
        # constructor: the multiphase/constrained solve passes ``segments=`` (one
        # phase per leg), the plain direct-collocation solve does not.
        mp_segments = []  # the segments the multiphase (constrained) construction receives

        class FakeCollocation:
            def __init__(self, aircraft, *, scheme=None, segments=None,
                         n_segments=None, max_duration=None, **_kwargs):
                if segments is not None:
                    mp_segments.append(segments)
                self.segment_durations_s = [10.0]
                self.last_dense_states_geo = None
                self.last_solve_timings = SOLVE_TIMINGS

            def optimize_free_time(self, initial_state, target_state, max_duration):
                return (60.0, np.array([[15000.0, 0.0, 1.0]]),
                        np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]))

        proc = {
            "procedureUid": "P", "airportIcao": "K", "runwayIdent": "05L", "branchId": "B",
            "approachCourseDeg": 180.0, "nominalSpeedKt": 180.0,
            "glidepath": {"angleDeg": 3.0, "tchFt": 50.0},
            "waypoints": [
                {"ident": "FAF", "role": "FAF", "legType": "IF",
                 "latDeg": 51.05, "lonDeg": -114.0, "altitude": None, "distanceFromStartM": 0.0},
                {"ident": "RW", "role": "MAPt", "legType": "TF",
                 "latDeg": 51.0, "lonDeg": -114.0, "altitude": None, "distanceFromStartM": 5000.0},
            ],
        }

        def payload(optimizer):
            return {
                "optimizer": optimizer, "nSegments": 2, "arrivalTimeS": 120.0, "dtS": 0.25,
                "initialState": {"lat": 51.2, "lon": -114.0, "altM": 1300.0, "speedMps": 95.0,
                                 "headingDeg": 180.0, "flightPathDeg": -3.0, "aircraftType": "A320"},
                "targetState": {"lat": 51.0, "lon": -114.0, "altM": 200.0, "speedMps": 70.0,
                                "headingDeg": 180.0, "flightPathDeg": -3.0},
                "procedureConstraint": proc,
            }

        original = optimization_backend.CollocationOptimizer
        optimization_backend.CollocationOptimizer = FakeCollocation
        try:
            backend = optimization_backend.OptimizationBackend()
            r_norm = backend.optimize(payload("casadiMultiphaseNormalizedFullTransport"))
            r_plain = backend.optimize(payload("casadiDirectCollocation"))  # not a multiphase scheme
        finally:
            optimization_backend.CollocationOptimizer = original

        # multiphase scheme: constructed WITH real segments + enforced flag True
        self.assertTrue(r_norm["procedureConstraintEnforced"])
        self.assertEqual(len(mp_segments), 1)
        self.assertGreaterEqual(len(mp_segments[0]), 1)
        # other scheme: not enforced, and constructed WITHOUT segments so the
        # segments-based (multiphase) construction did not repeat
        self.assertFalse(r_plain["procedureConstraintEnforced"])
        self.assertEqual(len(mp_segments), 1)

    def test_optimize_logs_whole_flow_timing_to_server_log(self):
        # The backend times the ENTIRE flow (build + solve + playback) and
        # writes a breakdown to the server log (stderr).  When the optimiser
        # exposes a cold-start / free-time split, both appear in the line.
        class FakeCasadiDirectCollocationOptimizer:
            def __init__(self, aircraft, *, scheme="hermiteSimpson", segments=None,
                         n_segments=None, max_duration=None, **_kwargs):
                self.last_solve_timings = {
                    "buildS": 0.05, "solverSetupS": 0.15,
                    "coldStartS": 0.4,
                    "freeTimeSolveS": 0.6,
                    "solveTotalS": 1.2,
                }

            def optimize_free_time(self, initial_state, target_state, max_duration):
                return (
                    51.0,
                    np.array([[15000.0, 0.0, 1.0]]),
                    np.array([[51.0, -114.0, 1000.0, 130.0, 0.3, -0.05]]),
                )

        original = optimization_backend.CollocationOptimizer
        optimization_backend.CollocationOptimizer = (
            FakeCasadiDirectCollocationOptimizer
        )
        buffer = io.StringIO()
        try:
            with redirect_stderr(buffer):
                result = optimization_backend.OptimizationBackend().optimize({
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
            optimization_backend.CollocationOptimizer = original

        log = buffer.getvalue()
        self.assertIn("optimization timing", log)
        self.assertIn("optimizer=casadiDirectCollocationNormalized", log)
        self.assertIn("build=", log)
        self.assertIn("coldStart=0.400s", log)
        self.assertIn("freeTime=0.600s", log)
        self.assertIn("playback=", log)
        self.assertIn("total=", log)
        # the resolved scheme + fitting are announced BEFORE the solve/timing line
        self.assertIn("optimizer config", log)
        self.assertIn("scheme=hermiteSimpsonNormalized ", log)
        self.assertIn("fitting=hermiteSimpson", log)
        self.assertLess(log.index("optimizer config"), log.index("optimization timing"))
        # the drift guard measured the (fake-controls) rollout end vs the target — a large
        # drift is expected here and must be BOTH in the response and loudly flagged
        self.assertIn("playbackDriftM", result)
        self.assertGreater(result["playbackDriftM"], optimization_backend.PLAYBACK_DRIFT_WARN_M)
        self.assertIn("playback terminal drift", log)

    def test_log_optimizer_config_decomposes_scheme_and_fitting(self):
        def line_for(name, constrained=False):
            buffer = io.StringIO()
            with redirect_stderr(buffer):
                optimization_backend.log_optimizer_config(name, constrained=constrained)
            return buffer.getvalue()

        multiphase = line_for("casadiMultiphaseNormalizedFullTransportTrapezoidal", constrained=True)
        self.assertIn("scheme=trapezoidalNormalizedFullTransport", multiphase)
        self.assertIn("fitting=trapezoidal", multiphase)
        self.assertIn("dynamics=geodeticNormalized", multiphase)
        self.assertIn("transport=full", multiphase)
        self.assertIn("constrained=True", multiphase)

        local_enu = line_for("casadiDirectCollocationLocalEnu")
        self.assertIn("scheme=localEnu ", local_enu)
        self.assertIn("fitting=rk4", local_enu)
        self.assertIn("dynamics=localEnu", local_enu)

        reanchored = line_for("casadiDirectCollocationReanchoredEnu")
        self.assertIn("fitting=shooting", reanchored)
        self.assertIn("constrained=False", reanchored)

    def test_optimize_rejects_unknown_optimizer(self):
        # the six archived optimizers are unknown names now (code_review F13)
        for name in ("notReal", "casadiIpopt", "transcription", "leastSquaresTranscription",
                     "warmStartTranscription", "variableTimeWarmStartTranscription", "singleShooting"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "optimizer must be one of"):
                optimization_backend.OptimizationBackend().optimize({
                    "optimizer": name,
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
