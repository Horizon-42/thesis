import unittest

from aeroviz_backend.http_server import AeroVizBackendApp


class TestAeroVizBackendApp(unittest.TestCase):
    def test_health_endpoint_returns_unified_backend_identity(self):
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=FakeOptimizationBackend(),
        )

        status, payload = app.handle_get("/health")

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"ok": True, "service": "aeroviz-backend"})

    def test_aircraft_endpoint_uses_simulation_namespace(self):
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=FakeOptimizationBackend(),
        )

        status, payload = app.handle_get("/simulation/aircraft")

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertIn("aircraft", payload)

    def test_trajectory_endpoint_parses_query_and_delegates(self):
        observed_backend = FakeObservedTrajectoryBackend()
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=FakeOptimizationBackend(),
            observed_trajectory_backend=observed_backend,
        )

        status, payload = app.handle_get(
            "/trajectories?airport=krdu&runway=23R&verdict=fail&limit=200&seed=42"
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["czml"][0]["id"], "document")
        self.assertEqual(observed_backend.calls, [("krdu", "23R", "fail", 200, 42)])

    def test_trajectory_endpoint_reports_invalid_query(self):
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=FakeOptimizationBackend(),
            observed_trajectory_backend=FakeObservedTrajectoryBackend(),
        )

        status, payload = app.handle_get(
            "/trajectories?airport=KRDU&limit=not-a-number"
        )

        self.assertEqual(status, 400)
        self.assertIn("limit", payload["error"])

    def test_simulation_routes_delegate_to_simulation_backend(self):
        simulation_backend = FakeSimulationBackend()
        app = AeroVizBackendApp(
            simulation_backend=simulation_backend,
            optimization_backend=FakeOptimizationBackend(),
        )

        reset_status, reset_payload, reset_log = app.handle_post(
            "/simulation/reset",
            {"state": {"aircraftType": "A320"}},
        )
        step_status, step_payload, step_log = app.handle_post(
            "/simulation/step",
            {"dtS": 0.2},
        )

        self.assertEqual(reset_status, 200)
        self.assertEqual(reset_payload["route"], "reset")
        self.assertEqual(reset_log, "reset")
        self.assertEqual(step_status, 200)
        self.assertEqual(step_payload["route"], "step")
        self.assertIsNone(step_log)
        self.assertEqual(
            simulation_backend.calls,
            [
                ("reset", {"state": {"aircraftType": "A320"}}),
                ("step", {"dtS": 0.2}),
            ],
        )

    def test_optimization_route_delegates_to_optimization_backend(self):
        optimization_backend = FakeOptimizationBackend()
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=optimization_backend,
        )
        request_payload = {
            "initialState": {"aircraftType": "A320"},
            "targetState": {},
        }

        status, payload, log_event = app.handle_post(
            "/optimization/run",
            request_payload,
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"ok": True, "finalTimeS": 12.0})
        self.assertIsNone(log_event)
        self.assertEqual(optimization_backend.calls, [request_payload])

    def test_optimization_session_routes_delegate(self):
        optimization_backend = FakeOptimizationBackend()
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=optimization_backend,
        )

        open_status, open_payload, _ = app.handle_post("/optimization/session/open", {})
        close_status, close_payload, _ = app.handle_post("/optimization/session/close", {})

        self.assertEqual(open_status, 200)
        self.assertEqual(open_payload, {"ok": True, "sessions": 1})
        self.assertEqual(close_status, 200)
        self.assertEqual(close_payload, {"ok": True, "sessions": 0})
        self.assertEqual(
            [name for name, _ in optimization_backend.session_calls],
            ["open", "close"],
        )

    def test_dynamics_comparison_session_routes_delegate(self):
        dynamics_comparison_backend = FakeDynamicsComparisonBackend()
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=FakeOptimizationBackend(),
            dynamics_comparison_backend=dynamics_comparison_backend,
        )

        open_status, open_payload, _ = app.handle_post(
            "/dynamics-comparison/session/open", {}
        )
        close_status, close_payload, _ = app.handle_post(
            "/dynamics-comparison/session/close", {}
        )

        self.assertEqual(open_status, 200)
        self.assertEqual(open_payload, {"ok": True, "sessions": 1})
        self.assertEqual(close_status, 200)
        self.assertEqual(close_payload, {"ok": True, "sessions": 0})

    def test_dynamics_comparison_route_delegates_to_its_backend(self):
        dynamics_comparison_backend = FakeDynamicsComparisonBackend()
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=FakeOptimizationBackend(),
            dynamics_comparison_backend=dynamics_comparison_backend,
        )
        request_payload = {
            "initialState": {"aircraftType": "A320"},
            "control": {"thrustN": 70000.0},
        }

        status, payload, log_event = app.handle_post(
            "/dynamics-comparison/run",
            request_payload,
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"ok": True, "route": "comparison"})
        self.assertIsNone(log_event)
        self.assertEqual(dynamics_comparison_backend.calls, [("run", request_payload)])

    def test_dynamics_comparison_history_routes_delegate(self):
        dynamics_comparison_backend = FakeDynamicsComparisonBackend()
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=FakeOptimizationBackend(),
            dynamics_comparison_backend=dynamics_comparison_backend,
        )

        get_status, get_payload = app.handle_get("/dynamics-comparison/history")
        avg_status, avg_payload, _ = app.handle_post("/dynamics-comparison/history/average", {})
        clear_status, clear_payload, _ = app.handle_post("/dynamics-comparison/history/clear", {})

        self.assertEqual(get_status, 200)
        self.assertEqual(get_payload, {"ok": True, "historyCount": 3})
        self.assertEqual(avg_status, 200)
        self.assertEqual(avg_payload, {"ok": True, "route": "average"})
        self.assertEqual(clear_status, 200)
        self.assertEqual(clear_payload, {"ok": True, "historyCount": 0})
        self.assertEqual(
            [name for name, _ in dynamics_comparison_backend.calls],
            ["history_count", "average", "clear"],
        )

    def test_legacy_root_simulation_routes_are_not_exposed(self):
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=FakeOptimizationBackend(),
        )

        get_status, _ = app.handle_get("/aircraft")
        post_status, _, _ = app.handle_post("/step", {})

        self.assertEqual(get_status, 404)
        self.assertEqual(post_status, 404)


class FakeSimulationBackend:
    def __init__(self):
        self.calls = []

    def reset(self, payload):
        self.calls.append(("reset", payload))
        return {"ok": True, "route": "reset"}

    def step(self, payload):
        self.calls.append(("step", payload))
        return {"ok": True, "route": "step"}


class FakeOptimizationBackend:
    def __init__(self):
        self.calls = []
        self.session_calls = []
        self._sessions = 0

    def optimize(self, payload):
        self.calls.append(payload)
        return {"ok": True, "finalTimeS": 12.0}

    def open_session(self, payload=None):
        self.session_calls.append(("open", payload))
        self._sessions += 1
        return {"ok": True, "sessions": self._sessions}

    def close_session(self, payload=None):
        self.session_calls.append(("close", payload))
        self._sessions = max(0, self._sessions - 1)
        return {"ok": True, "sessions": self._sessions}


class FakeObservedTrajectoryBackend:
    def __init__(self):
        self.calls = []

    def query(self, airport, *, runway=None, verdict=None, limit=200, seed=0):
        self.calls.append((airport, runway, verdict, limit, seed))
        return {
            "schemaVersion": "observed-trajectories-v1",
            "czml": [{"id": "document"}, {"id": "TEST"}],
            "verdicts": None,
            "evaluation": None,
        }


class FakeDynamicsComparisonBackend:
    def __init__(self):
        self.calls = []
        self._sessions = 0

    def run(self, payload):
        self.calls.append(("run", payload))
        return {"ok": True, "route": "comparison"}

    def open_session(self, payload=None):
        self.calls.append(("open_session", payload))
        self._sessions += 1
        return {"ok": True, "sessions": self._sessions}

    def close_session(self, payload=None):
        self.calls.append(("close_session", payload))
        self._sessions = max(0, self._sessions - 1)
        return {"ok": True, "sessions": self._sessions}

    def history_count(self, payload=None):
        self.calls.append(("history_count", payload))
        return {"ok": True, "historyCount": 3}

    def average(self, payload=None):
        self.calls.append(("average", payload))
        return {"ok": True, "route": "average"}

    def clear(self, payload=None):
        self.calls.append(("clear", payload))
        return {"ok": True, "historyCount": 0}


if __name__ == "__main__":
    unittest.main()
