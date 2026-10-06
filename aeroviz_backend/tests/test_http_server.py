import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from aeroviz_backend.http_server import AeroVizBackendApp, AeroVizRequestHandler, ServedFile
from aeroviz_backend.traffic_jobs import BadJobRequest, JobBusy, JobNotFound


class TestAeroVizBackendApp(unittest.TestCase):
    def test_health_endpoint_returns_unified_backend_identity(self):
        app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=FakeOptimizationBackend(),
        )

        status, payload = app.handle_get("/health")

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"ok": True, "service": "aeroviz-backend"})

    def test_a_sets_intent_is_its_one_campaign_and_none_or_several_are_named(self):
        """Outline §6.2 item 4: the run's one campaign (its id, title, intent, design and the run's line), read from the file
        at each request; no campaign, several, or no run asked: an error naming them. The answers are the frontend's
        fixture (`aeroviz-4d/src/data/__tests__/fixtures/training_intent/answers.json`, its reader's tests read it;
        ``AEROVIZ_WRITE_FIXTURES=1`` writes it), and the frontend's path is this route (a pinned mirror)."""
        import json
        import os
        import re
        import tempfile
        from pathlib import Path

        frontend = Path(__file__).resolve().parents[2] / "aeroviz-4d" / "src" / "data"
        path_constant = re.search(r'export const TRAINING_SET_INTENT_PATH = "([^"]*)"',
                                  (frontend / "trainingSetIntent.ts").read_text(encoding="utf-8"))
        self.assertEqual(path_constant.group(1), "/experiments/intent")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intents.json"
            path.write_text(json.dumps({"campaigns": {
                "one": {"title": "T", "intent": "I", "design": "D", "runs": {"set_a": "line a", "shared": "x"}},
                "two": {"title": "T2", "intent": "I2", "design": "D2", "runs": {"shared": "y"}},
                "question": {"title": "Q", "intent": "a question with no run of its own", "design": "D3"}}}), encoding="utf-8")
            app = AeroVizBackendApp(simulation_backend=FakeSimulationBackend(),
                                    optimization_backend=FakeOptimizationBackend(), experiment_intents=path)
            answers = {name: app.handle_get(f"/experiments/intent?run={run}")
                       for name, run in (("one", "set_a"), ("none", "nowhere"), ("several", "shared"))}
            self.assertEqual(answers["one"], (200, {
                "ok": True, "run": "set_a", "campaign": "one", "title": "T", "intent": "I", "design": "D", "line": "line a"}))
            self.assertEqual((answers["none"][0], answers["none"][1]["campaigns"]), (404, []))
            self.assertIn("nowhere is a run of no campaign", answers["none"][1]["error"])
            self.assertEqual((answers["several"][0], answers["several"][1]["campaigns"]), (409, ["one", "two"]))
            self.assertIn("2 campaigns (one, two)", answers["several"][1]["error"])
            self.assertEqual(app.handle_get("/experiments/intent")[0], 400)
            data = json.loads(path.read_text())
            data["campaigns"]["three"] = {"title": "T3", "intent": "I3", "design": "D4", "runs": {"set_b": "line b"}}
            path.write_text(json.dumps(data), encoding="utf-8")                # written since: answered at once
            self.assertEqual(app.handle_get("/experiments/intent?run=set_b")[1]["campaign"], "three")
        fixture = frontend / "__tests__" / "fixtures" / "training_intent" / "answers.json"
        text = json.dumps({name: {"status": status, "body": body} for name, (status, body) in answers.items()}, indent=1) + "\n"
        if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text(text, encoding="utf-8")
        self.assertEqual(fixture.read_text(encoding="utf-8"), text,
                         f"{fixture} is not what the backend answers now: AEROVIZ_WRITE_FIXTURES=1 writes it again")
        # the registry itself: a published set of stage B
        app = AeroVizBackendApp(simulation_backend=FakeSimulationBackend(), optimization_backend=FakeOptimizationBackend())
        status, payload = app.handle_get("/experiments/intent?run=prior_base_val_20261006")
        self.assertEqual((status, payload["campaign"]), (200, "prior_sets_20261006"))

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
            "&flight_key=flight-b&flight_key=flight-a"
        )
        # The comparison overlay's reference request: the same route, the arrival window.
        window_status, window_payload = app.handle_get(
            "/trajectories?airport=krdu&window=arrival&flight_key=flight-b"
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["czml"][0]["id"], "document")
        self.assertEqual(window_status, 200)
        self.assertEqual(window_payload["trackWindow"], "arrival")
        self.assertEqual(
            observed_backend.calls,
            [
                ("krdu", "23R", "fail", 200, 42, ["flight-b", "flight-a"], None),
                ("krdu", None, None, 200, 0, ["flight-b"], "arrival"),
            ],
        )

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


class TestAeroVizRequestHandler(unittest.TestCase):
    def test_a_page_that_closed_the_connection_first_is_one_log_line_not_an_error(self):
        # the Training view aborts a live-executor request a newer click superseded: the 409 finds the socket closed
        for gone in (BrokenPipeError, ConnectionResetError):
            body = b'{"clientId": "page", "seq": 1}'
            handler = AeroVizRequestHandler.__new__(AeroVizRequestHandler)
            handler.app = SupersedingApp()
            handler.request_version, handler.command, handler.path = "HTTP/1.1", "POST", "/autopilot/segment"
            handler.requestline, handler.client_address = "POST /autopilot/segment HTTP/1.1", ("page", 0)
            handler.headers = {"Content-Length": str(len(body))}
            handler.rfile, handler.wfile = io.BytesIO(body), ClosedSocket(gone)
            log = io.StringIO()
            with contextlib.redirect_stderr(log):
                handler.do_POST()
            self.assertEqual(log.getvalue().splitlines(),
                             ["[aeroviz-backend] client gone status=409 method=POST path=/autopilot/segment"])


class FakeTrafficJobs:
    """The job manager's interface, answering from a script (no process, no roster)."""

    def __init__(self, file=None):
        self.calls = []
        self.busy = False
        self.file_path = file

    def arrivals(self, airport, day):
        self.calls.append(("arrivals", airport, day))
        if airport == "KSEA":
            raise FileNotFoundError("no arrivals roster for KSEA")
        if day == "bad":
            raise ValueError("date must be YYYY-MM-DD, got 'bad'")
        return {"airport": airport, "date": day, "arrivals": []}

    def start(self, request):
        self.calls.append(("start", request))
        if self.busy:
            raise JobBusy("a traffic job is running")
        if request.get("mode") == "nope":
            raise BadJobRequest("mode must be 'm1' or 'm2'")
        return {"jobId": "20261006T120000123456Z-0123abcd"}

    def status(self, job_id):
        self.calls.append(("status", job_id))
        if job_id == "missing":
            raise JobNotFound("no traffic job 'missing'")
        return {"state": "running", "progress": {"done": 1, "total": 3, "current": "K"}, "error": None}

    def cancel(self, job_id):
        self.calls.append(("cancel", job_id))
        if job_id == "missing":
            raise JobNotFound("no traffic job 'missing'")
        return {"state": "cancelled", "progress": {"done": 1, "total": 3, "current": "K"}, "error": None}

    def file(self, job_id, name):
        self.calls.append(("file", job_id, name))
        if name == "unlisted.json":
            raise JobNotFound("'unlisted.json' is not a file of job's comparison index")
        return self.file_path


class TestTrafficJobRoutes(unittest.TestCase):
    def setUp(self):
        self.jobs = FakeTrafficJobs()
        self.app = AeroVizBackendApp(
            simulation_backend=FakeSimulationBackend(),
            optimization_backend=FakeOptimizationBackend(),
            traffic_jobs=self.jobs,
        )

    def test_arrivals_route_parses_the_query_and_maps_errors(self):
        status, payload = self.app.handle_get("/traffic/arrivals?airport=KRDU&date=2026-05-21")
        self.assertEqual((status, payload["airport"], payload["date"]), (200, "KRDU", "2026-05-21"))
        self.assertEqual(self.app.handle_get("/traffic/arrivals?airport=KSEA&date=2026-05-21")[0], 404)
        self.assertEqual(self.app.handle_get("/traffic/arrivals?airport=KRDU&date=bad")[0], 400)
        self.assertEqual(self.app.handle_get("/traffic/arrivals?airport=KRDU")[0], 400)           # date is required
        self.assertEqual(self.app.handle_get("/traffic/arrivals?date=2026-05-21")[0], 400)        # so is the airport

    def test_start_route_returns_the_job_id_and_refuses_a_second_job_and_a_bad_request(self):
        request = {"mode": "m1", "airport": "KRDU", "flightKey": "K"}
        self.assertEqual(self.app.handle_post("/traffic/jobs", request),
                         (200, {"jobId": "20261006T120000123456Z-0123abcd"}, None))
        self.jobs.busy = True
        status, payload, _log = self.app.handle_post("/traffic/jobs", request)
        self.assertEqual(status, 409)
        self.assertIn("running", payload["error"])
        self.jobs.busy = False
        self.assertEqual(self.app.handle_post("/traffic/jobs", {"mode": "nope"})[0], 400)
        self.assertEqual(self.jobs.calls[0], ("start", request))

    def test_status_and_cancel_routes_name_the_job(self):
        job = "20261006T120000123456Z-0123abcd"
        status, payload = self.app.handle_get(f"/traffic/jobs/{job}")
        self.assertEqual((status, payload["state"], payload["progress"]["done"]), (200, "running", 1))
        status, payload, _log = self.app.handle_post(f"/traffic/jobs/{job}/cancel", {})
        self.assertEqual((status, payload["state"]), (200, "cancelled"))
        self.assertEqual(self.app.handle_get("/traffic/jobs/missing")[0], 404)
        self.assertEqual(self.app.handle_post("/traffic/jobs/missing/cancel", {})[0], 404)
        self.assertEqual([c[0] for c in self.jobs.calls], ["status", "cancel", "status", "cancel"])

    def test_file_route_serves_the_listed_file_and_404s_the_rest(self):
        job = "20261006T120000123456Z-0123abcd"
        self.jobs.file_path = Path("/jobs/comparison/comparison_KRDU_05L_g.czml")
        status, payload = self.app.handle_get(f"/traffic/jobs/{job}/files/comparison_KRDU_05L_g.czml")
        self.assertEqual((status, payload), (200, ServedFile(self.jobs.file_path)))
        self.assertEqual(self.jobs.calls[-1], ("file", job, "comparison_KRDU_05L_g.czml"))
        self.assertEqual(self.app.handle_get(f"/traffic/jobs/{job}/files/unlisted.json")[0], 404)
        # a name with a slash is not one segment of the route; an encoded one reaches the manager, which refuses it
        self.assertEqual(self.app.handle_get(f"/traffic/jobs/{job}/files/a/b")[0], 404)
        self.app.handle_get(f"/traffic/jobs/{job}/files/..%2Fstate.json")
        self.assertEqual(self.jobs.calls[-1], ("file", job, "../state.json"))
        # cancel is a POST route only
        self.assertEqual(self.app.handle_get(f"/traffic/jobs/{job}/cancel")[0], 404)

    def test_a_served_file_goes_out_as_its_own_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            czml = Path(tmp) / "comparison.czml"
            czml.write_bytes(b'[{"id": "document"},\n {"id": "x"}]')
            self.jobs.file_path = czml
            handler = AeroVizRequestHandler.__new__(AeroVizRequestHandler)
            handler.app = self.app
            handler.request_version, handler.command = "HTTP/1.1", "GET"
            handler.path = "/traffic/jobs/20261006T120000123456Z-0123abcd/files/comparison.czml"
            handler.requestline, handler.client_address = f"GET {handler.path} HTTP/1.1", ("page", 0)
            handler.headers = {}
            handler.wfile = io.BytesIO()
            handler.do_GET()
            sent = handler.wfile.getvalue()
        head, _, body = sent.partition(b"\r\n\r\n")
        self.assertIn(b"200", head.splitlines()[0])
        self.assertIn(b"Content-Type: application/json", head)
        self.assertIn(b"Access-Control-Allow-Origin: *", head)       # the page is served from another origin
        self.assertEqual(body, b'[{"id": "document"},\n {"id": "x"}]')


class TestServedFileGone(unittest.TestCase):
    def test_a_file_that_a_prune_removed_after_the_route_checked_it_is_a_404_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            gone = Path(tmp) / "comparison_KRDU_05L_g.czml"       # listed by the index, but its job is pruned
            app = AeroVizBackendApp(
                simulation_backend=FakeSimulationBackend(), optimization_backend=FakeOptimizationBackend(),
                traffic_jobs=FakeTrafficJobs(file=gone))
            handler = AeroVizRequestHandler.__new__(AeroVizRequestHandler)
            handler.app = app
            handler.request_version, handler.command = "HTTP/1.1", "GET"
            handler.path = "/traffic/jobs/20261006T120000123456Z-0123abcd/files/comparison_KRDU_05L_g.czml"
            handler.requestline, handler.client_address = f"GET {handler.path} HTTP/1.1", ("page", 0)
            handler.headers = {}
            handler.wfile = io.BytesIO()
            log = io.StringIO()
            with contextlib.redirect_stderr(log):
                handler.do_GET()                                  # no FileNotFoundError out of the handler
            head, _, body = handler.wfile.getvalue().partition(b"\r\n\r\n")
        self.assertIn(b"404", head.splitlines()[0])
        self.assertIn(b"pruned", body)
        self.assertIn("status=404", log.getvalue())


class TestJobsRoot(unittest.TestCase):
    def test_each_backend_port_has_its_own_jobs_root(self):
        from aeroviz_backend.traffic_jobs import DEFAULT_JOBS_ROOT, traffic_jobs_root
        self.assertEqual(traffic_jobs_root(8765), DEFAULT_JOBS_ROOT / "8765")
        self.assertNotEqual(traffic_jobs_root(8765), traffic_jobs_root(8766))


class SupersedingApp:
    def handle_post(self, path, payload):
        return 409, {"ok": False, "error": "a newer request came in"}, None


class ClosedSocket:
    def __init__(self, error: type[OSError]) -> None:
        self.error = error

    def write(self, data: bytes) -> int:
        raise self.error()


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

    def query(
        self,
        airport,
        *,
        runway=None,
        verdict=None,
        limit=200,
        seed=0,
        flight_keys=None,
        window=None,
    ):
        self.calls.append(
            (airport, runway, verdict, limit, seed, flight_keys, window)
        )
        return {
            "schemaVersion": "observed-trajectories-v2",
            "trackWindow": window or "full",
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
