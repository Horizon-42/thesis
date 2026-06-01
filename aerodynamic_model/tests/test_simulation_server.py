import json
import math
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import unittest
from urllib import request


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from simulation_server import (  # noqa: E402
    GeodeticState,
    SimulationServer,
    SimulationSession,
    make_request_handler,
)
from coordinates_convertor import GeodeticCoordinate  # noqa: E402
from simulator import Control  # noqa: E402


class TestSimulationServerCore(unittest.TestCase):
    def test_enu_velocity_components_follow_simulator_angle_convention(self):
        # This protects the simulator convention: psi=0 is East, psi=pi/2 is North.
        east_velocity = SimulationServer.get_enu_velocity_components(
            V=120.0,
            gamma=0.0,
            psi=0.0,
        )
        north_velocity = SimulationServer.get_enu_velocity_components(
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
        ecef_velocity = SimulationServer.enu_velocity_to_ecef_velocity(
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

        ecef_velocity = SimulationServer.enu_velocity_to_ecef_velocity(original_enu, geo)
        restored_enu = SimulationServer.ecef_velocity_to_enu_velocity(ecef_velocity, geo)

        self.assertAlmostEqual(restored_enu[0], original_enu[0])
        self.assertAlmostEqual(restored_enu[1], original_enu[1])
        self.assertAlmostEqual(restored_enu[2], original_enu[2])

    def test_step_integrates_local_east_motion_into_longitude(self):
        # This protects the original step core: psi=0 should move east in geodetic output.
        server = SimulationServer()
        state = GeodeticState(
            latitude=0.0,
            longitude=0.0,
            altitude=1000.0,
            V=120.0,
            psi=0.0,
            gamma=0.0,
            m=10000.0,
        )

        next_state = server.step(state, Control(12000.0, 0.0, 1.0), 0.2)

        self.assertGreater(next_state.longitude, state.longitude)
        self.assertAlmostEqual(next_state.latitude, state.latitude, places=6)
        self.assertGreater(next_state.V, 0.0)

    def test_session_reset_reads_frontend_payload_names(self):
        # This protects the browser/server JSON boundary without bypassing field mapping.
        session = SimulationSession()

        snapshot = session.reset({
            "state": {
                "lon": -114.0203,
                "lat": 51.1139,
                "altM": 1084.0,
                "speedMps": 135.0,
                "headingDeg": 12.0,
                "flightPathDeg": -3.0,
                "massKg": 12000.0,
            },
            "control": {
                "thrustN": 15000.0,
                "bankDeg": 5.0,
                "loadFactor": 1.1,
            },
        })

        self.assertTrue(snapshot["ok"])
        self.assertEqual(snapshot["elapsedS"], 0.0)
        self.assertAlmostEqual(snapshot["state"]["lon"], -114.0203)
        self.assertAlmostEqual(snapshot["state"]["lat"], 51.1139)
        self.assertAlmostEqual(snapshot["state"]["headingDeg"], 12.0)
        self.assertAlmostEqual(snapshot["control"]["bankDeg"], 5.0)
        self.assertAlmostEqual(snapshot["control"]["loadFactor"], 1.1)

    def test_session_step_advances_elapsed_time_and_position(self):
        # This protects the stateful server loop used by repeated frontend /step calls.
        session = SimulationSession()
        session.reset({
            "state": {
                "lon": 0.0,
                "lat": 0.0,
                "altM": 1000.0,
                "speedMps": 120.0,
                "headingDeg": 0.0,
                "flightPathDeg": 0.0,
                "massKg": 10000.0,
            }
        })

        snapshot = session.step({"dtS": 0.2})

        self.assertAlmostEqual(snapshot["elapsedS"], 0.2)
        self.assertGreater(snapshot["state"]["lon"], 0.0)
        self.assertTrue(math.isfinite(snapshot["aero"]["liftCoefficient"]))


class TestSimulationHttpApi(unittest.TestCase):
    def setUp(self):
        handler = make_request_handler(SimulationSession())
        self.http_server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.http_server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.http_server.shutdown()
        self.http_server.server_close()
        self.thread.join(timeout=2)

    def test_health_endpoint_returns_ok(self):
        # This protects the lightweight readiness check used before browser coupling.
        with request.urlopen(f"{self.base_url}/health", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(response.status, 200)
        self.assertEqual(payload, {"ok": True, "service": "aeroviz-simulation"})

    def test_reset_endpoint_returns_frontend_snapshot_shape(self):
        # This protects the real HTTP input/output shape consumed by pilotClient.ts.
        payload = {
            "state": {
                "lon": 0.0,
                "lat": 0.0,
                "altM": 1000.0,
                "speedMps": 120.0,
                "headingDeg": 0.0,
                "flightPathDeg": 0.0,
                "massKg": 10000.0,
            }
        }

        response_payload = self._post_json("/reset", payload)

        self.assertTrue(response_payload["ok"])
        self.assertIn("state", response_payload)
        self.assertIn("control", response_payload)
        self.assertIn("aero", response_payload)
        self.assertEqual(response_payload["state"]["lon"], 0.0)

    def _post_json(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
