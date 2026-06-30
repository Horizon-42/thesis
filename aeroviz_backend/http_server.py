from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aeroviz_backend.dynamics_comparison_backend import DynamicsComparisonBackend
from aeroviz_backend.isolated_backend import (
    IsolatedDynamicsComparisonBackend,
    IsolatedOptimizationBackend,
)
from aeroviz_backend.optimization_backend import OptimizationBackend
from aeroviz_backend.simulation_backend import SimulationBackend, aircraft_catalog


class AeroVizBackendApp:
    def __init__(
        self,
        simulation_backend: SimulationBackend | None = None,
        optimization_backend: OptimizationBackend | None = None,
        dynamics_comparison_backend: DynamicsComparisonBackend | None = None,
    ) -> None:
        # The simulation endpoints run in-process (they are high-frequency and use
        # only casadi function evaluation, not the crash-prone NLP construction).
        # The optimizer and dynamics comparison BUILD casadi NLPs, which can abort
        # the whole process natively, so by default they run in an isolated worker
        # subprocess (see isolated_backend). Tests inject their own backends, so
        # they stay in-process and pay no subprocess cost.
        self.simulation_backend = simulation_backend or SimulationBackend()
        self.optimization_backend = (
            optimization_backend or IsolatedOptimizationBackend()
        )
        self.dynamics_comparison_backend = (
            dynamics_comparison_backend or IsolatedDynamicsComparisonBackend()
        )

    def handle_get(self, path: str) -> tuple[int, dict[str, Any]]:
        if path == "/health":
            return 200, {"ok": True, "service": "aeroviz-backend"}
        if path == "/simulation/aircraft":
            return 200, aircraft_catalog()
        if path == "/dynamics-comparison/history":
            return 200, self.dynamics_comparison_backend.history_count()
        return 404, {"ok": False, "error": "not found"}

    def handle_post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> tuple[int, dict[str, Any], str | None]:
        if path == "/simulation/reset":
            return 200, self.simulation_backend.reset(payload), "reset"
        if path == "/simulation/step":
            return 200, self.simulation_backend.step(payload), None
        if path == "/optimization/run":
            return 200, self.optimization_backend.optimize(payload), None
        # Tab lifecycle: keep the casadi worker resident while the Optimize/Compare
        # tab is open, and decommission it (free its memory) when the tab closes.
        if path == "/optimization/session/open":
            return 200, self.optimization_backend.open_session(payload), None
        if path == "/optimization/session/close":
            return 200, self.optimization_backend.close_session(payload), None
        if path == "/dynamics-comparison/run":
            return 200, self.dynamics_comparison_backend.run(payload), None
        if path == "/dynamics-comparison/session/open":
            return 200, self.dynamics_comparison_backend.open_session(payload), None
        if path == "/dynamics-comparison/session/close":
            return 200, self.dynamics_comparison_backend.close_session(payload), None
        if path == "/dynamics-comparison/history/average":
            return 200, self.dynamics_comparison_backend.average(payload), None
        if path == "/dynamics-comparison/history/clear":
            return 200, self.dynamics_comparison_backend.clear(payload), None
        return 404, {"ok": False, "error": "not found"}, None


class AeroVizRequestHandler(BaseHTTPRequestHandler):
    app = AeroVizBackendApp()
    server_version = "AeroVizBackendHTTP/0.1"
    _live_log_active = False

    def do_OPTIONS(self) -> None:
        self._send_empty(204)

    def do_GET(self) -> None:
        status, payload = self.app.handle_get(self.path)
        self._send_json(payload, status=status)
        if status >= 400:
            self._log_error(status, payload.get("error", "not found"))

    def do_POST(self) -> None:
        optimization_started_at: float | None = None
        try:
            payload = self._read_json()
            if self.path == "/optimization/run":
                optimization_started_at = time.monotonic()
                self._log_optimization_start(payload)
            status, response_payload, log_event = self.app.handle_post(
                self.path,
                payload,
            )
            self._send_json(response_payload, status=status)
            if optimization_started_at is not None:
                self._log_optimization_done(
                    status,
                    response_payload,
                    time.monotonic() - optimization_started_at,
                )
            if log_event and status < 400:
                self._log_snapshot(log_event, response_payload)
            elif status >= 400:
                self._log_error(status, response_payload.get("error", "not found"))
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            if optimization_started_at is not None:
                self._log_optimization_failed(
                    400,
                    str(exc),
                    time.monotonic() - optimization_started_at,
                )
            self._log_error(400, str(exc))
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)
            if optimization_started_at is not None:
                self._log_optimization_failed(
                    500,
                    str(exc),
                    time.monotonic() - optimization_started_at,
                )
            self._log_error(500, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _log_snapshot(self, event: str, snapshot: dict[str, Any]) -> None:
        state = snapshot["state"]
        control = snapshot["control"]
        aero = snapshot["aero"]
        if "loadFactor" in control:
            control_text = f"loadFactor={control['loadFactor']:.3f} "
        else:
            control_text = f"alpha={control['attackDeg']:.2f}deg "
        # state['headingDeg'] holds psi (math-ENU: 0 = East, CCW). Log the true compass
        # bearing (0 = North, CW) as 'heading', keeping the raw psi alongside it.
        compass_heading = (90.0 - state["headingDeg"]) % 360.0
        line = (
            "[aeroviz-backend] "
            f"{event} "
            f"t={snapshot['elapsedS']:.3f}s "
            f"lat={state['lat']:.7f} "
            f"lon={state['lon']:.7f} "
            f"alt={state['altM']:.2f}m "
            f"speed={state['speedMps']:.2f}m/s "
            f"heading={compass_heading:.2f}deg "
            f"psi={state['headingDeg']:.2f}deg "
            f"fpa={state['flightPathDeg']:.2f}deg "
            f"mass={state['massKg']:.1f}kg "
            f"type={state['aircraftType']} "
            f"thrust={control['thrustN']:.1f}N "
            f"bank={control['bankDeg']:.2f}deg "
            f"{control_text}"
            f"Cl={aero['liftCoefficient']:.4f} "
            f"Cd={aero['dragCoefficient']:.4f}"
        )

        if event == "frame":
            sys.stderr.write("\r" + line + "\033[K")
            sys.stderr.flush()
            type(self)._live_log_active = True
            return

        self._finish_live_log_line()
        sys.stderr.write(line + "\n")
        sys.stderr.flush()

    def _log_error(self, status: int, message: str) -> None:
        self._finish_live_log_line()
        sys.stderr.write(
            "[aeroviz-backend] "
            f"error status={status} method={self.command} path={self.path} "
            f"message={message}\n"
        )
        sys.stderr.flush()

    def _log_optimization_start(self, payload: dict[str, Any]) -> None:
        self._finish_live_log_line()
        sys.stderr.write(
            "[aeroviz-backend] optimization start "
            f"segments={payload.get('nSegments', 'default')} "
            f"maxIterations={payload.get('maxIterations', 'default')}\n"
        )
        sys.stderr.flush()

    def _log_optimization_done(
        self,
        status: int,
        payload: dict[str, Any],
        elapsed_s: float,
    ) -> None:
        self._finish_live_log_line()
        sys.stderr.write(
            "[aeroviz-backend] optimization done "
            f"status={status} "
            f"elapsed={elapsed_s:.3f}s "
            f"finalTime={payload.get('finalTimeS', 'n/a')} "
            f"states={len(payload.get('states', []))} "
            f"controls={len(payload.get('controls', []))}\n"
        )
        sys.stderr.flush()

    def _log_optimization_failed(
        self,
        status: int,
        message: str,
        elapsed_s: float,
    ) -> None:
        self._finish_live_log_line()
        sys.stderr.write(
            "[aeroviz-backend] optimization failed "
            f"status={status} elapsed={elapsed_s:.3f}s message={message}\n"
        )
        sys.stderr.flush()

    def _finish_live_log_line(self) -> None:
        if type(self)._live_log_active:
            sys.stderr.write("\n")
            type(self)._live_log_active = False

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self._send_cors_headers()
        self.end_headers()

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def make_request_handler(
    app: AeroVizBackendApp | None = None,
) -> type[AeroVizRequestHandler]:
    class BoundAeroVizRequestHandler(AeroVizRequestHandler):
        pass

    BoundAeroVizRequestHandler.app = app or AeroVizBackendApp()
    return BoundAeroVizRequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AeroViz backend server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    http_server = ThreadingHTTPServer(
        (args.host, args.port),
        make_request_handler(),
    )
    print(f"AeroViz backend listening on http://{args.host}:{args.port}", flush=True)
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http_server.server_close()


if __name__ == "__main__":
    main()
