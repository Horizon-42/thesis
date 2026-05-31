"""
HTTP bridge for AeroViz pilot mode.

Run from aeroviz-4d/:

    python python/pilot_sim_server.py

The server keeps the existing aerodynamic_model.simulator as the source of
flight dynamics.  The browser sends controls, and this process returns the next
WGS84 pose sample.  The lon/lat update uses a fixed placeholder display scale;
airport-specific local-frame <-> WGS84 conversion is a future concern.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aerodynamic_model.simulator import Atmosphere, Control, Simulator, State  # noqa: E402


DEFAULT_STATE = State(
    x=-78.7873,
    y=35.878659,
    h=1000.0,
    V=120.0,
    psi=0.0,
    gamma=0.0,
    m=10000.0,
)
DEFAULT_CONTROL = Control(
    thrust=12000.0,
    bank_rad=0.0,
    load_factor=1.0,
)
DEFAULT_DT_S = 0.2
MAX_DT_S = 2.0
# Temporary visual scale only.  This keeps the WGS84 API operable without
# adding the real local tangent-plane conversion layer yet.
WGS84_DEGREES_PER_SIM_METRE = 1.0 / 111_320.0


@dataclass(frozen=True)
class PilotStateResponse:
    lon: float
    lat: float
    altM: float
    speedMps: float
    headingDeg: float
    flightPathDeg: float
    massKg: float


@dataclass(frozen=True)
class PilotControlResponse:
    thrustN: float
    bankDeg: float
    loadFactor: float


class PilotSimulationSession:
    def __init__(self) -> None:
        self.simulator = Simulator()
        self.atmosphere = Atmosphere(rho0=1.225, H=8500.0)
        self.lon = DEFAULT_STATE.x
        self.lat = DEFAULT_STATE.y
        self.state = State(
            x=0.0,
            y=0.0,
            h=DEFAULT_STATE.h,
            V=DEFAULT_STATE.V,
            psi=DEFAULT_STATE.psi,
            gamma=DEFAULT_STATE.gamma,
            m=DEFAULT_STATE.m,
        )
        self.control = DEFAULT_CONTROL
        self.elapsed_s = 0.0

    def reset(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        state_payload = _as_mapping(payload.get("state"))
        control_payload = _as_mapping(payload.get("control"))

        self.lon = _read_float(state_payload, "lon", DEFAULT_STATE.x)
        self.lat = _read_float(state_payload, "lat", DEFAULT_STATE.y)
        self.state = State(
            x=0.0,
            y=0.0,
            h=max(0.0, _read_float(state_payload, "altM", DEFAULT_STATE.h)),
            V=max(1.0, _read_float(state_payload, "speedMps", DEFAULT_STATE.V)),
            psi=math.radians(_read_float(state_payload, "headingDeg", 0.0)),
            gamma=math.radians(_read_float(state_payload, "flightPathDeg", 0.0)),
            m=max(1.0, _read_float(state_payload, "massKg", DEFAULT_STATE.m)),
        )
        self.control = _read_control(control_payload, self.control)
        self.elapsed_s = 0.0
        return self._snapshot()

    def step(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        self.control = _read_control(_as_mapping(payload.get("control")), self.control)
        dt_s = _clamp(_read_float(payload, "dtS", DEFAULT_DT_S), 0.001, MAX_DT_S)

        solution = self.simulator.simulate(
            initial_state=self.state,
            control=self.control,
            atmosphere=self.atmosphere,
            t_span=(0.0, dt_s),
            t_eval=[dt_s],
        )
        if not solution.success:
            raise ValueError(solution.message)

        x, y, h, velocity, heading, flight_path, mass = solution.y[:, -1]
        self.lon += float(x) * WGS84_DEGREES_PER_SIM_METRE
        self.lat += float(y) * WGS84_DEGREES_PER_SIM_METRE
        self.state = State(
            x=0.0,
            y=0.0,
            h=max(0.0, float(h)),
            V=max(1.0, float(velocity)),
            psi=float(heading),
            gamma=float(flight_path),
            m=max(1.0, float(mass)),
        )
        self.elapsed_s += dt_s
        return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        coefficients = self.simulator.get_aerodynamic_coefficients(
            self.state.h,
            self.state.V,
            self.state.m,
            self.control,
            self.atmosphere,
        )
        return {
            "ok": True,
            "elapsedS": self.elapsed_s,
            "state": asdict(
                PilotStateResponse(
                    lon=self.lon,
                    lat=self.lat,
                    altM=self.state.h,
                    speedMps=self.state.V,
                    headingDeg=_normalize_degrees(math.degrees(self.state.psi)),
                    flightPathDeg=math.degrees(self.state.gamma),
                    massKg=self.state.m,
                )
            ),
            "control": asdict(
                PilotControlResponse(
                    thrustN=self.control.thrust,
                    bankDeg=math.degrees(self.control.bank_rad),
                    loadFactor=self.control.load_factor,
                )
            ),
            "aero": {
                "liftCoefficient": coefficients[0],
                "dragCoefficient": coefficients[1],
            },
        }


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _read_control(payload: dict[str, Any], fallback: Control) -> Control:
    return Control(
        thrust=_clamp(_read_float(payload, "thrustN", fallback.thrust), 0.0, 80000.0),
        bank_rad=math.radians(
            _clamp(
                _read_float(payload, "bankDeg", math.degrees(fallback.bank_rad)),
                -60.0,
                60.0,
            )
        ),
        load_factor=_clamp(
            _read_float(payload, "loadFactor", fallback.load_factor),
            0.2,
            3.0,
        ),
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_degrees(value: float) -> float:
    return value % 360.0


class PilotRequestHandler(BaseHTTPRequestHandler):
    session = PilotSimulationSession()
    server_version = "AeroVizPilotHTTP/0.1"

    def do_OPTIONS(self) -> None:
        self._send_empty(204)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"ok": True, "service": "aeroviz-pilot-sim"})
            return
        self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/reset":
                self._send_json(self.session.reset(payload))
                return
            if self.path == "/step":
                self._send_json(self.session.step(payload))
                return
            self._send_json({"ok": False, "error": "not found"}, status=404)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:  # pragma: no cover - last line of HTTP defense
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[pilot-sim] " + format % args + "\n")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AeroViz pilot mode simulator server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), PilotRequestHandler)
    print(f"Pilot simulator server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
