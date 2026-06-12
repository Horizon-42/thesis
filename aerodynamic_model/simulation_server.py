from coordinates_convertor import CoordinateConverter, GeodeticCoordinate, ECEFCoordinate, ENUCoordinate, ENUUnitVectors
from simulator import Simulator, Control, Atmosphere, State
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import math
import sys
from typing import Any

@dataclass
class GeodeticState:
    latitude: float
    longitude: float
    altitude: float
    V: float
    psi: float
    gamma: float
    m: float

DEFAULT_STATE = GeodeticState(
    latitude=35.878659,
    longitude=-78.7873,
    altitude=1000.0,
    V=120.0,
    psi=0.0,
    gamma=0.0,
    m=10000.0,
)

DEFAULT_CONTROL = Control(
    thrust=12000.0,
    bank_rad=0.0,
    attack_rad=0.0,
)

DEFAULT_DT = 0.2
MAX_DT = 2.0

class SimulationServer():
    def __init__(self):
        self.simulator = Simulator()
        self.atmosphere = Atmosphere()
    
    @staticmethod
    def get_enu_velocity_components(V: float, gamma: float, psi: float) -> tuple[float, float, float]:
        # Step 1: read the simulator velocity state in the old local ENU frame.
        # psi/gamma are radians; psi=0 points East and psi=pi/2 points North.
        V_east = V * math.cos(gamma) * math.cos(psi)
        V_north = V * math.cos(gamma) * math.sin(psi)
        V_up = V * math.sin(gamma)
        return (V_east, V_north, V_up)

    @staticmethod
    def enu_velocity_to_ecef_velocity(enu_velocity: tuple[float, float, float], geo_S: GeodeticCoordinate) -> tuple[float, float, float]:
        # Step 2: expand old-ENU components into the global ECEF basis.
        # e_hat/n_hat/u_hat are local ENU axes, but each axis is expressed as
        # an ECEF unit vector. Multiplying and summing gives one ECEF vector.
        V_east, V_north, V_up = enu_velocity
        lat_rad = math.radians(geo_S.latitude)
        lon_rad = math.radians(geo_S.longitude)
        e_hat = (-math.sin(lon_rad), math.cos(lon_rad), 0)
        n_hat = (-math.sin(lat_rad) * math.cos(lon_rad), -math.sin(lat_rad) * math.sin(lon_rad), math.cos(lat_rad))
        u_hat = (math.cos(lat_rad) * math.cos(lon_rad), math.cos(lat_rad) * math.sin(lon_rad), math.sin(lat_rad))

        return (
            V_east * e_hat[0] + V_north * n_hat[0] + V_up * u_hat[0],
            V_east * e_hat[1] + V_north * n_hat[1] + V_up * u_hat[1],
            V_east * e_hat[2] + V_north * n_hat[2] + V_up * u_hat[2],
        )
    
    @staticmethod
    def ecef_velocity_to_enu_velocity(ecef_velocity: tuple[float, float, float], geo_S: GeodeticCoordinate) -> tuple[float, float, float]:
        # Step 3: project the same physical ECEF vector onto a new local ENU
        # frame. This gives the updated local velocity components after moving.
        lat_rad = math.radians(geo_S.latitude)
        lon_rad = math.radians(geo_S.longitude)

        # project from old enu to new enu by new ENU unit vectors expressed in ECEF. This is the inverse of step 2.
        e_hat = (-math.sin(lon_rad), math.cos(lon_rad), 0)
        n_hat = (-math.sin(lat_rad) * math.cos(lon_rad), -math.sin(lat_rad) * math.sin(lon_rad), math.cos(lat_rad))
        u_hat = (math.cos(lat_rad) * math.cos(lon_rad), math.cos(lat_rad) * math.sin(lon_rad), math.sin(lat_rad))
        V_east = e_hat[0] * ecef_velocity[0] + e_hat[1] * ecef_velocity[1] + e_hat[2] * ecef_velocity[2]
        V_north = n_hat[0] * ecef_velocity[0] + n_hat[1] * ecef_velocity[1] + n_hat[2] * ecef_velocity[2]
        V_up = u_hat[0] * ecef_velocity[0] + u_hat[1] * ecef_velocity[1] + u_hat[2] * ecef_velocity[2]

        return (V_east, V_north, V_up)

    def step(self, state: GeodeticState, control: Control, dt: float) -> GeodeticState:
        # Convert geodetic state to ECEF for simulation
        geo_P = GeodeticCoordinate(state.latitude, state.longitude, state.altitude)
        geo_S = GeodeticCoordinate(state.latitude, state.longitude, 0.0)  # Reference point for ENU is at the same lat/lon but sea level
        enu_P = CoordinateConverter.geodetic_to_enu(geo_P, geo_S)  # Using the same point as reference

        # Create state vector for simulator (x, y, h, V, psi, gamma, m)
        state_vec = State(enu_P.east, enu_P.north, state.altitude, state.V, state.psi, state.gamma, state.m)

        # Simulate one time step
        solution = self.simulator.simulate(
            initial_state=state_vec,
            control=control,
            atmosphere=self.atmosphere,
            t_span=(0.0, dt),
            t_eval=[dt],
        )
        if not solution.success:
            raise ValueError(solution.message)

        new_state_vec = [float(value) for value in solution.y[:, -1]]

        # Convert back to geodetic coordinates
        new_enu_P = ENUCoordinate(new_state_vec[0], new_state_vec[1], new_state_vec[2])
        new_geo = CoordinateConverter.enu_to_geodetic(new_enu_P, geo_S)  # Convert back to geodetic using the same reference point

        # Keep velocity as the same physical vector while the local ENU frame
        # moves: old ENU components -> ECEF vector -> new ENU components.
        old_enu_velocity = self.get_enu_velocity_components(
            V=new_state_vec[3],
            gamma=new_state_vec[5],
            psi=new_state_vec[4]
        )
        new_ecef_velocity = self.enu_velocity_to_ecef_velocity(old_enu_velocity, geo_S)
        new_enu_velocity = self.ecef_velocity_to_enu_velocity(new_ecef_velocity, new_geo)
        V_east, V_north, V_up = new_enu_velocity
        V = math.sqrt(V_east**2 + V_north**2 + V_up**2)
        horizontal_V = math.hypot(V_east, V_north)

        return GeodeticState(
            latitude=new_geo.latitude,
            longitude=new_geo.longitude,
            altitude=new_geo.altitude,
            V=V,
            psi=math.atan2(V_north, V_east),
            gamma=math.atan2(V_up, horizontal_V),
            m=new_state_vec[6]
        )

class SimulationSession:
    def __init__(self):
        self.server = SimulationServer()
        self.state = DEFAULT_STATE
        self.control = DEFAULT_CONTROL
        self.elapsed = 0.0

    def reset(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        state_payload = _as_mapping(payload.get("state"))
        control_payload = _as_mapping(payload.get("control"))

        self.state = GeodeticState(
            latitude=_read_float(state_payload, "lat", DEFAULT_STATE.latitude),
            longitude=_read_float(state_payload, "lon", DEFAULT_STATE.longitude),
            altitude=max(0.0, _read_float(state_payload, "altM", DEFAULT_STATE.altitude)),
            V=max(1.0, _read_float(state_payload, "speedMps", DEFAULT_STATE.V)),
            psi=math.radians(_read_float(state_payload, "headingDeg", math.degrees(DEFAULT_STATE.psi))),
            gamma=math.radians(_read_float(state_payload, "flightPathDeg", math.degrees(DEFAULT_STATE.gamma))),
            m=max(1.0, _read_float(state_payload, "massKg", DEFAULT_STATE.m)),
        )
        self.control = _read_control(control_payload, DEFAULT_CONTROL)
        self.elapsed = 0.0
        return self.snapshot()

    def step(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        self.control = _read_control(_as_mapping(payload.get("control")), self.control)
        dt = _clamp(_read_float(payload, "dtS", DEFAULT_DT), 0.001, MAX_DT)
        self.state = self.server.step(self.state, self.control, dt)
        self.elapsed += dt
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        Cl, Cd = self.server.simulator.get_aerodynamic_coefficients(
            self.control.attack_rad,
        )
        return {
            "ok": True,
            "elapsedS": self.elapsed,
            "state": {
                "lon": self.state.longitude,
                "lat": self.state.latitude,
                "altM": self.state.altitude,
                "speedMps": self.state.V,
                "headingDeg": _normalize_degrees(math.degrees(self.state.psi)),
                "flightPathDeg": math.degrees(self.state.gamma),
                "massKg": self.state.m,
            },
            "control": {
                "thrustN": self.control.thrust,
                "bankDeg": math.degrees(self.control.bank_rad),
                "attackDeg": math.degrees(self.control.attack_rad),
            },
            "aero": {
                "liftCoefficient": Cl,
                "dragCoefficient": Cd,
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
        bank_rad=math.radians(_clamp(
            _read_float(payload, "bankDeg", math.degrees(fallback.bank_rad)),
            -60.0,
            60.0,
        )),
        attack_rad=math.radians(_clamp(
            _read_float(payload, "attackDeg", math.degrees(fallback.attack_rad)),
            -10.0,
            18.0,
        )),
    )

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def _normalize_degrees(value: float) -> float:
    return value % 360.0

def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")

class SimulationRequestHandler(BaseHTTPRequestHandler):
    session = SimulationSession()
    server_version = "AeroVizSimulationHTTP/0.1"
    _live_log_active = False

    def do_OPTIONS(self) -> None:
        self._send_empty(204)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"ok": True, "service": "aeroviz-simulation"})
            return
        self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/reset":
                snapshot = self.session.reset(payload)
                self._send_json(snapshot)
                self._log_snapshot("reset", snapshot)
                return
            if self.path == "/step":
                snapshot = self.session.step(payload)
                self._send_json(snapshot)
                self._log_snapshot("frame", snapshot)
                return
            self._send_json({"ok": False, "error": "not found"}, status=404)
            self._log_error(404, "not found")
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            self._log_error(400, str(exc))
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)
            self._log_error(500, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress BaseHTTPRequestHandler access logs such as:
        # "POST /step HTTP/1.1" 200. The simulation logs below carry the
        # per-frame state and error context that are useful while debugging.
        return

    def _log_snapshot(self, event: str, snapshot: dict[str, Any]) -> None:
        state = _as_mapping(snapshot.get("state"))
        control = _as_mapping(snapshot.get("control"))
        aero = _as_mapping(snapshot.get("aero"))
        elapsed = _safe_float(snapshot.get("elapsedS"))
        line = (
            "[simulation-server] "
            f"{event} "
            f"t={elapsed:.3f}s "
            f"lat={_safe_float(state.get('lat')):.7f} "
            f"lon={_safe_float(state.get('lon')):.7f} "
            f"alt={_safe_float(state.get('altM')):.2f}m "
            f"speed={_safe_float(state.get('speedMps')):.2f}m/s "
            f"heading={_safe_float(state.get('headingDeg')):.2f}deg "
            f"fpa={_safe_float(state.get('flightPathDeg')):.2f}deg "
            f"mass={_safe_float(state.get('massKg')):.1f}kg "
            f"thrust={_safe_float(control.get('thrustN')):.1f}N "
            f"bank={_safe_float(control.get('bankDeg')):.2f}deg "
            f"alpha={_safe_float(control.get('attackDeg')):.2f}deg "
            f"Cl={_safe_float(aero.get('liftCoefficient')):.4f} "
            f"Cd={_safe_float(aero.get('dragCoefficient')):.4f}"
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
            "[simulation-server] "
            f"error status={status} method={self.command} path={self.path} "
            f"message={message}\n"
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

def make_request_handler(session: SimulationSession | None = None) -> type[SimulationRequestHandler]:
    class BoundSimulationRequestHandler(SimulationRequestHandler):
        pass

    BoundSimulationRequestHandler.session = session or SimulationSession()
    return BoundSimulationRequestHandler

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AeroViz aerodynamic simulation server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    http_server = ThreadingHTTPServer((args.host, args.port), make_request_handler())
    print(f"Simulation server listening on http://{args.host}:{args.port}", flush=True)
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http_server.server_close()

if __name__ == "__main__":
    main()
