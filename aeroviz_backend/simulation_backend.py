from __future__ import annotations

import math
from typing import Any

from aeroviz_backend import paths  # noqa: F401

from aerodynamic_model.casadi_simulator import CasadiSimulator
from aircraft.aircraft_sets import AIRCRAFT_PRESETS, AircraftSpec, A320
from common import Atmosphere, Control, GeodeticState, LoadFactorControl
from geodetic_simulator import GeodeticSimulator
from simulator_simple import LoadFactorSimulator


DEFAULT_STATE = GeodeticState(
    latitude=35.878659,
    longitude=-78.7873,
    altitude=1000.0,
    V=120.0,
    psi=0.0,
    gamma=0.0,
    m=A320.mass_kg,
)

DEFAULT_CONTROL = Control(
    thrust=A320.approach_thrust_guess_n,
    bank_rad=0.0,
    attack_rad=0.0,
)
DEFAULT_AIRCRAFT_TYPE = A320.code
DEFAULT_SIMULATION_MODE = "alpha"
SUPPORTED_SIMULATION_MODES = ("alpha", "loadFactor", "casadi")

DEFAULT_DT = 0.2
MAX_DT = 2.0
MIN_LOAD_FACTOR = 0.0
MAX_LOAD_FACTOR = 3.0


class CasadiGeodeticSimulator:
    def __init__(self, aircraft: AircraftSpec) -> None:
        self.simulator = CasadiSimulator(aircraft, DEFAULT_DT)

    def step(
        self,
        state: GeodeticState,
        control: LoadFactorControl,
        dt: float,
    ) -> GeodeticState:
        next_state = self.simulator.step(state, control, dt)
        if next_state.altitude < 0.0:
            raise ValueError("simulation stopped: altitude below 0")
        return GeodeticState(
            latitude=next_state.latitude,
            longitude=next_state.longitude,
            altitude=next_state.altitude,
            V=next_state.V,
            psi=next_state.psi,
            gamma=next_state.gamma,
            m=next_state.m,
        )


class SimulationBackend:
    def __init__(self) -> None:
        self.geodetic_simulator = GeodeticSimulator()
        self.simulation_mode = DEFAULT_SIMULATION_MODE
        self.state = DEFAULT_STATE
        self.control = DEFAULT_CONTROL
        self.elapsed = 0.0

    def reset(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        state_payload = read_optional_mapping(payload, "state")
        control_payload = read_optional_mapping(payload, "control")
        simulation_mode = read_simulation_mode(payload, DEFAULT_SIMULATION_MODE)
        aircraft = read_aircraft(state_payload, DEFAULT_AIRCRAFT_TYPE)

        self.simulation_mode = simulation_mode
        self.geodetic_simulator = make_geodetic_simulator(aircraft, simulation_mode)
        self.state = read_geodetic_state(state_payload, DEFAULT_STATE, aircraft)
        self.control = read_simulation_control(
            control_payload,
            aircraft,
            simulation_mode,
            default_control_for_mode(aircraft, simulation_mode),
        )
        self.elapsed = 0.0
        return self.snapshot()

    def step(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        aircraft = self.geodetic_simulator.simulator.aircraft
        simulation_mode = read_simulation_mode(payload, self.simulation_mode)
        if simulation_mode != self.simulation_mode:
            self.simulation_mode = simulation_mode
            self.geodetic_simulator = make_geodetic_simulator(aircraft, simulation_mode)

        self.control = read_simulation_control(
            read_optional_mapping(payload, "control"),
            aircraft,
            simulation_mode,
            fallback_control_for_mode(self.control, aircraft, simulation_mode),
        )
        dt = clamp(read_float(payload, "dtS", DEFAULT_DT), 0.001, MAX_DT)
        integrator_dt = clamp(
            read_float(payload, "integratorDtS", dt),
            0.001,
            MAX_DT,
        )

        remaining_dt = dt
        while remaining_dt > 1e-12:
            step_dt = min(integrator_dt, remaining_dt)
            self.state = self.geodetic_simulator.step(
                self.state,
                self.control,
                step_dt,
            )
            remaining_dt -= step_dt
        self.elapsed += dt
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        Cl, Cd, actual_load_factor = read_snapshot_aero(
            self.geodetic_simulator,
            self.state,
            self.control,
            self.simulation_mode,
        )
        return {
            "ok": True,
            "elapsedS": self.elapsed,
            "simulationMode": self.simulation_mode,
            "state": format_geodetic_state(
                self.state,
                self.geodetic_simulator.simulator.aircraft.code,
            ),
            "control": format_control(self.control),
            "aero": {
                "liftCoefficient": Cl,
                "dragCoefficient": Cd,
                "actualLoadFactor": actual_load_factor,
            },
        }


def read_optional_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if isinstance(value, dict):
        return value
    raise ValueError(f"{key} must be an object")


def read_required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    raise ValueError(f"{key} must be an object")


def read_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def read_positive_int(payload: dict[str, Any], key: str, default: int) -> int:
    number = read_float(payload, key, float(default))
    if not number.is_integer() or number < 1:
        raise ValueError(f"{key} must be a positive integer")
    return int(number)


def read_simulation_mode(payload: dict[str, Any], default: str) -> str:
    value = payload.get("simulationMode", default)
    if not isinstance(value, str):
        raise ValueError("simulationMode must be a string")

    mode = value.strip()
    if mode not in SUPPORTED_SIMULATION_MODES:
        valid_values = ", ".join(SUPPORTED_SIMULATION_MODES)
        raise ValueError(f"simulationMode must be one of {valid_values}")
    return mode


def read_simulation_control(
    payload: dict[str, Any],
    aircraft: AircraftSpec,
    simulation_mode: str,
    fallback: Control | LoadFactorControl,
) -> Control | LoadFactorControl:
    if simulation_mode in ("loadFactor", "casadi"):
        return read_load_factor_control(
            payload,
            (
                fallback
                if isinstance(fallback, LoadFactorControl)
                else default_load_factor_control_for_aircraft(aircraft)
            ),
            aircraft,
        )
    return read_control(
        payload,
        fallback
        if isinstance(fallback, Control)
        else default_control_for_aircraft(aircraft),
        aircraft,
    )


def read_control(
    payload: dict[str, Any],
    fallback: Control,
    aircraft: AircraftSpec,
) -> Control:
    return Control(
        thrust=clamp(
            read_float(payload, "thrustN", fallback.thrust),
            0.0,
            aircraft.max_thrust_n,
        ),
        bank_rad=math.radians(
            clamp(
                read_float(payload, "bankDeg", math.degrees(fallback.bank_rad)),
                -60.0,
                60.0,
            )
        ),
        attack_rad=math.radians(
            clamp(
                read_float(payload, "attackDeg", math.degrees(fallback.attack_rad)),
                -10.0,
                18.0,
            )
        ),
    )


def read_load_factor_control(
    payload: dict[str, Any],
    fallback: LoadFactorControl,
    aircraft: AircraftSpec,
) -> LoadFactorControl:
    return LoadFactorControl(
        thrust=clamp(
            read_float(payload, "thrustN", fallback.thrust),
            0.0,
            aircraft.max_thrust_n,
        ),
        bank_rad=math.radians(
            clamp(
                read_float(payload, "bankDeg", math.degrees(fallback.bank_rad)),
                -60.0,
                60.0,
            )
        ),
        load_factor=clamp(
            read_float(payload, "loadFactor", fallback.load_factor),
            MIN_LOAD_FACTOR,
            MAX_LOAD_FACTOR,
        ),
    )


def default_control_for_aircraft(aircraft: AircraftSpec) -> Control:
    return Control(
        thrust=aircraft.approach_thrust_guess_n,
        bank_rad=DEFAULT_CONTROL.bank_rad,
        attack_rad=DEFAULT_CONTROL.attack_rad,
    )


def default_load_factor_control_for_aircraft(aircraft: AircraftSpec) -> LoadFactorControl:
    return LoadFactorControl(
        thrust=aircraft.approach_thrust_guess_n,
        bank_rad=DEFAULT_CONTROL.bank_rad,
        load_factor=1.0,
    )


def default_control_for_mode(
    aircraft: AircraftSpec,
    simulation_mode: str,
) -> Control | LoadFactorControl:
    if simulation_mode in ("loadFactor", "casadi"):
        return default_load_factor_control_for_aircraft(aircraft)
    return default_control_for_aircraft(aircraft)


def fallback_control_for_mode(
    control: Control | LoadFactorControl,
    aircraft: AircraftSpec,
    simulation_mode: str,
) -> Control | LoadFactorControl:
    if simulation_mode in ("loadFactor", "casadi"):
        return (
            control
            if isinstance(control, LoadFactorControl)
            else default_load_factor_control_for_aircraft(aircraft)
        )
    return (
        control
        if isinstance(control, Control)
        else default_control_for_aircraft(aircraft)
    )


def make_geodetic_simulator(
    aircraft: AircraftSpec,
    simulation_mode: str,
) -> GeodeticSimulator | CasadiGeodeticSimulator:
    if simulation_mode == "casadi":
        return CasadiGeodeticSimulator(aircraft)
    if simulation_mode == "loadFactor":
        return GeodeticSimulator(
            aircraft,
            simulator=LoadFactorSimulator(aircraft),
        )
    return GeodeticSimulator(aircraft)


def read_aircraft(payload: dict[str, Any], default_code: str) -> AircraftSpec:
    value = payload.get("aircraftType", default_code)
    if not isinstance(value, str):
        raise ValueError("aircraftType must be a string")

    code = value.strip().upper()
    aircraft = AIRCRAFT_PRESETS.get(code)
    if aircraft is None:
        valid_codes = ", ".join(sorted(AIRCRAFT_PRESETS))
        raise ValueError(f"aircraftType must be one of {valid_codes}")
    return aircraft


def read_geodetic_state(
    payload: dict[str, Any],
    fallback: GeodeticState,
    aircraft: AircraftSpec,
) -> GeodeticState:
    return GeodeticState(
        latitude=read_float(payload, "lat", fallback.latitude),
        longitude=read_float(payload, "lon", fallback.longitude),
        altitude=max(0.0, read_float(payload, "altM", fallback.altitude)),
        V=max(1.0, read_float(payload, "speedMps", fallback.V)),
        psi=math.radians(read_float(payload, "headingDeg", math.degrees(fallback.psi))),
        gamma=math.radians(
            read_float(payload, "flightPathDeg", math.degrees(fallback.gamma))
        ),
        m=aircraft.mass_kg,
    )


def format_geodetic_state(state: GeodeticState, aircraft_type: str) -> dict[str, Any]:
    return {
        "lon": state.longitude,
        "lat": state.latitude,
        "altM": state.altitude,
        "speedMps": state.V,
        "headingDeg": normalize_degrees(math.degrees(state.psi)),
        "flightPathDeg": math.degrees(state.gamma),
        "massKg": state.m,
        "aircraftType": aircraft_type,
    }


def format_control(control: Control | LoadFactorControl) -> dict[str, float]:
    formatted = {
        "thrustN": control.thrust,
        "bankDeg": math.degrees(control.bank_rad),
    }
    if isinstance(control, LoadFactorControl):
        formatted["loadFactor"] = control.load_factor
    else:
        formatted["attackDeg"] = math.degrees(control.attack_rad)
    return formatted


def read_snapshot_aero(
    geodetic_simulator: GeodeticSimulator | CasadiGeodeticSimulator,
    state: GeodeticState,
    control: Control | LoadFactorControl,
    simulation_mode: str,
) -> tuple[float, float, float]:
    simulator = geodetic_simulator.simulator
    if simulation_mode == "casadi" and isinstance(control, LoadFactorControl):
        return casadi_snapshot_aero(simulator, state, control)
    if simulation_mode == "loadFactor" and isinstance(control, LoadFactorControl):
        Cl, Cd, stalled = simulator.get_aerodynamic_coefficients(
            control.load_factor,
            state.V,
            geodetic_simulator.atmosphere,
            state.altitude,
            state.m,
        )
        if stalled:
            rho = geodetic_simulator.atmosphere.get_ISA_density(state.altitude)
            actual_load_factor = (
                0.5 * rho * state.V**2 * simulator.Cl_max * simulator.S
                / (state.m * simulator.g)
            )
        else:
            actual_load_factor = control.load_factor
        return Cl, Cd, actual_load_factor

    Cl, Cd = simulator.get_aerodynamic_coefficients(control.attack_rad)
    rho = geodetic_simulator.atmosphere.get_ISA_density(state.altitude)
    normal_force = (
        0.5 * rho * state.V**2 * Cl * simulator.S
        + control.thrust * math.sin(control.attack_rad)
    )
    return Cl, Cd, normal_force / (state.m * simulator.g)


def casadi_snapshot_aero(
    simulator: CasadiSimulator,
    state: GeodeticState,
    control: LoadFactorControl,
) -> tuple[float, float, float]:
    rho = Atmosphere().get_ISA_density(state.altitude)
    params = simulator.aero_params
    cl_req = control.load_factor * state.m * simulator.g / (
        0.5 * rho * params.S * state.V**2
    )
    cl = min(cl_req, params.Cl_max)
    r = cl_req / params.Cl_max
    cd_stall = 0.0
    if r > params.stall_threshold:
        stall_fraction = min(r, 1.0)
        x = clamp(
            (stall_fraction - params.stall_threshold)
            / (1.0 - params.stall_threshold),
            0.0,
            1.0,
        )
        cd_stall = params.k_stall * x * x * (3.0 - 2.0 * x)
    cd = params.Cd0 + params.k * cl**2 + cd_stall
    actual_load_factor = (
        0.5 * rho * state.V**2 * cl * params.S
        / (state.m * simulator.g)
    )
    return cl, cd, actual_load_factor


def aircraft_catalog() -> dict[str, Any]:
    return {
        "ok": True,
        "aircraft": [
            {
                "code": aircraft.code,
                "name": aircraft.name,
                "category": aircraft.category,
                "massKg": aircraft.mass_kg,
                "wingAreaM2": aircraft.wing_area_m2,
                "maxThrustN": aircraft.max_thrust_n,
                "approachThrustGuessN": aircraft.approach_thrust_guess_n,
                "terminalSpeedKt": aircraft.terminal_speed_kt,
                "terminalSpeedMinKt": aircraft.terminal_speed_min_kt,
                "terminalSpeedMaxKt": aircraft.terminal_speed_max_kt,
                "finalApproachMinNm": aircraft.final_approach_min_nm,
                "finalApproachMaxNm": aircraft.final_approach_max_nm,
                "finalApproachLateralHalfWidthNm": (
                    aircraft.final_approach_lateral_half_width_nm
                ),
                "finalApproachGlideAngleDeg": (
                    aircraft.final_approach_glide_angle_deg
                ),
                "thresholdCrossingHeightM": aircraft.threshold_crossing_height_m,
            }
            for aircraft in AIRCRAFT_PRESETS.values()
        ],
    }


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_degrees(value: float) -> float:
    return value % 360.0
