from __future__ import annotations

import math
from typing import Any

from aeroviz_backend import paths  # noqa: F401

from aircraft_sets import AIRCRAFT_PRESETS, AircraftSpec, A320
from geodetic_simulator import GeodeticSimulator, GeodeticState
from simulator import Control


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

DEFAULT_DT = 0.2
MAX_DT = 2.0


class SimulationBackend:
    def __init__(self) -> None:
        self.geodetic_simulator = GeodeticSimulator()
        self.state = DEFAULT_STATE
        self.control = DEFAULT_CONTROL
        self.elapsed = 0.0

    def reset(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        state_payload = read_optional_mapping(payload, "state")
        control_payload = read_optional_mapping(payload, "control")
        aircraft = read_aircraft(state_payload, DEFAULT_AIRCRAFT_TYPE)

        self.geodetic_simulator = GeodeticSimulator(aircraft)
        self.state = read_geodetic_state(state_payload, DEFAULT_STATE, aircraft)
        self.control = read_control(
            control_payload,
            default_control_for_aircraft(aircraft),
            aircraft,
        )
        self.elapsed = 0.0
        return self.snapshot()

    def step(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        aircraft = self.geodetic_simulator.simulator.aircraft
        self.control = read_control(
            read_optional_mapping(payload, "control"),
            self.control,
            aircraft,
        )
        dt = clamp(read_float(payload, "dtS", DEFAULT_DT), 0.001, MAX_DT)
        self.state = self.geodetic_simulator.step(self.state, self.control, dt)
        self.elapsed += dt
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        Cl, Cd = self.geodetic_simulator.simulator.get_aerodynamic_coefficients(
            self.control.attack_rad,
        )
        return {
            "ok": True,
            "elapsedS": self.elapsed,
            "state": format_geodetic_state(
                self.state,
                self.geodetic_simulator.simulator.aircraft.code,
            ),
            "control": format_control(self.control),
            "aero": {
                "liftCoefficient": Cl,
                "dragCoefficient": Cd,
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


def default_control_for_aircraft(aircraft: AircraftSpec) -> Control:
    return Control(
        thrust=aircraft.approach_thrust_guess_n,
        bank_rad=DEFAULT_CONTROL.bank_rad,
        attack_rad=DEFAULT_CONTROL.attack_rad,
    )


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


def format_control(control: Control) -> dict[str, float]:
    return {
        "thrustN": control.thrust,
        "bankDeg": math.degrees(control.bank_rad),
        "attackDeg": math.degrees(control.attack_rad),
    }


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
