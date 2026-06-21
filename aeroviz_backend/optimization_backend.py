from __future__ import annotations

from typing import Any

from aeroviz_backend import paths  # noqa: F401
from aeroviz_backend.simulation_backend import (
    DEFAULT_AIRCRAFT_TYPE,
    DEFAULT_DT,
    DEFAULT_STATE,
    MAX_DT,
    clamp,
    format_control,
    format_geodetic_state,
    read_aircraft,
    read_float,
    read_geodetic_state,
    read_positive_int,
    read_required_mapping,
)

from geodetic_simulator import GeodeticSimulator, GeodeticState
from casadi_optimizer import CasadiOptimizer
from casadi_direct_collocation_optimizer import CasadiDirectCollocationOptimizer
from common import LoadFactorControl
from least_squares_transcription_optimizor import LeastSquaresTranscriptionOptimizor
from single_shooting_optimizor import SingleShootingOptimizor
from simulator import Control
from transcription_optimizor import TranscriptionOptimizor
from variable_time_warm_start_transcription_optimizor import (
    VariableTimeWarmStartTranscriptionOptimizor,
)
from warm_start_transcription_optimizor import WarmStartTranscriptionOptimizor


DEFAULT_N_SEGMENTS = 10
DEFAULT_MAX_ITERATIONS = 1000
MIN_ARRIVAL_TIME_S = 1.0
MAX_ARRIVAL_TIME_S = 1000.0
MIN_OPTIMIZATION_DT = 0.001
DEFAULT_OPTIMIZER = "casadiDirectCollocation"
SUPPORTED_OPTIMIZERS = (
    "casadiDirectCollocation",
    "casadiIpopt",
    "transcription",
    "leastSquaresTranscription",
    "warmStartTranscription",
    "variableTimeWarmStartTranscription",
    "singleShooting",
)
CASADI_OPTIMIZERS = ("casadiIpopt", "casadiDirectCollocation")


class OptimizationBackend:
    def __init__(self) -> None:
        self._casadi_optimizer_key = None
        self._casadi_optimizer = None

    def optimize(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        initial_payload = read_required_mapping(payload, "initialState")
        target_payload = read_required_mapping(payload, "targetState")
        n_segments = read_positive_int(payload, "nSegments", DEFAULT_N_SEGMENTS)
        max_iterations = read_positive_int(
            payload,
            "maxIterations",
            DEFAULT_MAX_ITERATIONS,
        )
        optimizer_name = read_optimizer(payload)
        arrival_time_s = read_arrival_time_s(payload)
        dt = clamp(read_float(payload, "dtS", DEFAULT_DT), MIN_OPTIMIZATION_DT, MAX_DT)
        aircraft = read_aircraft(initial_payload, DEFAULT_AIRCRAFT_TYPE)

        initial_state = read_geodetic_state(initial_payload, DEFAULT_STATE, aircraft)
        target_state = read_geodetic_state(target_payload, initial_state, aircraft)
        optimizer = self.make_optimizer(
            optimizer_name,
            GeodeticSimulator(aircraft),
            n_segments,
            dt,
            max_iterations,
            arrival_time_s=arrival_time_s,
        )
        if optimizer_name == "casadiDirectCollocation":
            # Direct collocation includes T as a decision variable, so
            # one solve returns both the optimal trajectory and the
            # optimal arrival time -- no outer bisection needed.
            final_time, node_control, node_state = optimizer.optimize_free_time(
                initial_state,
                target_state,
                arrival_time_s,
            )
        elif optimizer_name == "casadiIpopt":
            # Multiple shooting uses a fixed-time NLP; finding the
            # shortest feasible duration still requires bisecting on T.
            final_time, node_control, node_state = optimizer.optimize_time_to_target(
                initial_state,
                target_state,
                arrival_time_s,
            )
        else:
            final_time, node_control, node_state = optimizer.optimize_trajectory(
                initial_state,
                target_state,
            )

        return {
            "ok": True,
            "finalTimeS": float(final_time),
            "nSegments": n_segments,
            "dtS": dt,
            "optimizer": optimizer_name,
            "controls": [
                format_optimizer_control(optimizer_name, control_values)
                for control_values in node_control
            ],
            "states": format_node_states(node_state, initial_state.m, aircraft.code),
        }

    def make_optimizer(
        self,
        optimizer_name: str,
        geodetic_simulator: GeodeticSimulator,
        n_segments: int,
        dt: float,
        max_iterations: int,
        arrival_time_s: float,
    ) -> Any:
        if optimizer_name in CASADI_OPTIMIZERS:
            # CasADi optimisers recompile a (potentially large) NLP at
            # construction.  Caching one instance per (aircraft, mesh,
            # dt, arrival_time, optimizer_name) avoids the cost on every
            # request.  ``optimizer_name`` is included in the key
            # because the multiple-shooting and direct-collocation
            # solvers have different NLP layouts.
            aircraft = geodetic_simulator.simulator.aircraft
            key = (optimizer_name, aircraft.code, n_segments, dt, arrival_time_s)
            if self._casadi_optimizer_key != key:
                self._casadi_optimizer_key = key
                self._casadi_optimizer = make_optimizer(
                    optimizer_name,
                    geodetic_simulator,
                    n_segments,
                    dt,
                    max_iterations,
                    arrival_time_s,
                )
            return self._casadi_optimizer

        return make_optimizer(
            optimizer_name,
            geodetic_simulator,
            n_segments,
            dt,
            max_iterations,
            arrival_time_s,
        )


def read_optimizer(payload: dict[str, Any]) -> str:
    value = payload.get("optimizer", DEFAULT_OPTIMIZER)
    if not isinstance(value, str):
        raise ValueError("optimizer must be a string")

    optimizer = value.strip()
    if optimizer not in SUPPORTED_OPTIMIZERS:
        valid_values = ", ".join(SUPPORTED_OPTIMIZERS)
        raise ValueError(f"optimizer must be one of {valid_values}")
    return optimizer


def make_optimizer(
    optimizer_name: str,
    geodetic_simulator: GeodeticSimulator,
    n_segments: int,
    dt: float,
    max_iterations: int,
    arrival_time_s: float,
) -> Any:
    if optimizer_name == "casadiIpopt":
        return CasadiOptimizer(
            n_segments=n_segments,
            dt=dt,
            max_duration=arrival_time_s,
            aircraft=geodetic_simulator.simulator.aircraft,
        )

    if optimizer_name == "casadiDirectCollocation":
        return CasadiDirectCollocationOptimizer(
            n_segments=n_segments,
            dt=dt,
            max_duration=arrival_time_s,
            aircraft=geodetic_simulator.simulator.aircraft,
        )

    if optimizer_name == "singleShooting":
        return SingleShootingOptimizor(
            geodetic_simulator,
            n_control_segments=n_segments,
            dt=dt,
            max_iterations=max_iterations,
        )

    if optimizer_name == "leastSquaresTranscription":
        return LeastSquaresTranscriptionOptimizor(
            geodetic_simulator,
            n_segments=n_segments,
            dt=dt,
            arrival_time_s=arrival_time_s,
            max_iterations=max_iterations,
        )

    if optimizer_name == "warmStartTranscription":
        return WarmStartTranscriptionOptimizor(
            geodetic_simulator,
            n_segments=n_segments,
            dt=dt,
            arrival_time_s=arrival_time_s,
            max_iterations=max_iterations,
        )

    if optimizer_name == "variableTimeWarmStartTranscription":
        return VariableTimeWarmStartTranscriptionOptimizor(
            geodetic_simulator,
            n_segments=n_segments,
            dt=dt,
            arrival_time_s=arrival_time_s,
            max_iterations=max_iterations,
        )

    return TranscriptionOptimizor(
        geodetic_simulator,
        n_segments=n_segments,
        dt=dt,
        arrival_time_s=arrival_time_s,
        max_iterations=max_iterations,
    )


def format_optimizer_control(
    optimizer_name: str,
    control_values: Any,
) -> dict[str, float]:
    # Both CasADi optimisers use the load-factor parameterisation
    # (T, mu, n_cmd); the alpha-based optimisers use (T, mu, alpha).
    if optimizer_name in CASADI_OPTIMIZERS:
        return format_control(LoadFactorControl(*control_values))
    return format_control(Control(*control_values))


def read_arrival_time_s(payload: dict[str, Any]) -> float:
    if "arrivalTimeS" not in payload:
        raise ValueError("arrivalTimeS must be a number")
    arrival_time_s = read_float(payload, "arrivalTimeS", 0.0)
    if not MIN_ARRIVAL_TIME_S <= arrival_time_s <= MAX_ARRIVAL_TIME_S:
        raise ValueError(
            f"arrivalTimeS must be between {MIN_ARRIVAL_TIME_S} and "
            f"{MAX_ARRIVAL_TIME_S}"
        )
    return arrival_time_s


def format_node_states(
    node_state: Any,
    mass: float,
    aircraft_code: str,
) -> list[dict[str, Any]]:
    if node_state is None:
        return []
    return [
        format_geodetic_state(
            _array_to_geodetic_state(state_values, mass),
            aircraft_code,
        )
        for state_values in node_state
    ]


def _array_to_geodetic_state(values: Any, mass: float) -> GeodeticState:
    latitude, longitude, altitude, V, psi, gamma = [float(value) for value in values]
    return GeodeticState(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        V=V,
        psi=psi,
        gamma=gamma,
        m=mass,
    )
