from __future__ import annotations

from typing import Any

from aeroviz_backend import paths  # noqa: F401
from aeroviz_backend.simulation_backend import (
    DEFAULT_AIRCRAFT_TYPE,
    DEFAULT_STATE,
    format_control,
    format_geodetic_state,
    read_aircraft,
    read_geodetic_state,
    read_positive_int,
    read_required_mapping,
)

from geodetic_simulator import GeodeticSimulator, GeodeticState
from simulator import Control
from transcription_optimizor import TranscriptionOptimizor


DEFAULT_N_SEGMENTS = 10
DEFAULT_MAX_ITERATIONS = 1000


class OptimizationBackend:
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
        aircraft = read_aircraft(initial_payload, DEFAULT_AIRCRAFT_TYPE)

        initial_state = read_geodetic_state(initial_payload, DEFAULT_STATE, aircraft)
        target_state = read_geodetic_state(target_payload, initial_state, aircraft)
        optimizer = TranscriptionOptimizor(
            GeodeticSimulator(aircraft),
            n_segments=n_segments,
            max_iterations=max_iterations,
        )
        final_time, node_control, node_state = optimizer.optimize_trajectory(
            initial_state,
            target_state,
        )

        return {
            "ok": True,
            "finalTimeS": float(final_time),
            "nSegments": n_segments,
            "controls": [
                format_control(Control(*control_values))
                for control_values in node_control
            ],
            "states": [
                format_geodetic_state(
                    _array_to_geodetic_state(state_values),
                    aircraft.code,
                )
                for state_values in node_state
            ],
        }


def _array_to_geodetic_state(values: Any) -> GeodeticState:
    latitude, longitude, altitude, V, psi, gamma, m = [float(value) for value in values]
    return GeodeticState(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        V=V,
        psi=psi,
        gamma=gamma,
        m=m,
    )
