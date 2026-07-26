"""flight_scenarios — build neutral modeling inputs from observed trajectory data.

This package is the **seam between the data plane and the modeling plane**. It reads a
manifest-rostered, model-ready harvest arrival and an aircraft identity, and produces a
serializable :class:`FlightScenario`:

    FlightScenario = initial GeodeticState  (lat, lon, alt, V, psi, gamma, m)
                   + aircraft Aircraft
                   + AeroParams
                   + source metadata

The same record feeds **both consumers**, neither of which it depends on:

    arrivals manifest ─► flight_scenarios ─► 4dTrajectory/optimization (problem instance)
                              │            └► ts_transformer            (training example)
                            ▼
              aircraft · aerodynamic_model · geokit

See ``flight_scenarios/README.md`` for the architecture and the start-state math.
"""

from __future__ import annotations

from .build import build_scenario, build_scenarios_from_arrivals, load_model_arrivals
from .datum import flights_to_msl, geoid_undulation_m
from .fitted_approach import FittedApproach, TimedFittedPoint, fit_flight_final_approach
from .identity import flight_key
from .runway_target import threshold_target_state
from .scenario import FlightScenario, aircraft_for_code, load_scenarios, save_scenarios
from .start_state import (
    final_state_from_track,
    initial_state_from_track,
    state_samples_from_track,
)

__all__ = [
    "FlightScenario",
    "build_scenario",
    "build_scenarios_from_arrivals",
    "load_model_arrivals",
    "flights_to_msl",
    "geoid_undulation_m",
    "FittedApproach",
    "TimedFittedPoint",
    "fit_flight_final_approach",
    "initial_state_from_track",
    "final_state_from_track",
    "state_samples_from_track",
    "threshold_target_state",
    "aircraft_for_code",
    "flight_key",
    "save_scenarios",
    "load_scenarios",
]
