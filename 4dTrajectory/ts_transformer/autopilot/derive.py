"""Method A (executor design §9, the E7 plan): the mildest values the design's constraints allow, found by
flying the executor's own manoeuvre rather than read from data.

- ``heading_time_constant_s`` (τ_ψ): the largest the per-step word envelope admits (``r_turn |τ_ψ − L| ≤ δψ − s/2``,
  `params.ExecutorParams.check`; down to 0.5 s) — the gentlest roll-out whose steady lag in a turn still keeps the
  flown track inside the word a lead earlier;
- ``roll_rate_deg_s`` (p): the least bank rate (a `ROLL_RATE_STEP_DEG_S` grid) at which the executor's own
  largest turn on its heading law (`largest_own_turn_deg`: from a heading word that just reaches the final — 90°
  plus the tolerance off the course — to its own intercept of the final), level and at a held speed, passes its
  target by no more than the heading tolerance at every speed of `ROLL_CHECK_SPEEDS_MPS` (`turn_overshoot_deg`
  simulates it on an A320 at 1500 m). Heading words themselves turn a step or two at a time.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from aircraft.aero_params import aero_params_for_aircraft
from flight_scenarios.scenario import aircraft_for_code
from geokit import metres_per_deg_lon
from ts_transformer.autopilot import inverse
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import AirportCharts, read_state, wrap180
from ts_transformer.autopilot.lateral import heading_rate
from ts_transformer.autopilot.measure import rounded
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import Plant
from ts_transformer.instructions.spec import VocabularySpec

ROLL_CHECK_SPEEDS_MPS = (60.0, 80.0, 100.0, 120.0, 140.0)
ROLL_RATE_STEP_DEG_S = 0.5
ROLL_RATE_MAX_DEG_S = 20.0


def heading_time_constant_s(turn_rate_deg_s: float, spec: VocabularySpec, cycle_s: float) -> float:
    """The largest τ_ψ the per-step word envelope admits (``r_turn |τ_ψ − L| ≤ δψ − s/2``), down to 0.5 s."""
    slack_s = (spec.heading_tolerance_deg - spec.heading_step_deg / 2) / turn_rate_deg_s
    tau = rounded(spec.heading_lead_s + slack_s, 0.5, math.floor)
    if tau < 2.0 * cycle_s:
        raise ValueError(f"τ_ψ {tau:g} s would be under 2 Δt: the lead {spec.heading_lead_s:g} s is too short")
    return tau


def largest_own_turn_deg(spec: VocabularySpec) -> float:
    """The largest turn the executor's heading law flies on its own: from a heading word that just reaches the final
    (`envelope.heading_converges`: 90° plus the tolerance off the course) to its own intercept of it."""
    return 90.0 + spec.heading_tolerance_deg + spec.intercept_angle_deg


def _a320_level(speed_mps: float) -> tuple[FlightInputs, AirportCharts]:
    """A level A320 at 1500 m flying north (method A's test flight), and a chart to read it in."""
    aircraft = aircraft_for_code("A320")
    aero = aero_params_for_aircraft(aircraft)
    f64 = torch.float64
    inputs = FlightInputs(
        initial_state=torch.tensor([[35.0, -78.0, 1500.0, speed_mps, math.pi / 2.0, 0.0, 62000.0]], dtype=f64),
        aero_params=torch.tensor([[aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall]], dtype=f64),
        frame_params=torch.tensor([[35.0, -78.0, 100.0, 0.0]], dtype=f64),
        max_thrust_n=torch.tensor([aircraft.engine.max_thrust_total_n], dtype=f64))
    charts = AirportCharts(lat0_deg=torch.tensor([35.0], dtype=f64), lon0_deg=torch.tensor([-78.0], dtype=f64),
                           m_per_deg_lon=torch.tensor([metres_per_deg_lon(35.0)], dtype=f64))
    return inputs, charts


def turn_overshoot_deg(params: ExecutorParams, speed_mps: float, turn_deg: float) -> float:
    """How far past the target the executor's own heading law turns (a right turn of ``turn_deg`` from
    wings level, level flight, speed held), degrees."""
    inputs, charts = _a320_level(speed_mps)
    plant = Plant(inputs)
    state, bank = inputs.initial_state, torch.zeros(1, dtype=torch.float64)
    target = torch.tensor([turn_deg], dtype=torch.float64)
    furthest = 0.0
    for _cycle in range(int(4 * turn_deg / params.turn_rate_deg_s + 60)):
        now = read_state(state, charts)
        attitude = inverse.attitude(now, heading_rate(now.track_deg, target, params), torch.zeros(1, dtype=torch.float64),
                                    bank, bank_cap_rad=math.radians(params.bank_cap_deg),
                                    bank_rate_rad_s=math.radians(params.bank_rate_deg_s), cycle_s=params.cycle_s)
        thrust = inverse.thrust(now, torch.zeros(1, dtype=torch.float64), attitude.load_factor, inputs.aero_params,
                                inputs.max_thrust_n)
        bank = attitude.bank_rad
        state = plant.step(state, torch.stack((thrust.fraction, bank, attitude.load_factor), dim=1), params.cycle_s)
        furthest = max(furthest, float(-wrap180(target - read_state(state, charts).track_deg)))
    return furthest


def roll_rate_deg_s(params: ExecutorParams, spec: VocabularySpec) -> tuple[float, dict[str, float]]:
    """The least bank rate that keeps the executor's own largest turn inside the heading tolerance (module
    docstring); returns it with the overshoot at each speed."""
    rate = ROLL_RATE_STEP_DEG_S
    while rate <= ROLL_RATE_MAX_DEG_S:
        trial = replace(params, bank_rate_deg_s=rate)
        overshoot = {f"{speed:g}": turn_overshoot_deg(trial, speed, largest_own_turn_deg(spec))
                     for speed in ROLL_CHECK_SPEEDS_MPS}
        if max(overshoot.values()) <= spec.heading_tolerance_deg:
            return rate, overshoot
        rate += ROLL_RATE_STEP_DEG_S
    raise ValueError(f"no bank rate up to {ROLL_RATE_MAX_DEG_S:g}°/s keeps the executor's turn inside the "
                     "heading tolerance")
