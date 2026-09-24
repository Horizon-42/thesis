"""Method A (executor design §9, the E7 plan): the mildest values the design's constraints allow, found by
flying the executor's own manoeuvre rather than read from data.

- ``heading_time_constant_s`` (τ_ψ): the ease-out of the executor's OWN turns (its intercept, a go-around, the capture,
  the line) — the heading words have their own law, which arrives on each word a lead after it is heard
  (`lateral.word_rate`). Chosen, not derived: the lead L, the time a word gives to arrive, so the executor's own turns
  settle on the scale the words do (at least 2Δt, `params.ExecutorParams.check`);
- ``roll_rate_deg_s`` (p): the least bank rate (a `ROLL_RATE_STEP_DEG_S` grid) at which, at every speed of
  `ROLL_CHECK_SPEEDS_MPS` (A320, level, at 1500 m, speed held), (1) the executor's own largest turn on its heading law
  (`largest_own_turn_deg`: from a heading word that just reaches the final — 90° plus the tolerance off the course —
  to its own intercept of the final) passes its target by no more than the heading tolerance (`turn_overshoot_deg`),
  and (2) a typical turn said word by word is flown inside every word's envelope (`follow_excess_deg`): a
  `FOLLOW_TURN_DEG` turn at the steady rate r_turn (or what the bank cap allows at the speed), read the way the
  labeller reads an observed track — the data plane's centred velocity fit and the labeller's track average, taken
  here as two centred moving averages of the heading (an approximation: the fit is a least-squares line over
  positions; the review of 2026-09-24 found a least-squares weighting moves no word at 5° cells) — then
  `labeller.lateral.per_step_words` at the lead; flown on the word law (each word heard on the cycle that starts its
  row) and read as the judge reads a flown track (the track average alone). The turn is an ideal one — the observed
  aircraft at its rate at once — so (2) prices the executor's roll-in and its roll-out before each word (the word
  law's stopping limit, `lateral.stopping_rate_deg_s`). Turns faster than r_turn up to the bank cap, and a word of two
  grid steps out of straight flight, are left to the train replay and the sensitivity: an ideal turn at the bank cap
  cannot be caught up by any roll rate (the executor is at the cap too), and the labeller's smoothed track never
  steps two cells from straight flight.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from aircraft.aero_params import aero_params_for_aircraft
from flight_scenarios.scenario import aircraft_for_code
from flight_scenarios.start_state import DEFAULT_WINDOW_S as VELOCITY_FIT_WINDOW_S
from geokit import metres_per_deg_lon
from ts_transformer.autopilot import inverse
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import AirportCharts, read_state, wrap180
from ts_transformer.autopilot.lateral import heading_rate, word_rate
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import Plant
from ts_transformer.instructions import envelope
from ts_transformer.instructions.labeller.lateral import per_step_words
from ts_transformer.instructions.piecewise import moving_average
from ts_transformer.instructions.spec import VocabularySpec

ROLL_CHECK_SPEEDS_MPS = (60.0, 80.0, 100.0, 120.0, 140.0)
ROLL_RATE_STEP_DEG_S = 0.5
ROLL_RATE_MAX_DEG_S = 20.0
#: The typical turn the follow check flies, and the straight flight before and after it (seconds).
FOLLOW_TURN_DEG = 90.0
FOLLOW_STRAIGHT_S = 40.0


def heading_time_constant_s(spec: VocabularySpec, cycle_s: float) -> float:
    """τ_ψ, the executor's own turns' ease-out: the lead, the time a heading word gives to arrive (module docstring);
    at least 2Δt."""
    if spec.heading_lead_s < 2.0 * cycle_s:
        raise ValueError(f"τ_ψ {spec.heading_lead_s:g} s would be under 2 Δt: the lead is too short")
    return spec.heading_lead_s


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


def as_labelled(heading_per_s: np.ndarray, spec: VocabularySpec) -> np.ndarray:
    """An observed heading sampled once a second, read the way the labeller reads a track (module docstring): the
    velocity fit as a centred moving average, the 2 s grid, the labeller's track average. An approximation, stated
    there."""
    fitted = moving_average(heading_per_s, int(round(VELOCITY_FIT_WINDOW_S)))
    return moving_average(fitted[:: int(round(spec.step_s))], spec.rows(spec.track_smoothing_s))


def turn_words(spec: VocabularySpec, rate_deg_s: float) -> tuple[list[tuple[int, float]], int]:
    """The words the labeller reads off a `FOLLOW_TURN_DEG` right turn at ``rate_deg_s`` between two straight legs,
    from compass 0°, and the sentence's rows."""
    seconds = np.arange(0.0, 2 * FOLLOW_STRAIGHT_S + FOLLOW_TURN_DEG / rate_deg_s, 1.0)
    observed = as_labelled(np.clip((seconds - FOLLOW_STRAIGHT_S) * rate_deg_s, 0.0, FOLLOW_TURN_DEG), spec)
    return per_step_words(observed, len(observed) - 1, spec.heading_step_deg, spec.rows_exact(spec.heading_lead_s)), \
        len(observed)


def fly_words(params: ExecutorParams, spec: VocabularySpec, speed_mps: float, said: list[tuple[int, float]],
              rows: int) -> np.ndarray:
    """A level A320 at ``speed_mps`` flying heading words ``(row, target)`` on the executor's word law, each heard on
    the cycle that starts its row; the flown track at the sentence's rows as the judge reads it (the labeller's track
    average, `flown_track` → `read.smooth`), unwrapped from compass 0°."""
    inputs, charts = _a320_level(speed_mps)
    plant = Plant(inputs)
    state, bank = inputs.initial_state, torch.zeros(1, dtype=torch.float64)
    step_rows = int(round(spec.step_s / params.cycle_s))
    flown, heard = [], 0
    for cycle in range(rows * step_rows):
        now = read_state(state, charts)
        if cycle % step_rows == 0:
            flown.append(float(now.track_deg[0]))
            while heard + 1 < len(said) and said[heard + 1][0] <= cycle // step_rows:
                heard += 1
        time_s = cycle * params.cycle_s
        error = wrap180(torch.tensor([said[heard][1]], dtype=torch.float64) - now.track_deg)
        to_go = torch.tensor([said[heard][0] * spec.step_s + spec.heading_lead_s - time_s], dtype=torch.float64)
        attitude = inverse.attitude(now, word_rate(error, to_go, now.ground_speed_mps, params, spec),
                                    torch.zeros(1, dtype=torch.float64), bank,
                                    bank_cap_rad=math.radians(params.bank_cap_deg),
                                    bank_rate_rad_s=math.radians(params.bank_rate_deg_s), cycle_s=params.cycle_s)
        thrust = inverse.thrust(now, torch.zeros(1, dtype=torch.float64), attitude.load_factor, inputs.aero_params,
                                inputs.max_thrust_n)
        bank = attitude.bank_rad
        state = plant.step(state, torch.stack((thrust.fraction, bank, attitude.load_factor), dim=1), params.cycle_s)
    return moving_average(np.unwrap(np.asarray(flown), period=360.0), spec.rows(spec.track_smoothing_s))


def words_excess_deg(spec: VocabularySpec, said: list[tuple[int, float]], flown: np.ndarray) -> float:
    """How far past the heading tolerance ``flown`` lies at worst over every word's judged rows
    (`envelope.heading_word_rows`), degrees; at most 0 when every row is inside."""
    rows = envelope.heading_word_rows([row for row, _ in said], spec.rows_exact(spec.heading_lead_s), len(flown))
    return max(float(np.abs(flown[first:stop] - target).max()) - spec.heading_tolerance_deg
               for (_, target), (first, stop) in zip(said, rows) if stop > first)


def follow_excess_deg(params: ExecutorParams, spec: VocabularySpec, speed_mps: float) -> float:
    """The worst excess past the words' envelopes at ``speed_mps`` of a typical turn said word by word (module
    docstring): at the steady rate r_turn, or what the bank cap allows at this speed."""
    rate = min(params.turn_rate_deg_s,
               math.degrees(GRAVITY_MPS2 * math.tan(math.radians(params.bank_cap_deg)) / speed_mps))
    said, rows = turn_words(spec, rate)
    return words_excess_deg(spec, said, fly_words(params, spec, speed_mps, said, rows))


def roll_rate_deg_s(params: ExecutorParams, spec: VocabularySpec) -> tuple[float, dict[str, dict[str, float]]]:
    """The least bank rate that keeps the executor's own largest turn inside the heading tolerance and follows a
    typical turn's words inside their envelopes (module docstring); returns it with both measures at each speed."""
    rate = ROLL_RATE_STEP_DEG_S
    while rate <= ROLL_RATE_MAX_DEG_S:
        trial = replace(params, bank_rate_deg_s=rate)
        overshoot = {f"{speed:g}": turn_overshoot_deg(trial, speed, largest_own_turn_deg(spec))
                     for speed in ROLL_CHECK_SPEEDS_MPS}
        excess = {f"{speed:g}": follow_excess_deg(trial, spec, speed) for speed in ROLL_CHECK_SPEEDS_MPS}
        if max(overshoot.values()) <= spec.heading_tolerance_deg and max(excess.values()) <= 0.0:
            return rate, {"overshoot_deg": overshoot, "follow_excess_deg": excess}
        rate += ROLL_RATE_STEP_DEG_S
    raise ValueError(f"no bank rate up to {ROLL_RATE_MAX_DEG_S:g}°/s keeps the executor's own turn inside the heading "
                     "tolerance and a typical turn's words inside their envelopes")
