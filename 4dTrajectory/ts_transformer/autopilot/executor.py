"""The executor's cycle (executor design §3, §7): state → words in force → rates → controls → one cycle.

Each cycle, in the design's order of limits:

1. the lateral law gives the track rate, the vertical law the path-angle rate (`autopilot.lateral`,
   `autopilot.vertical`);
2. the inverse sets bank and load factor for them (`autopilot.inverse.attitude`: bank cap and rate,
   the grader's load band);
3. the speed law gives the airspeed rate, above the stall floor at that load factor (`autopilot.speed`);
4. the inverse sets the thrust for it (`autopilot.inverse.thrust`: the thrust box);
5. the plant integrates the cycle (`autopilot.plant`).

The first cycle's bank is not rate-limited: step 0's words describe what the aircraft is already doing,
so a flight entering the slice in a turn keeps turning instead of rolling level first.

A flight is DONE at the end of the cycle in which it passes its pointed runway's threshold plane while
captured, touches the threshold elevation before reaching it, leaves the dynamics (a non-finite state
or no airspeed), or reaches its time limit; the batch stops when every flight is done. What each
outcome means is judged afterwards (`autopilot.judge`), from the recorded states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ts_transformer.autopilot import inverse
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import AirportCharts, read_state
from ts_transformer.autopilot.lateral import Lateral, Runways, relative
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import Plant
from ts_transformer.autopilot.sentence import DistanceClock, Sentences, TimeClock, TrackClock
from ts_transformer.autopilot.speed import Speed
from ts_transformer.autopilot.vertical import Vertical
from ts_transformer.instructions.words import ALTITUDE, ANGLE, HEADING, Words

#: Every limit the cycle can meet (§8.1), in the order they are applied.
LIMITS = ("bank_cap", "bank_rate", "load_factor", "path_rate_limited", "stall_floor", "thrust_max", "thrust_min",
          "stall")
#: What the laws were doing each cycle.
MODES = ("captured", "tracking", "bent", "intercepting", "go_around", "level_captured")


@dataclass(frozen=True)
class Flown:
    """What the batch flew, one row per cycle boundary (``states``: ``T + 1``) or per cycle (the rest)."""

    states: torch.Tensor                    # [B, T+1, 7] geodetic
    commands: torch.Tensor                  # [B, T, 3] thrust fraction, bank (the dynamics' sign), load factor
    wanted: torch.Tensor                    # [B, T, 3] track rate deg/s, path-angle rate rad/s, speed rate m/s²,
                                            # as the laws asked before their own limits (γ̇_max, the stall floor)
    limits: dict[str, torch.Tensor]         # LIMITS → [B, T] bool
    modes: dict[str, torch.Tensor]          # MODES → [B, T] bool, the state AFTER the cycle's law
    done_cycle: torch.Tensor                # [B] long: the cycle at whose end the flight was done (T − 1: never)
    sentence_s: torch.Tensor                # [B, T]: the sentence time each cycle's words were looked up at
    cycle_s: float


def fly(inputs: FlightInputs, sentences: Sentences, clock: TimeClock | DistanceClock | TrackClock, runways: Runways,
        charts: AirportCharts, approach_ias_mps: torch.Tensor, params: ExecutorParams, words: Words, *,
        time_limit_s: torch.Tensor, early_words: bool = False) -> Flown:
    """Fly every flight's sentence, its words said on ``clock``, until each is done or its time limit
    (``early_words``: a probe's negative delays, `ExecutorParams.check`)."""
    spec = words.spec
    params.check(spec, early_words=early_words)
    batch = len(time_limit_s)
    cycles = int(math.ceil(float(time_limit_s.max()) / params.cycle_s))
    device = inputs.initial_state.device
    plant = Plant(inputs)
    lateral = Lateral(batch, params, spec, device)
    vertical = Vertical(batch, params, words, device)
    speed = Speed(approach_ias_mps, params, spec)

    state = inputs.initial_state
    bank = torch.zeros(batch, dtype=state.dtype, device=device)
    states, commands, wanted = [state], [], []
    limits: dict[str, list[torch.Tensor]] = {name: [] for name in LIMITS}
    modes: dict[str, list[torch.Tensor]] = {name: [] for name in MODES}
    sentence_times = []
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    done_cycle = torch.full((batch,), cycles - 1, dtype=torch.long, device=device)
    for cycle in range(cycles):
        now = read_state(state, charts)
        sentence_s = clock.now(cycle, now)
        force = sentences.at(sentence_s, params.delays)
        sentence_times.append(sentence_s)
        bank_rate = math.inf if cycle == 0 else math.radians(params.bank_rate_deg_s)
        track_rate, lateral_modes = lateral.rate(now, force.heading_deg, force.issued_step[:, HEADING],
                                                 force.approach, force.runway, runways, bank, bank_rate)
        e0, n0, course, elevation = runways.pointed(force.runway)
        before, right, _off = relative(now, e0, n0, course)
        gamma_rate, gamma_wanted, vertical_modes = vertical.rate(now, force.altitude_m, force.land,
                                                   force.angle_class, force.angle_deg,
                                                   force.issued_step[:, [ALTITUDE, ANGLE]], before, elevation,
                                                   torch.hypot(before, right), lateral.captured,
                                                   lateral_modes["go_around"])
        attitude = inverse.attitude(now, track_rate, gamma_rate, bank, bank_cap_rad=math.radians(params.bank_cap_deg),
                                    bank_rate_rad_s=bank_rate, cycle_s=params.cycle_s)
        accel, accel_wanted, speed_modes = speed.rate(now, force.speed_mps, force.unspecified,
                                                      lateral_modes["go_around"], attitude.load_factor,
                                                      inputs.aero_params, torch.hypot(before, right))
        thrust = inverse.thrust(now, accel, attitude.load_factor, inputs.aero_params, inputs.max_thrust_n)
        bank = attitude.bank_rad
        command = torch.stack((thrust.fraction, bank, attitude.load_factor), dim=1)
        state = plant.step(state, command, params.cycle_s)

        states.append(state)
        commands.append(command)
        wanted.append(torch.stack((track_rate, gamma_wanted, accel_wanted), dim=1))
        for name, value in {**attitude.binds, **thrust.binds, **speed_modes,
                            "path_rate_limited": vertical_modes["path_rate_limited"]}.items():
            limits[name].append(value)
        for name, value in {**lateral_modes, "level_captured": vertical_modes["level_captured"]}.items():
            modes[name].append(value)

        after = read_state(state, charts)
        before, _right, _off = relative(after, e0, n0, course)
        finished = ((lateral.captured & (before <= 0.0)) | ((before > 0.0) & (after.height_m < elevation))
                    | ~torch.isfinite(state).all(dim=1) | (after.speed_mps <= 0.0)
                    | ((cycle + 1) * params.cycle_s >= time_limit_s))
        done_cycle = torch.where(finished & ~done, cycle, done_cycle)
        done = done | finished
        if bool(done.all()):
            break

    def stack(rows: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(rows, dim=1)

    return Flown(states=stack(states), commands=stack(commands), wanted=stack(wanted),
                 limits={name: stack(rows) for name, rows in limits.items()},
                 modes={name: stack(rows) for name, rows in modes.items()},
                 done_cycle=done_cycle, sentence_s=stack(sentence_times), cycle_s=params.cycle_s)
