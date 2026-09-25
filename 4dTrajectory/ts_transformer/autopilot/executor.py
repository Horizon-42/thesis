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
from ts_transformer.autopilot.frame import AirportCharts, Kinematics, read_state
from ts_transformer.autopilot.lateral import Lateral, Runways, relative
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import Plant
from ts_transformer.autopilot.sentence import DistanceClock, Sentences, TimeClock, TrackClock, WordsNow
from ts_transformer.autopilot.speed import Speed
from ts_transformer.autopilot.vertical import Vertical
from ts_transformer.instructions.words import ALTITUDE, ANGLE, HEADING, Words

#: Every limit the cycle can meet (§8.1), in the order they are applied.
LIMITS = ("bank_cap", "bank_rate", "load_factor", "path_rate_limited", "stall_floor", "thrust_max", "thrust_min",
          "stall")
#: What the laws were doing each cycle.
MODES = ("captured", "tracking", "bent", "intercepting", "intercepting_off_word", "go_around", "level_captured",
         "aim_left_tube")


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
    sentence_s: torch.Tensor                # [B, T]: each cycle's sentence time; its words are the ones of the cycle
                                            # that started its row (`sentence`, `judge.words_said`)
    cycle_s: float


class Executor:
    """The executor's cycle over a batch, one cycle at a time: `fly` runs it over whole sentences, a closed loop
    interleaves it with the speaker (the prior says a step's words, the executor flies the step). The words of a cycle
    are given to `cycle`; everything else — the laws' state, the bank, what is recorded — lives here."""

    def __init__(self, inputs: FlightInputs, runways: Runways, charts: AirportCharts, approach_ias_mps: torch.Tensor,
                 params: ExecutorParams, words: Words, *, time_limit_s: torch.Tensor) -> None:
        spec = words.spec
        params.check(spec)
        self.params, self.words, self.runways, self.charts = params, words, runways, charts
        self.inputs, self.time_limit_s = inputs, time_limit_s
        batch = len(time_limit_s)
        self.cycles = int(math.ceil(float(time_limit_s.max()) / params.cycle_s))
        device = inputs.initial_state.device
        self.plant = Plant(inputs)
        self.lateral = Lateral(batch, params, spec, device)
        self.vertical = Vertical(batch, params, words, device)
        self.speed = Speed(approach_ias_mps, spec)
        self.state = inputs.initial_state
        self.bank = torch.zeros(batch, dtype=self.state.dtype, device=device)
        self.states, self.commands, self.wanted = [self.state], [], []
        self.limits: dict[str, list[torch.Tensor]] = {name: [] for name in LIMITS}
        self.modes: dict[str, list[torch.Tensor]] = {name: [] for name in MODES}
        self.sentence_times: list[torch.Tensor] = []
        self.step_rows = int(round(spec.step_s / params.cycle_s))
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.done_cycle = torch.full((batch,), self.cycles - 1, dtype=torch.long, device=device)
        self.count = 0                                   # cycles flown

    def now(self) -> Kinematics:
        """The batch's state at the start of the next cycle."""
        return read_state(self.state, self.charts)

    @property
    def runway_locked(self) -> torch.Tensor:
        """``[B]`` bool: flights whose runway pointer may not change now — cleared since the last go-around, or captured
        (executor design §4.6; `Lateral.rate` refuses the change) — for a speaker that must not say it."""
        return self.lateral.cleared | self.lateral.captured

    def cycle(self, force: WordsNow, sentence_s: torch.Tensor) -> None:
        """Fly one cycle under the words ``force``, at sentence time ``sentence_s`` (recorded)."""
        params, cycle = self.params, self.count
        now = self.now()
        self.sentence_times.append(sentence_s)
        bank_rate = math.inf if cycle == 0 else math.radians(params.bank_rate_deg_s)
        track_rate, lateral_modes = self.lateral.rate(now, force.heading_deg, force.issued_step[:, HEADING],
                                                      force.approach, force.runway, self.runways, self.bank, bank_rate,
                                                      cycle * params.cycle_s)
        e0, n0, course, elevation, crossing = self.runways.pointed(force.runway)
        before, right, _off = relative(now, e0, n0, course)
        gamma_rate, gamma_wanted, vertical_modes = self.vertical.rate(now, force.altitude_m, force.land,
                                                        force.angle_class, force.angle_deg,
                                                        force.issued_step[:, [ALTITUDE, ANGLE]], before, elevation,
                                                        crossing, torch.hypot(before, right), self.lateral.captured,
                                                        lateral_modes["go_around"])
        attitude = inverse.attitude(now, track_rate, gamma_rate, self.bank,
                                    bank_cap_rad=math.radians(self.words.spec.turn_bank_max_deg),
                                    bank_rate_rad_s=bank_rate, cycle_s=params.cycle_s)
        accel, accel_wanted, speed_modes = self.speed.rate(now, force.speed_mps, force.unspecified,
                                                           lateral_modes["go_around"], attitude.load_factor,
                                                           self.inputs.aero_params, torch.hypot(before, right))
        thrust = inverse.thrust(now, accel, attitude.load_factor, self.inputs.aero_params, self.inputs.max_thrust_n)
        self.bank = attitude.bank_rad
        command = torch.stack((thrust.fraction, self.bank, attitude.load_factor), dim=1)
        self.state = self.plant.step(self.state, command, params.cycle_s)

        self.states.append(self.state)
        self.commands.append(command)
        self.wanted.append(torch.stack((track_rate, gamma_wanted, accel_wanted), dim=1))
        for name, value in {**attitude.binds, **thrust.binds, **speed_modes,
                            "path_rate_limited": vertical_modes["path_rate_limited"]}.items():
            self.limits[name].append(value)
        for name, value in {**lateral_modes, "level_captured": vertical_modes["level_captured"],
                            "aim_left_tube": vertical_modes["aim_left_tube"]}.items():
            self.modes[name].append(value)

        after = read_state(self.state, self.charts)
        before, _right, _off = relative(after, e0, n0, course)
        finished = ((self.lateral.captured & (before <= 0.0)) | ((before > 0.0) & (after.height_m < elevation))
                    | ~torch.isfinite(self.state).all(dim=1) | (after.speed_mps <= 0.0)
                    | ((cycle + 1) * params.cycle_s >= self.time_limit_s))
        self.done_cycle = torch.where(finished & ~self.done, cycle, self.done_cycle)
        self.done = self.done | finished
        self.count += 1

    def flown(self) -> Flown:
        def stack(rows: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(rows, dim=1)

        return Flown(states=stack(self.states), commands=stack(self.commands), wanted=stack(self.wanted),
                     limits={name: stack(rows) for name, rows in self.limits.items()},
                     modes={name: stack(rows) for name, rows in self.modes.items()},
                     done_cycle=self.done_cycle, sentence_s=stack(self.sentence_times), cycle_s=self.params.cycle_s)


def fly(inputs: FlightInputs, sentences: Sentences, clock: TimeClock | DistanceClock | TrackClock, runways: Runways,
        charts: AirportCharts, approach_ias_mps: torch.Tensor, params: ExecutorParams, words: Words, *,
        time_limit_s: torch.Tensor) -> Flown:
    """Fly every flight's sentence, its words said on ``clock``, until each is done or its time limit."""
    executor = Executor(inputs, runways, charts, approach_ias_mps, params, words, time_limit_s=time_limit_s)
    for cycle in range(executor.cycles):
        sentence_s = clock.now(cycle, executor.now())
        if cycle % executor.step_rows == 0:
            step_start_s = sentence_s                  # every word is heard once a step (`sentence`)
        executor.cycle(sentences.at(step_start_s), sentence_s)
        if bool(executor.done.all()):
            break
    return executor.flown()
