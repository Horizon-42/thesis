"""The executor's cycle (vocabulary §5; executor design §3, §7): state → words in force → rates → controls → one cycle.

Each cycle, in the design's order of limits:

1. the lateral law gives the track rate, the vertical law the path-angle rate (`autopilot.lateral`,
   `autopilot.vertical`);
2. the inverse sets bank and load factor for them (`autopilot.inverse.attitude`: bank cap and rate,
   the grader's load band);
3. the speed law gives the airspeed rate, above the stall floor at that load factor (`autopilot.speed`);
4. the inverse sets the thrust for it (`autopilot.inverse.thrust`: the thrust box);
5. the plant integrates the cycle (`autopilot.plant`).

The bank starts at 0 and its limit and roll rate hold from the first cycle (vocabulary §5.3, D84).

A flight is DONE at the end of the cycle in which the judge ends it (vocabulary §5.8, D79; the tests are `ends`, the
judge's own): an APPROACH CROSSING of the runway in force R (the go-around state G false, the threshold plane crossed
lined up — the track within the vocabulary's lined-up angle of the course — and within the landing screen's lateral
limit, `lateral.Runways`), below R's threshold elevation while before it, a lined-up crossing of another candidate with G
false inside that runway's own limit, a dynamics failure (a non-finite state, no airspeed, or the stall cut-off bound in
the cycle) — or its time limit; the batch stops when every flight is done. A crossing while G is true is not an end: the
flight flies on. Each go-around word heard gives its
flight `GO_AROUND_EXTRA_S` more time (`extend_time_limit`; the cycles are laid out for ``reserve_s``). What each outcome
means is judged afterwards (`autopilot.judge`), from the recorded states and the runway and go-around state of every
cycle.

THREE WAYS TO FLY (executor design §12.4), each checked against the spec's reference tracks
(`experiments/executor_conformance.py`): a SINGLE-AIRCRAFT BATCH starts every flight at the batch's cycle 0; a
MULTI-AIRCRAFT BATCH gives each flight its own start cycle (``start_cycle``) — before it the flight waits (its state,
its bank and every law's state are held, its rows are not its own), its first cycle is its own cycle 0 (the heading word anchored afresh, its time and time limit counted from it), and `flown` hands each flight's
rows from its own cycle 0, so it reads as the flight flown alone; the SINGLE FLIGHT is `autopilot.single`, plain Python.
A flight can also be HALTED (`halt`): from then on it is held as a waiting flight is, its command row repeating the last
one it flew — where a closed loop that once flew it in an executor of its own would have stopped that executor.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, fields
from typing import Any

import torch

from ts_transformer.autopilot import ends, inverse
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import AirportCharts, Kinematics, read_state
from ts_transformer.autopilot.lateral import Lateral, Runways, relative
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import Plant
from ts_transformer.autopilot.sentence import Sentences, WordsNow
from ts_transformer.autopilot.speed import Speed
from ts_transformer.autopilot.vertical import Vertical
from ts_transformer.instructions.words import ALTITUDE, ANGLE, HEADING, RUNWAY, Words

#: Every limit the cycle can meet (§8.1), in the order they are applied.
LIMITS = ("bank_cap", "bank_rate", "load_factor", "path_rate_limited", "stall_floor", "thrust_max", "thrust_min",
          "stall")
#: What the laws were doing each cycle: the go-around state, a level captured.
MODES = ("go_around", "level_captured")
#: The time a go-around adds to its flight's limit (vocabulary §5.8; multi-aircraft design §6.6 step 8, the user 2026-10-02).
GO_AROUND_EXTRA_S = 900.0


@dataclass(frozen=True)
class Flown:
    """What the batch flew, one row per cycle boundary (``states``: ``T + 1``) or per cycle (the rest)."""

    states: torch.Tensor                    # [B, T+1, 7] geodetic
    commands: torch.Tensor                  # [B, T, 3] thrust fraction, bank (the dynamics' sign), load factor
    wanted: torch.Tensor                    # [B, T, 3] track rate deg/s, path-angle rate rad/s, speed rate m/s²,
                                            # as the laws asked before their own limits (γ̇_max, the stall floor)
    limits: dict[str, torch.Tensor]         # LIMITS → [B, T] bool
    modes: dict[str, torch.Tensor]          # MODES → [B, T] bool, the state AFTER the cycle's law
    runway: torch.Tensor                    # [B, T] long: the runway in force R during each cycle
    done_cycle: torch.Tensor                # [B] long: the cycle at whose end the flight was done (T − 1: never)
    sentence_s: torch.Tensor                # [B, T]: each cycle's sentence time; its words are the ones of the cycle
                                            # that started its row (`sentence`, `judge.words_said`)
    cycle_s: float


class Executor:
    """The executor's cycle over a batch, one cycle at a time: `fly` runs it over whole sentences, a closed loop
    interleaves it with the speaker (the prior says a step's words, the executor flies the step). The words of a cycle
    are given to `cycle`; everything else — the laws' state, the bank, what is recorded — lives here."""

    def __init__(self, inputs: FlightInputs, runways: Runways, charts: AirportCharts, approach_ias_mps: torch.Tensor,
                 params: ExecutorParams, words: Words, *, step_s: float, time_limit_s: torch.Tensor,
                 start_cycle: torch.Tensor | None = None, reserve_s: float = 0.0) -> None:
        """``step_s``: the sentence's row interval (a row's words are heard on the cycle that starts it);
        ``reserve_s``: how much each flight's time limit may still grow (its go-arounds, `GO_AROUND_EXTRA_S` each)."""
        spec = words.spec
        params.check(spec, step_s)
        self.params, self.words, self.runways, self.charts = params, words, runways, charts
        self.inputs, self.time_limit_s = inputs, time_limit_s
        #: the time a flight's limit may still be extended by (`extend_time_limit`): the cycles are laid out for it
        self.reserve_s = reserve_s
        self.most_s = time_limit_s + reserve_s
        batch = len(time_limit_s)
        device = inputs.initial_state.device
        #: each flight's first cycle (module docstring: a multi-aircraft batch); every flight's 0 when not given
        self.start = (torch.zeros(batch, dtype=torch.long, device=device) if start_cycle is None
                      else start_cycle.to(device=device, dtype=torch.long))
        if bool((self.start < 0).any()):
            raise ValueError("a flight's start cycle is negative")
        self.staggered = bool((self.start != 0).any())
        own_cycles = torch.ceil(self.most_s / params.cycle_s).long()
        self.cycles = (int((self.start + own_cycles).max()) if self.staggered
                       else int(math.ceil(float(self.most_s.max()) / params.cycle_s)))
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
        self.runway_rows: list[torch.Tensor] = []
        self.step_rows = int(round(step_s / params.cycle_s))
        # the step the runway column's last word was written at (a new go-around word extends the time limit)
        self.runway_issued = torch.full((batch,), -1, dtype=torch.long, device=device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        # the flight's own cycle (from its start) at whose end it was done; never done: its last
        self.done_cycle = self.cycles - 1 - self.start
        self.count = 0                                   # cycles flown
        self.halted = torch.zeros(batch, dtype=torch.bool, device=device)
        self.last_command = torch.zeros((batch, 3), dtype=self.state.dtype, device=device)

    def extend_time_limit(self, seconds: torch.Tensor) -> None:
        """Give each flight ``seconds`` (``[B]``, 0 for most) more time (a go-around's, multi-aircraft design §6.6 step 8
        item 9) — never past the reserve the executor was laid out for, and never once a flight is done."""
        seconds = seconds.to(device=self.time_limit_s.device, dtype=self.time_limit_s.dtype)
        extended = self.time_limit_s + seconds
        if bool((extended > self.most_s).any()):
            raise ValueError("a time limit extended past the executor's reserve")
        if bool(((seconds != 0.0) & self.done).any()):
            raise ValueError("a time limit extended for a flight already done")
        self.time_limit_s = extended

    def halt(self, flights: torch.Tensor) -> None:
        """Hold the flights ``flights`` (``[B]`` bool) from the next cycle on (module docstring)."""
        self.halted = self.halted | flights.to(self.halted.device)

    def take(self, flights: torch.Tensor) -> Executor:
        """A copy of the flights ``flights`` (``[K]`` long, repeats permitted; vocabulary §6 item 5, D97 (2)): everything
        the executor holds of them — their inputs, runways and charts, time limits and reserve, first cycles, state,
        bank, the laws' state, done and halted, end cycles and the record so far — and nothing of the other flights. The
        cycles flown and their layout are the batch's. Flown on, each copy flies as its original would (a flight's
        states do not depend on the other flights of its batch, D97 (3))."""
        flights = torch.as_tensor(flights, dtype=torch.long, device=self.state.device)
        if flights.ndim != 1 or not len(flights) or bool(((flights < 0) | (flights >= len(self.state))).any()):
            raise ValueError(f"a copy takes one or more of the batch's {len(self.state)} flights, got {flights.tolist()}")

        def pick(rows: torch.Tensor) -> torch.Tensor:
            return rows[flights].clone()

        def picked(record: Any) -> Any:                  # a dataclass of per-flight tensors
            return type(record)(**{f.name: pick(getattr(record, f.name)) for f in fields(record)})

        out = copy.copy(self)
        out.inputs, out.runways, out.charts = picked(self.inputs), picked(self.runways), picked(self.charts)
        out.plant = Plant(out.inputs)
        for name in ("time_limit_s", "most_s", "start", "state", "bank", "runway_issued", "done", "done_cycle",
                     "halted", "last_command"):
            setattr(out, name, pick(getattr(self, name)))
        out.staggered = bool((out.start != 0).any())
        for name in ("lateral", "vertical", "speed"):
            law = copy.copy(getattr(self, name))
            for held in law.PER_FLIGHT:
                value = getattr(law, held)
                setattr(law, held, None if value is None else pick(value))   # the lateral word before any is heard
            setattr(out, name, law)
        out.speed.approach_ias_mps = pick(self.speed.approach_ias_mps)
        out.states, out.commands, out.wanted, out.sentence_times, out.runway_rows = (
            [pick(rows) for rows in record]
            for record in (self.states, self.commands, self.wanted, self.sentence_times, self.runway_rows))
        out.limits = {name: [pick(rows) for rows in record] for name, record in self.limits.items()}
        out.modes = {name: [pick(rows) for rows in record] for name, record in self.modes.items()}
        return out

    def own_cycle(self) -> torch.Tensor:
        """``[B]`` long: each flight's own cycle about to be flown (negative: it has not started)."""
        return self.count - self.start

    def _held(self) -> list[tuple[Any, str, torch.Tensor | None]]:
        """Every per-flight quantity a cycle changes (the state, the bank, each law's `PER_FLIGHT`), as it is now."""
        out: list[tuple[Any, str, torch.Tensor | None]] = [(self, "state", self.state), (self, "bank", self.bank),
                                                            (self, "runway_issued", self.runway_issued)]
        for law in (self.lateral, self.vertical, self.speed):
            out += [(law, name, getattr(law, name)) for name in law.PER_FLIGHT]
        return out

    def now(self) -> Kinematics:
        """The batch's state at the start of the next cycle."""
        return read_state(self.state, self.charts)

    def cycle(self, force: WordsNow, sentence_s: torch.Tensor) -> None:
        """Fly one cycle under the words ``force``, at sentence time ``sentence_s`` (recorded; each flight's own, from
        its start). In a multi-aircraft batch a flight that has not started takes no part: whatever ``force`` says for
        it, it ends the cycle as it began, with nothing limited, in no mode and not done."""
        params, cycle = self.params, self.count
        now = self.now()
        self.sentence_times.append(sentence_s)
        frozen: torch.Tensor | None = None
        if self.staggered:
            own = self.own_cycle()
            waiting, fresh = own < 0, own == 0
            frozen = waiting | self.halted
            time_s: float | torch.Tensor = own.to(self.state.dtype) * params.cycle_s
        else:
            waiting = fresh = None
            if bool(self.halted.any()):
                frozen = self.halted
            time_s = cycle * params.cycle_s
        bank_rate = math.radians(params.bank_rate_deg_s)          # from the first cycle (D84)
        held = self._held() if frozen is not None else []
        # a go-around heard this cycle gives its flight more time (`extend_time_limit`), not while it waits or is held
        issued = force.issued_step[:, RUNWAY]
        heard_go_around = force.go_around & (issued != self.runway_issued) & ~self.done
        if frozen is not None:
            heard_go_around = heard_go_around & ~frozen
        self.runway_issued = issued.clone()
        if bool(heard_go_around.any()):
            self.extend_time_limit(torch.where(heard_go_around, GO_AROUND_EXTRA_S, 0.0).to(self.time_limit_s.dtype))
        self.speed.hear_go_around(heard_go_around, now)
        track_rate = self.lateral.rate(now, force.heading_rel_deg, force.issued_step[:, HEADING], force.runway,
                                       self.runways, time_s, fresh=fresh)
        e0, n0, course, elevation = self.runways.pointed(force.runway)
        before, right, _off_course = relative(now, e0, n0, course)
        gamma_rate, gamma_wanted, vertical_modes = self.vertical.rate(now, force.level_m, self.charts.elevation_m,
                                                                      force.no_level_off,
                                                                      force.angle_class, force.angle_deg,
                                                                      force.issued_step[:, [ALTITUDE, ANGLE]],
                                                                      force.go_around, self.inputs.aero_params,
                                                                      self.inputs.max_thrust_n)
        attitude = inverse.attitude(now, track_rate, gamma_rate, self.bank,
                                    bank_cap_rad=math.radians(self.words.spec.turn_bank_max_deg),
                                    bank_rate_rad_s=bank_rate, cycle_s=params.cycle_s)
        accel, accel_wanted, speed_modes = self.speed.rate(now, force.speed_mps, force.unspecified, force.go_around,
                                                           attitude.load_factor, self.inputs.aero_params,
                                                           torch.hypot(before, right))
        thrust = inverse.thrust(now, accel, attitude.load_factor, self.inputs.aero_params, self.inputs.max_thrust_n)
        self.bank = attitude.bank_rad
        command = torch.stack((thrust.fraction, self.bank, attitude.load_factor), dim=1)
        self.state = self.plant.step(self.state, command, params.cycle_s)
        limits = {**attitude.binds, **thrust.binds, **speed_modes, "path_rate_limited": vertical_modes["path_rate_limited"]}
        modes = {"go_around": force.go_around.clone(), "level_captured": vertical_modes["level_captured"]}
        if frozen is not None and bool(frozen.any()):
            for owner, name, value in held:
                after = getattr(owner, name)
                if value is not None:
                    keep = frozen.view(-1, *([1] * (after.dim() - 1)))
                    setattr(owner, name, torch.where(keep, value, after))
            command = torch.where(self.halted[:, None], self.last_command, command)
            limits = {name: value & ~frozen for name, value in limits.items()}
            modes = {name: value & ~frozen for name, value in modes.items()}
        self.last_command = command

        self.states.append(self.state)
        self.commands.append(command)
        self.wanted.append(torch.stack((track_rate, gamma_wanted, accel_wanted), dim=1))
        self.runway_rows.append(force.runway.clone())
        for name, value in limits.items():
            self.limits[name].append(value)
        for name, value in modes.items():
            self.modes[name].append(value)

        after = read_state(self.state, self.charts)
        # where the judge ends the flight (vocabulary §5.8, D79; `ends`, the judge's tests on the same rows): a crossing
        # of a candidate's plane, interpolated, an approach crossing of R or a crossing of another candidate
        runways = self.runways
        all_before, all_right, _ = runways.relative(now)
        all_past, all_right_after, all_off_after = runways.relative(after)
        fraction = (all_before / (all_before - all_past)).clamp(0.0, 1.0)
        in_force = torch.arange(all_before.shape[1], device=all_before.device)[None, :] == force.runway[:, None]
        approach, other = ends.crossing_ends(all_right + fraction * (all_right_after - all_right), all_off_after,
                                             force.go_around[:, None], in_force, runways.landing_limit_m,
                                             runways.on_runway_m, self.words.spec)
        crossed = (ends.plane_crossed(all_before, all_past) & (approach | other)).any(dim=1)
        past = relative(after, e0, n0, course)[0]
        ended = (crossed | ends.ground_contact(past, after.height_m - elevation)
                 | ends.dynamics_failure(torch.isfinite(self.state).all(dim=1), after.speed_mps, limits["stall"]))
        if self.staggered:
            finished = (ended | ((own + 1).to(self.state.dtype) * params.cycle_s >= self.time_limit_s)) & ~frozen
            self.done_cycle = torch.where(finished & ~self.done, own, self.done_cycle)
        else:
            finished = ended | ((cycle + 1) * params.cycle_s >= self.time_limit_s)
            if frozen is not None:
                finished = finished & ~frozen
            self.done_cycle = torch.where(finished & ~self.done, cycle, self.done_cycle)
        self.done = self.done | finished
        self.count += 1

    def flown(self) -> Flown:
        """What was flown, each flight's rows from its own cycle 0 (a multi-aircraft batch's later starters end
        earlier: their last row repeats to the batch's length)."""
        def stack(rows: list[torch.Tensor]) -> torch.Tensor:
            out = torch.stack(rows, dim=1)
            if not self.staggered:
                return out
            at = (self.start[:, None] + torch.arange(out.shape[1], device=out.device)[None, :]).clamp(
                max=out.shape[1] - 1)
            return torch.gather(out, 1, at.view(*at.shape, *([1] * (out.dim() - 2))).expand_as(out))

        return Flown(states=stack(self.states), commands=stack(self.commands), wanted=stack(self.wanted),
                     limits={name: stack(rows) for name, rows in self.limits.items()},
                     modes={name: stack(rows) for name, rows in self.modes.items()}, runway=stack(self.runway_rows),
                     done_cycle=self.done_cycle, sentence_s=stack(self.sentence_times), cycle_s=self.params.cycle_s)


def fly(inputs: FlightInputs, sentences: Sentences, runways: Runways, charts: AirportCharts,
        approach_ias_mps: torch.Tensor, params: ExecutorParams, words: Words, *, time_limit_s: torch.Tensor,
        reserve_s: float) -> Flown:
    """Fly every flight's sentence on its own rows (D57: a cycle's sentence time is the time flown, a step's words heard
    on the cycle that starts it), until each is done or its time limit (``reserve_s``: the time its go-arounds may add,
    `Executor`)."""
    executor = Executor(inputs, runways, charts, approach_ias_mps, params, words, step_s=sentences.step_s,
                        time_limit_s=time_limit_s, reserve_s=reserve_s)
    for cycle in range(executor.cycles):
        sentence_s = torch.full_like(executor.now().e_m, cycle * params.cycle_s)
        executor.cycle(sentences.at(sentence_s), sentence_s)
        if bool(executor.done.all()):
            break
    return executor.flown()
