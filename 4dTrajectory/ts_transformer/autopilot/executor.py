"""The executor's cycle (vocabulary §5; executor design §3): state → words in force → rates → commands → one cycle.

Each cycle, in the design's order (§3; each step reads the one before it):

1. the lateral law gives the track rate and the heading error (`autopilot.lateral`); the inverse sets the bank command
   for them (`autopilot.inverse.bank`: φ_phys, the stopping bank, the roll rate p through the bank's lag);
2. the vertical law gives the path-angle rate at the cycle's mean bank (`autopilot.vertical`); the inverse sets the
   load-factor command for it (`inverse.load`: the load band and the stall);
3. the speed law gives the airspeed rate, above the stall speed at that load factor (`autopilot.speed`);
4. the inverse sets the thrust command for it (`inverse.thrust`: the thrust box);
5. the observer moves the executor's estimate of the actuators with the commands sent (`inverse.observe`, §4.2); the
   plant integrates the cycle, actuators and all (`autopilot.plant`).

The actuators start at bank 0, load factor 1 and the first cycle's thrust (§4.2), in the plant and in the observer
alike; the bank's limit and roll rate hold from the first cycle (vocabulary §5.3, D84). The plant's actuators are the
aircraft's, recorded as flown (`Flown.actuators`); the laws read only the observer.

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
from typing import Any, Callable, Sequence

import torch

from ts_transformer.autopilot import ends, inverse
from ts_transformer.autopilot.inverse import BANK, LOAD, THRUST
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import AirportCharts, Kinematics, read_state
from ts_transformer.autopilot.lateral import Lateral, Runways, relative
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import Plant
from ts_transformer.autopilot.sentence import Sentences, WordsNow
from ts_transformer.autopilot.speed import Speed
from ts_transformer.autopilot.vertical import Vertical
from ts_transformer.instructions.words import ALTITUDE, ANGLE, HEADING, RUNWAY, SPEED, Words
from ts_transformer.outputs.envelope import MAX_THRUST_FRACTION, MIN_THRUST_FRACTION

#: Every limit the cycle can meet, in the order they are applied; "stall": the wing could not give the load factor the
#: actuator reached by the cycle's end (the dynamics' stall cut-off, a dynamics failure).
LIMITS = ("bank_cap", "bank_rate", "load_factor", "stall_floor", "thrust_max", "thrust_min", "stall")
#: What the laws were doing each cycle: the go-around state, a level captured.
MODES = ("go_around", "level_captured")
#: The time a go-around adds to its flight's limit (vocabulary §5.8; multi-aircraft design §6.6 step 8, the user 2026-10-02).
GO_AROUND_EXTRA_S = 900.0


@dataclass(frozen=True)
class Flown:
    """What the batch flew, one row per cycle boundary (``states``: ``T + 1``) or per cycle (the rest)."""

    states: torch.Tensor                    # [B, T+1, 7] geodetic
    commands: torch.Tensor                  # [B, T, 3] thrust fraction, bank (the dynamics' sign), load factor, as sent
    actuators: torch.Tensor                 # [B, T+1, 3] what the actuators had reached at each cycle boundary (flown)
    wanted: torch.Tensor                    # [B, T, 3] track rate deg/s, path-angle rate rad/s, speed rate m/s²,
                                            # as the laws asked before the inverse's limits (the speed's before the stall)
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

    def __init__(self, inputs: FlightInputs, runways: Runways, charts: AirportCharts, approach_speeds: torch.Tensor,
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
        self.plant = Plant(inputs, params)
        self.lateral = Lateral(batch, params, spec, device)
        self.vertical = Vertical(batch, params, words, device)
        self.speed = Speed(approach_speeds, params, spec)
        self.state = inputs.initial_state
        #: the plant's actuators (the aircraft's) and the observer's estimate of them: thrust fraction, bank, load factor;
        #: the thrust is the first cycle's (`cycle`), NaN until then
        self.actuators = torch.tensor([math.nan, 0.0, 1.0], dtype=self.state.dtype, device=device).repeat(batch, 1)
        self.observed = self.actuators.clone()
        self.states, self.commands, self.wanted = [self.state], [], []
        self.actuator_rows: list[torch.Tensor] = []     # the actuators at each cycle's start, then the last end
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

    def take(self, flights: torch.Tensor, device: torch.device | None = None) -> Executor:
        """A copy of the flights ``flights`` (``[K]`` long, repeats permitted; vocabulary §6 item 5, D97 (2)): everything
        the executor holds of them — their inputs, runways and charts, time limits and reserve, first cycles, state,
        bank, the laws' state, done and halted, end cycles and the record so far — and nothing of the other flights. The
        cycles flown and their layout are the batch's. Flown on, each copy flies as its original would (a flight's
        states do not depend on the other flights of its batch, D97 (3)). ``device``: the copy's (this executor's when
        not given; multi-aircraft control D182: a copy held in the host's memory, flown on the GPU later)."""
        flights = torch.as_tensor(flights, dtype=torch.long, device=self.state.device)
        if flights.ndim != 1 or not len(flights) or bool(((flights < 0) | (flights >= len(self.state))).any()):
            raise ValueError(f"a copy takes one or more of the batch's {len(self.state)} flights, got {flights.tolist()}")
        device = self.state.device if device is None else torch.device(device)

        def pick(rows: torch.Tensor) -> torch.Tensor:
            return rows[flights].clone().to(device)

        def picked(record: Any) -> Any:                  # a dataclass of per-flight tensors
            return type(record)(**{f.name: pick(getattr(record, f.name)) for f in fields(record)})

        out = copy.copy(self)
        out.inputs, out.runways, out.charts = picked(self.inputs), picked(self.runways), picked(self.charts)
        out.plant = Plant(out.inputs, self.params)
        for name in ("time_limit_s", "most_s", "start", "state", "actuators", "observed", "runway_issued", "done",
                     "done_cycle", "halted", "last_command"):
            setattr(out, name, pick(getattr(self, name)))
        out.staggered = bool((out.start != 0).any())
        for name in ("lateral", "vertical", "speed"):
            law = copy.copy(getattr(self, name))
            for held in law.PER_FLIGHT:
                value = getattr(law, held)
                setattr(law, held, None if value is None else pick(value))   # the lateral word before any is heard
            setattr(out, name, law)
        out.speed.approach_speeds = pick(self.speed.approach_speeds)
        out.states, out.commands, out.wanted, out.sentence_times, out.runway_rows, out.actuator_rows = (
            [pick(rows) for rows in record]
            for record in (self.states, self.commands, self.wanted, self.sentence_times, self.runway_rows,
                           self.actuator_rows))
        out.limits = {name: [pick(rows) for rows in record] for name, record in self.limits.items()}
        out.modes = {name: [pick(rows) for rows in record] for name, record in self.modes.items()}
        return out

    @staticmethod
    def together(parts: Sequence[tuple[Executor, torch.Tensor, int]], device: torch.device) -> Executor:
        """Flights of several executors flown on as one multi-aircraft batch (module docstring; multi-aircraft control
        D188: a backward batch's searches, each at its own point), on ``device``: ``parts`` each an executor, its
        flights (``[K]`` long, repeats permitted) and the cycles its clock moves later by (below 0: its first cycles,
        before any of its flights started, are left out), so that every part is at one cycle. Each flight keeps what
        `take` keeps, its first cycle moved with its part: it flies on as its original would. The cycles are laid out
        again for the batch's flights."""
        counts = {executor.count + shift for executor, _, shift in parts}
        if len(counts) != 1:
            raise ValueError(f"the parts' clocks, moved, are at the cycles {sorted(counts)}, not one")
        first = parts[0][0]
        index = [torch.as_tensor(flights, dtype=torch.long, device=e.state.device) for e, flights, _ in parts]
        device = torch.device(device)

        def rows(get: Callable[[Executor], torch.Tensor]) -> torch.Tensor:
            return torch.cat([get(e)[i].to(device) for (e, _, _), i in zip(parts, index)])

        def records(get: Callable[[Executor], list[torch.Tensor]]) -> list[torch.Tensor]:
            # each part's cycles stacked, its flights taken and moved by its shift; the batch's cycles one by one again
            length = len(get(first)) + parts[0][2]
            if length == 0:
                return []
            template = next(get(e)[0] for e, _, _ in parts if get(e))     # the latest part has cycles: one exists
            blocks = []
            for (e, _, shift), i in zip(parts, index):
                kept = (torch.stack(get(e), dim=1)[i][:, max(-shift, 0):] if get(e)
                        else template.new_zeros((len(i), 0, *template.shape[1:])))
                before = template.new_zeros((len(i), length - kept.shape[1], *template.shape[1:]))   # no flight's own
                blocks.append(torch.cat((before.to(kept.device), kept), dim=1).to(device))
            return list(torch.cat(blocks).unbind(1))

        def law_rows(law: str, name: str) -> torch.Tensor | None:
            values = [getattr(getattr(e, law), name) for e, _, _ in parts]
            if all(value is None for value in values):
                return None
            # a part with no cycle flown: its flights start fresh at their own cycle 0 (`Lateral.word_error`)
            return torch.cat([(value[i] if value is not None else torch.zeros(len(i), dtype=torch.float64,
                                                                              device=i.device)).to(device)
                              for value, i in zip(values, index)])

        out = copy.copy(first)
        for name in ("inputs", "runways", "charts"):
            records_of = [getattr(e, name) for e, _, _ in parts]
            setattr(out, name, type(records_of[0])(**{f.name: torch.cat([getattr(r, f.name)[i].to(device)
                                                                          for r, i in zip(records_of, index)])
                                                       for f in fields(records_of[0])}))
        out.plant = Plant(out.inputs, first.params)
        for name in ("time_limit_s", "most_s", "state", "actuators", "observed", "runway_issued", "done", "halted",
                     "last_command"):
            setattr(out, name, rows(lambda e, name=name: getattr(e, name)))
        out.start = torch.cat([(e.start[i] + shift).to(device) for (e, _, shift), i in zip(parts, index)])
        if bool((out.start < 0).any()):
            raise ValueError("a part's cycles left out are a flight's own")
        out.staggered = bool((out.start != 0).any())
        out.cycles = (int((out.start + torch.ceil(out.most_s / first.params.cycle_s).long()).max()) if out.staggered
                      else int(math.ceil(float(out.most_s.max()) / first.params.cycle_s)))
        out.done_cycle = torch.where(out.done, rows(lambda e: e.done_cycle), out.cycles - 1 - out.start)
        for law_name in ("lateral", "vertical", "speed"):
            law = copy.copy(getattr(first, law_name))
            for held in law.PER_FLIGHT:
                setattr(law, held, law_rows(law_name, held))
            setattr(out, law_name, law)
        out.count = counts.pop()
        if out.count == 0:                  # a batch that has flown no cycle anchors its first as a new batch does
            out.lateral.word_deg = None
        out.speed.approach_speeds = rows(lambda e: e.speed.approach_speeds)
        out.states, out.commands, out.wanted, out.sentence_times, out.runway_rows, out.actuator_rows = (
            records(lambda e, name=name: getattr(e, name))
            for name in ("states", "commands", "wanted", "sentence_times", "runway_rows", "actuator_rows"))
        out.limits = {name: records(lambda e, name=name: e.limits[name]) for name in first.limits}
        out.modes = {name: records(lambda e, name=name: e.modes[name]) for name in first.modes}
        return out

    def own_cycle(self) -> torch.Tensor:
        """``[B]`` long: each flight's own cycle about to be flown (negative: it has not started)."""
        return self.count - self.start

    def _held(self) -> list[tuple[Any, str, torch.Tensor | None]]:
        """Every per-flight quantity a cycle changes (the state, the bank, each law's `PER_FLIGHT`), as it is now."""
        out: list[tuple[Any, str, torch.Tensor | None]] = [(self, "state", self.state), (self, "actuators", self.actuators),
                                                            (self, "observed", self.observed),
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
        aero, decay = self.inputs.aero_params, inverse.decays(params, self.state)
        observed = self.observed
        # 1. lateral: the track rate, then the bank through its lag
        track_rate, error = self.lateral.rate(now, force.heading_rel_deg, force.issued_step[:, HEADING], force.runway,
                                              self.runways, time_s, fresh=fresh)
        ceiling = inverse.load_ceiling(now, observed, aero, self.inputs.max_thrust_n, params.cycle_s)
        bank = inverse.bank(now, track_rate, error, observed[:, BANK], inverse.bank_limit_rad(now, ceiling), params,
                            decay[BANK])
        # 2. vertical: the path-angle rate at the cycle's mean bank, then the load factor through its lag
        e0, n0, course, elevation = self.runways.pointed(force.runway)
        gamma_rate, vertical_modes = self.vertical.rate(now, force.level_m, self.charts.elevation_m, force.no_level_off,
                                                        force.angle_class, force.angle_deg,
                                                        force.issued_step[:, [ALTITUDE, ANGLE]], force.go_around, aero,
                                                        self.inputs.max_thrust_n, time_s, bank.mean, observed[:, LOAD],
                                                        ceiling)
        load = inverse.load(now, gamma_rate, bank.mean, observed[:, LOAD], ceiling, decay[LOAD])
        # 3. speed at that load factor; 4. the thrust for it, through its lag (the first cycle's thrust: in effect)
        accel, accel_wanted, speed_modes = self.speed.rate(now, force.speed_mps, force.unspecified, force.go_around,
                                                           force.issued_step[:, SPEED], time_s, load.wanted, aero)
        solved = inverse.thrust_fraction(now, accel, load.wanted, aero, self.inputs.max_thrust_n)
        first = self.observed[:, THRUST].isnan()
        start_thrust = solved.clamp(MIN_THRUST_FRACTION, MAX_THRUST_FRACTION)
        self.observed = torch.cat((torch.where(first, start_thrust, observed[:, THRUST])[:, None], observed[:, 1:]), 1)
        self.actuators = torch.cat((torch.where(first, start_thrust, self.actuators[:, THRUST])[:, None],
                                    self.actuators[:, 1:]), 1)
        thrust = inverse.thrust(solved, self.observed[:, THRUST], decay[THRUST])
        command = torch.stack((thrust.command, bank.command, load.command), dim=1)
        # 5. the observer, and the plant
        self.observed = inverse.observe(self.observed, command, decay)
        start_actuators = self.actuators
        self.state, self.actuators = self.plant.step(self.state, self.actuators, command, params.cycle_s)
        after = read_state(self.state, self.charts)
        limits = {**bank.binds, **load.binds, **speed_modes, **thrust.binds,
                  "stall": inverse.stalled(after, self.actuators[:, LOAD], aero)}
        modes = {"go_around": force.go_around.clone(), "level_captured": vertical_modes["level_captured"]}
        if frozen is not None and bool(frozen.any()):
            for owner, name, value in held:
                after_value = getattr(owner, name)
                if value is not None:
                    keep = frozen.view(-1, *([1] * (after_value.dim() - 1)))
                    setattr(owner, name, torch.where(keep, value, after_value))
            command = torch.where(self.halted[:, None], self.last_command, command)
            start_actuators = torch.where(frozen[:, None], self.actuators, start_actuators)
            limits = {name: value & ~frozen for name, value in limits.items()}
            modes = {name: value & ~frozen for name, value in modes.items()}
            after = read_state(self.state, self.charts)
        self.last_command = command

        self.actuator_rows.append(start_actuators)
        self.states.append(self.state)
        self.commands.append(command)
        self.wanted.append(torch.stack((track_rate, gamma_rate, accel_wanted), dim=1))
        self.runway_rows.append(force.runway.clone())
        for name, value in limits.items():
            self.limits[name].append(value)
        for name, value in modes.items():
            self.modes[name].append(value)

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

        return Flown(states=stack(self.states), commands=stack(self.commands),
                     actuators=stack([*self.actuator_rows, self.actuators]), wanted=stack(self.wanted),
                     limits={name: stack(rows) for name, rows in self.limits.items()},
                     modes={name: stack(rows) for name, rows in self.modes.items()}, runway=stack(self.runway_rows),
                     done_cycle=self.done_cycle, sentence_s=stack(self.sentence_times), cycle_s=self.params.cycle_s)


def fly(inputs: FlightInputs, sentences: Sentences, runways: Runways, charts: AirportCharts,
        approach_speeds: torch.Tensor, params: ExecutorParams, words: Words, *, time_limit_s: torch.Tensor,
        reserve_s: float) -> Flown:
    """Fly every flight's sentence on its own rows (D57: a cycle's sentence time is the time flown, a step's words heard
    on the cycle that starts it), until each is done or its time limit (``reserve_s``: the time its go-arounds may add,
    `Executor`)."""
    executor = Executor(inputs, runways, charts, approach_speeds, params, words, step_s=sentences.step_s,
                        time_limit_s=time_limit_s, reserve_s=reserve_s)
    for cycle in range(executor.cycles):
        sentence_s = torch.full_like(executor.now().e_m, cycle * params.cycle_s)
        executor.cycle(sentences.at(sentence_s), sentence_s)
        if bool(executor.done.all()):
            break
    return executor.flown()
