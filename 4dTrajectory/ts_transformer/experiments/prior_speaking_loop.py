"""The step of a speaker's closed loop (prior design §7 item 7; D106 item 1): the prior speaks, the executor flies, one
Δ row at a time — shared by free generation (`prior_free_generation`) and the post-training's window loop, so neither
holds a copy of it.

A `SpeakingLoop` holds a batch of flights started at their first predicted step through the start of a closed loop
(vocabulary §6 item 5, D67: `autopilot.start`; the most go-arounds of a flight is the loop's, D68). The rows before the
first predicted step are the flight's observed rows as the start gives them back (`start.start_moved`'s: a moved
start's moved rows; with `NO_MOVE`, the closed-loop sentence's, bit for bit), each encoded by `observe` one at a time,
as the speaker reads them; of the closed-loop sentence the loop reads only its first row and its first predicted step
(D82: its rows alone). From the first predicted step on, `step` says one row:
the inputs from the states the executor flew (D32, §2) through the prior's one function of a loop's row
(`prior.loop.LoopRows`, D96 item 4, with each flight's own landings, D105), the speaker under its masks, the bound of D68
("go-around" forbidden after a flight's last) and the caller's masks, with the caller's random numbers and input of the
added modules; the loop flies the row for Δ seconds. A flight ends when the executor is done with it (its crossing, its
time limit, or the dynamics), or when the caller ends it (`end`: a loss of separation, post-training D93); an ended
flight is halted and keeps, as its inputs, the finite state of the row it ended in: the speaker still says its rows
(a batch is said together), which are not its sentence's. Its outcome is the judge's (`Loop.outcome`, item 6). This
module imports nothing else of `autopilot/` (`tests/test_architecture.py`, D69).

**Join ticks** (multi-aircraft control D150, §6.3 item 4; prior §7 item 7). The flights of the loop may join it at their
own ticks (the start's `Loop.join_ticks`, read from the loop: one definition): at tick t a flight is absent before its
join tick j, observed from it to its first predicted step (its own rows 0 … s − 1, the start's observed rows), then
said while it is flown. `step` advances one tick for every flight, each by its own clock (the speaker's row of several
roles, `Speaker.say`; the executor's step once the loop's clock reaches tick s, where a flight of join tick 0 starts);
`observe` is the phase of a loop whose join ticks are all 0. Each flight's records, words, states and sentence are of its
own rows from its row 0 (`sentences`, `said`, `states`, `generated`). A landing may be added to chosen flights' landings
while the loop runs (`add_landing`, `LoopRows.add_landing`); a copy keeps them and the join ticks. With every join tick 0
the loop is the loop without join ticks, bit for bit.

**Rows that leave** (multi-aircraft control D193 (3), S6). A join (`together`) keeps in the speaker and the executor only
the flights still flown. A flight no longer flown leaves them: nothing says or flies it after. The loop keeps what is
read of it later as a `Left` record: its outcome, its states, its landings, its sentence's inputs and its permitted
words. Each flight keeps its place in ``order``, so a caller reads a flight that left as it reads one still held
(`said`, `states`, `outcome`, `landings`, `sentences`, `permitted`, `generated`). A copy (`copy`) keeps each flight as
it is: held or left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.judge import TIMEOUT, Outcome
from ts_transformer.autopilot.start import Loop
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import STATE_COLUMNS, ClosedLoopSentence
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words
from ts_transformer.prior.batch import RowTensors, SentenceRows
from ts_transformer.prior.inputs import own_flight_key
from ts_transformer.prior.landings import Landing, LandingIndex, utc_s
from ts_transformer.prior.loop import LoopRows
from ts_transformer.prior.model import COPIED, Prior
from ts_transformer.prior.procedure import Final
from ts_transformer.prior.speaker import ABSENT, OBSERVED, SAID, Permitted, Speaker, go_around_bound

#: The predicted rows the speaker's cache has room for at first (it grows as the flights need, `model.Past.grown`; the
#: loop's time limit is not the speaker's to read, vocabulary D90).
FIRST_ROWS = 128


def flight_numbers(seed: int, sample: int, index: int) -> np.random.Generator:
    """A flight's own source of random numbers in free generation (§4 "Drawing", D96): from the seed, the sample and
    its place in the split's signals — the same whatever flights share its batch."""
    return np.random.default_rng([seed, sample, index])


@dataclass
class Generated:
    """One flight's sentence: the words said from its first predicted step to the row it ended in, its states on the 2 s
    rows from its row 0 (observed before the first predicted step, flown from it; D51's layout), and the judge's
    outcome."""

    index: int                         # the flight's place in the split's signals
    words: np.ndarray                  # [M, 5]
    #: [rows, 6] `STATE_COLUMNS` on the 2 s rows from row 0, to the first row at or after the cycle the executor was done in
    states: np.ndarray
    outcome: str
    crossing: dict[str, Any] | None
    timed_out: bool
    go_arounds: int
    go_around_probability: np.ndarray  # [M]: of "go-around" in the distribution each runway word was drawn from
    go_around_permitted: np.ndarray    # [M] bool: the masks permitted "go-around" (G false, D68's bound not met)
    #: [M] bool: inside the region of the runway in force (the FAF and the LPV cone) under which the row's runway word was
    #: drawn (D72)
    on_final: np.ndarray
    #: by column the procedure masks rule (`ProcedureMasks.columns`): [M, words] bool, the classes they blocked at each
    #: row (`Speaker.procedure_blocked`)
    blocked: dict[int, np.ndarray]


@dataclass(frozen=True)
class Left:
    """What a loop keeps of a flight that has left its speaker and executor (module docstring, S6). The flight is no
    longer flown, so these records do not change."""

    outcome: Outcome | None            # the judge's; None: the caller ended the flight (`SpeakingLoop.end`)
    states: np.ndarray                 # `SpeakingLoop.states`
    landings: LandingIndex             # its landings when it left (its reward reads those before its first step)
    inputs: dict[str, np.ndarray]      # the inputs of its sentence's rows, by `SentenceRows` field (`_own_inputs`)
    permitted: Permitted               # `SpeakingLoop.permitted`


class SpeakingLoop:
    """The closed loop of the flights ``order`` of ``loop`` (module docstring). ``sentences``, ``observed`` (each
    flight's observed rows before its first predicted step as the start gave them back, `STATE_COLUMNS`) and
    ``flights`` (their records in the split's signals: airport, entry time, key) by their place in the signals; ``landings`` each flight's,
    in ``order`` (D105); ``finals`` each airport's (the procedure masks); ``attention``: how its copies read the front
    of the speaker's cache they hold once (`Speaker`; multi-aircraft control D190 (2)). ``executor_device``: the device that the
    start's ``loop`` (the executor) was started on (multi-aircraft control S1). The executor stays on this device, and
    so do its copies and joins. The speaker stays on ``device``. A loop started on another device is refused. None: the
    executor follows the speaker, and each copy and join moves it to the speaker's device (the behaviour before S1)."""

    def __init__(self, model: Prior, loop: Loop, order: Sequence[int], sentences: Mapping[int, ClosedLoopSentence],
                 observed: Mapping[int, np.ndarray], flights: Mapping[int, Mapping[str, Any]], geometries: Mapping[str, AirportGeometry],
                 landings: Sequence[LandingIndex], finals: Mapping[str, Sequence[Final]], words: Words, *,
                 interval_s: float, variant: str, device: torch.device, temperature: float = 1.0,
                 attention: str = COPIED, executor_device: torch.device | None = None) -> None:
        if executor_device is not None and loop.executor.state.device.type != torch.device(executor_device).type:
            raise ValueError(f"the loop's executor is on {loop.executor.state.device}, not on {executor_device}")
        #: the device of the executor and of its copies and joins (class docstring); None: the speaker's device
        self.executor_device = None if executor_device is None else torch.device(executor_device)
        self.loop, self.order, self.words = loop, list(order), words
        count = len(self.order)
        rows = [sentences[i].rows for i in self.order]                   # the rows alone are read (D82)
        self.every, self.start = interval_rows(interval_s, words.spec.step_s), rows[0].start
        if any(r.start != self.start for r in rows):
            raise ValueError("the flights of a loop start their first predicted step at one Δ row")
        self.flights = [flights[i] for i in self.order]
        self.geometries = [geometries[f["airport"]] for f in self.flights]
        self.finals = [finals[g.code] for g in self.geometries]
        #: each flight's join tick (module docstring): the start's
        self.join_ticks = loop.join_ticks.copy()
        self.speaker = Speaker(model, words, self.finals, capacity=int(self.join_ticks.max()) + self.start + FIRST_ROWS,
                               temperature=temperature, attention=attention)
        # the inputs of a row: the prior's one function of a loop's row (D96 item 4), each flight's own landings (D105)
        self.rows_of = LoopRows(self.geometries, landings, [own_flight_key(f) for f in self.flights],
                                np.array([utc_s(f["entry_time_utc"]) for f in self.flights]),
                                np.array([r.first_row for r in rows]), self.start, variant=variant,
                                interval_s=interval_s, step_s=words.spec.step_s, device=device,
                                join_ticks=self.join_ticks)
        shape = (self.start * self.every, len(STATE_COLUMNS))
        wrong = [i for i in self.order if np.shape(observed[i]) != shape]
        if wrong:
            raise ValueError(f"flights {wrong[:5]}: observed rows of shape {np.shape(observed[wrong[0]])}, not the {shape} "
                             f"before the first predicted step (the start's, `start_moved`)")
        self.observed = np.stack([observed[i] for i in self.order])                 # [B, start·every, 6]
        #: the tick the next `observe` or `step` reads (the Δ row of a flight of join tick 0)
        self.t = 0
        #: the flights still flown: not done by the executor, not ended by the caller
        self.alive = np.ones(count, dtype=bool)
        #: the flights the caller ended (`end`)
        self.ended = np.zeros(count, dtype=bool)
        #: the flights the speaker and the executor hold (module docstring, S6), and the records of those that left
        self.held = np.ones(count, dtype=bool)
        self.left: dict[int, Left] = {}
        self._slot = np.arange(count)
        self.current: np.ndarray | None = None
        self.before: np.ndarray | None = None
        # each flight's records, of the rows of its sentence
        self._inputs: list[RowTensors] = []                     # every row's inputs (CPU), observed and said
        self._said: list[list[np.ndarray]] = [[] for _ in self.order]
        self._flown: list[list[np.ndarray]] = [[] for _ in self.order]
        self._probability: list[list[float]] = [[] for _ in self.order]
        self._permitted: list[list[bool]] = [[] for _ in self.order]
        self._on_final: list[list[bool]] = [[] for _ in self.order]
        self._blocked: list[list[dict[int, np.ndarray]]] = [[] for _ in self.order]

    @property
    def observing(self) -> bool:
        """Whether an observed row is still to be encoded (`observe`) before the first predicted step: the phase of a
        loop whose join ticks are all 0 (module docstring)."""
        return self.t < self.start and not self.join_ticks.any()

    def roles(self) -> np.ndarray:
        """``[B]`` each flight's role at the next tick (`speaker.ABSENT`, `OBSERVED` or `SAID`, by its own clock)."""
        own = self.t - self.join_ticks
        return np.where(own < 0, ABSENT, np.where(own < self.start, OBSERVED, SAID))

    def own_row(self) -> np.ndarray:
        """``[B]`` each flight's own Δ row at the next tick (negative: it has not joined)."""
        return self.t - self.join_ticks

    def joined(self) -> np.ndarray:
        """``[B]`` bool: the flights that have joined at the next tick (observed or said: not absent)."""
        return self.own_row() >= 0

    def said_now(self) -> np.ndarray:
        """``[B]`` bool: the flights said at the next tick (from their first predicted step on)."""
        return self.own_row() >= self.start

    def observe(self, extra: Any = None) -> RowTensors:
        """Encode the next observed row (its states, the 2 s row before it; row 0 has no motion, D60), with the caller's
        input of the added modules ``extra``; its inputs."""
        if not self.observing:
            raise ValueError(f"the {self.start} observed rows are encoded; the loop says its rows (`step`)")
        r = self.t * self.every
        tensors, at = self.rows_of(self.t, self.observed[:, r], self.observed[:, max(r - 1, 0)], r > 0,
                                   self.speaker.heard)
        self.speaker.observe(tensors, [at], extra)
        self._inputs.append(_on_cpu(tensors))
        self.t += 1
        if not self.observing:                  # the first predicted step reads the flown states from here on
            self.current, self.before = self.loop.rows(), self.observed[:, self.start * self.every - 1]
            for b in range(len(self.order)):
                self._flown[b].append(self.current[b])
        return tensors

    def _at(self, current: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each flight's state at the next tick, the 2 s row before and whether that row is its own (module docstring):
        a said flight's flown (``current``, ``previous``), an observed one's observed (row 0 has no motion, D60), an
        absent one's its observed row 0 (finite: its row is not present)."""
        own = self.own_row()
        r = np.clip(own, 0, self.start - 1) * self.every
        index = np.arange(len(self.order))
        at, before = self.observed[index, r].copy(), self.observed[index, np.maximum(r - 1, 0)].copy()
        known = (own >= 0) & (r > 0)
        said = own >= self.start
        if said.any():
            at[said], before[said], known[said] = current[said], previous[said], True
        return at, before, known

    def step(self, numbers: np.ndarray, caller: Mapping[int, np.ndarray] | None = None, extra: Any = None
             ) -> np.ndarray:
        """Say one row of every flight and fly it (module docstring): ``numbers`` ``[B, 5]`` the caller's uniform
        numbers (`Speaker.speak`), ``caller`` its masks by column (the runway column's joined with the bound of D68),
        ``extra`` its input of the added modules. ``[B, 5]`` the words said (an ended flight's too, which are not its
        sentence's; `UNCHANGED` for a flight absent or observed at the tick, or one that left, S6). A row the executor
        refuses (`start.RowRefused`, `start.GoAroundBeyondMost`) is refused whole: the loop is as it was. With join
        ticks, a tick advances every flight by its own clock (module docstring)."""
        if self.observing:
            raise ValueError(f"{self.start - self.t} observed rows are still to be encoded (`observe`)")
        if not self.alive.any():
            raise ValueError("every flight of the loop has ended")
        held = self.inside
        roles = self.roles()
        said_now = roles == SAID
        current, previous = self.current, self.before
        beginning = np.zeros(len(self.order), dtype=bool)
        if self.join_ticks.any():
            # the flights at their first predicted step, each from the state its executor holds it in: on copies, kept
            # once the row is (a row refused leaves the loop as it was)
            beginning = self.own_row() == self.start
            rows_now = self._each(self.loop.rows())
            if current is None:                              # the first tick of a loop with join ticks
                current, previous = rows_now, self.observed[:, self.start * self.every - 1].copy()
            else:
                current, previous = current.copy(), previous.copy()
            current[beginning] = rows_now[beginning]
            previous[beginning] = self.observed[beginning, self.start * self.every - 1]
            at_now, before_now, known = self._at(current, previous)
            known = known[held]
        else:
            at_now, before_now, known = current, previous, True
        # the speaker and the executor read and say the held flights only
        tensors, at = self.rows_of(self.t, at_now[held], before_now[held], known, self.speaker.heard)
        masks = {c: mask[held] for c, mask in (caller or {}).items()}
        bound = go_around_bound(self.speaker.go_arounds, self.words, int(self.speaker.n_candidates.max()),
                                self.loop.most_go_arounds)
        masks[RUNWAY] = bound & masks[RUNWAY] if RUNWAY in masks else bound
        before = self.speaker.in_force            # the runway in force under which the row's runway word is drawn
        flown: list[tuple[np.ndarray, np.ndarray]] = []
        flying = self.t >= self.start                        # the executor's clock: its step 0 is tick s
        if flying and self.loop.steps != self.t - self.start:
            raise ValueError(f"the loop's executor is at its step {self.loop.steps}, not tick {self.t}'s")
        # the executor flies the row before the speaker keeps it: a row it refuses (vocabulary D80) leaves the speaker,
        # the records and the loop's row as they were
        accept = lambda said: flown.append(self.loop.step(said)) if flying else None      # noqa: E731
        spoken = (self.speaker.speak(tensors, at, numbers[held], masks, extra, accept=accept) if said_now[held].all()
                  else self.speaker.say(tensors, at, roles[held], numbers[held], masks, extra, accept=accept))
        words_row = np.full((len(self.order), len(COLUMNS)), UNCHANGED, dtype=spoken.dtype)
        words_row[held] = spoken
        self._inputs.append(_on_cpu(tensors))
        self.current, self.before = current, previous
        for b in np.flatnonzero(beginning):
            self._flown[b].append(self.current[b])
        if not flying:
            self.t += 1
            return words_row
        (rows, done), = flown
        rows, done = self._each(rows), self._each(done, fill=False)
        speaking = self.alive & said_now
        for b in np.flatnonzero(speaking):
            k = self._slot[b]                                # its place in the speaker and the executor
            final = self.finals[b][before[k].runway] if before[k] is not None else None
            self._said[b].append(words_row[b])
            self._flown[b] += list(rows[b])
            self._probability[b].append(float(self.speaker.go_around_probability[-1][k]))
            self._permitted[b].append(bool(self.speaker.go_around_permitted[-1][k]))
            self._on_final[b].append(final is not None and bool(final.inside(np.array(at.e_m[k]), np.array(at.n_m[k]))))
            self._blocked[b].append({c: mask[k] for c, mask in self.speaker.procedure_blocked[-1].items()})
        self._halt(done)
        moved = (self.alive & said_now)[:, None]
        self.before = np.where(moved, rows[:, -2] if self.every > 1 else self.current, self.before)
        self.current = np.where(moved, rows[:, -1], self.current)
        self.t += 1
        return words_row

    @property
    def inside(self) -> np.ndarray:
        """The places in ``order`` of the flights the speaker and the executor hold, in their order there."""
        return np.flatnonzero(self.held)

    def _each(self, values: np.ndarray, fill: Any = np.nan) -> np.ndarray:
        """``values`` of the held flights (the speaker's or the executor's, in their order) at each flight's place in
        ``order``. A flight that left gets ``fill``."""
        out = np.full((len(self.order), *values.shape[1:]), fill, dtype=values.dtype)
        out[self.held] = values
        return out

    def add_landing(self, flights: Sequence[int], landing: Landing) -> None:
        """``landing`` added to the landings of the flights ``flights`` (places in ``order``; module docstring): the rows
        after its time count it (`LoopRows.add_landing`)."""
        self.rows_of.add_landing([self._slot[b] for b in flights if self.held[b]], landing)

    def end(self, flights: np.ndarray) -> None:
        """The caller ends the flights ``flights`` (``[B]`` bool) after the row said last (a loss of separation,
        post-training D93): halted as a done flight is. Its outcome is the caller's: the judge has none for a flight the
        executor is not done with (`said`, `states` give its sentence; `generated` refuses it)."""
        flights = np.asarray(flights, dtype=bool)
        self.ended |= flights & self.alive
        self._halt(flights)

    def _halt(self, flights: np.ndarray) -> None:
        # a halted flight keeps, as its inputs, the finite state of the row it ended in: the executor flies a done
        # flight on (a non-finite state is one of its ends) and the speaker still says its rows
        self.alive &= ~flights
        self.loop.halt(~self.alive[self.held])

    def permitted(self, flights: Sequence[int]) -> list[Permitted]:
        """The speaker's records of the words every mask permitted at each row said to each of ``flights`` (places in
        ``order``; §7 item 3), each the flight's own: the rows of its sentence, to the row it ended in, and in each
        column the classes of its airport. A record does not change with the other flights of the loop or with the
        rows said after the flight ended (multi-aircraft control S6). The loss reads the same words from it as from the
        speaker's whole record (`train.masked_log_probability` finds a row by its time). One call reads the speaker
        once."""
        flights = list(flights)
        held = [b for b in flights if b not in self.left]
        record = self.speaker.permitted([self._slot[b] for b in held]) if held else None
        place = {b: k for k, b in enumerate(held)}
        out = []
        for b in flights:
            if b in self.left:
                out.append(self.left[b].permitted)
                continue
            k, rows = place[b], len(self._said[b])
            candidates = len(self.geometries[b].candidates)
            masks = tuple(mask[k: k + 1, :rows, : len(column_words(c, self.words, candidates))].copy()
                          for c, mask in enumerate(record.masks))
            out.append(Permitted(masks, record.time_s[k: k + 1, :rows].copy(), record.own[k: k + 1, :rows].copy(),
                                 record.temperature))
        return out

    def sentences(self, split: str, flights: Sequence[int] | None = None) -> list[SentenceRows]:
        """Each flight's rows as the loop said them (D106 item 3): the inputs of its observed rows and of each row it
        said to the row it ended in, the words said as the targets — for `train.masked_log_probability` with
        `permitted`, through `batch.collate`; ``split`` the flights' split (a label of the rows); of the flights
        ``flights`` only (places in ``order``, in that order; every flight when not given)."""
        out = []
        for b in range(len(self.flights)) if flights is None else flights:
            inputs = self.left[b].inputs if b in self.left else self._own_inputs(b)
            said = len(self._said[b])
            targets = np.full((self.start + said, len(COLUMNS)), UNCHANGED, dtype=np.int64)
            targets[self.start:] = np.array(self._said[b], dtype=np.int64).reshape(said, len(COLUMNS))
            out.append(SentenceRows(flight_key=self.flights[b]["dataset_id"], airport=self.geometries[b].code,
                                    split=split, first_step=self.start, targets=targets, **inputs))
        return out

    def _own_inputs(self, flight: int) -> dict[str, np.ndarray]:
        """The inputs of the rows of held ``flight``'s sentence (its own rows from its join tick to the row it ended
        in), by `SentenceRows` field: its candidates only."""
        said, join, k = len(self._said[flight]), int(self.join_ticks[flight]), self._slot[flight]
        rows = self._inputs[join: join + self.start + said]
        slots = len(self.geometries[flight].candidates)
        return {"time_s": np.array([r.time_s[k, 0] for r in rows]), "own": np.stack([r.own[k, 0] for r in rows]),
                "candidates": np.stack([r.candidates[k, 0, :slots] for r in rows]),
                "runway_in_force": np.array([r.runway_in_force[k, 0] for r in rows]),
                "go_around": np.array([r.go_around[k, 0] for r in rows]),
                "heading_in_force": np.stack([r.heading_in_force[k, 0] for r in rows]),
                "words_in_force": np.stack([r.words_in_force[k, 0] for r in rows]),
                "since": np.stack([r.since[k, 0] for r in rows])}

    def _leaving(self, flights: Sequence[int]) -> list[Left]:
        """The records of held ``flights``, no longer flown, as they leave the speaker and the executor (S6). Their
        arrays are read-only: copies and joins share them."""
        out = []
        for b, permitted in zip(flights, self.permitted(flights)):
            k = self._slot[b]
            left = Left(None if self.ended[b] else self.loop.outcome(k), self.states(b), self.rows_of.landings[k],
                        self._own_inputs(b), permitted)
            for value in [left.states, *left.inputs.values(), *permitted.masks, permitted.time_s, permitted.own]:
                value.flags.writeable = False
            out.append(left)
        return out

    def outcome(self, flight: int) -> Outcome:
        """``flight``'s outcome by the judge (`Loop.outcome`), held or left; refused for a flight the executor is not
        done with (still flown, or ended by the caller)."""
        if flight not in self.left:
            return self.loop.outcome(self._slot[flight])
        if self.left[flight].outcome is None:
            raise ValueError(f"flight {flight} was ended by the caller: it has no judge's outcome")
        return self.left[flight].outcome

    def landings(self, flight: int) -> LandingIndex:
        """``flight``'s landings as the loop counts them now (`add_landing`); a flight that left, those when it left."""
        return self.left[flight].landings if flight in self.left else self.rows_of.landings[self._slot[flight]]

    def in_force(self, flight: int) -> Any:
        """The words in force of held ``flight`` (the speaker's, `Speaker.in_force`)."""
        return self.speaker.in_force[self._slot[flight]]

    def copy(self, flights: Sequence[int], device: torch.device | None = None) -> SpeakingLoop:
        """A loop of copies of the flights ``flights`` (places in ``order``, repeats permitted; post-training D94): the
        start's own copy of its flights (`Loop.copy`, vocabulary D97), the speaker's (`Speaker.copy`), the inputs'
        (`LoopRows.select`) and every record. Flown on with the same numbers, masks and input of the added modules, a
        copy says what its original says and flies it within the executor's bound (vocabulary D97 (3)); the loop copied
        is unchanged. ``device``: the copy's (this loop's when not given; multi-aircraft control D182: a copy held in the
        host's memory, flown on the GPU later). A copy of a flight that left has its record (S6)."""
        index = list(flights)
        out = object.__new__(SpeakingLoop)
        out.executor_device = self.executor_device
        out.held = self.held[index].copy()
        out.left = {n: self.left[b] for n, b in enumerate(index) if b in self.left}
        out._slot = np.cumsum(out.held) - 1
        slots = self._slot[index][out.held]
        if len(slots):
            out.loop = self.loop.copy(slots, device if self.executor_device is None else self.executor_device)
            out.speaker = self.speaker.copy(slots, device)
            out.rows_of = self.rows_of.select(slots, device)
        else:                                       # every flight copied has left: no speaker or executor
            out.loop = out.speaker = out.rows_of = None
        out.order, out.words, out.every, out.start, out.t = [self.order[i] for i in index], self.words, self.every, \
            self.start, self.t
        out.join_ticks = self.join_ticks[index].copy()
        out.flights, out.geometries = [self.flights[i] for i in index], [self.geometries[i] for i in index]
        out.finals = [self.finals[i] for i in index]
        out.observed, out.alive, out.ended = self.observed[index].copy(), self.alive[index].copy(), self.ended[index].copy()
        out.current = None if self.current is None else self.current[index].copy()
        out.before = None if self.before is None else self.before[index].copy()
        out._inputs = [RowTensors(*(value[slots] for value in row)) for row in self._inputs] if len(slots) else []
        for name in ("_said", "_flown", "_probability", "_permitted", "_on_final", "_blocked"):
            setattr(out, name, [list(getattr(self, name)[i]) for i in index])
        return out

    @staticmethod
    def together(parts: Sequence[tuple[SpeakingLoop, Sequence[int]]], device: torch.device, room: int | None,
                 spent: SpeakingLoop | None = None) -> SpeakingLoop:
        """Flights of several loops, each at its own tick, said and flown on as one loop (multi-aircraft control D188: a
        backward batch's searches stepped together), on ``device``: ``parts`` each a loop from its first predicted step
        on and its flights (places in its ``order``, repeats permitted). Each part's clock is moved later by the ticks
        it is behind the latest, its flights' join ticks with it (module docstring: a flight joins the loop at its own
        tick), and the ticks before any flight joined are left out. Each flight keeps what `copy` keeps of it
        (`Loop.together`, `Speaker.together`, `LoopRows.together`) and its own rows: flown on with the same numbers,
        masks and input of the added modules, it says what it says in its own loop and flies it within the executor's
        bound. ``room``: the rows its speaker's cache has room for past the rows written (`Speaker.together`); ``spent``:
        a loop among the parts that nothing reads after (the one this loop replaces): its speaker's cache freed as it
        is copied. A flight no longer flown leaves the speaker and the executor here (module docstring, S6): the loop
        keeps its record (`Left`)."""
        index = [np.asarray(list(flights), dtype=np.int64) for _, flights in parts]
        for loop, _ in parts:
            if loop.t < loop.start or (loop.speaker is not None and len(loop.speaker.said_rows) != loop.t - loop.start):
                raise ValueError("a loop goes with others from its first predicted step on, a row said each tick since")
        tick = max(loop.t - int(loop.join_ticks[i].min()) for (loop, _), i in zip(parts, index))
        shifts = [tick - loop.t for loop, _ in parts]
        moved = [(loop, i, shift) for (loop, _), i, shift in zip(parts, index, shifts)]
        first = parts[0][0]
        if len({loop.executor_device for loop, _ in parts}) != 1:
            raise ValueError("the joined loops must hold their executors on the same device")
        out = object.__new__(SpeakingLoop)
        out.executor_device = first.executor_device
        # each flight still flown stays in the speaker and the executor; each other one leaves them with its record,
        # made once for a flight joined more than once (each part's records in one call)
        out.held = np.concatenate([loop.held[i] & loop.alive[i] for loop, i, _ in moved])
        records: dict[tuple[int, int], Left] = {}
        for loop, i, _ in moved:
            leaving = sorted({int(b) for b in i if loop.held[b] and not loop.alive[b]} - {b for key, b in records
                                                                                          if key == id(loop)})
            records.update({(id(loop), b): left for b, left in zip(leaving, loop._leaving(leaving))})
            records.update({(id(loop), b): left for b, left in loop.left.items()})
        out.left, n = {}, 0
        for loop, i, _ in moved:
            for b in i:
                if not (loop.held[b] and loop.alive[b]):
                    out.left[n] = records[(id(loop), int(b))]
                n += 1
        out._slot = np.cumsum(out.held) - 1
        kept = [(loop, loop._slot[i[loop.held[i] & loop.alive[i]]], shift) for loop, i, shift in moved]
        kept = [(loop, slots, shift) for loop, slots, shift in kept if len(slots)]
        if kept:
            out.loop = Loop.together([(loop.loop, slots, shift) for loop, slots, shift in kept],
                                     device if first.executor_device is None else first.executor_device)
            out.speaker = Speaker.together([(loop.speaker, slots, shift) for loop, slots, shift in kept], device, room,
                                           None if spent is None else spent.speaker)
            out.rows_of = LoopRows.together([(loop.rows_of, slots, shift) for loop, slots, shift in kept], device)
        else:                                       # no flight is flown: no speaker or executor
            out.loop = out.speaker = out.rows_of = None
        out.order = [loop.order[b] for loop, i, _ in moved for b in i]
        out.words, out.every, out.start, out.t = first.words, first.every, first.start, tick
        out.join_ticks = np.concatenate([loop.join_ticks[i] + shift for loop, i, shift in moved])
        for name in ("flights", "geometries", "finals"):
            setattr(out, name, [getattr(loop, name)[b] for loop, i, _ in moved for b in i])
        for name in ("observed", "alive", "ended"):
            setattr(out, name, np.concatenate([getattr(loop, name)[i] for loop, i, _ in moved]))
        out.current = np.concatenate([loop.current[i] for loop, i, _ in moved])
        out.before = np.concatenate([loop.before[i] for loop, i, _ in moved])
        out._inputs = _inputs_together(kept, tick) if kept else []
        for name in ("_said", "_flown", "_probability", "_permitted", "_on_final", "_blocked"):
            setattr(out, name, [list(getattr(loop, name)[b]) for loop, i, _ in moved for b in i])
        return out

    def said(self, flight: int) -> np.ndarray:
        """``[M, 5]`` the words said to ``flight`` (its place in ``order``) from its first predicted step to the row it
        ended in (or the row said last)."""
        return np.array(self._said[flight], dtype=np.int64).reshape(-1, len(COLUMNS))

    def states(self, flight: int) -> np.ndarray:
        """``[rows, 6]`` ``flight``'s states on the 2 s rows from its row 0: observed before the first predicted step,
        flown from it — to the first row at or after the cycle the executor was done with it in, or to the end of the
        row it was ended in (or the row said last)."""
        if flight in self.left:
            return self.left[flight].states
        flown = self._flown[flight]
        if not self.alive[flight] and not self.ended[flight]:            # done by the executor: to its end
            done = int(self.loop.executor.done_cycle[self._slot[flight]])
            flown = flown[: int(np.ceil((done + 1) / self.loop.row_cycles)) + 1]
        # an aircraft ended at its first predicted step before a row flown (a loop of later joins has no observe
        # phase to write that row) has its observed rows only
        return np.concatenate((self.observed[flight], np.array(flown).reshape(-1, self.observed.shape[2])))

    def generated(self, flights: Sequence[int] | None = None) -> list[Generated]:
        """The sentences of ``flights`` (places in ``order``; default every flight), each done by the executor
        (`Generated`, the judge's outcome); a flight still flown, or ended by the caller (its outcome is the caller's:
        `said`, `states`), is refused by name."""
        flights = range(len(self.order)) if flights is None else list(flights)
        open_ = [b for b in flights if self.alive[b] or self.ended[b]]
        if open_:
            raise ValueError(f"flights {open_} are still flown or were ended by the caller: no judge's outcome "
                             f"(`said`, `states`)")
        out = []
        for b in flights:
            outcome = self.outcome(b)               # why it ended: the judge's (vocabulary D90)
            said = self._said[b]
            out.append(Generated(index=self.order[b], words=self.said(b), states=self.states(b),
                                 outcome=outcome.outcome, crossing=outcome.crossing, timed_out=outcome.outcome == TIMEOUT,
                                 # the go-arounds of its own words: the speaker says an ended flight's rows too
                                 go_arounds=int(sum(row[RUNWAY] == RUNWAY_GO_AROUND for row in said)),
                                 go_around_probability=np.array(self._probability[b]),
                                 go_around_permitted=np.array(self._permitted[b], dtype=bool),
                                 on_final=np.array(self._on_final[b], dtype=bool),
                                 blocked={c: np.array([row[c] for row in self._blocked[b]], dtype=bool)
                                          for c in self._blocked[b][0]}))
        return out


def _inputs_together(parts: Sequence[tuple[SpeakingLoop, np.ndarray, int]], length: int) -> list[RowTensors]:
    """The inputs of every tick of a loop of flights together (`SpeakingLoop.together`), ``length`` ticks: ``parts``
    each a loop, its flights held on (their places in its speaker) and its shift; each part's inputs of those flights
    moved by its shift (a tick before its first: no row present); the candidates of every tick padded to the most of
    any (a copy's rows have its own batch's, fewer than those of the batch it was copied from)."""
    rows = [loop._inputs[max(-shift, 0):] for loop, _, shift in parts]
    most = max(r.valid.shape[1] for kept in rows for r in kept)

    def together(name: str) -> np.ndarray:                       # [B, length, ...]: one row a tick
        blocks = []
        for kept, (_, i, _) in zip(rows, parts):
            # the candidates' validity is a tick's [B, K]; every other input a tick's [B, 1, ...]; K padded to the most
            if name == "valid":
                ticks = [v if v.shape[1] == most else np.pad(v, ((0, 0), (0, most - v.shape[1])))
                         for v in (r.valid for r in kept)]
                block = np.stack(ticks, axis=1)[i]
            else:
                ticks = [getattr(r, name) for r in kept]
                if name == "candidates":
                    ticks = [c if c.shape[2] == most else np.pad(c, ((0, 0), (0, 0), (0, most - c.shape[2]), (0, 0)))
                             for c in ticks]
                block = np.concatenate(ticks, axis=1)[i]
            blocks.append(np.pad(block, [(0, 0), (length - block.shape[1], 0)] + [(0, 0)] * (block.ndim - 2)))
        return np.concatenate(blocks)

    values = {name: together(name) for name in RowTensors._fields}
    return [RowTensors(*(value[:, t] if name == "valid" else value[:, t: t + 1] for name, value in values.items()))
            for t in range(length)]


def _on_cpu(rows: RowTensors) -> RowTensors:
    """A row's inputs as numpy arrays on the CPU (the loop keeps every row's for `SpeakingLoop.sentences`)."""
    return RowTensors(*(value.detach().cpu().numpy() for value in rows))
