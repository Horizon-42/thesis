"""The step of a speaker's closed loop (prior design §7 item 7; D106 item 1): the prior speaks, the executor flies, one
Δ row at a time — shared by free generation (`prior_free_generation`) and the post-training's window loop, so neither
holds a copy of it.

A `SpeakingLoop` holds a batch of flights started at their first predicted step through the start of a closed loop
(vocabulary §6 item 5, D67: `autopilot.start`; the most go-arounds of a flight is the loop's, D68). The rows before the
first predicted step are the flight's observed rows (the closed-loop sentence's states there, D82: its rows alone), each
encoded by `observe` one at a time, as the speaker reads them. From the first predicted step on, `step` says one row:
the inputs from the states the executor flew (D32, §2) through the prior's one function of a loop's row
(`prior.loop.LoopRows`, D96 item 4, with each flight's own landings, D105), the speaker under its masks, the bound of D68
("go-around" forbidden after a flight's last) and the caller's masks, with the caller's random numbers and input of the
added modules; the loop flies the row for Δ seconds. A flight ends when the executor is done with it (its crossing, its
time limit, or the dynamics), or when the caller ends it (`end`: a loss of separation, post-training D93); an ended
flight is halted and keeps, as its inputs, the finite state of the row it ended in: the speaker still says its rows
(a batch is said together), which are not its sentence's. Its outcome is the judge's (`Loop.outcome`, item 6). This
module imports nothing else of `autopilot/` (`tests/test_architecture.py`, D69).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.start import Loop
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.labeller.interval import interval_rows
from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words
from ts_transformer.prior.batch import RowTensors, SentenceRows
from ts_transformer.prior.inputs import own_flight_key
from ts_transformer.prior.landings import LandingIndex, utc_s
from ts_transformer.prior.loop import LoopRows
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import Final
from ts_transformer.prior.speaker import Permitted, Speaker, go_around_bound

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


class SpeakingLoop:
    """The closed loop of the flights ``order`` of ``loop`` (module docstring). ``sentences`` and ``flights`` (their
    records in the split's signals: airport, entry time, key) by their place in the signals; ``landings`` each flight's,
    in ``order`` (D105); ``finals`` each airport's (the procedure masks)."""

    def __init__(self, model: Prior, loop: Loop, order: Sequence[int], sentences: Mapping[int, ClosedLoopSentence],
                 flights: Mapping[int, Mapping[str, Any]], geometries: Mapping[str, AirportGeometry],
                 landings: Sequence[LandingIndex], finals: Mapping[str, Sequence[Final]], words: Words, *,
                 interval_s: float, variant: str, device: torch.device, temperature: float = 1.0) -> None:
        self.loop, self.order, self.words = loop, list(order), words
        count = len(self.order)
        rows = [sentences[i].rows for i in self.order]                   # the rows alone are read (D82)
        self.every, self.start = interval_rows(interval_s, words.spec.step_s), rows[0].start
        if any(r.start != self.start for r in rows):
            raise ValueError("the flights of a loop start their first predicted step at one Δ row")
        self.flights = [flights[i] for i in self.order]
        self.geometries = [geometries[f["airport"]] for f in self.flights]
        self.finals = [finals[g.code] for g in self.geometries]
        self.speaker = Speaker(model, words, self.finals, capacity=self.start + FIRST_ROWS, temperature=temperature)
        # the inputs of a row: the prior's one function of a loop's row (D96 item 4), each flight's own landings (D105)
        self.rows_of = LoopRows(self.geometries, landings, [own_flight_key(f) for f in self.flights],
                                np.array([utc_s(f["entry_time_utc"]) for f in self.flights]),
                                np.array([r.first_row for r in rows]), self.start, variant=variant,
                                interval_s=interval_s, step_s=words.spec.step_s, device=device)
        self.observed = np.stack([r.states[: self.start * self.every] for r in rows])     # [B, start·every, 6]
        #: the Δ row the next `observe` or `step` reads
        self.t = 0
        #: the flights still flown: not done by the executor, not ended by the caller
        self.alive = np.ones(count, dtype=bool)
        #: the flights the caller ended (`end`)
        self.ended = np.zeros(count, dtype=bool)
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
        """Whether an observed row is still to be encoded (`observe`) before the first predicted step."""
        return self.t < self.start

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

    def step(self, numbers: np.ndarray, caller: Mapping[int, np.ndarray] | None = None, extra: Any = None
             ) -> np.ndarray:
        """Say one row of every flight and fly it (module docstring): ``numbers`` ``[B, 5]`` the caller's uniform
        numbers (`Speaker.speak`), ``caller`` its masks by column (the runway column's joined with the bound of D68),
        ``extra`` its input of the added modules. ``[B, 5]`` the words said (an ended flight's too, which are not its
        sentence's)."""
        if self.observing:
            raise ValueError(f"{self.start - self.t} observed rows are still to be encoded (`observe`)")
        if not self.alive.any():
            raise ValueError("every flight of the loop has ended")
        tensors, at = self.rows_of(self.t, self.current, self.before, True, self.speaker.heard)
        masks = dict(caller or {})
        bound = go_around_bound(self.speaker.go_arounds, self.words, int(self.speaker.n_candidates.max()),
                                self.loop.most_go_arounds)
        masks[RUNWAY] = bound & masks[RUNWAY] if RUNWAY in masks else bound
        before = self.speaker.in_force            # the runway in force under which the row's runway word is drawn
        words_row = self.speaker.speak(tensors, at, numbers, masks, extra)
        rows, done = self.loop.step(words_row)
        self._inputs.append(_on_cpu(tensors))
        for b in np.flatnonzero(self.alive):
            final = self.finals[b][before[b].runway] if before[b] is not None else None
            self._said[b].append(words_row[b])
            self._flown[b] += list(rows[b])
            self._probability[b].append(float(self.speaker.go_around_probability[-1][b]))
            self._permitted[b].append(bool(self.speaker.go_around_permitted[-1][b]))
            self._on_final[b].append(final is not None and bool(final.inside(np.array(at.e_m[b]), np.array(at.n_m[b]))))
            self._blocked[b].append({c: mask[b] for c, mask in self.speaker.procedure_blocked[-1].items()})
        self._halt(done)
        self.before = np.where(self.alive[:, None], rows[:, -2] if self.every > 1 else self.current, self.before)
        self.current = np.where(self.alive[:, None], rows[:, -1], self.current)
        self.t += 1
        return words_row

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
        self.loop.halt(~self.alive)

    def permitted(self) -> Permitted:
        """The speaker's records of the words every mask permitted at each row it said (§7 item 3; an ended flight's
        later rows too, which no row of its sentence asks)."""
        return self.speaker.permitted()

    def sentences(self, split: str) -> list[SentenceRows]:
        """Each flight's rows as the loop said them (D106 item 3): the inputs of its observed rows and of each row it
        said to the row it ended in, the words said as the targets — for `train.masked_log_probability` with
        `permitted`, through `batch.collate`; ``split`` the flights' split (a label of the rows)."""
        out = []
        for b, flight in enumerate(self.flights):
            said = len(self._said[b])
            rows = self._inputs[: self.start + said]
            slots = len(self.geometries[b].candidates)
            targets = np.full((len(rows), len(COLUMNS)), UNCHANGED, dtype=np.int64)
            targets[self.start:] = np.array(self._said[b], dtype=np.int64).reshape(said, len(COLUMNS))
            out.append(SentenceRows(
                flight_key=flight["dataset_id"], airport=self.geometries[b].code, split=split,
                first_step=self.start, time_s=np.array([r.time_s[b, 0] for r in rows]),
                own=np.stack([r.own[b, 0] for r in rows]), candidates=np.stack([r.candidates[b, 0, :slots] for r in rows]),
                runway_in_force=np.array([r.runway_in_force[b, 0] for r in rows]),
                go_around=np.array([r.go_around[b, 0] for r in rows]),
                heading_in_force=np.stack([r.heading_in_force[b, 0] for r in rows]),
                words_in_force=np.stack([r.words_in_force[b, 0] for r in rows]),
                since=np.stack([r.since[b, 0] for r in rows]), targets=targets))
        return out

    def said(self, flight: int) -> np.ndarray:
        """``[M, 5]`` the words said to ``flight`` (its place in ``order``) from its first predicted step to the row it
        ended in (or the row said last)."""
        return np.array(self._said[flight], dtype=np.int64).reshape(-1, len(COLUMNS))

    def states(self, flight: int) -> np.ndarray:
        """``[rows, 6]`` ``flight``'s states on the 2 s rows from its row 0: observed before the first predicted step,
        flown from it — to the first row at or after the cycle the executor was done with it in, or to the end of the
        row it was ended in (or the row said last)."""
        flown = self._flown[flight]
        if not self.alive[flight] and not self.ended[flight]:            # done by the executor: to its end
            done = int(self.loop.executor.done_cycle[flight])
            flown = flown[: int(np.ceil((done + 1) / self.loop.row_cycles)) + 1]
        return np.concatenate((self.observed[flight], np.array(flown)))

    def generated(self, flights: Sequence[int] | None = None) -> list[Generated]:
        """The sentences of ``flights`` (places in ``order``; default every flight), each done by the executor
        (`Generated`, the judge's outcome); a flight still flown, or ended by the caller (its outcome is the caller's:
        `said`, `states`), is refused by name."""
        flights = range(len(self.order)) if flights is None else list(flights)
        open_ = [b for b in flights if self.alive[b] or self.ended[b]]
        if open_:
            raise ValueError(f"flights {open_} are still flown or were ended by the caller: no judge's outcome "
                             f"(`said`, `states`)")
        timed_out = self.loop.timed_out()
        out = []
        for b in flights:
            outcome = self.loop.outcome(b)
            said = self._said[b]
            out.append(Generated(index=self.order[b], words=self.said(b), states=self.states(b),
                                 outcome=outcome.outcome, crossing=outcome.crossing, timed_out=bool(timed_out[b]),
                                 # the go-arounds of its own words: the speaker says an ended flight's rows too
                                 go_arounds=int(sum(row[RUNWAY] == RUNWAY_GO_AROUND for row in said)),
                                 go_around_probability=np.array(self._probability[b]),
                                 go_around_permitted=np.array(self._permitted[b], dtype=bool),
                                 on_final=np.array(self._on_final[b], dtype=bool),
                                 blocked={c: np.array([row[c] for row in self._blocked[b]], dtype=bool)
                                          for c in self._blocked[b][0]}))
        return out


def _on_cpu(rows: RowTensors) -> RowTensors:
    """A row's inputs as numpy arrays on the CPU (the loop keeps every row's for `SpeakingLoop.sentences`)."""
    return RowTensors(*(value.detach().cpu().numpy() for value in rows))
