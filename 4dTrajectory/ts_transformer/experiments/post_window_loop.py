"""The window loop of the post-training (post-training §2 items 1–3, §8 C4; D29–D31, D91, D93, D105, D110): a module
shared by the runners of stage C, not a runner.

Each window's commanded aircraft is flown from the observed rows that the start gives back (`autopilot.start.start_moved`:
window B's moved, every other window's its stored rows, vocabulary D97 (4); C9), which the speaking loop takes as they
are (prior §7 item 7, B12), through the prior's step of a speaker's
closed loop (prior §7 item 7,
`prior_speaking_loop.SpeakingLoop`: the start of a closed loop, the prior's one function of a loop's row with each
window's own landings, D105, the speaker, the executor; the most go-arounds of a flight 2, D91). The loop adds the
scene, row by row:

- **the traffic module's input** at every row, observed and said: the tokens of the other aircraft of the window's step
  (`post.edges.tokens`, D98) — the edge features' conformance is checked first in every process (§4 item 1, D104);
- **the speed-word mask** (§3), a mask of a caller, computed once for a row from the state at the start of the row
  (D110), with the aircraft's runway and G in force before the row's words;
- **the separation judge** after each row the executor flew (`post.traffic.commanded_loss`, VISUAL): a loss of
  separation that the commanded aircraft answers for ends its window there (D93);
- **the reward** of D30 (`post.reward`), its present landing direction from the window's landings (D105);
- **the faulty points** of the recorded aircraft (D114; vocabulary D111, C1's census): the steps at which a recorded
  aircraft reads one (`post.fault_census.reads_fault`: its row or the row before it is a fault row), and whether the
  other aircraft of a loss reads one at the event's step or in the 2 Δ before it (the commanded aircraft flies on the
  executor: it has none). No rule is applied: they are reported.

A window ends where its commanded aircraft is done (the judge's outcome) or loses separation. No time limit is read here
(vocabulary D90): the executor ends a flight. A window without other aircraft says and flies what free generation says
and flies for the same flight with the same random numbers (§2 item 1): the module gives zero and the speed mask
permits every word.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.start import Loop, Move, moved_signals
from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop
from ts_transformer.inference.separation import Loss
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, SPEED, Words
from ts_transformer.post.conformance import Checked, require_conforming_edges
from ts_transformer.post.edges import tokens
from ts_transformer.post.fault_census import STEPS_BEFORE_EVENT, reads_fault
from ts_transformer.post.landings import roster_key, window_landings
from ts_transformer.post.reward import present_runways, reward
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import INSERTED_SUFFIX, AircraftAt, Recorded, Window, recorded, utc_s
from ts_transformer.post.speed_mask import along_course_speeds, speed_check
from ts_transformer.post.traffic import commanded_loss, joined
from ts_transformer.post.traffic import traffic as separation_traffic
from ts_transformer.post.traffic_attention import traffic_of
from ts_transformer.prior.landings import LandingIndex
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import Final
from ts_transformer.prior.speaker import MOST_GO_AROUNDS

#: The outcome of a window ended by a loss of separation that its commanded aircraft answers for (D30: reward 0).
LOST_SEPARATION = "lost_separation"
#: The edge references checked in this process (§4 item 1: once in every process that computes edge features).
_CHECKED: dict[Path, Checked] = {}


def checked_edges(reference: Path) -> Checked:
    """`require_conforming_edges` of ``reference``, run once in this process."""
    reference = Path(reference).resolve()
    if reference not in _CHECKED:
        _CHECKED[reference] = require_conforming_edges(reference)
    return _CHECKED[reference]


@dataclass(frozen=True)
class WindowResult:
    """One window's end: the commanded flight (its place in the split's signals), the kind of window, the outcome (the
    judge's, or `LOST_SEPARATION`), the loss and the other aircraft's key where it ended so (with the step of the loss),
    the reward, the go-arounds said, the words said, the states on the 2 s rows from row 0 and the rows where the speed
    mask acted (D101)."""

    index: int
    kind: str
    outcome: str
    loss: Loss | None
    loss_step: int | None
    other: str | None
    reward: float
    go_arounds: int
    words: np.ndarray
    states: np.ndarray
    speed_mask_rows: int
    faulty_steps: int               # D114: the steps at which a recorded aircraft reads a faulty point
    loss_reads_fault: bool          # D114: the loss's other aircraft reads one at the event or in the 2 Δ before it


class WindowLoop:
    """The windows ``windows`` (one for each flight of ``order``, in that order) flown in one batch (module docstring):
    ``loop`` started for their commanded flights (`autopilot.start.start`), ``sentences`` and ``flights`` (their records
    in the split's signals) by their place in the signals, ``rosters`` each airport's roster landings (each window's own
    landings from them, D105), ``finals`` each airport's, ``edges_reference`` the edge features' reference (D104)."""

    def __init__(self, model: Prior, loop: Loop, order: Sequence[int], windows: Sequence[Window],
                 sentences: Mapping[int, ClosedLoopSentence], flights: Mapping[int, Mapping[str, Any]],
                 geometries: Mapping[str, AirportGeometry], rosters: Mapping[str, LandingIndex],
                 finals: Mapping[str, Sequence[Final]], words: Words, *, interval_s: float, variant: str,
                 edges_reference: Path, faults: Mapping[str, Mapping[str, frozenset[int]]],
                 observed: Mapping[int, np.ndarray], device: torch.device) -> None:
        checked_edges(edges_reference)
        if loop.most_go_arounds != MOST_GO_AROUNDS:
            raise ValueError(f"the loop was started for {loop.most_go_arounds} go-arounds a flight, not {MOST_GO_AROUNDS} "
                             f"(D91)")
        if [w.signal_index for w in windows] != list(order):
            raise ValueError("one window for each flight of the loop, in its order")
        self.order, self.windows, self.words, self.device = list(order), list(windows), words, device
        self.landings = [window_landings(w, rosters[w.scene.geometry.code]) for w in self.windows]
        self.speaking = SpeakingLoop(model, loop, order, sentences, observed, flights, geometries, self.landings, finals,
                                     words, interval_s=interval_s, variant=variant, device=device)
        self.step_s = words.spec.step_s
        self.every = self.speaking.every
        self.geometries = [w.scene.geometry for w in self.windows]
        self.finals = [finals[g.code] for g in self.geometries]
        self.separations = {code: airport_separation(g) for code, g in geometries.items()}
        self.keys = [roster_key(flights[i]["dataset_id"], flights[i]["airport"]) for i in self.order]
        for w, i in zip(self.windows, self.order):
            row0 = utc_s(flights[i]["entry_time_utc"]) + sentences[i].rows.first_row * self.step_s
            if w.commanded.key != flights[i]["dataset_id"] or abs(w.row0_s - row0) > 1e-6:
                raise ValueError(f"window of {w.commanded.key}: not flight {flights[i]['dataset_id']}'s row 0")
        self.speed_words = len(column_words(SPEED, words, 1))
        self.loss: list[Loss | None] = [None] * len(self.order)
        self.loss_step: list[int | None] = [None] * len(self.order)
        self.other: list[str | None] = [None] * len(self.order)
        self.speed_mask_rows = np.zeros(len(self.order), dtype=np.int64)
        #: each window's marked recorded aircraft (D114), by key: their fault rows (an inserted aircraft keeps its
        #: source's, under its own key)
        self.faults = [_window_faults(w, faults[w.scene.geometry.code]) for w in self.windows]
        #: each window's steps read so far, with the keys of the recorded aircraft reading a faulty point there
        self._reading: list[dict[int, frozenset[str]]] = [{} for _ in self.order]
        #: each window's commanded aircraft, other aircraft and judged scene at the loop's next row, kept from the judge
        #: after a row flown: the same state, aircraft and words in force that the next row's mask and tokens read
        self._next: dict[int, tuple[AircraftAt, AircraftAt, AircraftAt, Any]] = {}
        #: each window's tokens at every row, observed and said (``[N, features]`` each): the loss reads the words said
        #: with the same input of the traffic module (post-training C7)
        self._tokens: list[list[np.ndarray]] = [[] for _ in self.order]

    # ---- the aircraft of a row
    def _own(self, b: int) -> AircraftAt:
        """The commanded aircraft of window ``b`` at the loop's row: its state there and at the 2 s row before (observed
        before the first predicted step, flown from it), its runway and G in force before the row's words."""
        speaking = self.speaking
        if speaking.observing:
            r = speaking.t * self.every
            at, before, known = speaking.observed[b, r], speaking.observed[b, max(r - 1, 0)], r > 0
        else:
            at, before, known = speaking.current[b], speaking.before[b], True
        in_force = speaking.speaker.in_force[b]
        runway, go_around = (in_force.runway, in_force.go_around) if in_force is not None else (-1, False)
        window = self.windows[b]
        return AircraftAt.of([(window.commanded.key, tuple(at[:3]), tuple(before[:3]), known, runway,
                               window.commanded.category, False, go_around)])

    def _traffic(self, owns: Sequence[AircraftAt], others: Sequence[AircraftAt]):
        """The traffic module's input of the row (each window's tokens, recorded for the loss)."""
        rows = [tokens(own, other, g, self.separations[g.code], self.step_s)
                for own, other, g in zip(owns, others, self.geometries)]
        for b, row in enumerate(rows):
            self._tokens[b].append(row)
        return traffic_of([[row] for row in rows], self.device)

    def _scene(self, b: int, own: AircraftAt, others: AircraftAt):
        g = self.geometries[b]
        aircraft = joined(own, others)
        return aircraft, separation_traffic(aircraft, g, self.separations[g.code], self.finals[b], self.step_s)

    def _read(self, b: int, t: int, others: AircraftAt) -> None:
        """Window ``b``'s step ``t``: the recorded aircraft that read a faulty point there (once a step, D114)."""
        if t in self._reading[b]:
            return
        time_s = self.windows[b].step_s(t)
        scene = self.windows[b].scene
        self._reading[b][t] = frozenset(key for key in others.keys if key in self.faults[b] and bool(reads_fault(
            scene.flight(key), np.array([time_s]), self.faults[b][key])[0]))

    def _speed_masks(self, owns: Sequence[AircraftAt], others: Sequence[AircraftAt]) -> np.ndarray:
        """``[B, words]`` the speed-word mask of each window at the start of the row (D110); every word for a window
        whose aircraft is no longer flown."""
        out = np.ones((len(self.order), self.speed_words), dtype=bool)
        for b in np.flatnonzero(self.speaking.alive):
            aircraft, scene = self._next[b][2:] if b in self._next else self._scene(b, owns[b], others[b])
            check = speed_check(scene, along_course_speeds(aircraft, scene, self.step_s), self.separations[
                self.geometries[b].code], self.words, len(self.geometries[b].candidates))
            out[b] = check.permitted
            self.speed_mask_rows[b] += check.applies
        return out

    # ---- the loop
    def observe(self) -> None:
        """Encode the next observed row, with its traffic."""
        t = self.speaking.t
        owns = [self._own(b) for b in range(len(self.order))]
        others = [w.others_at(t) for w in self.windows]
        for b, other in enumerate(others):
            self._read(b, t, other)
        self.speaking.observe(self._traffic(owns, others))

    def step(self, numbers: np.ndarray) -> np.ndarray:
        """Say and fly one row of every window (``numbers`` ``[B, 5]``, `SpeakingLoop.step`), then judge the
        separation at the row flown to; a window whose commanded aircraft lost separation it answers for ends there."""
        t = self.speaking.t
        kept = [self._next[b] if b in self._next else None for b in range(len(self.order))]
        owns = [k[0] if k is not None else self._own(b) for b, k in enumerate(kept)]
        others = [k[1] if k is not None else w.others_at(t) for k, w in zip(kept, self.windows)]
        for b in np.flatnonzero(self.speaking.alive):
            self._read(b, t, others[b])
        said = self.speaking.step(numbers, {SPEED: self._speed_masks(owns, others)}, self._traffic(owns, others))
        self._next = {}
        ended = np.zeros(len(self.order), dtype=bool)
        for b in np.flatnonzero(self.speaking.alive):
            own, others_next = self._own(b), self.windows[b].others_at(t + 1)
            aircraft, scene = self._scene(b, own, others_next)
            self._next[b] = (own, others_next, aircraft, scene)
            self._read(b, t + 1, others_next)
            loss = commanded_loss(scene, aircraft.last_step, self.separations[self.geometries[b].code])
            if loss is not None:
                partner = loss.j if loss.i == 0 else loss.i
                self.loss[b], self.loss_step[b], self.other[b] = loss, t + 1, aircraft.keys[partner]
                ended[b] = True
        if ended.any():
            self.speaking.end(ended)
        return said

    def copy(self, windows: Sequence[int]) -> WindowLoop:
        """A loop of copies of the windows ``windows`` (places in ``order``, repeats permitted; post-training D94): the
        closed loop's own copy (`SpeakingLoop.copy`: the executor's loop, the speaker, the inputs and their records) and
        the window's own state — its scene, landings, separation judge, speed-mask counts and recorded tokens. Flown on
        with the same numbers, a copy says what its original says (D94); the loop copied is unchanged."""
        index = list(windows)
        out = object.__new__(WindowLoop)
        out.speaking = self.speaking.copy(index)
        out.order, out.words, out.device = [self.order[i] for i in index], self.words, self.device
        out.windows, out.landings = [self.windows[i] for i in index], [self.landings[i] for i in index]
        out.step_s, out.every, out.separations = self.step_s, self.every, self.separations
        out.geometries, out.finals = [self.geometries[i] for i in index], [self.finals[i] for i in index]
        out.keys, out.speed_words = [self.keys[i] for i in index], self.speed_words
        out.loss, out.loss_step = [self.loss[i] for i in index], [self.loss_step[i] for i in index]
        out.other, out.speed_mask_rows = [self.other[i] for i in index], self.speed_mask_rows[index].copy()
        out.faults, out._reading = [self.faults[i] for i in index], [dict(self._reading[i]) for i in index]
        out._next = {k: self._next[i] for k, i in enumerate(index) if i in self._next}
        out._tokens = [list(self._tokens[i]) for i in index]
        return out

    def end_step(self, b: int) -> int:
        """The Δ row at which window ``b`` ended (post-training §2 item 9, t_E): the row of its loss of separation, or
        the row after the last one said to its commanded aircraft (its judged outcome or its time limit)."""
        if self.speaking.alive[b]:
            raise ValueError(f"window {b} is still flown")
        if self.loss_step[b] is not None:
            return self.loss_step[b]
        return self.speaking.start + len(self.speaking.said(b))

    def samples(self, split: str) -> list[tuple[Any, Any, list[np.ndarray]]]:
        """Each window's sentence for the loss (post-training C7): its rows with the words said as targets
        (`SpeakingLoop.sentences`), the speaker's records of them and its tokens at each of those rows."""
        permitted = self.speaking.permitted()
        return [(rows, permitted.select([b]), self._tokens[b][: len(rows.time_s)])
                for b, rows in enumerate(self.speaking.sentences(split))]

    def run(self, numbers: Sequence[np.random.Generator]) -> list[WindowResult]:
        """Fly every window to its end, ``numbers`` each window's source of random numbers (five a row said, as free
        generation draws them); the windows' ends."""
        if len(numbers) != len(self.order):
            raise ValueError(f"{len(numbers)} sources of random numbers for {len(self.order)} windows")
        while self.speaking.observing:
            self.observe()
        return self.finish(numbers)

    def finish(self, numbers: Sequence[np.random.Generator]) -> list[WindowResult]:
        """Say and fly every window from its present row to its end, ``numbers`` each window's source (five a row)."""
        while self.speaking.alive.any():
            self.step(np.stack([n.random(len(COLUMNS)) for n in numbers]))
        return self.results()

    def results(self) -> list[WindowResult]:
        """Each window's end (`WindowResult`), every window ended."""
        if self.speaking.alive.any():
            raise ValueError("windows are still flown")
        done = [b for b in range(len(self.order)) if self.loss[b] is None]
        generated = dict(zip(done, self.speaking.generated(done))) if done else {}
        out = []
        for b, (window, index) in enumerate(zip(self.windows, self.order)):
            if self.loss[b] is not None:
                outcome, landed, go_arounds = LOST_SEPARATION, None, None
            else:
                g = generated[b]
                outcome, go_arounds = g.outcome, g.go_arounds
                landed = None if g.crossing is None else int(g.crossing["runway_index"])
            said = self.speaking.said(b)
            go_arounds = int((said[:, RUNWAY] == RUNWAY_GO_AROUND).sum()) if go_arounds is None else go_arounds
            present = present_runways(self.landings[b], self.geometries[b], window.first_step_s, self.keys[b])
            out.append(WindowResult(
                index=index, kind=window.kind, outcome=outcome, loss=self.loss[b], loss_step=self.loss_step[b],
                other=self.other[b], reward=reward(outcome, landed, go_arounds, present, self.loss[b] is not None),
                go_arounds=go_arounds, words=said, states=self.speaking.states(b),
                speed_mask_rows=int(self.speed_mask_rows[b]),
                faulty_steps=sum(bool(keys) for keys in self._reading[b].values()),
                loss_reads_fault=self.loss_step[b] is not None and any(
                    self.other[b] in self._reading[b][step]
                    for step in range(self.loss_step[b] - STEPS_BEFORE_EVENT, self.loss_step[b] + 1)
                    if step in self._reading[b])))
        return out


def _window_faults(window: Window, faults: Mapping[str, frozenset[int]]) -> dict[str, frozenset[int]]:
    """The fault rows of a window's marked recorded aircraft, by their key in its scene: its airport's ``faults`` (by
    the flights' keys), and an inserted aircraft's under its own key (its record is its source's)."""
    out = {key: rows for key, rows in faults.items() if key != window.commanded.key}
    for key, _ in window.moved:
        source = key.removesuffix(INSERTED_SUFFIX)
        if source != key and source in faults:
            out[key] = faults[source]
    return out


def start_move_of(window: Window) -> Move:
    """The start's move of ``window``'s commanded aircraft (`autopilot.start.Move`): window B's, `NO_MOVE` otherwise."""
    move = window.start_move
    return Move(move.turn_deg, move.height_m, move.speed_scale)


def moved_commanded(window: Window, signals, step_s: float) -> Recorded:
    """Window B's commanded aircraft on its moved record (`autopilot.start.moved_signals`, vocabulary D97 (4)): its
    observed rows to its first predicted step moved as the start moves them (no later row is kept) — what the rule of
    D113 judges at its first predicted step."""
    anchor = window.commanded.row_at(window.first_step_s)
    moved = moved_signals(signals, anchor, start_move_of(window))
    return recorded(moved, window.scene.geometry, window.scene.interval_s, step_s,
                    np.zeros(len(moved.time_s), dtype=bool), lambda _: window.commanded.category)
