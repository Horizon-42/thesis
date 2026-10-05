"""The window loop of the post-training (post-training §2 items 1–3, §8 C4; D29–D31, D91, D93, D105, D110): a module
shared by the runners of stage C, not a runner.

Each window's commanded aircraft is flown through the prior's step of a speaker's closed loop (prior §7 item 7,
`prior_speaking_loop.SpeakingLoop`: the start of a closed loop, the prior's one function of a loop's row with each
window's own landings, D105, the speaker, the executor; the most go-arounds of a flight 2, D91). The loop adds the
scene, row by row:

- **the traffic module's input** at every row, observed and said: the tokens of the other aircraft of the window's step
  (`post.edges.tokens`, D98) — the edge features' conformance is checked first in every process (§4 item 1, D104);
- **the speed-word mask** (§3), a mask of a caller, computed once for a row from the state at the start of the row
  (D110), with the aircraft's runway and G in force before the row's words;
- **the separation judge** after each row the executor flew (`post.traffic.commanded_loss`, VISUAL): a loss of
  separation that the commanded aircraft answers for ends its window there (D93);
- **the reward** of D30 (`post.reward`), its present landing direction from the window's landings (D105).

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

from ts_transformer.autopilot.start import Loop
from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop
from ts_transformer.inference.separation import Loss
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, SPEED, Words
from ts_transformer.post.conformance import Checked, require_conforming_edges
from ts_transformer.post.edges import tokens
from ts_transformer.post.landings import roster_key, window_landings
from ts_transformer.post.reward import present_runways, reward
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import AircraftAt, Window, utc_s
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


class WindowLoop:
    """The windows ``windows`` (one for each flight of ``order``, in that order) flown in one batch (module docstring):
    ``loop`` started for their commanded flights (`autopilot.start.start`), ``sentences`` and ``flights`` (their records
    in the split's signals) by their place in the signals, ``rosters`` each airport's roster landings (each window's own
    landings from them, D105), ``finals`` each airport's, ``edges_reference`` the edge features' reference (D104)."""

    def __init__(self, model: Prior, loop: Loop, order: Sequence[int], windows: Sequence[Window],
                 sentences: Mapping[int, ClosedLoopSentence], flights: Mapping[int, Mapping[str, Any]],
                 geometries: Mapping[str, AirportGeometry], rosters: Mapping[str, LandingIndex],
                 finals: Mapping[str, Sequence[Final]], words: Words, *, interval_s: float, variant: str,
                 edges_reference: Path, device: torch.device) -> None:
        checked_edges(edges_reference)
        if loop.most_go_arounds != MOST_GO_AROUNDS:
            raise ValueError(f"the loop was started for {loop.most_go_arounds} go-arounds a flight, not {MOST_GO_AROUNDS} "
                             f"(D91)")
        if [w.signal_index for w in windows] != list(order):
            raise ValueError("one window for each flight of the loop, in its order")
        self.order, self.windows, self.words, self.device = list(order), list(windows), words, device
        self.landings = [window_landings(w, rosters[w.scene.geometry.code]) for w in self.windows]
        self.speaking = SpeakingLoop(model, loop, order, sentences, flights, geometries, self.landings, finals, words,
                                     interval_s=interval_s, variant=variant, device=device)
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
        #: each window's commanded aircraft, other aircraft and judged scene at the loop's next row, kept from the judge
        #: after a row flown: the same state, aircraft and words in force that the next row's mask and tokens read
        self._next: dict[int, tuple[AircraftAt, AircraftAt, AircraftAt, Any]] = {}

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

    def _tokens(self, owns: Sequence[AircraftAt], others: Sequence[AircraftAt]):
        return traffic_of([[tokens(own, other, g, self.separations[g.code], self.step_s)]
                           for own, other, g in zip(owns, others, self.geometries)], self.device)

    def _scene(self, b: int, own: AircraftAt, others: AircraftAt):
        g = self.geometries[b]
        aircraft = joined(own, others)
        return aircraft, separation_traffic(aircraft, g, self.separations[g.code], self.finals[b], self.step_s)

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
        self.speaking.observe(self._tokens(owns, others))

    def step(self, numbers: np.ndarray) -> np.ndarray:
        """Say and fly one row of every window (``numbers`` ``[B, 5]``, `SpeakingLoop.step`), then judge the
        separation at the row flown to; a window whose commanded aircraft lost separation it answers for ends there."""
        t = self.speaking.t
        kept = [self._next[b] if b in self._next else None for b in range(len(self.order))]
        owns = [k[0] if k is not None else self._own(b) for b, k in enumerate(kept)]
        others = [k[1] if k is not None else w.others_at(t) for k, w in zip(kept, self.windows)]
        said = self.speaking.step(numbers, {SPEED: self._speed_masks(owns, others)}, self._tokens(owns, others))
        self._next = {}
        ended = np.zeros(len(self.order), dtype=bool)
        for b in np.flatnonzero(self.speaking.alive):
            own, others_next = self._own(b), self.windows[b].others_at(t + 1)
            aircraft, scene = self._scene(b, own, others_next)
            self._next[b] = (own, others_next, aircraft, scene)
            loss = commanded_loss(scene, aircraft.last_step, self.separations[self.geometries[b].code])
            if loss is not None:
                partner = loss.j if loss.i == 0 else loss.i
                self.loss[b], self.loss_step[b], self.other[b] = loss, t + 1, aircraft.keys[partner]
                ended[b] = True
        if ended.any():
            self.speaking.end(ended)
        return said

    def run(self, numbers: Sequence[np.random.Generator]) -> list[WindowResult]:
        """Fly every window to its end, ``numbers`` each window's source of random numbers (five a row said, as free
        generation draws them); the windows' ends."""
        if len(numbers) != len(self.order):
            raise ValueError(f"{len(numbers)} sources of random numbers for {len(self.order)} windows")
        while self.speaking.observing:
            self.observe()
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
                speed_mask_rows=int(self.speed_mask_rows[b])))
        return out
