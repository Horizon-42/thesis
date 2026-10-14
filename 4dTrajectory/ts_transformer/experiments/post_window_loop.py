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
- **the separation judge** after each row the executor flew (`post.traffic.step_losses`, VISUAL): a loss of
  separation that the commanded aircraft answers for ends its window there (D93);
- **the reward** of D30 (`post.reward`; the runway of a landing does not change it, D207);
- **the faulty points** of the recorded aircraft (D114; vocabulary D111, C1's census): the steps at which a recorded
  aircraft reads one (`post.fault_census.reads_fault`: its row or the row before it is a fault row), and whether the
  other aircraft of a loss reads one at the event's step or in the 2 Δ before it (the commanded aircraft flies on the
  executor: it has none). No rule is applied: they are reported.

A window ends where its commanded aircraft is done (the judge's outcome) or loses separation. No time limit is read here
(vocabulary D90): the executor ends a flight. A window without other aircraft says and flies what free generation says
and flies for the same flight with the same random numbers (§2 item 1): the module gives zero and the speed mask
permits every word.

**Windows of several commanded aircraft** (multi-aircraft control D144–D148, D150, D152; post-training §9 item 8). A
window may hold commanded aircraft after its anchor (`post.scene.Window.joined`): the rows of the batch are then each
window's commanded aircraft, each joining the loop at its own tick (its join step, the start's join ticks; prior §7
item 7), each with its own landings (`post.landings.commanded_landings`) and, in a compressed window, its moved entry
time. At each tick, for each commanded aircraft that has joined: its other aircraft are the window's recorded aircraft
and its other commanded aircraft that have joined (their states in the loop, their R and G from their words in force),
in that order, with a caller's token part beside the edge features (``token_part``, D152); the speed-word mask reads a
commanded leader as it reads a recorded one (D148). After each row flown, each window is judged once on all its
commanded aircraft still flown (first, in their order) and its recorded aircraft: of each loss, the commanded aircraft
said in the row that the caller's rule makes answer (``answering``; stage C's: the commanded aircraft that the rules
make responsible) become silent (D144): reward 0, flown on by a caller's mask that permits only "unchanged" in every
column, never answering again; an aircraft in its observed rows answers for nothing. A commanded aircraft that the
executor ends `landed` adds its landing (its landed runway, its crossing time) to the other commanded aircraft of its
window (D147 item 2; a loop landing on a sealed test day is left out and counted, `LandingIndex.with_landing`), and is
judged once more at the row flown to, over its threshold at its last state flown, so that the wake minimum at the
threshold behind a commanded leader is judged as behind a recorded one (`_judged`; the user, 2026-10-07). A window ends when every commanded aircraft is done or silent (D144): its silent ones are halted
then. Each commanded aircraft gives its own result (`results`). A stage C window (one commanded aircraft, joining at
tick 0) is flown as before, bit for bit: silent is its window's end.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.start import Loop, Move, moved_signals
from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop
from ts_transformer.inference.separation import Loss, Traffic
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.labeller.interval import OBSERVATION_S
from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, SPEED, UNCHANGED, Words
from ts_transformer.post.conformance import Checked, require_conforming_edges
from ts_transformer.post.edges import TOKEN_FEATURES, Pairs, each_tokens, window_tokens
from ts_transformer.post.fault_census import STEPS_BEFORE_EVENT, reads_fault
from ts_transformer.post.landings import commanded_landings, roster_key
from ts_transformer.post.reward import LANDED, reward
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import INSERTED_SUFFIX, AircraftAt, Recorded, Window, recorded, utc_s
from ts_transformer.post.speed_mask import along_course_speeds, speed_check
from ts_transformer.post.traffic import answered_loss, scene_aircraft, step_losses
from ts_transformer.post.traffic import traffic as separation_traffic
from ts_transformer.post.traffic_attention import traffic_of
from ts_transformer.prior.inputs import motion
from ts_transformer.prior.landings import Landing, LandingIndex
from ts_transformer.prior.model import COPIED, Prior
from ts_transformer.prior.procedure import Final
from ts_transformer.prior.speaker import MOST_GO_AROUNDS

#: The outcome of a window ended by a loss of separation that its commanded aircraft answers for (D30: reward 0).
LOST_SEPARATION = "lost_separation"
#: The edge references checked in this process (§4 item 1: once in every process that computes edge features).
_CHECKED: dict[Path, Checked] = {}

#: A caller's rule of who answers for a loss (module docstring): ``(window, step, loss, aircraft, commanded)`` → the
#: places in ``aircraft`` (the window's judged aircraft, its ``commanded`` commanded ones first) of the commanded
#: aircraft that answer.
Answering = Callable[[Window, int, Loss, AircraftAt, int], tuple[int, ...]]
#: A caller's token part (module docstring): ``(loop, row, others)`` → ``[N, width]`` the part of each other aircraft
#: of the row (``others``: for each, its row in the batch, None for a recorded aircraft), with its ``width``.
TokenPartOf = Callable[["WindowLoop", int, Sequence["int | None"]], np.ndarray]
#: A caller's reading of what a value network reads at a row beside the model (post-training D171): ``(loop, row b, tick
#: t, b's aircraft, its other aircraft) -> (one row for each other aircraft, a number)``, kept apart from the tokens.
ValueReaderOf = Callable[["WindowLoop", int, int, AircraftAt, AircraftAt], tuple[np.ndarray, float]]


def responsible(window: Window, step: int, loss: Loss, aircraft: AircraftAt, commanded: int) -> tuple[int, ...]:
    """Stage C's rule of who answers (D93): the commanded aircraft that the rules make responsible."""
    return tuple(k for k in loss.responsible if k < commanded)


def checked_edges(reference: Path) -> Checked:
    """`require_conforming_edges` of ``reference``, run once in this process."""
    reference = Path(reference).resolve()
    if reference not in _CHECKED:
        _CHECKED[reference] = require_conforming_edges(reference)
    return _CHECKED[reference]


@dataclass(frozen=True)
class WindowResult:
    """One commanded aircraft's end (a window's, in stage C): the commanded flight (its place in the split's signals),
    the kind of window, the outcome (the judge's, or `LOST_SEPARATION`), the loss and the other aircraft's key where it
    ended so (with the step of the loss), the reward, the go-arounds said, the words said, the states on the 2 s rows
    from row 0 and the rows where the speed mask acted (D101)."""

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
    #: the judge's crossing (its runway index, its cycle row, …) of an aircraft the executor ended `landed`, a silent
    #: one's too (the window's later judgement over its threshold, `_judged`, and stage D's readouts read it); else None
    crossing: Mapping[str, Any] | None = None


@dataclass
class _Tick:
    """A window's aircraft at one tick (kept from the judge after a row flown for the next row's mask and tokens): its
    commanded rows that have joined and are flown, their aircraft, the recorded aircraft, and the judged set (the
    commanded aircraft, then the recorded ones), its scene, motions and speeds along the course, each aircraft's own,
    computed once for the window (multi-aircraft control D189 (1)); ``of_row`` each row's view once taken at this tick
    (`WindowLoop._of_row`, D181 (59))."""

    t: int
    rows: list[int]
    owns: list[AircraftAt]
    recorded: AircraftAt
    aircraft: AircraftAt
    scene: Any
    moving: Any = None                  # `prior.inputs.motion`'s
    speeds: np.ndarray | None = None
    #: each row's tokens (in ``rows``' order), every pair of the window computed at once when first read (D189 (3))
    tokens: list[np.ndarray] | None = None
    of_row: dict[int, _View] = field(default_factory=dict)


class _View(NamedTuple):
    """A commanded row's view of its window at a tick (`WindowLoop._of_row`): its aircraft, its other aircraft (the
    recorded ones, then its window's other commanded ones), their rows (None for a recorded one), and the set (itself
    first), its scene, motions and speeds along the course, the window's reordered."""

    own: AircraftAt
    other: AircraftAt
    placed: list[int | None]
    aircraft: AircraftAt
    scene: Any
    moving: Any
    speeds: np.ndarray


class WindowLoop:
    """The windows ``windows`` flown in one batch (module docstring): ``loop`` started for their commanded flights
    (`autopilot.start.start`; each flight once, with its window's join step as its join tick), ``order`` its order,
    ``sentences`` and ``flights`` (their records in the split's signals) by their place in the signals, ``rosters`` each
    airport's roster landings (each commanded aircraft's own landings from them, D105), ``finals`` each airport's,
    ``edges_reference`` the edge features' reference (D104); ``answering`` the rule of who answers for a loss
    (`responsible`: stage C's), ``token_part`` a caller's token part and its ``part_width`` (none: stage C's)."""

    def __init__(self, model: Prior, loop: Loop, order: Sequence[int], windows: Sequence[Window],
                 sentences: Mapping[int, ClosedLoopSentence], flights: Mapping[int, Mapping[str, Any]],
                 geometries: Mapping[str, AirportGeometry], rosters: Mapping[str, LandingIndex],
                 finals: Mapping[str, Sequence[Final]], words: Words, *, interval_s: float, variant: str,
                 edges_reference: Path, faults: Mapping[str, Mapping[str, frozenset[int]]],
                 observed: Mapping[int, np.ndarray], device: torch.device, answering: Answering = responsible,
                 token_part: TokenPartOf | None = None, part_width: int = 0,
                 value_reader: ValueReaderOf | None = None, attention: str = COPIED,
                 executor_device: torch.device | None = None) -> None:
        """``value_reader``: what a value network reads at each row beside the model (D171; `values`), never in the
        model's tokens; None: nothing read. ``attention``: how a window's copies read the front of the speaker's cache
        they hold once (`SpeakingLoop`; multi-aircraft control D190 (2)). ``executor_device``: the device of the executor
        (`SpeakingLoop`; multi-aircraft control S1). Stage D gives it in `multi_train.loop_options`. None: the executor
        follows the speaker."""
        checked_edges(edges_reference)
        if loop.most_go_arounds != MOST_GO_AROUNDS:
            raise ValueError(f"the loop was started for {loop.most_go_arounds} go-arounds a flight, not {MOST_GO_AROUNDS} "
                             f"(D91)")
        if (token_part is None) != (part_width == 0):
            raise ValueError("a token part comes with its width, and only with it")
        self.order, self.windows, self.words, self.device = list(order), list(windows), words, device
        self.answering, self.token_part, self.part_width = answering, token_part, part_width
        place = {index: b for b, index in enumerate(self.order)}
        members = [[place.get(i, -1) for i in w.signal_indices] for w in self.windows]
        if sorted(b for rows in members for b in rows) != list(range(len(self.order))):
            raise ValueError("one window for each flight of the loop: each flight commanded by one window, once")
        #: each window's rows of the batch (its commanded aircraft, the anchor first), and each row's window and member
        self.members = members
        self.window_of = np.zeros(len(self.order), dtype=np.int64)
        self.member_of = np.zeros(len(self.order), dtype=np.int64)
        for w, rows in enumerate(members):
            self.window_of[rows], self.member_of[rows] = w, np.arange(len(rows))
        joins = [int(self.windows[w].join_steps()[m]) for w, m in zip(self.window_of, self.member_of)]
        if not np.array_equal(loop.join_ticks, joins):
            raise ValueError("the loop's join ticks are not the windows' join steps")
        #: each row's join step in its window's ticks, and each window's tick 0 in the loop's clock (moved when loops
        #: are flown together, `together`: a window's ticks are its own, the loop's tick less its origin)
        self.joins = np.array(joins, dtype=np.int64)
        self.origins = np.zeros(len(self.windows), dtype=np.int64)
        #: each row's record as its window replays it, and its shift (a compressed window's)
        self.records: list[Recorded] = [self.windows[w].commanded_all[m] for w, m in zip(self.window_of, self.member_of)]
        shifts = [0.0 if m == 0 else self.windows[w].joined[m - 1].shift_s
                  for w, m in zip(self.window_of, self.member_of)]
        # the latest a row of the loop is said, from its window's row 0 (a batch is said together: a window's rows go on
        # while another's are flown): an aircraft's first predicted step and its time limit with every go-around's
        # (the executor's), the latest of any, and one Δ more (a row's time is its flight's, the window's to rounding);
        # each commanded aircraft's landings those its rows can count (D189 (0))
        most = loop.executor.most_s.cpu().numpy()
        reach = max(self.records[b].first_step_s - self.windows[w].row0_s + float(most[b])
                    for b, w in enumerate(self.window_of)) + interval_s
        landings = [commanded_landings(window, rosters[window.scene.geometry.code], window.row0_s + reach)
                    for window in self.windows]
        row_landings = [landings[w][m] for w, m in zip(self.window_of, self.member_of)]
        # a moved aircraft (a compressed window) reads its landings at its moved time (multi-aircraft control §6.3 item 2)
        moved_flights = dict(flights)
        for b, shift in enumerate(shifts):
            if shift:
                i = self.order[b]
                moved_flights[i] = {**flights[i], "entry_time_utc": _utc_text(utc_s(flights[i]["entry_time_utc"]) + shift)}
        self.speaking = SpeakingLoop(model, loop, order, sentences, observed, moved_flights, geometries, row_landings,
                                     finals, words, interval_s=interval_s, variant=variant, device=device,
                                     attention=attention, executor_device=executor_device)
        self.step_s = words.spec.step_s
        self.every = self.speaking.every
        self.geometries = [self.windows[w].scene.geometry for w in self.window_of]
        self.finals = [finals[g.code] for g in self.geometries]
        self.separations = {code: airport_separation(g) for code, g in geometries.items()}
        #: each airport's pairs of runways and wake categories, their token features computed once (D189 (1))
        self.pairs = {code: Pairs(separation) for code, separation in self.separations.items()}
        self.keys = [roster_key(flights[i]["dataset_id"], flights[i]["airport"]) for i in self.order]
        for b, i in enumerate(self.order):
            row0 = utc_s(flights[i]["entry_time_utc"]) + sentences[i].rows.first_row * self.step_s + shifts[b]
            if self.records[b].key != flights[i]["dataset_id"] or abs(self.records[b].first_step_s - OBSERVATION_S
                                                                         - row0) > 1e-6:
                raise ValueError(f"window of {self.records[b].key}: not flight {flights[i]['dataset_id']}'s row 0")
        self.speed_words = len(column_words(SPEED, words, 1))
        self.loss: list[Loss | None] = [None] * len(self.order)
        self.loss_step: list[int | None] = [None] * len(self.order)
        self.other: list[str | None] = [None] * len(self.order)
        #: the rows that answered for a loss (silent, D144)
        self.silent = np.zeros(len(self.order), dtype=bool)
        self.speed_mask_rows = np.zeros(len(self.order), dtype=np.int64)
        #: each window's marked recorded aircraft (D114), by key: their fault rows (an inserted aircraft keeps its
        #: source's, under its own key)
        self.faults = [_window_faults(w, faults[w.scene.geometry.code]) for w in self.windows]
        #: each window's steps read so far, with the keys of the recorded aircraft reading a faulty point there
        self._reading: list[dict[int, frozenset[str]]] = [{} for _ in self.windows]
        #: each window's aircraft at the loop's next tick, kept from the judge after a row flown: the same states,
        #: aircraft and words in force that the next row's masks and tokens read
        self._next: dict[int, _Tick] = {}
        #: each row's tokens at every row of its own, observed and said (``[N, features]`` each): the loss reads the
        #: words said with the same input of the traffic module (post-training C7)
        self._tokens: list[list[np.ndarray]] = [[] for _ in self.order]
        #: each row's value reading at every row of its own (D171; ``value_reader``), and the present row's, pending
        self.value_reader = value_reader
        self._values: list[list[tuple[np.ndarray, float]]] = [[] for _ in self.order]
        self._pending: list[tuple[np.ndarray, float] | None] = [None] * len(self.order)

    # ---- the aircraft of a row
    def _own(self, b: int) -> AircraftAt:
        """Row ``b``'s commanded aircraft at the loop's tick: its state there and at the 2 s row before (observed before
        its first predicted step, flown from it), its runway and G in force before the row's words."""
        speaking = self.speaking
        own_row = int(speaking.t - speaking.join_ticks[b])
        if own_row < speaking.start:
            r = own_row * self.every
            at, before, known = speaking.observed[b, r], speaking.observed[b, max(r - 1, 0)], r > 0
        else:
            at, before, known = speaking.current[b], speaking.before[b], True
        in_force = speaking.in_force(b)
        runway, go_around = (in_force.runway, in_force.go_around) if in_force is not None else (-1, False)
        record = self.records[b]
        return AircraftAt.of([(record.key, tuple(at[:3]), tuple(before[:3]), known, runway, record.category, False,
                               go_around)])

    def _tick(self, w: int, t: int) -> _Tick:
        """Window ``w``'s aircraft at its tick ``t`` (the loop's present tick in its window's ticks): its commanded rows
        that have joined and are flown, their aircraft, its recorded aircraft, and the judged set and scene."""
        joined = self.speaking.joined()
        rows = [b for b in self.members[w] if joined[b] and self.speaking.alive[b]]
        owns = [self._own(b) for b in rows]
        recorded = self.windows[w].others_at(t)
        if not rows:
            return _Tick(t, rows, owns, recorded, recorded, None)
        aircraft = scene_aircraft(_concat(owns), recorded)
        g = self.windows[w].scene.geometry
        scene = separation_traffic(aircraft, g, self.separations[g.code], self.finals[rows[0]], self.step_s)
        moving = motion(aircraft.at, aircraft.before, aircraft.known, self.step_s)
        return _Tick(t, rows, owns, recorded, aircraft, scene, moving, along_course_speeds(moving.ground_speed_mps,
                                                                                         scene))

    def _of_row(self, tick: _Tick, b: int) -> _View:
        """Row ``b``'s view of its window at ``tick`` (`_View`): the window's set, scene, motions and speeds reordered
        (``b`` first, the recorded aircraft, the other commanded ones: the order its mask and tokens read), not computed
        again (multi-aircraft control D189 (1)): each value is its aircraft's alone, and a window's aircraft are of one
        airport, its finals (`post.scene.Window` refuses a commanded aircraft of another); taken once a tick (D181
        (59))."""
        n, recorded = len(tick.rows), len(tick.recorded)
        if n == 1:                                           # stage C: the window's judged set is the row's own
            return _View(tick.owns[0], tick.recorded, [None] * recorded, tick.aircraft, tick.scene, tick.moving,
                         tick.speeds)
        if b not in tick.of_row:
            k = tick.rows.index(b)
            order = [k, *range(n, n + recorded), *(c for c in range(n) if c != k)]
            aircraft = tick.aircraft.take(order)
            tick.of_row[b] = _View(tick.owns[k], aircraft.take(range(1, len(order))),
                                   [None] * recorded + [c for c in tick.rows if c != b], aircraft,
                                   tick.scene.take(order), tick.moving._make(value[order] for value in tick.moving),
                                   tick.speeds[order])
        return tick.of_row[b]

    def _ticks(self, t: int) -> list[_Tick | None]:
        """Each window's aircraft at the loop's tick ``t``, from the judge's (kept after the row flown) where it holds
        them; None for a window without a row joined and flown."""
        out: list[_Tick | None] = []
        for w in range(len(self.windows)):
            kept, own = self._next.get(w), t - int(self.origins[w])
            tick = kept if kept is not None and kept.t == own else self._tick(w, own)
            out.append(tick if tick.rows else None)
        return out

    def _traffic(self, t: int, ticks: Sequence[_Tick | None], joined: np.ndarray) -> tuple[Any, list[np.ndarray]]:
        """The traffic module's input of the row at the loop's tick ``t`` and the tokens of each row the speaker holds,
        in its order (`_keep` records them for the loss once the row is kept): a row that has not joined has none; a
        row no longer flown (its rows are said with the batch until it leaves, not its sentence's) reads its window's
        recorded aircraft, as stage C's ended window does. A row that left has no tokens (S6)."""
        alone = self._alone(t, ticks, joined)
        rows: list[np.ndarray] = []
        for b in self.speaking.inside:
            tick = ticks[self.window_of[b]]
            if not joined[b]:
                rows.append(np.zeros((0, len(TOKEN_FEATURES) + self.part_width), dtype=np.float32))
                continue
            w = int(self.window_of[b])
            g = self.geometries[b]
            if b in alone:
                own, other, row = alone[b]
                placed = [None] * len(other)
            else:                       # the window's tokens, every pair of it at once (D189 (3))
                view = self._of_row(tick, b)
                own, other, placed = view.own, view.other, view.placed
                if tick.tokens is None:
                    tick.tokens = window_tokens(tick.aircraft, len(tick.rows), g, self.pairs[g.code], tick.moving,
                                                tick.scene.along_m)
                row = tick.tokens[tick.rows.index(b)]
            if self.value_reader is not None:
                self._pending[b] = self.value_reader(self, b, t - int(self.origins[w]), own, other)
            if self.token_part is not None:
                row = np.concatenate((row, np.asarray(self.token_part(self, b, placed), dtype=np.float32)), axis=1)
            rows.append(row)
        return traffic_of([[row] for row in rows], self.device, self.part_width), rows

    def _alone(self, t: int, ticks: Sequence[_Tick | None], joined: np.ndarray
               ) -> dict[int, tuple[AircraftAt, AircraftAt, np.ndarray]]:
        """The joined rows that are not flown at the loop's tick ``t`` and have not left the loop (S6), by row: the
        aircraft of the row, the recorded aircraft of its window, and the tokens of the recorded aircraft as the row
        reads them. A row that is not flown reads only the recorded aircraft of its window, as stage C's ended window
        does. The rows of one window share these aircraft, so `each_tokens` calculates their tokens in one call
        (multi-aircraft control S2). Each row gets the tokens of `tokens`, bit for bit."""
        out = {}
        for w, members in enumerate(self.members):
            tick = ticks[w]
            lone = [b for b in members if joined[b] and self.speaking.held[b] and (tick is None or b not in tick.rows)]
            if not lone:
                continue
            recorded = self.windows[w].others_at(t - int(self.origins[w]))
            owns = [self._own(b) for b in lone]
            g = self.windows[w].scene.geometry
            found = each_tokens(_concat(owns), recorded, g, self.pairs[g.code], self.step_s)
            out.update({b: (own, recorded, row) for b, own, row in zip(lone, owns, found)})
        return out

    def _keep(self, rows: Sequence[np.ndarray], joined: np.ndarray, applies: np.ndarray | None = None) -> None:
        """A row kept (``rows``: the tokens of the rows the speaker holds, `_traffic`; ``joined``: the rows that had
        joined at it): each joined row's tokens of it recorded, and the rows whose speed-word mask acted there
        counted."""
        for b, row in zip(self.speaking.inside, rows):
            if joined[b]:
                self._tokens[b].append(row)
                if self.value_reader is not None:
                    self._values[b].append(self._pending[b])
        if applies is not None:
            self.speed_mask_rows += applies

    def _read(self, w: int, t: int, recorded: AircraftAt) -> None:
        """Window ``w``'s step ``t``: the recorded aircraft that read a faulty point there (once a step, D114)."""
        if t in self._reading[w]:
            return
        time_s = self.windows[w].step_s(t)
        scene = self.windows[w].scene
        self._reading[w][t] = frozenset(key for key in recorded.keys if key in self.faults[w] and bool(reads_fault(
            scene.flight(key), np.array([time_s]), self.faults[w][key])[0]))

    def _speed_masks(self, ticks: Sequence[_Tick | None]) -> tuple[np.ndarray, np.ndarray]:
        """``[B, words]`` the speed-word mask of each row at the start of the row (D110; every word for a row that is not
        said, no longer flown or silent), and ``[B]`` whether it acted (counted by `_keep` once the row is kept)."""
        out = np.ones((len(self.order), self.speed_words), dtype=bool)
        applies = np.zeros(len(self.order), dtype=np.int64)
        for b in np.flatnonzero(self.speaking.alive & self.speaking.said_now() & ~self.silent):
            view = self._of_row(ticks[self.window_of[b]], b)
            check = speed_check(view.scene, view.speeds, self.separations[self.geometries[b].code], self.words,
                                len(self.geometries[b].candidates))
            out[b] = check.permitted
            applies[b] = check.applies
        return out, applies

    def _masks(self, speed: np.ndarray) -> dict[int, np.ndarray]:
        """The masks of a caller of the row: the speed-word mask, and a silent row's "unchanged" alone in every column
        (D144) while it is flown (a row ended with its window is said on as before: its rows are not its sentence's)."""
        silent = self.silent & self.speaking.alive
        if not silent.any():
            return {SPEED: speed}
        most = int(self.speaking.speaker.n_candidates.max())
        out = {}
        for c in range(len(COLUMNS)):
            values = column_words(c, self.words, most)
            mask = speed.copy() if c == SPEED else np.ones((len(self.order), len(values)), dtype=bool)
            mask[silent] = values == UNCHANGED
            out[c] = mask
        return out

    # ---- the loop
    def observe(self) -> None:
        """Encode the next observed row, with its traffic (a loop whose join ticks are all 0)."""
        t = self.speaking.t
        ticks, joined = self._ticks(t), self.speaking.joined()
        for w, tick in enumerate(ticks):
            own = t - int(self.origins[w])
            self._read(w, own, tick.recorded if tick is not None else self.windows[w].others_at(own))
        traffic, rows = self._traffic(t, ticks, joined)
        self.speaking.observe(traffic)
        self._keep(rows, joined)

    def step(self, numbers: np.ndarray) -> np.ndarray:
        """Say and fly one row of every window (``numbers`` ``[B, 5]``, `SpeakingLoop.step`), then judge the
        separation at the row flown to (module docstring). A row the executor refuses is refused whole: the window's
        records are as they were."""
        t = self.speaking.t
        ticks = self._ticks(t)
        for w, tick in enumerate(ticks):
            if tick is not None:
                self._read(w, t - int(self.origins[w]), tick.recorded)
        joined, said_now = self.speaking.joined(), self.speaking.said_now()
        masks, applies = self._speed_masks(ticks)
        traffic, rows = self._traffic(t, ticks, joined)
        # a row the executor refuses (vocabulary D80) leaves the shared step as it was (prior §7 items 3 and 7): the
        # window's records of it are kept only once it is kept
        alive_before = self.speaking.alive.copy()
        said = self.speaking.step(numbers, self._masks(masks), traffic)
        self._keep(rows, joined, applies)
        landed = self._landed(alive_before & ~self.speaking.alive & ~self.speaking.ended)
        self._next = {}
        ending = np.zeros(len(self.order), dtype=bool)
        for w, rows_w in enumerate(self.members):
            if not self.speaking.alive[rows_w].any():
                continue
            after = t + 1 - int(self.origins[w])                # the row flown to, in the window's ticks
            tick = self._tick(w, after)
            self._next[w] = tick
            self._read(w, after, tick.recorded)
            if tick.rows:
                judged, scene, commanded = self._judged(tick, [b for b in landed if self.window_of[b] == w])
                found = step_losses(scene, judged.last_step, self.separations[self.geometries[rows_w[0]].code])
                for k, b in enumerate(tick.rows):
                    if self.silent[b] or not said_now[b]:       # an observed one answers for nothing (D145)
                        continue
                    loss = next((loss for loss in found if k in self.answering(self.windows[w], after, loss,
                                                                                   judged, commanded)),
                                None) if commanded > 1 or self.answering is not responsible else \
                        answered_loss(found, 0)
                    if loss is not None:
                        partner = loss.j if loss.i == k else loss.i
                        self.loss[b], self.loss_step[b], self.other[b] = loss, after, judged.keys[partner]
                        self.silent[b] = True
            alive = self.speaking.alive[rows_w]
            if (self.silent[rows_w] | ~alive).all():          # every commanded aircraft done or silent: its end
                ending[rows_w] = alive
        if ending.any():
            self.speaking.end(ending)
        return said

    def _landed(self, done: np.ndarray) -> list[int]:
        """The rows the executor ended in the row just flown (``done``): one judged `landed` adds its landing on its
        landed runway at its crossing time to the other commanded aircraft of its window (D147 item 2; not a silent
        one's). The rows judged `landed` in a window of other commanded aircraft (silent ones too), for the judge
        (`_judged`)."""
        out = []
        for b in np.flatnonzero(done):
            others = [c for c in self.members[self.window_of[b]] if c != b]
            if not others:
                continue
            outcome = self.speaking.outcome(b)
            if outcome.outcome != LANDED:
                continue
            out.append(int(b))
            if self.silent[b]:
                continue
            g = self.geometries[b]
            time_s = self.records[b].first_step_s + outcome.crossing["at_row"] * self.speaking.loop.params.cycle_s
            landing = Landing(time_s, g.candidates[int(outcome.crossing["runway_index"])].ident, self.keys[b])
            self.speaking.add_landing(others, landing)
        return out

    def _judged(self, tick: _Tick, landed: Sequence[int]) -> tuple[AircraftAt, Any, int]:
        """The judged set of a window at the row flown to, its scene and its count of commanded aircraft: the tick's
        (its commanded aircraft still flown, then its recorded ones), and with ``landed`` (its rows judged `landed` in
        the row just flown) each of those once more, over its threshold (``last_step``) at its last state flown on its
        landed runway, after the commanded ones still flown: the wake minimum at the threshold behind a commanded leader
        (post-training's judge, as for a recorded leader at its last row; the user, 2026-10-07, stage D's requests item
        7). It is never asked to answer (it is not said)."""
        if not landed:
            return tick.aircraft, tick.scene, len(tick.rows)
        over = []
        for b in landed:
            states = self.speaking.states(b)
            runway = int(self.speaking.outcome(b).crossing["runway_index"])
            over.append(AircraftAt.of([(self.records[b].key, tuple(states[-1, :3]), tuple(states[-2, :3]), True, runway,
                                        self.records[b].category, True, False)]))
        b, n = tick.rows[0], len(tick.rows)
        g = self.geometries[b]
        landing = _concat(over)                   # computed alone and put in its place (D189 (1))
        scene = separation_traffic(landing, g, self.separations[g.code], self.finals[b], self.step_s)
        rest = range(n, len(tick.aircraft))
        return (scene_aircraft(_concat([tick.aircraft.take(range(n)), landing]), tick.aircraft.take(rest)),
                Traffic.joined([tick.scene.take(range(n)), scene, tick.scene.take(rest)]), n + len(landed))

    def copy(self, windows: Sequence[int], device: torch.device | None = None) -> WindowLoop:
        """A loop of copies of the windows ``windows`` (places in ``windows``, repeats permitted; post-training D94): the
        closed loop's own copy of every row of each (`SpeakingLoop.copy`: the executor's loop, the speaker, the inputs
        and their records) and the window's own state — its scene, landings, separation judge, silent rows, speed-mask
        counts, recorded tokens and value readings (D171). Flown on with the same numbers, a copy says what its original says (D94); the loop
        copied is unchanged. ``device``: the copy's (this loop's when not given; multi-aircraft control D182: the second
        pass's copies held in the host's memory until their point is tried)."""
        index = list(windows)
        rows = [b for w in index for b in self.members[w]]
        out = object.__new__(WindowLoop)
        out.speaking = self.speaking.copy(rows, device)
        out.order, out.words = [self.order[b] for b in rows], self.words
        out.device = self.device if device is None else torch.device(device)
        out.answering, out.token_part, out.part_width = self.answering, self.token_part, self.part_width
        out.windows = [self.windows[w] for w in index]
        out.members, k = [], 0
        for w in index:
            out.members.append(list(range(k, k + len(self.members[w]))))
            k += len(self.members[w])
        out.window_of = np.array([n for n, w in enumerate(index) for _ in self.members[w]], dtype=np.int64)
        out.member_of = self.member_of[rows].copy()
        out.joins, out.origins = self.joins[rows].copy(), self.origins[index].copy()
        out.records = [self.records[b] for b in rows]
        out.step_s, out.every, out.separations, out.pairs = self.step_s, self.every, self.separations, self.pairs
        out.geometries, out.finals = [self.geometries[b] for b in rows], [self.finals[b] for b in rows]
        out.keys, out.speed_words = [self.keys[b] for b in rows], self.speed_words
        out.loss, out.loss_step = [self.loss[b] for b in rows], [self.loss_step[b] for b in rows]
        out.other, out.speed_mask_rows = [self.other[b] for b in rows], self.speed_mask_rows[rows].copy()
        out.silent = self.silent[rows].copy()
        out.faults, out._reading = [self.faults[w] for w in index], [dict(self._reading[w]) for w in index]
        out._next = {}
        for n, w in enumerate(index):
            if w in self._next:
                tick = self._next[w]
                start = out.members[n][0]
                out._next[n] = replace(tick, rows=[start + self.members[w].index(b) for b in tick.rows], of_row={})
        out._tokens = [list(self._tokens[b]) for b in rows]
        out.value_reader = self.value_reader
        out._values, out._pending = [list(self._values[b]) for b in rows], [self._pending[b] for b in rows]
        return out

    @staticmethod
    def together(parts: Sequence[tuple[WindowLoop, Sequence[int]]], device: torch.device, room: int | None,
                 spent: WindowLoop | None = None) -> WindowLoop:
        """Windows of several loops, each at its own tick, flown on as one loop (multi-aircraft control D188: a backward
        batch's searches stepped together), on ``device``: ``parts`` each a loop and its windows (places in its
        ``windows``, repeats permitted), the loops of one batch (one model, rules and token part). The closed loop's own
        join (`SpeakingLoop.together`: each part's clock moved later by the ticks it is behind) and each window's own
        state as `copy` keeps it; a window keeps its own ticks (its origin in the loop's clock moved with its part).
        Flown on with the same numbers, a window says and flies what it would in its own loop. ``room``: the rows its
        speaker's cache has room for past the rows written (`SpeakingLoop.together`); ``spent``: a loop among the parts
        that nothing reads after (the one this loop replaces), its cache freed as it is copied."""
        index = [list(windows) for _, windows in parts]
        rows = [[b for w in i for b in loop.members[w]] for (loop, _), i in zip(parts, index)]
        first = parts[0][0]
        out = object.__new__(WindowLoop)
        out.speaking = SpeakingLoop.together([(loop.speaking, r) for (loop, _), r in zip(parts, rows)], device, room,
                                             None if spent is None else spent.speaking)
        shifts = [out.speaking.t - loop.speaking.t for loop, _ in parts]
        moved = list(zip([loop for loop, _ in parts], index, rows, shifts))
        out.order = [loop.order[b] for loop, _, r, _ in moved for b in r]
        out.words, out.device = first.words, torch.device(device)
        out.answering, out.token_part, out.part_width = first.answering, first.token_part, first.part_width
        out.windows = [loop.windows[w] for loop, i, _, _ in moved for w in i]
        out.members, out._next, k, n = [], {}, 0, 0
        for loop, i, _, _ in moved:
            for w in i:
                out.members.append(list(range(k, k + len(loop.members[w]))))
                if w in loop._next:
                    tick = loop._next[w]
                    out._next[n] = replace(tick, rows=[k + loop.members[w].index(b) for b in tick.rows], of_row={})
                k, n = k + len(loop.members[w]), n + 1
        out.window_of = np.array([n for n, rows_w in enumerate(out.members) for _ in rows_w], dtype=np.int64)
        out.member_of = np.concatenate([loop.member_of[r] for loop, _, r, _ in moved])
        out.joins = np.concatenate([loop.joins[r] for loop, _, r, _ in moved])
        out.origins = np.concatenate([loop.origins[i] + shift for loop, i, _, shift in moved])
        out.step_s, out.every, out.speed_words = first.step_s, first.every, first.speed_words
        out.separations = {code: s for loop, _, _, _ in moved for code, s in loop.separations.items()}
        out.pairs = {code: pairs for loop, _, _, _ in moved for code, pairs in loop.pairs.items()}
        for name in ("records", "geometries", "finals", "keys", "loss", "loss_step", "other", "_pending"):
            setattr(out, name, [getattr(loop, name)[b] for loop, _, r, _ in moved for b in r])
        out.speed_mask_rows = np.concatenate([loop.speed_mask_rows[r] for loop, _, r, _ in moved])
        out.silent = np.concatenate([loop.silent[r] for loop, _, r, _ in moved])
        out.faults = [loop.faults[w] for loop, i, _, _ in moved for w in i]
        out._reading = [dict(loop._reading[w]) for loop, i, _, _ in moved for w in i]
        out._tokens = [list(loop._tokens[b]) for loop, _, r, _ in moved for b in r]
        out.value_reader = first.value_reader
        out._values = [list(loop._values[b]) for loop, _, r, _ in moved for b in r]
        return out

    def fault_readings(self, w: int) -> Mapping[int, frozenset[str]]:
        """Window ``w``'s steps read so far, each with the keys of its recorded aircraft that read a faulty point there
        (D114: its row or the row before it is a fault row; an empty set at a step where none does) — what its results
        count (`WindowResult.faulty_steps`: the steps with one; ``loss_reads_fault``: the loss's other aircraft among
        them at the event or in the 2 Δ before it). A read-only view (post-training C26, frontend D177 (15))."""
        return MappingProxyType(self._reading[w])

    def end_step(self, b: int) -> int:
        """The Δ row of its own at which row ``b`` ended (post-training §2 item 9, t_E): the row of its loss of
        separation, or the row after the last one said to it (its judged outcome or its time limit)."""
        if self.speaking.alive[b]:
            raise ValueError(f"window {b} is still flown")
        if self.loss_step[b] is not None:
            return self.loss_step[b] - int(self.joins[b])
        return self.speaking.start + len(self.speaking.said(b))

    def own_row(self, b: int, tick: int) -> int:
        """Row ``b``'s own Δ row at its window's tick ``tick``."""
        return tick - int(self.joins[b])

    def samples(self, split: str, rows: Sequence[int] | None = None) -> list[tuple[Any, Any, list[np.ndarray]]]:
        """Each row's sentence for the loss (post-training C7): its rows with the words said as targets
        (`SpeakingLoop.sentences`), its own records of them (`SpeakingLoop.permitted`) and its tokens at each of those
        rows; of the rows ``rows`` only (in that order; every row when not given), the others never built
        (multi-aircraft control D181 (58): a branch's copies are read for their varied aircraft only)."""
        rows = list(range(len(self.order))) if rows is None else list(rows)
        return [(sentence, permitted, self._tokens[b][: len(sentence.time_s)])
                for b, sentence, permitted in zip(rows, self.speaking.sentences(split, rows),
                                                  self.speaking.permitted(rows))]

    def values(self, split: str) -> list[tuple[list[np.ndarray], np.ndarray]]:
        """Each row's value readings (D171, ``value_reader``) at the rows of its sentence for the loss (`samples`):
        the part of each row and the numbers of its rows."""
        if self.value_reader is None:
            raise ValueError("a loop without a value reader reads no value")
        lengths = [len(rows.time_s) for rows in self.speaking.sentences(split)]
        return [([part for part, _ in self._values[b][:n]], np.array([x for _, x in self._values[b][:n]],
                                                                    dtype=np.float32))
                for b, n in enumerate(lengths)]

    def run(self, numbers: Sequence[np.random.Generator]) -> list[WindowResult]:
        """Fly every window to its end, ``numbers`` each row's source of random numbers (five a row it is said, as free
        generation draws them); the rows' ends."""
        if len(numbers) != len(self.order):
            raise ValueError(f"{len(numbers)} sources of random numbers for {len(self.order)} windows")
        while self.speaking.observing:
            self.observe()
        return self.finish(numbers)

    def finish(self, numbers: Sequence[np.random.Generator]) -> list[WindowResult]:
        """Say and fly every window from its present row to its end, ``numbers`` each row's source (five a row it is
        said)."""
        while self.speaking.alive.any():
            said_now = self.speaking.said_now()
            self.step(np.stack([n.random(len(COLUMNS)) if said else np.zeros(len(COLUMNS))
                                for n, said in zip(numbers, said_now)]))
        return self.results()

    def results(self, rows: Sequence[int] | None = None) -> list[WindowResult]:
        """Each row's end (`WindowResult`), of the rows ``rows`` (in that order; every row when not given), each ended.
        A row's end is its own: it does not change with the other rows of the loop (multi-aircraft control S6)."""
        rows = list(range(len(self.order))) if rows is None else list(rows)
        if self.speaking.alive[rows].any():
            raise ValueError("windows are still flown")
        done = [b for b in rows if self.loss[b] is None]
        generated = dict(zip(done, self.speaking.generated(done))) if done else {}
        out = []
        for b in rows:
            index, w = self.order[b], int(self.window_of[b])
            window = self.windows[w]
            crossing = None
            if self.loss[b] is not None:
                outcome, go_arounds = LOST_SEPARATION, None
                if not self.speaking.ended[b]:                 # a silent aircraft the executor ended
                    judged = self.speaking.outcome(b)
                    crossing = judged.crossing if judged.outcome == LANDED else None
            else:
                g = generated[b]
                outcome, go_arounds = g.outcome, g.go_arounds
                crossing = g.crossing if outcome == LANDED else None
            said = self.speaking.said(b)
            go_arounds = int((said[:, RUNWAY] == RUNWAY_GO_AROUND).sum()) if go_arounds is None else go_arounds
            out.append(WindowResult(
                index=index, kind=window.kind, outcome=outcome, loss=self.loss[b], loss_step=self.loss_step[b],
                other=self.other[b], reward=reward(outcome, go_arounds, self.loss[b] is not None),
                go_arounds=go_arounds, words=said, states=self.speaking.states(b),
                speed_mask_rows=int(self.speed_mask_rows[b]),
                faulty_steps=sum(bool(keys) for keys in self._reading[w].values()),
                loss_reads_fault=self.loss_step[b] is not None and any(
                    self.other[b] in self._reading[w][step]
                    for step in range(self.loss_step[b] - STEPS_BEFORE_EVENT, self.loss_step[b] + 1)
                    if step in self._reading[w]), crossing=crossing))
        return out


def _concat(items: Sequence[AircraftAt]) -> AircraftAt:
    """Several sets of aircraft as one, in order."""
    out = items[0]
    for item in items[1:]:
        out = scene_aircraft(out, item)
    return out


def _utc_text(time_s: float) -> str:
    """A UTC epoch as the signals write an entry time."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(time_s, tz=timezone.utc).isoformat().replace("+00:00", "Z")


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
