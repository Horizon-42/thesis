"""The prior commanding every aircraft of a time window (multi-aircraft design §2.2 "窗口内由模型指挥", §6.6 step 7) —
shared by M3's and M4's second pass, not a runner.

**Windows** (step 7 item 1): an airport's flights are chained into segments on the steps their rows hang on
(`prior.scene.SceneIndex.segments`, design §2.3); from each segment's first step a window opens every `WINDOW_EVERY_S`
up to its last step, `WINDOW_S` long. The prior commands the flights with a sentence whose first row hangs on a step
inside the window and that fly on their own dynamics (the replay gate's group, `autopilot.replay`); everything else in
the air — before, after, the background, a flight the executor cannot fly — is replayed along its record. Windows
overlap, so a flight is commanded in two of them, once in each. A window's scene runs from its first commanded aircraft's
first row to the last one's time limit.

What the loop reads is placed by the one-aircraft code (`traffic_speaking`, in the edge features' source hash): each
commanded aircraft's view of its window is a one-speaking `traffic_speaking.Scene` (`Window.scene`) — none of the files
in `traffic_scene_data.EDGE_SOURCES` is changed, so every traffic prior trained so far still loads.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from geokit import NM_M

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Executor
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import PSI, AirportCharts
from ts_transformer.autopilot.judge import flown_track, outcome_of
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import Spoken
from ts_transformer.experiments.prior_free_generation import BELOW_GLIDEPATH
from ts_transformer.experiments.traffic_census import Track
from ts_transformer.experiments.traffic_labelled import own_end
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION, Controlled, Judging, Run, join
from ts_transformer.experiments.traffic_scene_data import FlightRows
from ts_transformer.experiments.traffic_speaking import (
    HISTORY_S, MASK_COLUMNS, Aircraft, OthersAt, Scene, SceneAirport, edge_rows, others_at, scene_landings,
    speaking_masks,
)
from ts_transformer.inference.scene_edges import SceneRows, scene_edge_blocks
from ts_transformer.inference.separation import VISUAL, Traffic
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import load_sentences, load_signals
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import (
    APPROACH, APPROACH_CLEARED, RUNWAY, SPEED, UNCHANGED, Words, compass_from_math_rad,
)
from ts_transformer.prior.generate import rows_for
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import below_floor
from ts_transformer.prior.scene import N_LOOK, SceneIndex, hang, hung_span
from ts_transformer.prior.scene_data import Node
from ts_transformer.prior.window_speaker import WindowSpeaker

#: A window's length and how often one opens (design §8: 20 minutes, every 10).
WINDOW_S = 1_200.0
WINDOW_EVERY_S = 600.0
#: Windows whose flights are rebuilt at a time while drawing (a rebuild opens the flights' tracks).
DRAW_CHUNK = 64


@dataclass(frozen=True)
class Window:
    """One window of an airport's scene: the step it opens on, the flights the prior commands (by the step their first
    row hangs on, then key), the others replayed (in the air at a step from the first commanded aircraft's first row to
    the end of the last one's time limit) and — an augmented window's — flights moved in time or inserted (their rows
    and tracks, by their keys among the commanded or the others)."""

    airport: SceneAirport
    opens_s: float
    commanded: tuple[str, ...]
    others: tuple[str, ...]
    moved: tuple[tuple[FlightRows, Track], ...] = ()

    def rows(self, key: str) -> FlightRows:
        """A flight of the window as the samples place it (a moved one's own)."""
        return next((rows for rows, _ in self.moved if rows.presence.dataset_id == key), None) \
            or self.airport.flights.flights[key]

    def track(self, key: str) -> Track:
        """A flight of the window as the judge replays it (a moved one's own)."""
        return next((track for _, track in self.moved if track.key == key), None) or self.airport.tracks[key]

    def first_step_s(self, key: str, step_s: float) -> float:
        """The step a commanded aircraft's first row hangs on."""
        return float(hang(self.rows(key).presence.start_s, step_s))

    def scene(self, key: str, step_s: float) -> Scene:
        """Commanded aircraft ``key``'s view of the window as a one-speaking scene: it speaking, every other aircraft of
        the window among the others — the other commanded ones first, in the window's order, then the replayed."""
        if self.moved:
            raise NotImplementedError("an augmented window's scene view (design §6.6 step 7.5)")
        others = tuple(k for k in self.commanded if k != key) + self.others
        return Scene(self.airport, key, self.first_step_s(key, step_s), others)


def window_tiles(airport: SceneAirport, step_s: float) -> list[tuple[float, tuple[str, ...]]]:
    """Every window of ``airport``'s segments (module docstring) that holds a flight with a sentence: its opening step and
    those flights, by the step their first row hangs on (then key) — which of them fly is the draw's to find."""
    out = []
    for segment in SceneIndex(f.presence for f in airport.flights.flights.values()).segments(step_s):
        spans = [hung_span(p, step_s) for p in segment]
        first, last = min(s for s, _ in spans), max(e for _, e in spans)
        entering = sorted((s, p.dataset_id) for p, (s, _) in zip(segment, spans) if p.speaking)
        opens = first
        while opens <= last:
            keys = tuple(k for s, k in entering if opens <= s < opens + WINDOW_S)
            if keys:
                out.append((opens, keys))
            opens += WINDOW_EVERY_S
    return out


def window_of(airport: SceneAirport, opens_s: float, commanded: Sequence[str], limits_s: Sequence[float],
              step_s: float) -> Window:
    """The window opening at ``opens_s`` with ``commanded`` (each flying ``limits_s`` from its first predicted step):
    everything else in the air from the first one's first row to the end of the last one's time limit replayed."""
    firsts = [float(hang(airport.flights.flights[k].presence.start_s, step_s)) for k in commanded]
    start = min(firsts)
    end = max(first + N_LOOK * step_s + limit for first, limit in zip(firsts, limits_s))
    taken = set(commanded)
    others = tuple(k for k, t in airport.tracks.items()
                   if k not in taken and t.first_step_s <= end and t.last_step_s >= start)
    return Window(airport, opens_s, tuple(commanded), others)


@dataclass(frozen=True)
class WindowDraw:
    """Windows drawn from a split (`draw_windows`): each window's opening and commanded flights, the commanded flights
    as one replay batch (window after window, each window's in its order) and what the draw read."""

    openings: list[tuple[str, float, tuple[str, ...]]]     # (airport, opening step, commanded) per window
    batch: replay.Batch
    counts: dict[str, Any]

    def members(self) -> list[range]:
        """Each window's commanded flights' places in `batch`."""
        out, at = [], 0
        for _, _, commanded in self.openings:
            out.append(range(at, at + len(commanded)))
            at += len(commanded)
        return out


def draw_windows(directory: Path, split: str, spec: VocabularySpec, words: Words, airports: Mapping[str, SceneAirport],
                 *, per_airport: int, seed: int, step_s: float) -> WindowDraw:
    """``per_airport`` windows of each airport of ``airports`` (in order), drawn from its tiles (`window_tiles`) in a
    permutation of one generator seeded ``seed``: in each, the flights that fly on their own dynamics are commanded (each
    re-read and checked against its stored sentence, as `replay.draw` does); a tile none of whose flights flies is passed
    over and counted; refused when an airport runs short."""
    sentences = load_sentences(directory, split, spec)
    stored = {int(index): k for k, index in enumerate(sentences["signal_index"])}
    index = {s.dataset_id: i for i, s in enumerate(load_signals(directory, split))}
    rng = np.random.default_rng(seed)
    openings: list[tuple[str, float, tuple[str, ...]]] = []
    parts: list[replay.Batch] = []
    counts: dict[str, Any] = {"split": split, "seed": seed, "per_airport": per_airport, "airports": {}}
    for code, airport in airports.items():
        tiles = window_tiles(airport, step_s)
        order = rng.permutation(len(tiles))
        taken, read, passed = 0, 0, 0
        # of the windows taken: their flights with a sentence, and those replayed as the executor cannot fly them
        entering_unflown: Counter = Counter(entering=0, not_flown=0)
        for start in range(0, len(order), DRAW_CHUNK):
            if taken == per_airport:
                break
            chunk = [tiles[int(i)] for i in order[start: start + DRAW_CHUNK]]
            candidates = sorted({index[k] for _, keys in chunk for k in keys})
            drawn = replay.draw_flights(directory, split, candidates, per_airport=0, seed=0)
            flies = {s.dataset_id: n for n, s in enumerate(drawn.signals)}
            readings: dict[str, Any] = {}
            keep: list[int] = []
            for opens, keys in chunk:
                if taken == per_airport:
                    break
                read += 1
                commanded = tuple(k for k in keys if k in flies)
                if not commanded:
                    passed += 1
                    continue
                entering_unflown.update({"entering": len(keys), "not_flown": len(keys) - len(commanded)})
                for k in commanded:
                    if k not in readings:
                        readings[k] = _reading(drawn.signals[flies[k]], index[k], drawn.geometries, sentences, stored,
                                               spec, words)
                    keep.append(flies[k])
                openings.append((code, opens, commanded))
                taken += 1
            if keep:
                parts.append(replay.batch_of(drawn, keep, [readings[drawn.signals[n].dataset_id] for n in keep]))
        if taken < per_airport:
            raise ValueError(f"{code}: {taken} windows with a flight that flies, {per_airport} wanted")
        counts["airports"][code] = {"tiles": len(tiles), "read": read, "passed_over": passed, "windows": taken,
                                    "commanded": sum(len(c) for a, _, c in openings if a == code),
                                    **dict(entering_unflown)}
    batch = replay.Batch(**{f.name: [item for part in parts for item in getattr(part, f.name)]
                            for f in dataclasses.fields(replay.Batch) if f.name != "drawn"}, drawn=counts)
    return WindowDraw(openings, batch, counts)


def _reading(signals: Any, i: int, geometries: Mapping[str, Any], sentences: Mapping[str, np.ndarray],
             stored: Mapping[int, int], spec: VocabularySpec, words: Words) -> Any:
    """A commanded flight re-read and checked against its stored sentence (MIRROR of `replay.draw`'s check: the replay
    module is in the executor's source hash)."""
    reading = read_flight(signals, geometries[signals.airport], spec, words)
    k = stored[i]
    grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]
    if not np.array_equal(reading.words, grid) or reading.runway_index != int(sentences["runway_index"][k]):
        raise ValueError(f"{signals.dataset_id}: the re-read sentence differs from the stored one")
    return reading


# ---- the loop (design §6.6 step 7 item 3)

@dataclass
class _State:
    """A commanded aircraft at one of its steps, as the judge reads it (`traffic_speaking.speaking_aircraft`'s reading of
    a flown state, one at a time): position, unwrapped compass track, ground speed, the executor's capture after the cycle
    before; and, once judged there, the runway in force after that step's words with its approach-clock reading."""

    t_s: float
    e_m: float
    n_m: float
    height_m: float
    track_deg: float
    ground_speed_mps: float
    captured: bool
    runway: str | None = None
    along_m: float = math.nan
    angle_deg: float = math.nan
    right_m: float = math.nan


class _Unwrapped:
    """A track unwrapped one state at a time: MIRROR of `np.unwrap` (period 2π) over a whole sequence, as
    `autopilot.judge.flown_track` unwraps one — the same arithmetic, so the same values element for element (tested)."""

    def __init__(self, first_rad: float) -> None:
        self.last, self.correction = np.float64(first_rad), np.float64(0.0)

    def next(self, rad: float) -> np.float64:
        rad = np.float64(rad)
        dd = rad - self.last
        ddmod = np.mod(dd - (-np.pi), 2 * np.pi) + (-np.pi)
        if ddmod == -np.pi and dd > 0:
            ddmod = np.float64(np.pi)
        correct = np.float64(0.0) if abs(dd) < np.pi else ddmod - dd
        self.correction = self.correction + correct
        self.last = rad
        return rad + self.correction


def _rows_of(batch: Any, index: np.ndarray) -> Any:
    """A dataclass of per-flight tensors (the executor's inputs, runways, charts) at ``index``."""
    where = torch.as_tensor(index)
    return type(batch)(**{f.name: getattr(batch, f.name)[where] for f in dataclasses.fields(batch)})


@dataclass
class Commanded:
    """What became of one commanded aircraft (`WindowLoop.results`): the words it said (``[steps, 6]``, `UNCHANGED`
    where a column says nothing), its outcome under the loop's reading (the executor judge's, `BELOW_GLIDEPATH`, or
    `traffic_loop.LOST_SEPARATION` when the judge ended it — then ``end`` is that end), its own end (the executor's or the
    glidepath edge's, whatever the judge did), the steps said up to its judged end (``counted``: what it is trained on),
    and where its executor's states are (``group``, ``place``)."""

    key: str
    window: int
    said: np.ndarray
    outcome: str
    own: str
    end: dict[str, Any] | None
    counted: int
    group: int
    place: int


class WindowLoop:
    """Every commanded aircraft of a batch of windows flown with the prior speaking, judged as it flies (module
    docstring and design §6.6 step 7 item 3). ``windows``: one per sample (a window spoken to `K` times is here `K`
    times); ``flights``: their commanded aircraft window after window, each window's in its order, with each one's
    executor inputs (``inputs``, ``runways``, ``charts``, ``approach_ias_mps`` over ``flights``, from its first predicted
    step), time limit and geometry; ``alone``: each aircraft speaks seeing no other (its edge features its own, no
    separation mask) and is still judged in its window.

    **A step** (`step`), at the batch step every speaking aircraft's rows reach:

    1. every landing in ``(t − step, t]`` of each window (a commanded aircraft's crossing, a replayed one's roster time) is
       checked against the aircraft behind it (`traffic_loop.Judging.landing`) — an ended aircraft still flying lands too;
    2. the aircraft still spoken to speak, round by round from the front of the approach clock (`WindowSpeaker`);
    3. each window's aircraft on the step are judged (`Judging.step`), their runway the one in force after this step's
       words (`speaking_aircraft`'s reading); one ended here flies this step's words and then nothing more — passive;
    4. each executor flies the step: the aircraft that first spoke at one step share one (`Executor`'s cycles count from
       its first, so a later aircraft cannot join an earlier one's), the executor's code untouched;
    5. an aircraft its executor is done with leaves after its own last state (`traffic_labelled.own_end`); under the
       procedure's altitudes one whose next state sank below the glidepath lower edge stops speaking there and is judged
       at that state for the last time (`prior_free_generation.glidepath_stops`' reading); the rest append the state
       they reached.

    With one commanded aircraft a window it says and flies, to its judged end, what `traffic_speaking.SceneLoop` says
    and flies, and ends where `traffic_speaking.judged` ends it (tests).
    """

    def __init__(self, model: Prior, windows: Sequence[Window], flights: Sequence[FlightSignals],
                 geometries: Sequence[AirportGeometry], inputs: FlightInputs, runways: Runways, charts: AirportCharts,
                 approach_ias_mps: torch.Tensor, limits: Sequence[float], words: Words, params: ExecutorParams,
                 landings: Any, *, generator: torch.Generator, temperature: float, procedure_masks: ProcedureMasks,
                 alone: bool = False, reading: str = VISUAL) -> None:
        step_s = words.spec.step_s
        self.windows, self.words, self.params, self.step_s, self.alone = list(windows), words, params, step_s, alone
        self.flights, self.geometries = list(flights), list(geometries)
        self.keys = [f.dataset_id for f in flights]
        self.window_of = np.array([w for w, window in enumerate(windows) for _ in window.commanded], dtype=np.int64)
        if [k for window in windows for k in window.commanded] != self.keys:
            raise ValueError("the flights are the windows' commanded aircraft, window after window, each in its order")
        count = len(flights)
        # the batch's steps: each window's first commanded aircraft's row 0 at the pre-roll's end
        origin = np.array([window.first_step_s(window.commanded[0], step_s) for window in windows])
        firsts = np.array([windows[w].first_step_s(k, step_s) for k, w in zip(self.keys, self.window_of)])
        entered = [[float(hang(window.rows(k).presence.start_s, step_s)) for k in window.others] for window in windows]
        before = max([0] + [int(round((o - s) / step_s)) for o, starts in zip(origin, entered) for s in starts])
        self.pre = min(int(HISTORY_S // step_s), before)
        self.origin = origin - self.pre * step_s               # each window's time at batch step 0
        self.start = np.rint((firsts - self.origin[self.window_of]) / step_s).astype(np.int64)
        nodes = [[Node(k, window.rows(k).presence.speaking, int(round((s - self.origin[w]) / step_s)),
                       **window.rows(k).node_rows) for k, s in zip(window.others, entered[w])]
                 for w, window in enumerate(windows)]
        per_aircraft = (None if landings is None else
                        [scene_landings(landings[windows[w].airport.flights.code], windows[w].scene(k, step_s))
                         for k, w in zip(self.keys, self.window_of)])
        self.speaker = WindowSpeaker(
            model, flights, geometries, landings, words, scenes=np.arange(count) if alone else self.window_of,
            starts=self.start, others=[[] for _ in range(count)] if alone else nodes, edges=self._edges,
            masks=self._masks, mask_columns=() if alone else MASK_COLUMNS,
            max_rows=rows_for(max(limits) + step_s, step_s), generator=generator, procedure_masks=procedure_masks,
            temperature=temperature, scene_landings=per_aircraft)
        #: per masked column, each aircraft's step: whether the separation masks took a word away there
        self.separation_masked = {c: np.zeros((count, rows_for(max(limits) + step_s, step_s) - N_LOOK), dtype=bool)
                                  for c in MASK_COLUMNS}
        # the executors, one per step aircraft first speak at
        device = inputs.initial_state.device
        self.approach_mps = approach_ias_mps.cpu().numpy()
        self.group = np.zeros(count, dtype=np.int64)
        self.place = np.zeros(count, dtype=np.int64)
        self.executors: list[tuple[int, np.ndarray, Executor, Spoken]] = []
        for g, first in enumerate(sorted(set((self.start + N_LOOK).tolist()))):
            index = np.flatnonzero(self.start + N_LOOK == first)
            self.group[index], self.place[index] = g, np.arange(len(index))
            executor = Executor(_rows_of(inputs, index), _rows_of(runways, index), _rows_of(charts, index),
                                approach_ias_mps[torch.as_tensor(index)], params, words,
                                time_limit_s=torch.tensor([limits[i] for i in index], dtype=torch.float64,
                                                          device=device))
            self.executors.append((first, index, executor, Spoken(len(index), words, device=device)))
        # the judge, a window at a time
        self.runs = [Run(reading) for _ in windows]
        self.judging = [Judging(window.airport.flights.separation, reading, step_s, run)
                        for window, run in zip(windows, self.runs)]
        self.replayed = [[window.track(k) for k in window.others] for window in windows]
        self.finals = ([procedure_masks.finals[g.code] for g in geometries] if procedure_masks.altitudes else None)
        # each aircraft (`_State` per step it has reached; its own end and the judge's)
        self.states: list[list[_State]] = [[] for _ in range(count)]
        self.unwrapped: list[_Unwrapped | None] = [None] * count
        self.ended_at = np.full(count, -1, dtype=np.int64)       # the judge's end (a step), −1: none
        self.judged_to = np.full(count, np.iinfo(np.int64).max)  # the last step judged (its own end)
        self.in_scene_to = np.full(count, np.iinfo(np.int64).max)  # the last step in the scene
        self.sentence_end = np.full(count, np.iinfo(np.int64).max)  # steps said: the glidepath edge stops it
        self.own: list[str | None] = [None] * count
        self.landing_s = np.full(count, math.nan)
        self.done = np.zeros(count, dtype=bool)                  # its executor is done with it
        self._others: tuple[int, dict[int, OthersAt]] = (-1, {})
        self._now: tuple[int, list[Aircraft] | None] = (-1, None)

    # -- the clock
    def window_time_s(self, w: int, step: int) -> float:
        return float(self.origin[w] + step * self.step_s)

    def _step_of(self, i: int, step: int) -> int:
        """Aircraft ``i``'s own step (0: its first predicted one) at batch step ``step``."""
        return int(step - self.start[i] - N_LOOK)

    @property
    def running(self) -> bool:
        return not all(bool(executor.done.all()) or executor.count == executor.cycles
                       for _, _, executor, _ in self.executors)

    def step(self) -> None:
        """One batch step (the class docstring's 1–5; the landings are checked once this step's words have put each
        aircraft's state on its runway, as `speaking_aircraft` reads a state's runway from its step's words)."""
        speaker, step = self.speaker, self.speaker.step
        for first, index, executor, _ in self.executors:
            if first == step:                                  # its first predicted step: the state flown from
                self._record(index, executor, step, captured=np.zeros(len(index), dtype=bool))
        k = step - self.start - N_LOOK
        speaking = (k >= 0) & (self.ended_at < 0) & ~self.done & (k < self.sentence_end)
        said = speaker.speak(self._rank(speaking, step), self._locked())
        for w in range(len(self.windows)):
            self._runways(w, step)
            self._landings(w, step)
            self._judge(w, step)
        self._fly(step, said)

    # -- 1. landings
    def _landings(self, w: int, step: int) -> None:
        t_s = self.window_time_s(w, step)
        members = np.flatnonzero(self.window_of == w)
        leaders: list[tuple[float, str, Controlled | Track]] = []
        for i in members:
            if t_s - self.step_s < self.landing_s[i] <= t_s:
                leaders.append((float(self.landing_s[i]), self.keys[i], self._controlled(i, self.in_scene_to[i])))
        for track in self.replayed[w]:
            if t_s - self.step_s < track.presence.landing_s <= t_s:
                leaders.append((track.presence.landing_s, track.key, track))
        for at, key, leader in sorted(leaders, key=lambda x: (x[0], x[1])):
            controlled, passive = self._here(w, step, at, exclude=key)
            replayed = [a for a in self.replayed[w]
                        if a.key != key and a.presence.times_s[0] <= at <= a.presence.times_s[-1]]
            self.judging[w].landing(at, leader, controlled, replayed, passive)
        for i in members:
            if self.runs[w].ended.get(self.keys[i]) is not None and self.ended_at[i] < 0:
                self.ended_at[i] = self._step_of(i, step)       # a wake shortfall at a landing ends it at this step

    def _here(self, w: int, step: int, at: float, exclude: str = "") -> tuple[list[Controlled], list[Controlled]]:
        """Window ``w``'s commanded aircraft in the scene at instant ``at`` of batch step ``step`` (on their steps up to
        this one): the judged ones and the passive ones."""
        controlled, passive = [], []
        for i in np.flatnonzero(self.window_of == w):
            k = self._step_of(i, step)
            if self.keys[i] == exclude or k < 0 or k > self.in_scene_to[i]:
                continue
            aircraft = self._controlled(i, k)
            if not aircraft.on_step(at):
                continue
            (controlled if self.ended_at[i] < 0 and k <= self.judged_to[i] else passive).append(aircraft)
        return controlled, passive

    def _controlled(self, i: int, last: int) -> Controlled:
        """Aircraft ``i`` on its steps ``last − 1`` and ``last`` (as many of them as it has)."""
        states = self.states[i][max(0, last - 1): last + 1]
        rows = self.windows[self.window_of[i]].rows(self.keys[i])
        return Controlled(rows.presence, np.array([s.t_s for s in states]), np.array([s.e_m for s in states]),
                          np.array([s.n_m for s in states]), np.array([s.height_m for s in states]),
                          tuple(s.runway for s in states), np.array([s.along_m for s in states]),
                          np.array([s.angle_deg for s in states]), np.array([s.right_m for s in states]),
                          np.array([s.ground_speed_mps for s in states])
                          * np.cos(np.radians(np.array([s.angle_deg for s in states]))),
                          np.array([s.captured for s in states], dtype=bool), rows.category, "flying", None)

    # -- 2. speaking
    def _aircraft_now(self, step: int) -> list[Aircraft | None]:
        """Every aircraft's executor state at the start of the step (read once a step; None before it flies)."""
        if self._now[0] != step:
            now: list[Aircraft | None] = [None] * len(self.keys)
            for first, index, executor, _ in self.executors:
                if first > step:
                    continue
                state = executor.now()
                e, n, h = state.e_m.cpu().numpy(), state.n_m.cpu().numpy(), state.height_m.cpu().numpy()
                track, ground = state.track_deg.cpu().numpy(), state.ground_speed_mps.cpu().numpy()
                captured = executor.lateral.captured.cpu().numpy()
                for p, i in enumerate(index):
                    now[i] = Aircraft(float(e[p]), float(n[p]), float(h[p]), float(track[p]), float(ground[p]),
                                      bool(captured[p]))
            self._now = (step, now)
        return self._now[1]

    def _clock(self, i: int, aircraft: Aircraft, pointer: int) -> float:
        """Aircraft ``i``'s place on the approach clock of its runway class ``pointer``."""
        window = self.windows[self.window_of[i]]
        geometry, separation = window.airport.flights.geometry, window.airport.flights.separation
        candidate = geometry.candidates[pointer - 1]
        before = relative_to_runway(np.array([aircraft.e_m]), np.array([aircraft.n_m]), np.array([aircraft.track_deg]),
                                    np.array([aircraft.height_m]), candidate).before_threshold_m
        return separation.along_nm[candidate.ident] * NM_M - float(before[0])

    def _rank(self, speaking: np.ndarray, step: int) -> np.ndarray:
        """Each speaking aircraft's round (−1: silent): in its window, the front of the approach clock first (on the
        runway in force before the step), those with none yet — at their first predicted step — after them (design
        §2.6). Alone, every one in round 0."""
        rank = np.full(len(speaking), -1, dtype=np.int64)
        if self.alone:
            rank[speaking] = 0
            return rank
        now = self._aircraft_now(step)
        value = self.speaker.value
        for w in range(len(self.windows)):
            members = [i for i in np.flatnonzero(self.window_of == w) if speaking[i]]
            order = sorted(members, key=lambda i: (value[i, RUNWAY] == 0,
                                                   -self._clock(i, now[i], int(value[i, RUNWAY]))
                                                   if value[i, RUNWAY] else 0.0, i))
            for r, i in enumerate(order):
                rank[i] = r
        return rank

    def _locked(self) -> np.ndarray:
        out = np.zeros(len(self.keys), dtype=bool)
        for _, index, executor, _ in self.executors:
            out[index] = executor.runway_locked.cpu().numpy()
        return out

    # -- 3. the judge
    def _runways(self, w: int, step: int) -> None:
        """Window ``w``'s commanded aircraft in the scene at the step: the runway in force after this step's words."""
        value = self.speaker.value
        for i in np.flatnonzero(self.window_of == w):
            k = self._step_of(i, step)
            if 0 <= k <= self.in_scene_to[i]:
                self._on_runway(i, self.states[i][k], int(value[i, RUNWAY]))

    def _judge(self, w: int, step: int) -> None:
        """Window ``w``'s aircraft on the step judged (the class docstring's 3)."""
        t_s = self.window_time_s(w, step)
        controlled, passive = self._here(w, step, t_s)
        replayed = [a for a in self.replayed[w] if a.on_step(t_s)]
        ended = self.judging[w].step(t_s, controlled, replayed, passive)
        for i in np.flatnonzero(self.window_of == w):
            if self.keys[i] in ended and self.ended_at[i] < 0:
                self.ended_at[i] = self._step_of(i, step)

    def _on_runway(self, i: int, state: _State, pointer: int) -> None:
        window = self.windows[self.window_of[i]]
        geometry, separation = window.airport.flights.geometry, window.airport.flights.separation
        candidate = geometry.candidates[pointer - 1]
        relative = relative_to_runway(np.array([state.e_m]), np.array([state.n_m]), np.array([state.track_deg]),
                                      np.array([state.height_m]), candidate)
        state.runway = candidate.ident
        state.along_m = separation.along_nm[candidate.ident] * NM_M - float(relative.before_threshold_m[0])
        state.angle_deg = float(relative.track_minus_course_deg[0])
        state.right_m = float(relative.right_of_course_m[0])

    # -- 4., 5. flying
    def _fly(self, step: int, said: np.ndarray) -> None:
        cycle_s = self.params.cycle_s
        flying: list[int] = []
        for g, (first, index, executor, spoken) in enumerate(self.executors):
            if first > step or bool(executor.done.all()) or executor.count == executor.cycles:
                continue
            was_done = executor.done.cpu().numpy().copy()
            heard = torch.full((len(index),), spoken.steps * self.step_s, dtype=torch.float64,
                               device=executor.done.device)
            spoken.say(np.where(said[index] > 0, said[index] - 1, UNCHANGED))
            for _ in range(executor.step_rows):
                if executor.count == executor.cycles:
                    break
                executor.cycle(spoken.at(heard), torch.full((len(index),), executor.count * cycle_s,
                                                            dtype=torch.float64, device=executor.done.device))
            self._record(index, executor, step + 1, captured=executor.lateral.captured.cpu().numpy())
            done = executor.done.cpu().numpy()
            newly = done & ~was_done
            if newly.any():
                self._own_ends(g, index[newly], step)
            for p, i in enumerate(index):
                if not done[p]:
                    if self.finals is not None:
                        self._glidepath(i, step)
                    flying.append(int(i))
        self.speaker.advance(np.array(flying, dtype=np.int64),
                             *self._positions(np.array(flying, dtype=np.int64)))

    def _record(self, index: np.ndarray, executor: Executor, step: int, captured: np.ndarray) -> None:
        """Every aircraft of ``index`` at batch step ``step``: its executor state (`flown_track`'s reading)."""
        states = executor.state.cpu().numpy()
        for p, i in enumerate(index):
            if step - self.start[i] - N_LOOK != len(self.states[i]):
                continue                                         # (none past its executor's last cycle)
            read = flown_track(states[p: p + 1], self.geometries[i])
            rad = float(np.radians(compass_from_math_rad(states[p: p + 1, PSI]))[0])
            if self.unwrapped[i] is None:
                self.unwrapped[i] = _Unwrapped(rad)
                track = np.float64(rad)
            else:
                track = self.unwrapped[i].next(rad)
            w = self.window_of[i]
            self.states[i].append(_State(self.window_time_s(w, step), float(read["e"][0]), float(read["n"][0]),
                                         float(read["height"][0]), float(np.degrees(track)),
                                         float(read["ground_speed"][0]), bool(captured[p])))

    def _own_ends(self, g: int, index: np.ndarray, step: int) -> None:
        """Aircraft its executor is done with in step ``step``: its own end (`traffic_labelled.own_end`) — the step it
        is last in the scene and, while the judge has not ended it, last judged — and its crossing when it landed."""
        _, members, executor, _ = self.executors[g]
        flown = executor.flown()
        step_rows = executor.step_rows
        for i in index:
            p = int(self.place[i])
            pointer = int(self.speaker.value[i, RUNWAY]) - 1
            ended = outcome_of(flown, p, self.geometries[i], pointer, self.words.spec)
            outcome, last_row = own_end(ended.outcome, ended.end_row, -1, step_rows)
            last = last_row // step_rows
            self.done[i] = True
            self.in_scene_to[i] = min(self.in_scene_to[i], last)
            if self.own[i] is None:                            # (the glidepath edge's stop comes first)
                self.own[i] = outcome
                if self.ended_at[i] < 0:
                    self.judged_to[i] = last
            if outcome == "landed":
                w = self.window_of[i]
                self.landing_s[i] = self.window_time_s(w, int(self.start[i]) + N_LOOK) + ended.crossing["at_row"] * \
                    flown.cycle_s

    def _glidepath(self, i: int, step: int) -> None:
        """Under the procedure's altitudes: aircraft ``i`` (flying step ``step``'s words) stopped where the state it
        reached sank below the glidepath lower edge of the runway in force during the step."""
        k = self._step_of(i, step)
        if self.ended_at[i] >= 0 or k >= self.sentence_end[i] or k > self.judged_to[i]:
            return
        state = self.states[i][k + 1]
        pointer = int(self.speaker.value[i, RUNWAY]) - 1
        below, _ = below_floor(self.finals[i][pointer], np.array([state.e_m]), np.array([state.n_m]),
                               np.array([state.height_m]), self.words.spec)
        if below[0]:
            self.sentence_end[i], self.judged_to[i], self.own[i] = k + 1, k + 1, BELOW_GLIDEPATH

    def _positions(self, index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where each aircraft of ``index`` is (its executor's state, as the speaker reads a flown row)."""
        e, n, h = np.zeros(len(index)), np.zeros(len(index)), np.zeros(len(index))
        for first, members, executor, _ in self.executors:
            here = np.isin(index, members)
            if not here.any():
                continue
            state = executor.now()
            at = self.place[index[here]]
            e[here] = state.e_m.cpu().numpy()[at]
            n[here] = state.n_m.cpu().numpy()[at]
            h[here] = state.height_m.cpu().numpy()[at]
        return e, n, h

    # -- what the speaker asks
    def _edges(self, first: int, last: int) -> np.ndarray:
        """Every speaking scene's edge features on batch steps ``first … last − 1``, read from two steps before (as
        `SceneLoop._edges`): each commanded aircraft's rows placed by `traffic_speaking.edge_rows` from its row 0, the
        replayed ones by the same call over the scene's first commanded aircraft — the one placement of the edge code's
        source hash — stacked in the speaker's places."""
        speaker = self.speaker
        low = max(first - 2, 0)
        out = np.zeros((len(speaker.others), last - first, speaker.aircraft, speaker.aircraft,
                        len(speaker.model.traffic_features)), dtype=np.float32)
        pointers = speaker.in_force[:, :, RUNWAY].cpu().numpy()
        by_airport: dict[str, list[tuple[int, SceneRows]]] = defaultdict(list)
        scenes = np.arange(len(self.keys)) if self.alone else self.window_of
        for b in range(len(speaker.others)):
            members = np.flatnonzero(scenes == b)
            parts = []
            for m, i in enumerate(members):
                window = self.windows[self.window_of[i]]
                others = window.others if m == 0 and not self.alone else ()
                scene = Scene(window.airport, self.keys[i], window.first_step_s(self.keys[i], self.step_s), others)
                parts.append(edge_rows(scene, speaker.e[i], speaker.n[i], speaker.h[i], pointers[i],
                                       int(speaker.rows[i]), int(self.start[i]), low, last, self.step_s))
            head = parts[0]                                   # the first commanded one, then its replayed ones
            rows = parts if len(parts) == 1 else [_aircraft(head, slice(0, 1)), *parts[1:]] + (
                [_aircraft(head, slice(1, None))] if len(head.category) > 1 else [])
            by_airport[self.windows[self.window_of[members[0]]].airport.flights.code].append((b, _stacked(rows)))
        for code, blocks in by_airport.items():
            stacked = _stacked([rows for _, rows in blocks])
            separation = next(w.airport.flights.separation for w in self.windows if w.airport.flights.code == code)
            sizes = [len(rows.category) for _, rows in blocks]
            for (b, _), size, block in zip(blocks, sizes, scene_edge_blocks(stacked, separation, sizes)):
                out[b, :, :size, :size] = block[first - low:]
        return out

    def _masks(self, column: int, chosen: np.ndarray, now: np.ndarray) -> np.ndarray:
        """``[N, classes]``: `traffic_speaking.speaking_masks` of each aircraft of the round (``now``) at the step, its
        window's others there — the replayed ones and the commanded ones not yet flying at their records
        (`traffic_speaking.others_at`), the commanded ones flying (or passive) at their executor states with their words
        in force, this step's so far (`WindowSpeaker.value`); one with no runway in force or said keeps every word."""
        speaker = self.speaker
        step = speaker.step
        classes = speaker.model.config.classes[column]
        out = np.ones((len(now), classes), dtype=bool)
        state = self._aircraft_now(step)
        value = speaker.value
        for i in np.flatnonzero(now):
            pointer = int(chosen[i, RUNWAY]) or int(value[i, RUNWAY])
            if not pointer:
                continue
            w = int(self.window_of[i])
            t_s = self.window_time_s(w, step)
            speed = int(value[i, SPEED]) - 1
            out[i] = speaking_masks(self.windows[w].scene(self.keys[i], self.step_s), column, classes, t_s, state[i],
                                    pointer - 1, speed if speed >= 0 else None,
                                    int(value[i, APPROACH]) - 1 == APPROACH_CLEARED, float(self.approach_mps[i]),
                                    self.words, self._step_of(i, step) == 0, self._others_at(i, step))
            self.separation_masked[column][i, self._step_of(i, step)] |= not out[i].all()
        return out

    def _others_at(self, i: int, step: int) -> OthersAt:
        """Aircraft ``i``'s others at the step as the masks read them (`_masks`)."""
        w = int(self.window_of[i])
        t_s = self.window_time_s(w, step)
        window = self.windows[w]
        if self._others[0] != step:
            self._others = (step, {})
        members = np.flatnonzero(self.window_of == w)
        flying = [j for j in members if j != i and self._step_of(j, step) >= 0 and not self.done[j]]
        waiting = [self.keys[j] for j in members if j != i and self._step_of(j, step) < 0]
        if w not in self._others[1]:
            self._others[1][w] = others_at(Scene(window.airport, self.keys[i], 0.0, tuple(waiting) + window.others),
                                           t_s, self.words)
        replayed = self._others[1][w]
        commanded = [j for j in flying if self.speaker.value[j, RUNWAY]]
        if not commanded:
            return replayed
        state = self._aircraft_now(step)
        levels = [self.words.speed_mps(c) for c in range(self.words.speed_unspecified)]
        traffic, speeds, targets, cleared = [], [], [], []
        geometry, separation = window.airport.flights.geometry, window.airport.flights.separation
        for j in commanded:
            now, value = state[j], self.speaker.value[j]
            candidate = geometry.candidates[int(value[RUNWAY]) - 1]
            relative = relative_to_runway(np.array([now.e_m]), np.array([now.n_m]), np.array([now.track_deg]),
                                          np.array([now.height_m]), candidate)
            angle = float(relative.track_minus_course_deg[0])
            traffic.append(Traffic(np.array([now.e_m]), np.array([now.n_m]), np.array([now.height_m]),
                                   (candidate.ident,),
                                   np.array([separation.along_nm[candidate.ident] * NM_M
                                             - float(relative.before_threshold_m[0])]),
                                   np.array([angle]), np.array([float(relative.right_of_course_m[0])]),
                                   np.array([now.captured]), (window.rows(self.keys[j]).category,)))
            speed = now.ground_speed_mps * math.cos(math.radians(angle))
            word = int(value[SPEED]) - 1
            speeds.append(speed)
            targets.append(speed if word < 0 else levels[word] if word < self.words.speed_unspecified
                           else float(self.approach_mps[j]))
            cleared.append(int(value[APPROACH]) - 1 == APPROACH_CLEARED)
        return OthersAt(join(replayed.traffic, *traffic), np.concatenate((replayed.speeds, speeds)),
                        np.concatenate((replayed.targets, targets)), np.concatenate((replayed.cleared, cleared)))

    # -- the end
    def results(self) -> list[Commanded]:
        """What became of each commanded aircraft (`Commanded`), once the loop has run."""
        out = []
        for i, key in enumerate(self.keys):
            _, members, executor, spoken = self.executors[self.group[i]]
            grid = spoken.sentences()[self.place[i]]
            said_steps = int(min(len(grid), self.sentence_end[i],
                                 int(executor.done_cycle[self.place[i]]) // executor.step_rows + 1))
            w = int(self.window_of[i])
            end = self.runs[w].ended.get(key)
            counted = said_steps if end is None else min(said_steps, int(self.ended_at[i]))
            out.append(Commanded(key, w, grid[:said_steps], LOST_SEPARATION if end is not None else self.own[i],
                                 self.own[i], end, counted, int(self.group[i]), int(self.place[i])))
        return out

    def close(self) -> None:
        """Let the speaker go (`traffic_speaking.SceneLoop.close`: the loop and the speaker hold each other)."""
        self.speaker.edges_of = self.speaker.masks_of = None
        self.speaker = None


def _aircraft(rows: SceneRows, part: slice) -> SceneRows:
    """``rows``' aircraft at ``part``."""
    return SceneRows(rows.time_s[part], rows.e_m[part], rows.n_m[part], rows.height_m[part], rows.along_m[part],
                     list(rows.runway)[part], list(rows.category)[part])


def _stacked(parts: Sequence[SceneRows]) -> SceneRows:
    """Scenes' (or aircraft's) rows on one aircraft axis, in order."""
    return SceneRows(*(np.concatenate([getattr(p, name) for p in parts]) for name in
                       ("time_s", "e_m", "n_m", "height_m", "along_m")),
                     [row for p in parts for row in p.runway], [c for p in parts for c in p.category])
