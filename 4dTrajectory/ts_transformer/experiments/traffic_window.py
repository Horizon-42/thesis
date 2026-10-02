"""The prior commanding every aircraft of a time window (multi-aircraft design §2.2 "窗口内由模型指挥", §6.6 step 7) —
shared by M3's and M4's second pass, not a runner.

**Windows** (step 7 item 1): an airport's flights are chained into segments on their rows' steps
(`prior.scene.SceneIndex.segments`, design §2.3; every row is on a step); from each segment's first step a window opens
every `WINDOW_EVERY_S` up to its last step, `WINDOW_S` long. The prior commands the flights with a sentence whose first
row is on a step inside the window and that fly on their own dynamics (the replay gate's group, `autopilot.replay`); everything else in
the air — before, after, the background, a flight the executor cannot fly — is replayed along its record. Windows
overlap, so a flight is commanded in two of them, once in each. A window's scene runs from its first commanded aircraft's
first row to the last one's time limit.

What the loop reads is placed by the one-aircraft code (`traffic_speaking`, in the edge features' source hash): each
commanded aircraft's view of its window is a one-speaking `traffic_speaking.Scene` (`Window.scene`).
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
from ts_transformer.experiments.traffic_go_around import GO_AROUND_EXTRA_S, PROBE_MARGIN, approach_altitude_m
from ts_transformer.experiments.traffic_labelled import own_end
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION, Controlled, Judging, Run, join
from ts_transformer.experiments.traffic_scene_data import FlightRows
from ts_transformer.experiments.traffic_speaking import (
    HISTORY_S, INSERTED, MASK_COLUMNS, Aircraft, OthersAt, Scene, SceneAirport, edge_rows, others_at, scene_landings,
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
    APPROACH, APPROACH_CLEARED, APPROACH_GO_AROUND, RUNWAY, SPEED, UNCHANGED, Words, compass_from_math_rad,
)
from ts_transformer.prior.generate import rows_for
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import below_floor
from ts_transformer.prior.scene import N_LOOK, Landings, SceneIndex
from ts_transformer.prior.scene_data import Node
from ts_transformer.prior.window_speaker import WindowSpeaker

#: A window's length and how often one opens (design §8: 20 minutes, every 10).
WINDOW_S = 1_200.0
WINDOW_EVERY_S = 600.0
#: Windows whose flights are rebuilt at a time while drawing (a rebuild opens the flights' tracks).
DRAW_CHUNK = 64


@dataclass(frozen=True)
class Window:
    """One window of an airport's scene: the step it opens on, the flights the prior commands (by their first row's step,
    then key), the others replayed (in the air at a step from the first commanded aircraft's first row to
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

    def first_step_s(self, key: str) -> float:
        """A commanded aircraft's first row's step."""
        return self.rows(key).presence.start_s

    def scene(self, key: str) -> Scene:
        """Commanded aircraft ``key``'s view of the window as a one-speaking scene (`view`): every other aircraft of the
        window among the others — the other commanded ones first, in the window's order, then the replayed."""
        return self.view(key, self.first_step_s(key),
                         tuple(k for k in self.commanded if k != key) + self.others)

    def view(self, key: str, first_step_s: float, others: tuple[str, ...]) -> Scene:
        """Commanded aircraft ``key`` speaking from ``first_step_s`` in a scene of this window with ``others``, the
        window's moved flights found by their keys. An inserted aircraft speaks as the flight it was inserted from: what
        a scene reads of its speaking flight is its category, the same."""
        return Scene(self.airport, key.removesuffix(INSERTED), first_step_s, others, self.moved)


def window_tiles(airport: SceneAirport, step_s: float) -> list[tuple[float, tuple[str, ...]]]:
    """Every window of ``airport``'s segments (module docstring) that holds a flight with a sentence: its opening step and
    those flights, by their first row's step (then key) — which of them fly is the draw's to find."""
    out = []
    for segment in SceneIndex(f.presence for f in airport.flights.flights.values()).segments():
        first, last = min(p.start_s for p in segment), max(p.end_s for p in segment)
        entering = sorted((p.start_s, p.dataset_id) for p in segment if p.speaking)
        opens = first
        while opens <= last:
            keys = tuple(k for s, k in entering if opens <= s < opens + WINDOW_S)
            if keys:
                out.append((opens, keys))
            opens += WINDOW_EVERY_S
    return out


def window_of(airport: SceneAirport, opens_s: float, commanded: Sequence[str], limits_s: Sequence[float],
              step_s: float, moved: Sequence[tuple[FlightRows, Track]] = (), left_out: Sequence[str] = ()) -> Window:
    """The window opening at ``opens_s`` with ``commanded`` (each flying ``limits_s`` from its first predicted step;
    ``moved``: flights moved in time or inserted, an augmented window's): everything else in the air from the first one's
    first row to the end of the last one's time limit replayed — but ``left_out`` (an inserted flight's source)."""
    window = Window(airport, opens_s, tuple(commanded), (), tuple(moved))
    firsts = [window.first_step_s(k) for k in commanded]
    start = min(firsts)
    end = max(first + N_LOOK * step_s + limit for first, limit in zip(firsts, limits_s))
    taken = set(commanded) | set(left_out)
    tracks = [window.track(k) for k in airport.tracks if k not in taken] + \
        [track for _, track in moved if track.key not in taken]
    others = tuple(t.key for t in tracks if t.first_step_s <= end and t.last_step_s >= start)
    return dataclasses.replace(window, others=others)


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
    rebuilt once — `replay.draw_flights`, a chunk of tiles at a time — re-read and checked against its stored sentence,
    as `replay.draw` does); a tile none of whose flights flies is passed over and counted; refused when an airport runs
    short."""
    sentences = load_sentences(directory, split, spec)
    stored = {int(index): k for k, index in enumerate(sentences["signal_index"])}
    # the split's signals, loaded once: a flight's arrays are views of the whole split's, and every `draw_flights`
    # loads its own — the batch keeps these, never those (each would pin a copy of the split)
    every = load_signals(directory, split)
    index = {s.dataset_id: i for i, s in enumerate(every)}
    rng = np.random.default_rng(seed)
    openings: list[tuple[str, float, tuple[str, ...]]] = []
    rebuilt: dict[str, tuple[Any, Any, str, Any] | None] = {}   # key → (signals, series, group, reading); None: no
    geometries: Mapping[str, Any] = {}
    paths: Mapping[str, Any] = {}
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
            new = sorted({k for _, keys in chunk for k in keys if k not in rebuilt})
            if new:
                drawn = replay.draw_flights(directory, split, [index[k] for k in new], per_airport=0, seed=0)
                geometries, paths = drawn.geometries, drawn.vertical_paths
                flies = dict(zip((s.dataset_id for s in drawn.signals), zip(drawn.series, drawn.groups)))
                del drawn
                for k in new:
                    rebuilt[k] = None
                    if k in flies:
                        series, group = flies[k]
                        signals = every[index[k]]
                        rebuilt[k] = (signals, series, group,
                                      _reading(signals, index[k], geometries, sentences, stored, spec, words))
            for opens, keys in chunk:
                if taken == per_airport:
                    break
                read += 1
                commanded = tuple(k for k in keys if rebuilt[k] is not None)
                if not commanded:
                    passed += 1
                    continue
                entering_unflown.update({"entering": len(keys), "not_flown": len(keys) - len(commanded)})
                openings.append((code, opens, commanded))
                taken += 1
        if taken < per_airport:
            raise ValueError(f"{code}: {taken} windows with a flight that flies, {per_airport} wanted")
        counts["airports"][code] = {"tiles": len(tiles), "read": read, "passed_over": passed, "windows": taken,
                                    "commanded": sum(len(c) for a, _, c in openings if a == code),
                                    **dict(entering_unflown)}
    counts["rebuilt"] = len(rebuilt)
    members = [rebuilt[k] for _, _, commanded in openings for k in commanded]
    batch = replay.Batch(signals=[m[0] for m in members], series=[m[1] for m in members],
                         readings=[m[3] for m in members], geometries=[geometries[m[0].airport] for m in members],
                         vertical_paths=[paths[m[0].airport] for m in members],
                         approach_ias_mps=[replay.flight_approach_ias_mps(m[1], m[2]) for m in members],
                         groups=[m[2] for m in members], drawn=counts)
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
    """What became of one commanded aircraft (`WindowLoop.results`): the words it said to its own end (``[steps, 6]``,
    `UNCHANGED` where a column says nothing; nothing after the judge ended it), its outcome under the loop's reading (its
    own end, or `traffic_loop.LOST_SEPARATION` when the judge ended it first — then ``end`` is that end), its own end
    (the executor judge's or the glidepath edge's, whatever the judge did), the steps said up to its judged end
    (``counted``: what it is trained on; `traffic_free_generation.judged_steps`' count), its first state's time, its
    crossing (None: it did not land), the runway in force at its end (a candidate's index), where its executor's
    states are (``group``, ``place``), the own step of its first go-around word (None: it said none; multi-aircraft
    design §6.6 step 8) and of a go-around a probe said for it (None: none; step 8 item 10)."""

    key: str
    window: int
    said: np.ndarray
    outcome: str
    own: str
    end: dict[str, Any] | None
    counted: int
    first_s: float
    landing_s: float | None
    runway: int
    group: int
    place: int
    go_around: int | None
    forced: int | None


_NEVER = np.iinfo(np.int64).max


@dataclass(frozen=True)
class Given:
    """A commanded aircraft's words given instead of spoken (multi-aircraft design §6.6 step 7.7): ``said`` (``[steps,
    6]``, its own steps from its first predicted one, as `Commanded.said`) up to its own step ``until`` — never past
    ``said``'s end; from there the prior speaks for it."""

    said: np.ndarray
    until: int

    def __post_init__(self) -> None:
        if not 0 <= self.until <= len(self.said):
            raise ValueError(f"a line given to own step {self.until} of {len(self.said)} said")


class WindowLoop:
    """Every commanded aircraft of a batch of windows flown with the prior speaking, judged as it flies (module
    docstring and design §6.6 step 7 item 3). ``windows``: one per sample (a window spoken to `K` times is here `K`
    times); ``flights``: their commanded aircraft window after window, each window's in its order, with each one's
    executor inputs (``inputs``, ``runways``, ``charts``, ``approach_ias_mps`` over ``flights``, from its first predicted
    step), time limit and geometry; ``alone``: each aircraft speaks seeing no other (its edge features its own, no
    separation mask) and is still judged in its window; ``given``: an aircraft's words given up to an own step
    (`Given`; None: spoken throughout) — said in its round, neither sampled nor masked; ``go_around_extra_s``: the time a
    go-around adds to its sentence (0: none — as the one-aircraft scene loop, whose module the traffic prior's edge-source
    hash holds, flies it); ``probing``: the aircraft a probe watches (multi-aircraft design §6.6 step 8 item 10; None:
    none) — once cleared, established on its final and under its approach altitude, and while it has said no go-around,
    a go-around is said for it at the first step its tightest margin at its step before was under ``probe_margin`` (`WindowSpeaker.speak`'s ``forced``:
    said in place of its draw; an infinite ``probe_margin``: at its first such step, alone or not — the longest sentences,
    a preflight's).

    **A step** (`step`), at the batch step every speaking aircraft's rows reach:

    1. the aircraft still spoken to speak, round by round from the front of the approach clock (`WindowSpeaker`);
    2. every aircraft of a window in the scene is put on the runway in force after this step's words
       (`speaking_aircraft`'s reading of a state's runway);
    3. every landing in ``(t − step, t]`` of the window (a commanded aircraft's crossing, a replayed one's roster time) is
       checked against the aircraft behind it (`traffic_loop.Judging.landing`) — a passive one lands too;
    4. the window's aircraft on the step are judged (`Judging.step`); one ended here (or at 3) flies this step's words,
       said before the judge read the step, and then nothing more — passive: in the scene, never ended again;
    5. the executor flies the step: ONE executor for the batch, each aircraft from its own first spoken step
       (`Executor`'s ``start_cycle``: a multi-aircraft batch, executor design §12.4). The aircraft that first spoke at
       one step are a COHORT and fly as the executor once given to each cohort flew them: a cohort whose aircraft are
       all done at a step's start, or that has flown its own cycles (its longest time limit, also within a step), is
       halted where that executor stopped (`Executor.halt`) and from then on is neither recorded nor flown;
    6. an aircraft whose flight ended in the step leaves after its own last state (`traffic_labelled.own_end`): one its
       executor is done with, and one that stalled or crossed a threshold plane in it, whose end `autopilot.judge.
       outcome_of` finds then — the executor flies on past a stall, an uncaptured crossing or another runway's, which
       `outcome_of` reads afterwards (**stated difference**: read as it happens, with the runway in force then; the
       one-aircraft judge reads it once the sentence is over, with the last runway it said — and one that only a later
       runway makes an end, a lined-up pass over runway A read once the aircraft is pointed at B, is found then, at its
       own earlier row, after the loop has judged past it). Under the procedure's
       altitudes one whose next state sank below the glidepath lower edge stops speaking there and is judged at that
       state for the last time (`prior_free_generation.glidepath_stops`' reading), then flies on passive. The rest append
       the state they reached, as does one whose own last state is the one reached (a time limit's).

    A window is judged from its first commanded aircraft's first predicted step while one of its commanded aircraft is
    still to be read (every state it is in the scene at judged, its landing checked); the loop runs on, judging only,
    once the executors are done, until every window is read.

    **The landing context** (the prior design's rule: the inputs are what is known before the step): a commanded
    aircraft's context is the airport's landings less every commanded aircraft of its window's recorded one, and each
    of theirs added as it lands in the loop (the rows not encoded yet read it again: `WindowSpeaker.set_context`).
    **Stated approximation**: a replayed aircraft's inputs are the samples' (`traffic_scene_data`, in the edge features'
    source hash), its landing context the recorded one, the commanded aircraft's recorded landings included.

    With one commanded aircraft a window it says and flies, to its judged end, what `traffic_speaking.SceneLoop` says
    and flies, and ends where `traffic_speaking.judged` ends it (tests).
    """

    def __init__(self, model: Prior, windows: Sequence[Window], flights: Sequence[FlightSignals],
                 geometries: Sequence[AirportGeometry], inputs: FlightInputs, runways: Runways, charts: AirportCharts,
                 approach_ias_mps: torch.Tensor, limits: Sequence[float], words: Words, params: ExecutorParams,
                 landings: Any, *, generator: torch.Generator, temperature: float, procedure_masks: ProcedureMasks,
                 alone: bool = False, reading: str = VISUAL, given: Sequence[Given | None] | None = None,
                 go_around_extra_s: float = GO_AROUND_EXTRA_S, probing: Sequence[bool] | None = None,
                 probe_margin: float = PROBE_MARGIN) -> None:
        step_s = words.spec.step_s
        self.windows, self.words, self.params, self.step_s, self.alone = list(windows), words, params, step_s, alone
        self.flights, self.geometries = list(flights), list(geometries)
        self.keys = [f.dataset_id for f in flights]
        self.given = [None] * len(flights) if given is None else list(given)
        if len(self.given) != len(flights):
            raise ValueError("a given line or None for every commanded aircraft")
        self.window_of = np.array([w for w, window in enumerate(windows) for _ in window.commanded], dtype=np.int64)
        if [k for window in windows for k in window.commanded] != self.keys:
            raise ValueError("the flights are the windows' commanded aircraft, window after window, each in its order")
        count = len(flights)
        self.members = [np.flatnonzero(self.window_of == w) for w in range(len(windows))]
        places = window_places(windows, step_s)
        self.pre, self.origin, self.start, nodes = places.pre, places.origin, places.start, places.nodes
        # a go-around gives its sentence ``go_around_extra_s`` more (`GO_AROUND_EXTRA_S`, multi-aircraft design §6.6 step
        # 8 item 9), as much of it as the model's positions hold after its own time limit (`extra_s`, stated in each
        # go-around's row)
        room_s = np.array([(model.config.max_rows - rows_for(limit + step_s, step_s)) * step_s for limit in limits])
        if bool((room_s < 0.0).any()):
            raise ValueError("a commanded aircraft's time limit is longer than the model's positions")
        self.go_around_extra_s = go_around_extra_s
        self.probing = np.zeros(len(flights), dtype=bool) if probing is None else np.asarray(probing, dtype=bool)
        if len(self.probing) != len(flights):
            raise ValueError("a probe flag for every commanded aircraft")
        self.probe_margin = probe_margin
        #: the own step a probe said a go-around for each aircraft at (−1: none)
        self.forced_step = np.full(len(flights), -1, dtype=np.int64)
        self.extra_s = np.minimum(go_around_extra_s, room_s)
        max_rows = rows_for(float(max(np.array(limits) + self.extra_s)) + step_s, step_s)
        self.speaker = WindowSpeaker(
            model, flights, geometries, landings, words, scenes=np.arange(count) if alone else self.window_of,
            starts=self.start, others=[[] for _ in range(count)] if alone else nodes, edges=self._edges,
            masks=self._masks, mask_columns=() if alone else MASK_COLUMNS, max_rows=max_rows, generator=generator,
            procedure_masks=procedure_masks, temperature=temperature,
            scene_landings=None if landings is None else self._contexts(landings))
        #: per masked column, each aircraft's step: whether the separation masks took a word away there
        self.separation_masked = {c: np.zeros((count, max_rows - N_LOOK), dtype=bool) for c in MASK_COLUMNS}
        # one executor for the batch, each aircraft from its first spoken step; its groups (module docstring, 5.)
        device = inputs.initial_state.device
        self.approach_mps = approach_ias_mps.cpu().numpy()
        self.first_spoken = self.start + N_LOOK                    # each aircraft's first spoken batch step
        first = int(self.first_spoken.min())
        step_rows = int(round(step_s / params.cycle_s))
        limit_s = torch.tensor(list(limits), dtype=torch.float64, device=device)
        executor = Executor(inputs, runways, charts, approach_ias_mps, params, words, time_limit_s=limit_s,
                            start_cycle=torch.as_tensor((self.first_spoken - first) * step_rows, device=device),
                            reserve_s=go_around_extra_s)
        #: each aircraft's time limit (a go-around's extended) and the own step of its first go-around word (−1: none)
        self.limit_s = np.array(list(limits), dtype=np.float64)
        self.go_around_step = np.full(count, -1, dtype=np.int64)
        #: at its first go-around: the runway in force (−1: none) and the executor's height, path angle and airspeed —
        #: what the reward's H reads (`traffic_window_generation.go_around_fields`)
        self.go_around_runway = np.full(count, -1, dtype=np.int64)
        self.go_around_state = np.full((count, 3), np.nan)
        #: each aircraft's tightest separation margin at each of its own steps judged (`separation.margins`; inf: none)
        self.margin = np.full((count, max_rows - N_LOOK), np.inf)
        #: each aircraft's executor (one: 0) and its place there
        self.group = np.zeros(count, dtype=np.int64)
        self.place = np.arange(count, dtype=np.int64)
        #: each aircraft's cohort (module docstring, 5.), its members, and each cohort's own cycles: its longest time
        #: limit's — the cycles its own executor had
        self.cohort = np.unique(self.first_spoken, return_inverse=True)[1].astype(np.int64)
        self.cohorts = [np.flatnonzero(self.cohort == c) for c in range(int(self.cohort.max()) + 1)]
        own_cycles = torch.ceil(limit_s / params.cycle_s).long().cpu().numpy()
        self.cohort_cycles = np.array([own_cycles[members].max() for members in self.cohorts])
        self.executors: list[tuple[int, np.ndarray, Executor, Spoken]] = [
            (first, self.place, executor, Spoken(count, words, device=device, start_step=self.first_spoken - first))]
        # the judge, a window at a time
        self.runs = [Run(reading) for _ in windows]
        self.judging = [Judging(window.airport.flights.separation, reading, step_s, run)
                        for window, run in zip(windows, self.runs)]
        self.replayed = [[window.track(k) for k in window.others] for window in windows]
        self.first_step = self.pre + N_LOOK                      # every window's first judged step
        self.last_step = np.full(len(windows), self.first_step)  # each window's last judged step
        self.finals = ([procedure_masks.finals[g.code] for g in geometries] if procedure_masks.altitudes else None)
        # each aircraft: a `_State` per step it has reached, its own end and the judge's
        self.states: list[list[_State]] = [[] for _ in range(count)]
        self.unwrapped: list[_Unwrapped | None] = [None] * count
        self.ended_at = np.full(count, -1, dtype=np.int64)       # the step the judge ended it at, −1: none
        self.judged_to = np.full(count, _NEVER)                  # the last step it is judged at (its own end's)
        self.in_scene_to = np.full(count, _NEVER)                # the last step it is in the scene at
        self.sentence_end = np.full(count, _NEVER)               # its steps: the glidepath edge stops it
        self.judged_state = np.full(count, -1, dtype=np.int64)   # the last of its steps read
        self.steps_flown = np.zeros(count, dtype=np.int64)       # its steps to its own end
        self.own: list[str | None] = [None] * count
        self.landing_s = np.full(count, math.nan)
        self.landing_checked = np.zeros(count, dtype=bool)
        self.left = np.zeros(count, dtype=bool)                  # its flight is over
        self._others: dict[tuple[int, int, tuple[str, ...]], OthersAt] = {}
        self._now: tuple[int, list[Aircraft] | None] = (-1, None)
        #: each aircraft's landing context switches: (its first row reading the new one, the landing added, its runway)
        self.context_changes: list[list[tuple[int, float, str]]] = [[] for _ in range(count)]

    def _contexts(self, landings: Mapping[str, Landings]) -> list[Landings]:
        """Each commanded aircraft's landings before its own is taken out (`WindowSpeaker`: `data.own_context`):
        `window_landings`."""
        return [window_landings(self.windows[w], key, landings, self.step_s) for key, w in zip(self.keys, self.window_of)]

    # -- the clock
    def window_time_s(self, w: int, step: int) -> float:
        return float(self.origin[w] + step * self.step_s)

    def _step_of(self, i: int, step: int) -> int:
        """Aircraft ``i``'s own step (0: its first predicted one) at batch step ``step``."""
        return int(step - self.start[i] - N_LOOK)

    def _to_read(self, i: int) -> bool:
        """Whether aircraft ``i`` still has a state in the scene to judge or a landing to check."""
        if not self.left[i]:
            return True
        last = min(int(self.in_scene_to[i]), len(self.states[i]) - 1)
        return self.judged_state[i] < last or (not math.isnan(self.landing_s[i]) and not self.landing_checked[i])

    def _live(self, w: int) -> bool:
        return any(self._to_read(i) for i in self.members[w])

    @property
    def running(self) -> bool:
        return any(self._live(w) for w in range(len(self.windows)))

    def step(self) -> None:
        """One batch step (the class docstring's 1–6)."""
        speaker, step = self.speaker, self.speaker.step
        starting = np.flatnonzero(self.first_spoken == step)    # their first predicted step: the state flown from
        if len(starting):
            self._record(starting, self.executors[0][2], step, captured=np.zeros(len(starting), dtype=bool))
        k = step - self.start - N_LOOK
        speaking = (k >= 0) & (self.ended_at < 0) & ~self.left & (k < self.sentence_end)
        given = self._given(k, speaking)
        forced = self._probes(k, speaking, given)
        said = speaker.speak(self._rank(speaking, step), self._locked(), given, forced)
        said_go_around = said[:, APPROACH] == APPROACH_GO_AROUND + 1
        probed = np.flatnonzero((forced[:, APPROACH] >= 0) & said_go_around)
        self.forced_step[probed] = k[probed]
        going = np.flatnonzero(speaking & said_go_around & (self.go_around_step < 0))
        if len(going):
            self._go_around(going, k)
        for w in range(len(self.windows)):
            if self._live(w):
                self._runways(w, step)
                self._landings(w, step)
                self._judge(w, step)
                self.last_step[w] = step
                self.runs[w].scene_seconds = float(step - self.first_step) * self.step_s
        self._fly(step, said)

    def _probes(self, k: np.ndarray, speaking: np.ndarray, given: np.ndarray | None) -> np.ndarray:
        """``[N, 6]``: the go-around a probe says this step for each aircraft it watches (class docstring), −1 elsewhere:
        speaking its own words (not given), cleared, established on its final (the executor's capture) and under the
        approach altitude of the runway in force (`traffic_go_around.approach_altitude_m`: inside the FAF on the
        glidepath — a go-around abandons an approach to landing; one said before is flown straight back onto the line,
        small tests ② / ②b), no go-around said yet, and its tightest margin at its own step before under
        ``probe_margin``."""
        out = np.full((len(k), 6), -1, dtype=np.int64)
        before = np.clip(k - 1, 0, self.margin.shape[1] - 1)
        captured = np.array([bool(states[-1].captured) if states else False for states in self.states])
        low = np.zeros(len(k), dtype=bool)
        for i in np.flatnonzero(captured & self.probing):
            pointer = int(self.speaker.value[i, RUNWAY]) - 1
            low[i] = self.states[i][-1].height_m < approach_altitude_m(self.geometries[i], pointer)
        captured &= low
        own = np.ones(len(k), dtype=bool) if given is None else given[:, 0] < 0
        watch = (self.probing & speaking & own & captured & (self.go_around_step < 0) & (k >= 1)
                 & (self.speaker.value[:, APPROACH] == APPROACH_CLEARED + 1)
                 & ((self.margin[np.arange(len(k)), before] < self.probe_margin) | (self.probe_margin == math.inf)))
        out[watch, APPROACH] = APPROACH_GO_AROUND + 1
        return out

    def _go_around(self, going: np.ndarray, k: np.ndarray) -> None:
        """The aircraft ``going`` said their first go-around at own steps ``k[going]``: their `extra_s` more time, on their
        executor and their cohort's own cycles (module docstring, 5.); the runway in force and the executor's state
        there, for the reward's H (`traffic_go_around`)."""
        self.go_around_step[going] = k[going]
        self.limit_s[going] += self.extra_s[going]
        executor = self.executors[0][2]
        now = executor.now()
        state = np.stack([x.cpu().numpy() for x in (now.height_m, now.gamma_rad, now.speed_mps)], axis=1)
        self.go_around_runway[going] = self.speaker.value[going, RUNWAY] - 1
        self.go_around_state[going] = state[self.place[going]]
        seconds = np.zeros(len(self.place))
        seconds[self.place[going]] = self.extra_s[going]
        executor.extend_time_limit(torch.as_tensor(seconds))
        for c in np.unique(self.cohort[going]):
            members = self.cohorts[int(c)]
            self.cohort_cycles[c] = int(np.ceil(self.limit_s[members] / self.params.cycle_s).max())

    def _given(self, k: np.ndarray, speaking: np.ndarray) -> np.ndarray | None:
        """``[N, 6]``: the speaking aircraft's given lines at their own steps ``k`` (`Given`) as the speaker's classes
        (0: unchanged, else the word + 1), −1 for the others; None when none is given."""
        out = np.full((len(k), 6), -1, dtype=np.int64)
        for i, line in enumerate(self.given):
            if line is not None and speaking[i] and k[i] < line.until:
                out[i] = np.where(line.said[k[i]] == UNCHANGED, 0, line.said[k[i]] + 1)
        return out if (out[:, 0] >= 0).any() else None

    # -- 2.–4. the judge
    def _runways(self, w: int, step: int) -> None:
        """Window ``w``'s commanded aircraft in the scene at the step: the runway in force after this step's words."""
        value = self.speaker.value
        for i in self.members[w]:
            k = self._step_of(i, step)
            if 0 <= k <= min(int(self.in_scene_to[i]), len(self.states[i]) - 1):
                self._on_runway(i, self.states[i][k], int(value[i, RUNWAY]))
                self.judged_state[i] = k

    def _landings(self, w: int, step: int) -> None:
        t_s = self.window_time_s(w, step)
        leaders: list[tuple[float, str, Controlled | Track, int]] = []
        for i in self.members[w]:
            if t_s - self.step_s < self.landing_s[i] <= t_s:
                leaders.append((float(self.landing_s[i]), self.keys[i], self._controlled(i, int(self.in_scene_to[i])),
                                int(i)))
                self.landing_checked[i] = True
        for track in self.replayed[w]:
            if t_s - self.step_s < track.presence.landing_s <= t_s:
                leaders.append((track.presence.landing_s, track.key, track, -1))
        for at, key, leader, i in sorted(leaders, key=lambda x: (x[0], x[1])):
            controlled, passive = self._here(w, step, at, exclude=key)
            replayed = [a for a in self.replayed[w]
                        if a.key != key and a.presence.times_s[0] <= at <= a.presence.times_s[-1]]
            # a commanded leader is passive when the judge ended it or it flew past its last judged step (a glidepath
            # stop), as `_here` reads it
            self.judging[w].landing(at, leader, controlled, replayed, passive,
                                    leader_passive=i >= 0 and (self.ended_at[i] >= 0
                                                               or self.in_scene_to[i] > self.judged_to[i]))
            self._mark_ended(w, step)                          # a follower ended here is passive at the next landing

    def _judge(self, w: int, step: int) -> None:
        t_s = self.window_time_s(w, step)
        controlled, passive = self._here(w, step, t_s)
        self.judging[w].step(t_s, controlled, [a for a in self.replayed[w] if a.on_step(t_s)], passive)
        margins = self.judging[w].margins
        for i in self.members[w]:
            own = self._step_of(i, step)
            if self.keys[i] in margins and 0 <= own < self.margin.shape[1]:
                self.margin[i, own] = margins[self.keys[i]]
        self._mark_ended(w, step)

    def _mark_ended(self, w: int, step: int) -> None:
        for i in self.members[w]:
            if self.ended_at[i] < 0 and self.keys[i] in self.runs[w].ended:
                self.ended_at[i] = self._step_of(i, step)

    def _here(self, w: int, step: int, at: float, exclude: str = "") -> tuple[list[Controlled], list[Controlled]]:
        """Window ``w``'s commanded aircraft in the scene at instant ``at`` of batch step ``step`` (on their steps up to
        this one): the judged ones and the passive ones."""
        controlled, passive = [], []
        for i in self.members[w]:
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
        return self._path(i, max(0, last - 1), last + 1)

    def path(self, i: int) -> Controlled:
        """Aircraft ``i`` on every step it was read at in the scene (a readout judging it again afterwards)."""
        return self._path(i, 0, int(self.judged_state[i]) + 1)

    def judged_until_s(self, i: int) -> float:
        """The last instant aircraft ``i`` was judged at (its own end's; past it, passive — a glidepath stop — or out)."""
        return self.states[i][int(min(self.judged_to[i], self.judged_state[i]))].t_s

    def _path(self, i: int, first: int, last: int) -> Controlled:
        states = self.states[i][first:last]
        rows = self.windows[self.window_of[i]].rows(self.keys[i])
        angle = np.array([s.angle_deg for s in states])
        return Controlled(rows.presence, np.array([s.t_s for s in states]), np.array([s.e_m for s in states]),
                          np.array([s.n_m for s in states]), np.array([s.height_m for s in states]),
                          tuple(s.runway for s in states), np.array([s.along_m for s in states]), angle,
                          np.array([s.right_m for s in states]),
                          np.array([s.ground_speed_mps for s in states]) * np.cos(np.radians(angle)),
                          np.array([s.captured for s in states], dtype=bool), rows.category,
                          self.own[i] or "flying", None if math.isnan(self.landing_s[i]) else float(self.landing_s[i]))

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

    # -- 1. speaking
    def _aircraft_now(self, step: int) -> list[Aircraft | None]:
        """Every aircraft's executor state at the start of the step (read once a step; None before it flies)."""
        if self._now[0] != step:
            now: list[Aircraft | None] = [None] * len(self.keys)
            _, index, executor, _ = self.executors[0]
            state = executor.now()
            e, n, h = state.e_m.cpu().numpy(), state.n_m.cpu().numpy(), state.height_m.cpu().numpy()
            track, ground = state.track_deg.cpu().numpy(), state.ground_speed_mps.cpu().numpy()
            captured = executor.lateral.captured.cpu().numpy()
            for p, i in enumerate(index):
                if self.first_spoken[i] <= step:
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
        for members in self.members:
            order = sorted((i for i in members if speaking[i]),
                           key=lambda i: (value[i, RUNWAY] == 0,
                                          -self._clock(i, now[i], int(value[i, RUNWAY])) if value[i, RUNWAY] else 0.0,
                                          i))
            for r, i in enumerate(order):
                rank[i] = r
        return rank

    def _locked(self) -> np.ndarray:
        out = np.zeros(len(self.keys), dtype=bool)
        for _, index, executor, _ in self.executors:
            out[index] = executor.runway_locked.cpu().numpy()
        return out

    # -- 5., 6. flying
    def _cohorts_halted(self, executor: Executor, *, at_step_start: bool) -> np.ndarray:
        """``[aircraft]`` bool: the aircraft whose cohort's own executor would have stopped by now — its own cycles flown,
        or, at a step's start, every aircraft of it done (module docstring, 5.)."""
        own = executor.own_cycle().cpu().numpy()
        stopped = np.array([own[members[0]] >= cycles for members, cycles in zip(self.cohorts, self.cohort_cycles)])
        if at_step_start:
            done = executor.done.cpu().numpy()
            stopped |= np.array([bool(done[members].all()) for members in self.cohorts])
        return stopped[self.cohort]

    def _fly(self, step: int, said: np.ndarray) -> None:
        cycle_s = self.params.cycle_s
        first, index, executor, spoken = self.executors[0]
        if first > step or executor.count == executor.cycles:
            self.speaker.advance(np.zeros(0, dtype=np.int64), *self._positions(np.zeros(0, dtype=np.int64)))
            return
        device = executor.done.device
        halted = self._cohorts_halted(executor, at_step_start=True)
        executor.halt(torch.as_tensor(halted, device=device))
        # the aircraft that fly this step: started, their cohort not halted
        flying = (self.first_spoken <= step) & ~halted
        heard = torch.as_tensor((spoken.steps - spoken.start) * self.step_s, dtype=torch.float64, device=device)
        spoken.say(np.where(said > 0, said - 1, UNCHANGED))
        stalled = torch.zeros(len(index), dtype=torch.bool, device=device)
        for _ in range(executor.step_rows):
            if executor.count == executor.cycles:
                break
            executor.halt(torch.as_tensor(self._cohorts_halted(executor, at_step_start=False), device=device))
            executor.cycle(spoken.at(heard), executor.own_cycle().clamp(min=0).to(torch.float64) * cycle_s)
            stalled |= executor.limits["stall"][-1]
        members = np.flatnonzero(flying)
        appended: list[int] = []
        if len(members):
            self._record(members, executor, step + 1, captured=executor.lateral.captured.cpu().numpy()[members])
            done, stalled = executor.done.cpu().numpy(), stalled.cpu().numpy()
            crossed = self._crossed(members, np.array([self._step_of(int(i), step) for i in members]))
            ending = [int(i) for i, cross in zip(members, crossed)
                      if not self.left[i] and (done[i] or stalled[i] or cross)]
            if ending:
                self._own_ends(ending, step)
            for i in members:
                if not self.left[i] and self.finals is not None:
                    self._glidepath(i, step)
                if self._step_of(i, step) + 1 <= self.in_scene_to[i]:
                    appended.append(int(i))                   # still in the scene (one's own last state included)
        index_appended = np.array(appended, dtype=np.int64)
        self.speaker.advance(index_appended, *self._positions(index_appended))

    def _record(self, index: np.ndarray, executor: Executor, step: int, captured: np.ndarray) -> None:
        """Every aircraft of ``index`` at batch step ``step``: its executor state (`flown_track`'s reading);
        ``captured`` in the order of ``index``."""
        states = executor.state.cpu().numpy()[self.place[index]]
        for p, i in enumerate(index):
            read = flown_track(states[p: p + 1], self.geometries[i])
            rad = float(np.radians(compass_from_math_rad(states[p: p + 1, PSI]))[0])
            if self.unwrapped[i] is None:
                self.unwrapped[i] = _Unwrapped(rad)
                track = np.float64(rad)
            else:
                track = self.unwrapped[i].next(rad)
            self.states[i].append(_State(self.window_time_s(int(self.window_of[i]), step), float(read["e"][0]),
                                         float(read["n"][0]), float(read["height"][0]), float(np.degrees(track)),
                                         float(read["ground_speed"][0]), bool(captured[p])))

    def _crossed(self, index: np.ndarray, own: np.ndarray) -> np.ndarray:
        """``[len(index)]`` bool: which aircraft of ``index`` passed a candidate runway's threshold plane in its own step
        ``own`` (each aircraft's, `_step_of`) — a crossing the judge may read as an end the executor does not stop at."""
        out = np.zeros(len(index), dtype=bool)
        by_geometry: dict[int, list[int]] = defaultdict(list)
        for p, i in enumerate(index):
            by_geometry[id(self.geometries[i])].append(p)
        for places in by_geometry.values():
            members = [int(index[p]) for p in places]
            steps = [int(own[p]) for p in places]
            e = np.array([[self.states[i][k].e_m, self.states[i][k + 1].e_m] for i, k in zip(members, steps)])
            n = np.array([[self.states[i][k].n_m, self.states[i][k + 1].n_m] for i, k in zip(members, steps)])
            for candidate in self.geometries[members[0]].candidates:
                d = relative_to_runway(e, n, np.zeros_like(e), np.zeros_like(e), candidate).before_threshold_m
                out[places] |= (d[:, 0] > 0.0) & (d[:, 1] <= 0.0)
        return out

    def _own_ends(self, ending: Sequence[int], step: int) -> None:
        """The aircraft ``ending`` whose flight may have ended in step ``step``: its own end
        (`outcome_of` with the runway in force, `traffic_labelled.own_end`) — the step it is last in the scene at and,
        while the judge has not ended it, last judged at; its crossing when it landed. One the executor flies on with
        no end found (a crossing that is none) flies on."""
        _, _, executor, _ = self.executors[0]
        flown = executor.flown()
        step_rows = executor.step_rows
        done = executor.done.cpu().numpy()
        for i in ending:
            p = int(self.place[i])
            ended = outcome_of(flown, p, self.geometries[i], int(self.speaker.value[i, RUNWAY]) - 1, self.words.spec)
            if not done[p] and ended.outcome == "timeout":
                continue                                        # nothing ended it yet
            outcome, last_row = own_end(ended.outcome, ended.end_row, -1, step_rows)
            last = last_row // step_rows
            self.left[i] = True
            self.steps_flown[i] = self._step_of(i, step) + 1
            self.in_scene_to[i] = min(self.in_scene_to[i], last)
            if self.own[i] is None:                             # (the glidepath edge's stop comes first)
                self.own[i] = outcome
                if self.ended_at[i] < 0:
                    self.judged_to[i] = min(self.judged_to[i], last)
            if outcome == "landed":
                w = int(self.window_of[i])
                self.landing_s[i] = self.states[i][0].t_s + ended.crossing["at_row"] * flown.cycle_s
                self._landed(i, w)

    def _landed(self, i: int, w: int) -> None:
        """Aircraft ``i``'s landing in the loop, added to the landing context of its window's other commanded
        aircraft (class docstring)."""
        if self.speaker.contexts[i] is None:
            return
        runway = self.geometries[i].candidates[int(self.speaker.value[i, RUNWAY]) - 1].ident
        for j in self.members[w]:
            if j != i:
                first = self.speaker.set_context(int(j), _with_landing(self.speaker.contexts[j],
                                                                       float(self.landing_s[i]), runway))
                self.context_changes[j].append((first, float(self.landing_s[i]), runway))

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
        if not len(index):
            return np.zeros(0), np.zeros(0), np.zeros(0)
        state = self.executors[0][2].now()
        at = self.place[index]
        return state.e_m.cpu().numpy()[at], state.n_m.cpu().numpy()[at], state.height_m.cpu().numpy()[at]

    # -- what the speaker asks
    def _edges(self, first: int, last: int) -> np.ndarray:
        """Every speaking scene's edge features on batch steps ``first … last − 1`` (`window_edges` of the speaker's
        rows)."""
        speaker = self.speaker
        return window_edges(self.windows, self.keys, self.window_of, self.alone, speaker.e, speaker.n, speaker.h,
                            speaker.in_force[:, :, RUNWAY].cpu().numpy(), speaker.rows, self.start, first, last,
                            self.step_s, speaker.aircraft, len(speaker.model.traffic_features))

    def _masks(self, column: int, chosen: np.ndarray, now: np.ndarray) -> np.ndarray:
        """``[N, classes]``: `traffic_speaking.speaking_masks` of each aircraft of the round (``now``) at the step, among
        its window's others (`_others_at`); one with no runway in force or said keeps every word."""
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
            speed = int(value[i, SPEED]) - 1
            out[i] = speaking_masks(self.windows[w].scene(self.keys[i]), column, classes,
                                    self.window_time_s(w, step), state[i], pointer - 1, speed if speed >= 0 else None,
                                    int(value[i, APPROACH]) - 1 == APPROACH_CLEARED, float(self.approach_mps[i]),
                                    self.words, self._step_of(i, step) == 0, self._others_at(i, step))
            self.separation_masked[column][i, self._step_of(i, step)] |= not out[i].all()
        return out

    def _others_at(self, i: int, step: int) -> OthersAt:
        """Aircraft ``i``'s others at the step as the masks read them: the replayed ones, and the commanded ones not
        speaking yet — before their first predicted step, or at it before their round — at their records with their
        labelled words, as the one-aircraft loop reads a replayed flight (`traffic_speaking.others_at`); the commanded
        ones in the scene past that at their executor states with their words in force, this step's so far
        (`WindowSpeaker.value`)."""
        w = int(self.window_of[i])
        window, t_s, value = self.windows[w], self.window_time_s(w, step), self.speaker.value
        waiting, commanded = [], []
        for j in self.members[w]:
            k = self._step_of(j, step)
            if j == i:
                continue
            if k < 0 or (k == 0 and not value[j, RUNWAY]):
                waiting.append(self.keys[j])
            elif k <= self.in_scene_to[j]:
                commanded.append(int(j))
        key = (w, step, tuple(waiting))
        if key not in self._others:
            self._others = {k: v for k, v in self._others.items() if k[1] == step}
            self._others[key] = others_at(window.view(self.keys[i], 0.0, tuple(waiting) + window.others),
                                          t_s, self.words)
        replayed = self._others[key]
        if not commanded:
            return replayed
        state = self._aircraft_now(step)
        levels = [self.words.speed_mps(c) for c in range(self.words.speed_unspecified)]
        traffic, speeds, targets, cleared = [], [], [], []
        geometry, separation = window.airport.flights.geometry, window.airport.flights.separation
        for j in commanded:
            now, force = state[j], value[j]
            candidate = geometry.candidates[int(force[RUNWAY]) - 1]
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
            word = int(force[SPEED]) - 1
            speeds.append(speed)
            targets.append(speed if word < 0 else levels[word] if word < self.words.speed_unspecified
                           else float(self.approach_mps[j]))
            cleared.append(int(force[APPROACH]) - 1 == APPROACH_CLEARED)
        return OthersAt(join(replayed.traffic, *traffic), np.concatenate((replayed.speeds, speeds)),
                        np.concatenate((replayed.targets, targets)), np.concatenate((replayed.cleared, cleared)))

    # -- the end
    def results(self) -> list[Commanded]:
        """What became of each commanded aircraft (`Commanded`) once the loop has run (each window's `Run` holds its
        scene time: its judged steps' span)."""
        out = []
        for i, key in enumerate(self.keys):
            spoken = self.executors[self.group[i]][3]
            grid = spoken.sentences()[self.place[i]]
            said = grid[: int(min(len(grid), self.sentence_end[i], self.steps_flown[i]))]
            w = int(self.window_of[i])
            end = self.runs[w].ended.get(key)
            first_s = self.states[i][0].t_s
            counted = len(said) if end is None else min(len(said), int(round((end["t_s"] - first_s) / self.step_s)))
            out.append(Commanded(key, w, said, LOST_SEPARATION if end is not None else self.own[i], self.own[i], end,
                                 counted, first_s, None if math.isnan(self.landing_s[i]) else float(self.landing_s[i]),
                                 int(self.speaker.value[i, RUNWAY]) - 1, int(self.group[i]), int(self.place[i]),
                                 None if self.go_around_step[i] < 0 else int(self.go_around_step[i]),
                                 None if self.forced_step[i] < 0 else int(self.forced_step[i])))
        return out

    def records(self) -> list[WindowRecord]:
        """What the speaker read of each commanded aircraft (`WindowRecord`), once the loop has run."""
        speaker, out = self.speaker, []
        for i, key in enumerate(self.keys):
            rows = int(speaker.rows[i])
            # its executor's steps (its group's: they may fly on past its own last, "unchanged" said for it there); its
            # last row is either a step it spoke at and ended in (an own end inside the step) or the state its last step
            # reached, where it said nothing — one its executor may not have flown
            said = self.executors[self.group[i]][3].sentences()[self.place[i]]
            steps = rows - N_LOOK
            if len(said) < steps - 1:
                raise ValueError(f"{key}: {rows} rows read, {len(said)} steps said")
            grid = np.full((steps, 6), UNCHANGED, dtype=said.dtype)
            grid[: min(steps, len(said))] = said[:steps]
            out.append(WindowRecord(key, int(self.window_of[i]), speaker.e[i, :rows].copy(), speaker.n[i, :rows].copy(),
                                    speaker.h[i, :rows].copy(), grid, list(self.context_changes[i])))
        return out

    def close(self) -> None:
        """Let the speaker go (`traffic_speaking.SceneLoop.close`: the loop and the speaker hold each other)."""
        self.speaker.edges_of = self.speaker.masks_of = None
        self.speaker = None


@dataclass(frozen=True)
class WindowRecord:
    """What the speaker read of one commanded aircraft (`WindowLoop.records`), once it has flown: its key and window, its
    positions over every row it was in the scene (``e``, ``n``, ``h``; observed to its first predicted step, then
    flown), the words it said at each of its rows from its first predicted one (``grid`` ``[rows − N_LOOK, 6]``,
    `UNCHANGED` where a column says nothing — silent steps too: the other aircraft read its words in force at every row;
    its last row is a step it spoke at when its flight ended inside that step, else the state its last step reached,
    silent) and its landing context's switches (the first row reading each, the landing added, its runway:
    `WindowSpeaker.set_context`)."""

    key: str
    window: int
    e: np.ndarray
    n: np.ndarray
    h: np.ndarray
    grid: np.ndarray
    context_changes: list[tuple[int, float, str]]


@dataclass(frozen=True)
class WindowPlaces:
    """Windows placed on one batch's steps (`window_places`): the pre-roll, each window's time at step 0, each commanded
    aircraft's step of its row 0 (window after window, each window's in its order) and each window's replayed
    aircraft as the speaker reads them (`Node.first_step`: the step of its row 0)."""

    pre: int
    origin: np.ndarray
    start: np.ndarray
    nodes: list[list[Node]]


def window_places(windows: Sequence[Window], step_s: float) -> WindowPlaces:
    """``windows`` on one batch's steps — the loop's and a trainer's (`WindowPlaces`): each window's first commanded
    aircraft's row 0 at the pre-roll's end, the pre-roll the longest any replayed aircraft is in the air before it, at
    most `HISTORY_S`."""
    origin = np.array([window.first_step_s(window.commanded[0]) for window in windows])
    window_of = np.array([w for w, window in enumerate(windows) for _ in window.commanded], dtype=np.int64)
    firsts = np.array([window.first_step_s(k) for window in windows for k in window.commanded])
    entered = [[window.rows(k).presence.start_s for k in window.others] for window in windows]
    before = max([0] + [int(round((o - s) / step_s)) for o, starts in zip(origin, entered) for s in starts])
    pre = min(int(HISTORY_S // step_s), before)
    origin = origin - pre * step_s
    start = np.rint((firsts - origin[window_of]) / step_s).astype(np.int64)
    nodes = [[Node(k, window.rows(k).presence.speaking, int(round((s - origin[w]) / step_s)), **window.rows(k).node_rows)
              for k, s in zip(window.others, entered[w])] for w, window in enumerate(windows)]
    return WindowPlaces(pre, origin, start, nodes)


def window_edges(windows: Sequence[Window], keys: Sequence[str], window_of: np.ndarray, alone: bool, e: np.ndarray,
                 n: np.ndarray, h: np.ndarray, pointers: np.ndarray, rows: np.ndarray, start: np.ndarray, first: int,
                 last: int, step_s: float, aircraft: int, width: int) -> np.ndarray:
    """``[S, last − first, aircraft, aircraft, width]``: every speaking scene's edge features on batch steps ``first … last
    − 1``, read from two steps before (as `SceneLoop._edges`) — the loop's and a trainer's: each commanded aircraft
    ``i`` (``keys``, of window ``window_of[i]``; ``alone``: each its own scene) from its row 0 at step ``start[i]`` over
    its first ``rows[i]`` rows at ``e``, ``n``, ``h`` with the runway class in force before each (``pointers``), placed
    by `traffic_speaking.edge_rows`, the replayed ones by the same call over the scene's first commanded aircraft — the
    one placement of the edge code's source hash — stacked in the speaker's places."""
    low = max(first - 2, 0)
    scenes = np.arange(len(keys)) if alone else np.asarray(window_of)
    count = int(scenes.max()) + 1
    out = np.zeros((count, last - first, aircraft, aircraft, width), dtype=np.float32)
    by_airport: dict[str, list[tuple[int, SceneRows]]] = defaultdict(list)
    for b in range(count):
        members = np.flatnonzero(scenes == b)
        parts = []
        for m, i in enumerate(members):
            window = windows[window_of[i]]
            others = window.others if m == 0 and not alone else ()
            scene = window.view(keys[i], window.first_step_s(keys[i]), others)
            parts.append(edge_rows(scene, e[i], n[i], h[i], pointers[i], int(rows[i]), int(start[i]), low, last,
                                   step_s))
        head = parts[0]                                   # the first commanded one, then its replayed ones
        stacked = parts if len(parts) == 1 else [_aircraft(head, slice(0, 1)), *parts[1:]] + (
            [_aircraft(head, slice(1, None))] if len(head.category) > 1 else [])
        by_airport[windows[window_of[members[0]]].airport.flights.code].append((b, _stacked(stacked)))
    for code, blocks in by_airport.items():
        together = _stacked([block for _, block in blocks])
        separation = next(w.airport.flights.separation for w in windows if w.airport.flights.code == code)
        sizes = [len(block.category) for _, block in blocks]
        for (b, _), size, block in zip(blocks, sizes, scene_edge_blocks(together, separation, sizes)):
            out[b, :, :size, :size] = block[first - low:]
    return out


def window_landings(window: Window, key: str, landings: Mapping[str, Landings], step_s: float) -> Landings:
    """Commanded aircraft ``key``'s landings of its window before any lands in the loop (`WindowLoop`'s landing
    context): the airport's, less every OTHER commanded aircraft's recorded landing — its own stays, for
    `data.own_context` to take out."""
    out = scene_landings(landings[window.airport.flights.code], window.scene(key))
    for other in window.commanded:
        if other != key:
            seen = window.rows(other).presence
            out = out.without(seen.landing_s, seen.runway)
    return out


def _with_landing(landings: Landings, time_s: float, runway: str) -> Landings:
    """``landings`` and one more (a commanded aircraft's landing in the loop)."""
    return Landings(np.sort(np.append(landings.times_s, time_s)),
                    {**landings.by_runway, runway: np.sort(np.append(landings.by_runway[runway], time_s))})


def _aircraft(rows: SceneRows, part: slice) -> SceneRows:
    """``rows``' aircraft at ``part``."""
    return SceneRows(rows.time_s[part], rows.e_m[part], rows.n_m[part], rows.height_m[part], rows.along_m[part],
                     list(rows.runway)[part], list(rows.category)[part])


def _stacked(parts: Sequence[SceneRows]) -> SceneRows:
    """Scenes' (or aircraft's) rows on one aircraft axis, in order."""
    return SceneRows(*(np.concatenate([getattr(p, name) for p in parts]) for name in
                       ("time_s", "e_m", "n_m", "height_m", "along_m")),
                     [row for p in parts for row in p.runway], [c for p in parts for c in p.category])
