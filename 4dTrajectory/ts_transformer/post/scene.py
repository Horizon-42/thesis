"""Windows of recorded traffic and their scenes (post-training §3, §8 C1; D29, D93, C32).

**A recorded flight** (`Recorded`) is a flight's stored signals (vocabulary §6 item 3, D97) as a scene replays it: its
positions in the airport frame and its MSL heights on the 2 s UTC rows, from the entry of the arrival slice (row 0) to
the last row before the observed threshold crossing. Nothing else of the signals is kept: their track, ground speed and
vertical rate are fits that use later rows, so a recorded aircraft's motion is its 2 s displacement
(`prior.inputs.motion`, prior §7 item 2).
Its R is the runway of its record, in force only at the steps after its own first predicted step (D23; the commanded
aircraft's runway word, said at its first predicted step, is in force from its next row: the same rule). Its G (the
go-around state, D92) comes from the labeller's reading of its record where it has a sentence: true at the rows after
each labelled go-around row up to the row where the runway is said again, as a word said at a row is in force from the
next (the user's decision, 2026-10-05); a flight without a sentence has G false. The labelled rows are a withheld
field (vocabulary §6 item 3): only the separation judge and the masks read G, never a model input. A flight's
first predicted step is `OBSERVATION_S` after its first row on the Δ grid (vocabulary §6 item 7; the sentence's first
row, `labeller.interval.first_interval_row`).

**A scene** (`Scene`) is every flight of one airport and one split, by their signals; a test day is never read (the
artefact refuses one, C32), nor a flight of another split. A stated limit: an aircraft outside the arrival slice
(before its entry, a departure, an overflight) is not in the data, and near the cut between two operating days a
scene can lack the aircraft of the next day when that day is of another split (`near_day_cut`).

**A window** (`Window`) is one commanded flight with a closed-loop sentence at the chosen Δ (every such flight, inside
the base's selection or not, D76) and its scene: at each step from the commanded aircraft's row 0, every other flight
of the airport and the split in the air then (D93: the window has no length of its own; it ends where the loop ends
it). Augmented windows: A, one recorded flight of the same airport and split from another time inserted with its
record shifted in time; D, the aircraft next ahead on the approach clock at the first predicted step with its record
shifted in time (§2 item 4). A shift is a whole number of Δ, so every row stays on the UTC grid. B, the commanded
aircraft's start moved (C9): its observed rows up to the first predicted step turned about the airport reference, raised
and their speed scaled, by the start of a closed loop (vocabulary §6 item 5, D97 (4)); the window holds the move as
three numbers (`StartMove`), and the loop's caller gives them to the start (`autopilot.start.Move`).

What is drawn from the record — the commanded aircraft's landed runway and landing time, the order on the approach
clock at the first predicted step — chooses windows and fills the census. It is never an input of the prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from ts_transformer.data.day_split import DaySplit, landing_day, operating_day_span_s, parse_utc
from ts_transformer.inference.runway_schedule import Separation, wake_category
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import closed_loop_indices, load_sentences, load_signals
from ts_transformer.instructions.labeller.interval import OBSERVATION_S, first_interval_row, interval_rows
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.post.runways import approach_clock_m

#: The kinds of window (§2 item 4): real, an inserted aircraft, the aircraft ahead moved. B comes with C9.
REAL, INSERTED, LEADER_MOVED, MOVED_START = "real", "A", "D", "B"
WINDOW_KINDS = (REAL, INSERTED, LEADER_MOVED, MOVED_START)
#: The key of an inserted flight: its own key with this suffix, so that it never stands for the flight it came from.
INSERTED_SUFFIX = "+inserted"


def utc_s(value: str) -> float:
    return parse_utc(value).timestamp()


@dataclass(frozen=True)
class Recorded:
    """One flight of a scene (module docstring): positions only; ``start_s`` is the UTC epoch of row 0, ``first_step_s``
    of its first predicted step, ``landing_s`` its recorded landing (for the drawing and the census, never an input)."""

    key: str
    airport: str
    runway_index: int
    category: str | None            # its CWT category; None: the record has no type
    start_s: float
    step_s: float
    first_step_s: float
    landing_s: float
    e_m: np.ndarray
    n_m: np.ndarray
    height_m: np.ndarray
    go_around: np.ndarray           # [rows] bool: G at each row (module docstring)

    @property
    def end_s(self) -> float:
        """The UTC epoch of its last row (the last before the observed threshold crossing)."""
        return self.start_s + (len(self.e_m) - 1) * self.step_s

    def row_at(self, time_s: float) -> int:
        """The row at ``time_s``, refused unless that time is one of its rows."""
        row = (time_s - self.start_s) / self.step_s
        if abs(row - round(row)) > 1e-6 or not 0 <= round(row) < len(self.e_m):
            raise ValueError(f"{self.key} has no row at {time_s:.1f}")
        return int(round(row))

    def at_step(self, time_s: float, interval_s: float) -> tuple[str, tuple, tuple, bool, int, str | None, bool, bool]:
        """Its fields of `AircraftAt` at ``time_s`` (a step of a scene at Δ = ``interval_s``): its R only after its own
        first predicted step, its G (module docstring)."""
        row = self.row_at(time_s)
        before = (self.e_m[row - 1], self.n_m[row - 1], self.height_m[row - 1]) if row else (0.0, 0.0, 0.0)
        return (self.key, (self.e_m[row], self.n_m[row], self.height_m[row]), before, row > 0,
                self.runway_index if time_s > self.first_step_s else -1, self.category,
                time_s + interval_s > self.end_s, bool(self.go_around[row]))

    def shifted(self, shift_s: float, interval_s: float, key: str | None = None) -> Recorded:
        """The same record ``shift_s`` later (a whole number of Δ = ``interval_s``, so that its rows stay on the grid)."""
        if abs(shift_s / interval_s - round(shift_s / interval_s)) > 1e-9:
            raise ValueError(f"a shift of {shift_s:g} s is not a whole number of {interval_s:g} s rows")
        return replace(self, key=key or self.key, start_s=self.start_s + shift_s,
                       first_step_s=self.first_step_s + shift_s, landing_s=self.landing_s + shift_s)


def go_around_rows(rows: int, go_around: Sequence[int], runway_again: Sequence[int]) -> np.ndarray:
    """``[rows]`` bool: G at each row from a labelled sentence's go-around rows and the rows where the runway is said
    again (module docstring): true after each go-around row, up to and including its runway's row."""
    if len(go_around) != len(runway_again) or any(not 0 <= g < a < rows for g, a in zip(go_around, runway_again)):
        raise ValueError(f"each go-around is ended by its runway word, at a later row of the {rows} rows")
    out = np.zeros(rows, dtype=bool)
    for g, a in zip(go_around, runway_again):
        out[g + 1:a + 1] = True
    return out


def recorded(signals: FlightSignals, geometry: AirportGeometry, interval_s: float, step_s: float, go_around: np.ndarray,
             category_of: Callable[[str], str] = wake_category) -> Recorded:
    """A flight's stored signals as a scene replays it (module docstring): ``go_around`` its G at each row
    (`go_around_rows`; every row false for a flight without a sentence), its type's CWT category by ``category_of``."""
    if np.shape(go_around) != np.shape(signals.time_s):
        raise ValueError(f"{signals.dataset_id}: G is given for each of its rows")
    if signals.airport != geometry.code:
        raise ValueError(f"{signals.dataset_id} lands at {signals.airport}, not {geometry.code}")
    times = np.asarray(signals.time_s, dtype=np.float64)
    if times[0] != 0.0 or not np.allclose(np.diff(times), step_s, rtol=0.0, atol=1e-9):
        raise ValueError(f"{signals.dataset_id}: its rows are not {step_s:g} s apart from row 0")
    start = utc_s(signals.entry_time_utc)
    first = first_interval_row(signals.entry_time_utc, interval_s, step_s)
    return Recorded(key=signals.dataset_id, airport=signals.airport,
                    runway_index=geometry.candidate_index(signals.runway),
                    category=None if signals.typecode is None else category_of(signals.typecode),
                    start_s=start, step_s=step_s, first_step_s=start + first * step_s + OBSERVATION_S,
                    landing_s=utc_s(signals.landing_time_utc),
                    e_m=np.asarray(signals.e_m, dtype=np.float64), n_m=np.asarray(signals.n_m, dtype=np.float64),
                    height_m=np.asarray(signals.altitude_m, dtype=np.float64),
                    go_around=np.asarray(go_around, dtype=bool))


@dataclass(frozen=True)
class StartMove:
    """Window B's move of the commanded aircraft's start (C9; vocabulary D97 (4), `autopilot.start.Move`, which
    `post/` does not import): a turn about the airport reference (compass degrees, clockwise), a change of height (m)
    and a scale of the speed (the displacements before the first predicted step stretched about it)."""

    turn_deg: float = 0.0
    height_m: float = 0.0
    speed_scale: float = 1.0


NO_START_MOVE = StartMove()


@dataclass(frozen=True)
class AircraftAt:
    """Aircraft of a window at one step: their states at the step and at the 2 s row before (``[N, 3]``: e, n, MSL
    height; the row before 0 where it does not exist), whether that row exists (their motion is known, `prior.inputs.motion`),
    their R (candidate index; −1 where not in force, D23), their CWT categories, whether the step is their last in the
    air (the last step before their threshold crossing) and their G (D92: the commanded aircraft's from its words in
    force, a recorded aircraft's from its labelled sentence)."""

    keys: tuple[str, ...]
    at: np.ndarray
    before: np.ndarray
    known: np.ndarray
    runway_index: np.ndarray
    category: tuple[str | None, ...]
    last_step: np.ndarray
    go_around: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.keys)
        if (self.at.shape != (count, 3) or self.before.shape != (count, 3)
                or not all(len(v) == count for v in (self.known, self.runway_index, self.category, self.last_step,
                                                     self.go_around))):
            raise ValueError("every field of an AircraftAt holds one value for each aircraft")
        if not all(np.asarray(v).dtype == bool for v in (self.known, self.last_step, self.go_around)):
            raise ValueError("known, last_step and go_around of an AircraftAt are bool arrays")

    def __len__(self) -> int:
        return len(self.keys)

    @classmethod
    def of(cls, items: Sequence[tuple[str, tuple, tuple, bool, int, str | None, bool, bool]]) -> AircraftAt:
        """From ``(key, at, before, known, runway_index, category, last_step, go_around)`` for each aircraft."""
        return cls(keys=tuple(i[0] for i in items), at=np.array([i[1] for i in items], dtype=np.float64).reshape(-1, 3),
                   before=np.array([i[2] for i in items], dtype=np.float64).reshape(-1, 3),
                   known=np.array([i[3] for i in items], dtype=bool),
                   runway_index=np.array([i[4] for i in items], dtype=np.int64),
                   category=tuple(i[5] for i in items), last_step=np.array([i[6] for i in items], dtype=bool),
                   go_around=np.array([i[7] for i in items], dtype=bool))


class Scene:
    """Every flight of one airport and one split (module docstring), and who is in the air at a time."""

    def __init__(self, geometry: AirportGeometry, split: str, flights: Sequence[Recorded], interval_s: float) -> None:
        if any(f.airport != geometry.code for f in flights):
            raise ValueError(f"a scene of {geometry.code} holds a flight of another airport")
        if len({f.key for f in flights}) != len(flights):
            raise ValueError("a flight is twice in one scene")
        self.geometry, self.split, self.interval_s = geometry, split, interval_s
        self.flights = tuple(flights)
        self.index = {f.key: k for k, f in enumerate(self.flights)}
        self.start_s = np.array([f.start_s for f in self.flights])
        self.end_s = np.array([f.end_s for f in self.flights])
        self.landing_s = np.array([f.landing_s for f in self.flights])

    def with_flights(self, flights: Sequence[Recorded]) -> Scene:
        return Scene(self.geometry, self.split, flights, self.interval_s)

    def flight(self, key: str) -> Recorded:
        """The flight ``key`` of the scene."""
        return self.flights[self.index[key]]

    def in_air(self, time_s: float) -> list[Recorded]:
        """The flights in the air at ``time_s`` (from their first row to their last), in the scene's order."""
        return [self.flights[k] for k in np.flatnonzero((self.start_s <= time_s) & (time_s <= self.end_s))]

    def others_at(self, time_s: float, without: str) -> AircraftAt:
        """The flights in the air at ``time_s`` but ``without`` (the commanded flight's key)."""
        return AircraftAt.of([f.at_step(time_s, self.interval_s) for f in self.in_air(time_s) if f.key != without])

    def between(self, start_s: float, end_s: float) -> list[Recorded]:
        """The flights in the air at some time from ``start_s`` to ``end_s``, in the scene's order."""
        return [self.flights[k] for k in np.flatnonzero((self.start_s <= end_s) & (self.end_s >= start_s))]


class MovedScene:
    """A scene of an augmented window: its airport's scene (``base``, shared, never copied) with flights replaced by
    moved records (by key) and flights added (an inserted one). The moved and added flights come after the base's in
    the order of the aircraft."""

    def __init__(self, base: Scene, *, replaced: Mapping[str, Recorded] = MappingProxyType({}),
                 added: Sequence[Recorded] = ()) -> None:
        if (any(key not in base.index for key in replaced) or any(f.key in base.index for f in added)
                or len({f.key for f in added}) != len(added)):
            raise ValueError("a moved flight is one of the scene's, an added one is not")
        self.base, self.replaced, self.added = base, dict(replaced), tuple(added)
        self.geometry, self.split, self.interval_s = base.geometry, base.split, base.interval_s

    @property
    def moved_flights(self) -> tuple[Recorded, ...]:
        return (*self.replaced.values(), *self.added)

    def flight(self, key: str) -> Recorded:
        """The flight ``key`` as this scene replays it."""
        if key in self.replaced:
            return self.replaced[key]
        added = {f.key: f for f in self.added}
        return added[key] if key in added else self.base.flights[self.base.index[key]]

    def in_air(self, time_s: float) -> list[Recorded]:
        return ([f for f in self.base.in_air(time_s) if f.key not in self.replaced]
                + [f for f in self.moved_flights if f.start_s <= time_s <= f.end_s])

    def others_at(self, time_s: float, without: str) -> AircraftAt:
        return AircraftAt.of([f.at_step(time_s, self.interval_s) for f in self.in_air(time_s) if f.key != without])

    def between(self, start_s: float, end_s: float) -> list[Recorded]:
        return ([f for f in self.base.between(start_s, end_s) if f.key not in self.replaced]
                + [f for f in self.moved_flights if f.start_s <= end_s and f.end_s >= start_s])


@dataclass(frozen=True)
class Window:
    """One window (module docstring): its kind, the commanded flight (its record, from which the start of a closed loop
    rebuilds it, without its labelled G: its G is its words' in the loop), its scene of the other aircraft and, for an
    augmented window, the flights moved (key, shift s)."""

    kind: str
    commanded: Recorded
    signal_index: int               # the commanded flight's place in the split's signals (its sentence's index)
    scene: Scene | MovedScene
    moved: tuple[tuple[str, float], ...] = field(default=())
    #: window B's move of the commanded aircraft's start (`StartMove`; every other window moves nothing)
    start_move: StartMove = field(default_factory=lambda: NO_START_MOVE)

    def __post_init__(self) -> None:
        if self.kind not in WINDOW_KINDS:
            raise ValueError(f"a window is one of {WINDOW_KINDS}, not {self.kind!r}")
        if (self.kind == MOVED_START) != (self.start_move != NO_START_MOVE):
            raise ValueError("window B, and only it, moves its commanded aircraft's start")
        if self.commanded.go_around.any():
            raise ValueError(f"{self.commanded.key}: the commanded aircraft's G comes from its words in force, never from "
                             "its labelled sentence (a withheld field)")

    @property
    def row0_s(self) -> float:
        """The UTC epoch of the commanded aircraft's row 0 on the Δ grid."""
        return self.commanded.first_step_s - OBSERVATION_S

    @property
    def first_step_s(self) -> float:
        return self.commanded.first_step_s

    def step_s(self, step: int) -> float:
        """The UTC epoch of the window's step ``step`` (step 0 is the commanded aircraft's row 0): a multiple of Δ."""
        return self.row0_s + step * self.scene.interval_s

    def others_at(self, step: int) -> AircraftAt:
        return self.scene.others_at(self.step_s(step), self.commanded.key)


def airport_scenes(directory: Path, split: str, spec: VocabularySpec, interval_s: float,
                   geometries: Mapping[str, AirportGeometry], category_of: Callable[[str], str] = wake_category
                   ) -> tuple[dict[str, Scene], list[FlightSignals]]:
    """Each airport's scene of ``split`` (its signals, the artefact's checks: C32; each flight's G from its labelled
    sentence, module docstring), and the split's signals in order."""
    signals = load_signals(directory, split)
    return scenes_of(signals, directory, split, spec, interval_s, geometries, category_of), signals


def scenes_of(signals: Sequence[FlightSignals], directory: Path, split: str, spec: VocabularySpec, interval_s: float,
              geometries: Mapping[str, AirportGeometry], category_of: Callable[[str], str] = wake_category
              ) -> dict[str, Scene]:
    """`airport_scenes` of the split's signals ``signals`` as the artefact's reader gives them (`load_signals`), read
    once by a caller that holds them already (post-training C13: the split's opened start)."""
    interval_rows(interval_s, spec.step_s)
    labelled = load_sentences(directory, split, spec, ("signal_index", "go_around_offsets", "go_around_row",
                                                       "runway_again_row"))
    offsets = labelled["go_around_offsets"]
    go_around = [np.zeros(s.n_rows, dtype=bool) for s in signals]          # no sentence, or no go-around: G false
    for k, index in enumerate(labelled["signal_index"].tolist()):
        rows = slice(int(offsets[k]), int(offsets[k + 1]))
        go_around[index] = go_around_rows(signals[index].n_rows, labelled["go_around_row"][rows].tolist(),
                                          labelled["runway_again_row"][rows].tolist())
    flights: dict[str, list[Recorded]] = {code: [] for code in geometries}
    for s, g in zip(signals, go_around):
        flights[s.airport].append(recorded(s, geometries[s.airport], interval_s, spec.step_s, g, category_of))
    return {code: Scene(geometries[code], split, items, interval_s) for code, items in flights.items()}


def real_windows(directory: Path, split: str, spec: VocabularySpec, interval_s: float, scenes: Mapping[str, Scene],
                 signals: Sequence[FlightSignals]) -> list[Window]:
    """Every flight of ``split`` with a closed-loop sentence at Δ = ``interval_s`` as the commanded flight of a real
    window, in signal order (the sentences' index only: no sentence is read)."""
    windows = []
    for index in sorted(closed_loop_indices(directory, split, interval_s, spec)):
        scene = scenes[signals[index].airport]
        own = scene.flights[scene.index[signals[index].dataset_id]]
        windows.append(Window(REAL, replace(own, go_around=np.zeros_like(own.go_around)), index, scene))
    return windows


def next_ahead(window: Window, separation: Separation, time_s: float | None = None) -> str | None:
    """The key of the aircraft next ahead of the commanded one on the approach clock at ``time_s`` (its first predicted
    step when None), on its recorded runway or a runway separated as one (`Separation.one_runway`), each on its record;
    None without one."""
    time_s = window.first_step_s if time_s is None else time_s
    geometry = window.scene.geometry
    own = window.commanded
    row = own.row_at(time_s)
    mine = float(approach_clock_m(separation, geometry, own.runway_index, own.e_m[row], own.n_m[row]))
    ident = geometry.candidates[own.runway_index].ident
    best, best_along = None, math.inf
    for other in window.scene.in_air(time_s):
        if other.key == own.key or not separation.one_runway(ident, geometry.candidates[other.runway_index].ident):
            continue
        r = other.row_at(time_s)
        along = float(approach_clock_m(separation, geometry, other.runway_index, other.e_m[r], other.n_m[r]))
        if mine < along < best_along:
            best, best_along = other.key, along
    return best


def grid_shift_s(rng: np.random.Generator, low_s: float, high_s: float, interval_s: float, *, nonzero: bool) -> float:
    """A shift drawn uniformly from the whole numbers of Δ in [``low_s``, ``high_s``] (0 left out when ``nonzero``)."""
    steps = np.arange(math.ceil(low_s / interval_s), math.floor(high_s / interval_s) + 1)
    if nonzero:
        steps = steps[steps != 0]
    if not len(steps):
        raise ValueError(f"no whole number of {interval_s:g} s rows in [{low_s:g}, {high_s:g}] s")
    return float(rng.choice(steps)) * interval_s


def inserted_window(window: Window, rng: np.random.Generator, *, landing_shift_s: tuple[float, float],
                    apart_s: float) -> Window | None:
    """Window A from ``window``: one flight of its scene whose recorded landing is more than ``apart_s`` from the
    commanded flight's, drawn uniformly, inserted with its record shifted so that its landing falls the drawn
    ``landing_shift_s`` (a whole number of Δ) from the commanded flight's recorded landing. None when no flight is that
    far apart. The ranges are proposals for the user (post-training §8 C1)."""
    if not apart_s > 0.0:
        raise ValueError(f"window A inserts a flight of another time: apart_s {apart_s:g} s must be positive")
    if window.kind != REAL:
        raise ValueError(f"window A is drawn from a real window, not from a window {window.kind}")
    scene, own = window.scene, window.commanded
    pool = np.flatnonzero(np.abs(scene.landing_s - own.landing_s) > apart_s)      # the commanded flight is never in it
    if not len(pool):
        return None
    source = scene.flights[int(pool[int(rng.integers(len(pool)))])]
    offset = grid_shift_s(rng, *landing_shift_s, scene.interval_s, nonzero=False)
    shift = own.landing_s - source.landing_s + offset
    shift = round(shift / scene.interval_s) * scene.interval_s
    moved = source.shifted(shift, scene.interval_s, key=source.key + INSERTED_SUFFIX)
    return replace(window, kind=INSERTED, scene=MovedScene(scene, added=(moved,)), moved=((moved.key, shift),))


def leader_moved_window(window: Window, separation: Separation, rng: np.random.Generator, *,
                        shift_s: tuple[float, float]) -> Window | None:
    """Window D from ``window``: the aircraft next ahead on the approach clock at the first predicted step
    (`next_ahead`) with its record shifted by a whole number of Δ drawn in ``shift_s`` (never 0). None without one. The
    range is a proposal for the user (post-training §8 C1)."""
    if window.kind != REAL:
        raise ValueError(f"window D is drawn from a real window, not from a window {window.kind}")
    leader = next_ahead(window, separation)
    if leader is None:
        return None
    scene = window.scene
    shift = grid_shift_s(rng, *shift_s, scene.interval_s, nonzero=True)
    moved = scene.flights[scene.index[leader]].shifted(shift, scene.interval_s)
    return replace(window, kind=LEADER_MOVED, scene=MovedScene(scene, replaced={leader: moved}),
                   moved=((leader, shift),))


def near_day_cut(window: Window, days: DaySplit, longest_s: float) -> bool:
    """Whether ``window``'s scene can lack aircraft at a cut between two operating days (module docstring), on either
    side of the commanded flight's own day, where the day beside it is not of its split: at the end of its day, when
    its recorded landing is less than ``longest_s`` (the longest record of the scene) before the cut, so that a flight
    of the next day can be in the air during the window; at the start of its day, when its row 0 is before the cut, so
    that the flights of the day before that land after its row 0 are in the air and missing. (A loop that flies longer
    than the record can reach the end cut too: a stated limit.)"""
    start, end = operating_day_span_s(landing_day_s(window.commanded.landing_s))

    def other_split(day: str) -> bool:
        return day not in days.listed or days.split_of(day) != window.scene.split

    return ((other_split(landing_day_s(end)) and window.commanded.landing_s >= end - longest_s)
            or (other_split(landing_day_s(start - 1.0)) and window.row0_s < start))


def landing_day_s(time_s: float) -> str:
    """The operating day of a UTC epoch (`data.day_split.landing_day`)."""
    return landing_day(datetime.fromtimestamp(time_s, tz=timezone.utc).isoformat())


def census(windows: Sequence[Window], separation: Mapping[str, Separation], days: DaySplit) -> dict:
    """The census of C1 for each airport of ``windows`` (one split): the windows, the other aircraft in the air at the
    first predicted step (their count's mean, quantiles and the share with none), the share with a leader in the air on
    the same runway or a runway separated as one (`next_ahead`: the windows that admit D), and the windows near the
    cut between two operating days (`near_day_cut`)."""
    out: dict[str, dict] = {}
    by_airport: dict[str, list[Window]] = {}
    for window in windows:
        by_airport.setdefault(window.scene.geometry.code, []).append(window)
    for code, items in sorted(by_airport.items()):
        scene = items[0].scene
        if any(w.kind != REAL for w in items):
            raise ValueError("a census counts real windows")
        longest = float(np.max(scene.end_s - scene.start_s))
        counts = np.array([len(w.others_at(int(round(OBSERVATION_S / scene.interval_s)))) for w in items])
        leaders = sum(next_ahead(w, separation[code]) is not None for w in items)
        near = sum(near_day_cut(w, days, longest) for w in items)
        out[code] = {
            "windows": len(items),
            "others_at_first_step": {"mean": float(counts.mean()), "p10": float(np.percentile(counts, 10)),
                                     "p50": float(np.percentile(counts, 50)), "p90": float(np.percentile(counts, 90)),
                                     "max": int(counts.max()), "share_none": float((counts == 0).mean())},
            "share_with_leader_in_air": leaders / len(items),
            "windows_admitting_D": leaders,
            "near_day_cut": near,
            "longest_record_s": longest,
        }
    return out


def moved_start_window(window: Window, rng: np.random.Generator, *, turn_deg: float, height_m: float,
                       speed_scale: float) -> Window:
    """Window B from the real ``window``: its commanded aircraft's start moved by a turn drawn uniformly in
    [−``turn_deg``, ``turn_deg``], a height in [−``height_m``, ``height_m``] and a speed scale in
    [1 − ``speed_scale``, 1 + ``speed_scale``]. The ranges are proposals for the user (post-training §8 C9). Whether
    it opens inside a loss of separation is read where its loop is started (the moved start is the start's)."""
    if window.kind != REAL:
        raise ValueError(f"window B is drawn from a real window, not from a window {window.kind}")
    if not (turn_deg >= 0.0 and height_m >= 0.0 and 0.0 <= speed_scale < 1.0):
        raise ValueError("the ranges of a moved start are non-negative, the speed's less than 1")
    move = StartMove(float(rng.uniform(-turn_deg, turn_deg)), float(rng.uniform(-height_m, height_m)),
                     float(rng.uniform(1.0 - speed_scale, 1.0 + speed_scale)))
    return replace(window, kind=MOVED_START, start_move=move)
