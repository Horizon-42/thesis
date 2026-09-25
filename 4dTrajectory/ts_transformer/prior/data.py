"""The sentence artefact as the prior's training data (prior design §3–§5): one flight at a time, a scene of one
aircraft (design §9 step 1).

A flight's row ``t`` (2 s) gives the model only what a controller could know before speaking at ``t`` (design §2):

- the state at row ``t`` (`STEP_FEATURES`, fixed SI scales, §4.1): airport-frame E, N, MSL height, time from row 0,
  and the motion from row ``t − 1`` to row ``t`` — ground speed, direction of motion (sin, cos), vertical rate —
  zero with a flag at row 0, which has no row before it. Never the signals' fitted track, ground speed or vertical
  rate: those are least-squares fits over a window centred on the row, 7.5 s of the future (the second version's V0
  measured what that future is worth, readouts §2);
- for every candidate runway of the airport (`relative_features`, §4.2): where the aircraft is relative to it
  (`instructions.airport.relative_to_runway`, the labeller's own function: before the threshold along the course,
  right of the centreline — both `asinh` of kilometres — height above the threshold), the past direction of motion
  − the course (sin, cos; zero with a flag at row 0) and, in the variants that have it, the airport's landings on
  it before ``t`` (`CONTEXT_FEATURES`: how many in the `scene.CONTEXT_WINDOW_S` before ``t``, how long since the
  last one, a flag when there is none) — from the tracks roster's landings less the sealed test days
  (`scene.context_landings`). Every candidate is measured, never only the landed one, so nothing gives the answer
  away;
- the words said so far (§4.3): the rows before `scene.N_LOOK` are only observed — the prior says nothing there and
  the labeller's words there are neither input nor target; at the first predicted step (row ``N_LOOK``) nothing has
  been said (``in_force`` 0 = none yet); from row ``N_LOOK + 1`` each column's word in force at the row before
  (``in_force``: the word + 1) and the steps since the prior said it (``since``, `log1p` / `SINCE_SCALE`, counted
  from the first predicted step at the earliest — that is when the prior said it);
- the aircraft's static attributes (§4.5): `STATIC_FEATURES`, none in this version;
- the context: the airport, and its candidate runways' geometry (`CANDIDATE_FEATURES`, one table per airport).

and asks for the step's six words (``targets``: 0 = unchanged, else the word + 1) from row ``N_LOOK`` on: at the
first predicted step every column's word in force there (none may be "unchanged": the model masks it), after it the
sentence's words as written. ``asked`` says which of them the loss counts: every column from row ``N_LOOK`` on for a
labelled flight; a closed-loop sentence (`chain_record`) is given its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from ts_transformer.data.runway_context import ENTRY_SECTORS
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import load_candidates, load_day_split, load_sentences, load_signals
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.prior.scene import CONTEXT_WINDOW_S, N_LOOK, Landings, context_landings, utc_s

POSITION_SCALE_M = 10_000.0
HEIGHT_SCALE_M = 1_000.0
SPEED_SCALE_MPS = 100.0
VERTICAL_RATE_SCALE_MPS = 10.0
TIME_SCALE_S = 600.0
#: The relative horizontal distances go in as asinh(d / RELATIVE_SCALE_M): linear within a few hundred metres of a
#: centreline (the corridor is 20 m + d·tan 0.45°), logarithmic out to the 25 km of the other candidates.
RELATIVE_SCALE_M = 1_000.0
LENGTH_SCALE_M = 1_000.0
SINCE_SCALE = 5.0
#: The time since a runway's last landing goes in as log1p(minutes) / LANDING_SINCE_SCALE (≈ 1 at two and a half hours).
LANDING_SINCE_SCALE = 5.0

STEP_FEATURES = ("e", "n", "height", "time", "ground_speed", "motion_sin", "motion_cos", "vertical_rate", "first_row")
RELATIVE_FEATURES = ("before_threshold", "right_of_centreline", "height_above_threshold",
                     "motion_minus_course_sin", "motion_minus_course_cos", "first_row")
CONTEXT_FEATURES = ("landings_in_window", "since_last_landing", "no_landing_before")
#: The per-aircraft static attributes (design §4.5): none in this version. A later quantity (a wake category, a
#: published approach speed) is appended here and nothing else changes.
STATIC_FEATURES: tuple[str, ...] = ()
CANDIDATE_FEATURES = ("threshold_e", "threshold_n", "elevation", "course_sin", "course_cos", "length", "valid")


@dataclass(frozen=True)
class Variant:
    """What the step-1 comparison switches (readouts §4, the rule written before the runs)."""

    landing_context: bool          # the candidates carry the airport's landings (`CONTEXT_FEATURES`)
    ordered_heads: bool            # a step's columns are said in order, each head seeing the earlier ones (§5)

    @property
    def relative_features(self) -> tuple[str, ...]:
        return RELATIVE_FEATURES + (CONTEXT_FEATURES if self.landing_context else ())


VARIANTS = {"full": Variant(True, True), "no-context": Variant(False, True), "unordered": Variant(True, False)}


def column_classes(words: Words, max_candidates: int) -> tuple[int, ...]:
    """Classes per column: "unchanged" + every value the column can take (the runway pointer: a candidate slot)."""
    values = {"runway": max_candidates, "approach": 3, "heading": words.n_heading,
              "altitude": words.altitude_land + 1, "angle": words.n_descent + 2,
              "speed": words.speed_unspecified + 1}
    return tuple(1 + values[name] for name in COLUMNS)


@dataclass(frozen=True)
class Flight:
    dataset_id: str
    airport: int
    features: np.ndarray       # [N, len(STEP_FEATURES)] float32
    relative: np.ndarray       # [N, K, len(variant.relative_features)] float32, K = the airport's candidates
    static: np.ndarray         # [len(STATIC_FEATURES)] float32
    in_force: np.ndarray       # [N, 6] int16
    since: np.ndarray          # [N, 6] float32
    targets: np.ndarray        # [N, 6] int16; the heads read the earlier columns' (teacher forcing)
    asked: np.ndarray          # [N, 6] bool: the targets the loss counts (never a row before N_LOOK)
    #: For the runway rules at the first predicted step (§8): its time (epoch seconds), the entry sector the flight
    #: came in from (its row 0), its past direction of motion (compass degrees).
    first_step_s: float
    sector: int
    course_deg: float
    #: In the final corridor to the end from the first predicted step on (the labeller's capture row ≤ N_LOOK).
    established: bool

    @property
    def rows(self) -> int:
        return len(self.targets)


@dataclass(frozen=True)
class Split:
    flights: list[Flight]
    airports: tuple[str, ...]
    candidates: np.ndarray     # [A, slots, len(CANDIDATE_FEATURES)] float32
    runways: tuple[tuple[str, ...], ...]     # each airport's candidates' idents, in slot order
    courses: tuple[tuple[float, ...], ...]   # and their courses (compass degrees, the geometry's)
    classes: tuple[int, ...]
    variant: str               # a `VARIANTS` name

    def subset(self, indices: Sequence[int]) -> Split:
        return Split([self.flights[i] for i in indices], self.airports, self.candidates, self.runways, self.courses,
                     self.classes, self.variant)


def candidate_table(geometries: Mapping[str, AirportGeometry], airports: Sequence[str], slots: int) -> np.ndarray:
    """Each airport's candidates in its own order (the sentence's pointer), padded to ``slots`` with ``valid`` 0."""
    table = np.zeros((len(airports), slots, len(CANDIDATE_FEATURES)), dtype=np.float32)
    for a, code in enumerate(airports):
        for c, candidate in enumerate(geometries[code].candidates):
            course = math.radians(candidate.course_deg)
            table[a, c] = (candidate.threshold_e_m / POSITION_SCALE_M, candidate.threshold_n_m / POSITION_SCALE_M,
                           candidate.elevation_m / HEIGHT_SCALE_M, math.sin(course), math.cos(course),
                           candidate.length_m / LENGTH_SCALE_M, 1.0)
    return table


def runway_names(geometries: Mapping[str, AirportGeometry], airports: Sequence[str]
                 ) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[float, ...], ...]]:
    """``(idents, courses)`` of each airport's candidates, in slot order."""
    return (tuple(tuple(c.ident for c in geometries[code].candidates) for code in airports),
            tuple(tuple(c.course_deg for c in geometries[code].candidates) for code in airports))


def entry_sector(signals: FlightSignals, geometry: AirportGeometry) -> int:
    """Which of the `ENTRY_SECTORS` bearing sectors, seen from the centroid of the airport's candidate thresholds, the
    flight's row 0 is in (the runway rules' "same entry direction")."""
    e0 = sum(c.threshold_e_m for c in geometry.candidates) / len(geometry.candidates)
    n0 = sum(c.threshold_n_m for c in geometry.candidates) / len(geometry.candidates)
    bearing = math.degrees(math.atan2(signals.e_m[0] - e0, signals.n_m[0] - n0)) % 360.0
    return int(bearing // (360.0 / ENTRY_SECTORS)) % ENTRY_SECTORS


def _motion(e: np.ndarray, north: np.ndarray, height: np.ndarray, time_s: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ground speed, direction of motion (compass degrees), vertical rate)`` of each row from the displacement since
    the row before, along the last axis (rows; a leading axis, flights); the first row given has none (zeros)."""
    dt = np.diff(time_s)
    de, dn = np.diff(e, axis=-1), np.diff(north, axis=-1)
    none = np.zeros(e.shape[:-1] + (1,))
    return (np.concatenate((none, np.hypot(de, dn) / dt), axis=-1),
            np.concatenate((none, np.degrees(np.arctan2(de, dn))), axis=-1),
            np.concatenate((none, np.diff(height, axis=-1) / dt), axis=-1))


def rows_inputs(e: np.ndarray, north: np.ndarray, height: np.ndarray, time_s: np.ndarray, entry_s: np.ndarray,
                first: int, geometry: AirportGeometry, contexts: Sequence[Landings] | None
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(features [F, m − first, step], relative [F, m − first, K, relative])`` of rows ``first … m − 1`` of ``F``
    flights of one airport whose positions so far are ``e``, ``north``, ``height`` (``[F, m]``) at ``time_s`` (``[m]``,
    the same rows for all; ``entry_s`` ``[F]``: each flight's epoch time of row 0). A row reads itself and the row before
    it (its motion), and the landings before its time — so rows can be added one at a time, as the prior speaks
    (`prior.generate`), and come out as the whole flight's; every value is computed element by element, so a flight's
    rows are the same whichever flights it is computed with. ``contexts``: each flight's landing context, its own landing
    already left out (`own_context`); None for a variant without it."""
    low, origin = max(first - 1, 0), time_s[0]
    e, north, height, time_s = e[:, low:], north[:, low:], height[:, low:], time_s[low:]
    speed, motion, climb = _motion(e, north, height, time_s)
    flag = (np.arange(low, low + e.shape[1]) == 0).astype(np.float64)   # row 0 has no row before it
    radians = np.radians(motion)
    features = np.stack([e / POSITION_SCALE_M, north / POSITION_SCALE_M, height / HEIGHT_SCALE_M,
                         np.broadcast_to((time_s - origin) / TIME_SCALE_S, e.shape), speed / SPEED_SCALE_MPS,
                         np.sin(radians) * (1.0 - flag), np.cos(radians) * (1.0 - flag),
                         climb / VERTICAL_RATE_SCALE_MPS, np.broadcast_to(flag, e.shape)], axis=-1)
    clock = entry_s[:, None] + time_s
    width = len(RELATIVE_FEATURES) + (len(CONTEXT_FEATURES) if contexts is not None else 0)
    relative = np.zeros(e.shape + (len(geometry.candidates), width))
    for k, candidate in enumerate(geometry.candidates):
        place = relative_to_runway(e, north, motion, height, candidate)
        off = np.radians(place.track_minus_course_deg)
        parts = [np.arcsinh(place.before_threshold_m / RELATIVE_SCALE_M),
                 np.arcsinh(place.right_of_course_m / RELATIVE_SCALE_M),
                 place.height_above_threshold_m / HEIGHT_SCALE_M,
                 np.sin(off) * (1.0 - flag), np.cos(off) * (1.0 - flag), np.broadcast_to(flag, e.shape)]
        if contexts is not None:
            since = np.stack([context.since_last(clock[f], candidate.ident) for f, context in enumerate(contexts)])
            count = np.stack([context.count_before(clock[f], CONTEXT_WINDOW_S, candidate.ident)
                              for f, context in enumerate(contexts)])
            none = np.isnan(since)
            parts += [np.log1p(count), np.where(none, 0.0, np.log1p(np.nan_to_num(since) / 60.0) / LANDING_SINCE_SCALE),
                      none.astype(np.float64)]
        relative[:, :, k] = np.stack(parts, axis=-1)
    keep = slice(first - low, None)
    return features[:, keep].astype(np.float32), relative[:, keep].astype(np.float32)


def row_inputs(e: np.ndarray, north: np.ndarray, height: np.ndarray, time_s: np.ndarray, entry_s: float, first: int,
               geometry: AirportGeometry, context: Landings | None) -> tuple[np.ndarray, np.ndarray]:
    """`rows_inputs` of one flight: ``(features [m − first, step], relative [m − first, K, relative])``; ``context``:
    its landing context (its own landing left out, `flight_steps`), None for a variant without it."""
    features, relative = rows_inputs(e[None], north[None], height[None], time_s, np.array([entry_s]), first, geometry,
                                     None if context is None else [context])
    return features[0], relative[0]


def step_inputs(signals: FlightSignals, n: int, geometry: AirportGeometry, context: Landings | None
                ) -> tuple[np.ndarray, np.ndarray]:
    """`row_inputs` of the first ``n`` rows of ``signals``."""
    return row_inputs(signals.e_m[:n], signals.n_m[:n], signals.altitude_m[:n], signals.time_s[:n],
                      utc_s(signals.entry_time_utc), 0, geometry, context)


def own_context(signals: FlightSignals, landings: Landings) -> Landings:
    """The airport's landings less the flight's own: a flight never counts its own landing (the roster's landing time is
    whole seconds, and a sentence's last rows can fall after it)."""
    return landings.without(utc_s(signals.landing_time_utc), signals.runway)


def issued_rows(grid: np.ndarray) -> np.ndarray:
    """The row each row's word in force was said at (``grid``'s step 0 writes every column)."""
    if (grid[0] == UNCHANGED).any():
        raise ValueError("step 0 does not write every column")
    rows = np.arange(len(grid))[:, None]
    return np.maximum.accumulate(np.where(grid != UNCHANGED, rows, 0), axis=0)


def sentence_steps(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(in_force, since, targets)`` of one sentence (``[N, 6]``, the artefact's words; N > `N_LOOK`)."""
    n = len(grid)
    if n <= N_LOOK:
        raise ValueError(f"a sentence of {n} rows has no predicted step (N_LOOK = {N_LOOK})")
    grid = grid.astype(np.int64)
    written = grid != UNCHANGED
    rows = np.arange(n)[:, None]
    issued = issued_rows(grid)
    filled = np.take_along_axis(grid, issued, axis=0)                    # the word in force at each row
    targets = np.zeros((n, 6), dtype=np.int16)
    targets[N_LOOK] = filled[N_LOOK] + 1                                 # the first predicted step says them all
    targets[N_LOOK + 1:] = np.where(written[N_LOOK + 1:], grid[N_LOOK + 1:] + 1, 0)
    in_force = np.zeros((n, 6), dtype=np.int16)
    in_force[N_LOOK + 1:] = filled[N_LOOK:-1] + 1
    said = np.maximum(issued, N_LOOK)                                    # the prior said them at N_LOOK at the earliest
    since = np.zeros((n, 6), dtype=np.float32)
    since[N_LOOK + 1:] = np.log1p(rows[N_LOOK + 1:] - said[N_LOOK:-1]) / SINCE_SCALE
    return in_force, since, targets


def flight_steps(signals: FlightSignals, grid: np.ndarray, geometry: AirportGeometry, landings: Landings | None
                 ) -> tuple[np.ndarray, ...]:
    """``(features, relative, static, in_force, since, targets)`` of one flight's sentence ``grid`` over its signals'
    first ``len(grid)`` rows; ``landings``: the airport's landing context (its own left out here), None for a variant
    without it."""
    if signals.airport != geometry.code:
        raise ValueError(f"{signals.dataset_id} lands at {signals.airport}, not {geometry.code}")
    context = own_context(signals, landings) if landings is not None else None
    features, relative = step_inputs(signals, len(grid), geometry, context)
    return (features, relative, np.zeros(len(STATIC_FEATURES), dtype=np.float32), *sentence_steps(grid))


def _first_step(signals: FlightSignals, geometry: AirportGeometry, capture_row: int) -> dict[str, float | int | bool]:
    """What the runway rules and the readout need of a flight (`Flight`): the first predicted step's time, the entry
    sector, the past direction of motion there, whether the labeller's capture row (the final corridor to the end) is
    at or before it — all observed."""
    _, motion, _ = _motion(signals.e_m[: N_LOOK + 1], signals.n_m[: N_LOOK + 1], signals.altitude_m[: N_LOOK + 1],
                           signals.time_s[: N_LOOK + 1])
    return {"first_step_s": utc_s(signals.entry_time_utc) + float(signals.time_s[N_LOOK]),
            "sector": entry_sector(signals, geometry), "course_deg": float(motion[N_LOOK]) % 360.0,
            "established": bool(capture_row <= N_LOOK)}


def _asked_from_first_step(rows: int) -> np.ndarray:
    asked = np.zeros((rows, 6), dtype=bool)
    asked[N_LOOK:] = True
    return asked


def flight_record(signals: FlightSignals, grid: np.ndarray, geometry: AirportGeometry, landings: Landings | None,
                  airport: int, capture_row: int) -> Flight:
    """One labelled flight as the prior reads it: its steps (`flight_steps`), every column asked from the first
    predicted step on, and what the runway rules and the readout need (`_first_step`)."""
    features, relative, static, in_force, since, targets = flight_steps(signals, grid, geometry, landings)
    return Flight(signals.dataset_id, airport, features, relative, static, in_force, since, targets,
                  _asked_from_first_step(len(grid)), **_first_step(signals, geometry, capture_row))


def chain_record(signals: FlightSignals, e: np.ndarray, north: np.ndarray, height: np.ndarray, said: np.ndarray,
                 classes: np.ndarray, asked: np.ndarray, geometry: AirportGeometry, landings: Landings | None,
                 airport: int, capture_row: int, step_s: float) -> Flight:
    """One closed-loop sentence as the prior reads it: the rows it read — the positions ``e``, ``north``, ``height``
    (``[N_LOOK + steps]``: observed to the first predicted step, then where the executor flew), on the vocabulary's
    step from row 0, and the words it said (``said``, ``[steps, 6]`` from row `N_LOOK`) as the words said so far — and
    the targets given for its steps (``classes``, ``asked``; design §9.3: its own words, every column). The same rows a
    speaker (`prior.generate.Speaker`) built as it spoke."""
    rows = N_LOOK + len(said)
    if not (len(e) == len(north) == len(height) == rows and classes.shape == asked.shape == said.shape):
        raise ValueError(f"{signals.dataset_id}: a sentence of {len(said)} steps has {len(e)} rows and targets "
                         f"{classes.shape}")
    context = own_context(signals, landings) if landings is not None else None
    features, relative = row_inputs(e, north, height, np.arange(rows) * step_s, utc_s(signals.entry_time_utc), 0,
                                    geometry, context)
    grid = np.full((rows, 6), UNCHANGED, dtype=np.int64)
    grid[0], grid[N_LOOK:] = said[0], said                   # the prior said its first step at row N_LOOK
    in_force, since, _ = sentence_steps(grid)
    targets = np.zeros((rows, 6), dtype=np.int16)
    targets[N_LOOK:] = classes
    mask = np.zeros((rows, 6), dtype=bool)
    mask[N_LOOK:] = asked
    return Flight(signals.dataset_id, airport, features, relative, np.zeros(len(STATIC_FEATURES), dtype=np.float32),
                  in_force, since, targets, mask, **_first_step(signals, geometry, capture_row))


def airport_landings(directory: Path, tracks: Mapping[str, Path]) -> dict[str, Landings]:
    """Every airport's context landings on its candidate runways (``tracks``: airport → its tracks roster), less the
    sealed test days of the artefact's day split."""
    days = load_day_split(directory)
    return {code: context_landings(tracks[code], [c.ident for c in geometry.candidates], days).landings()
            for code, geometry in load_candidates(directory).items()}


def load_split(directory: Path, split: str, spec: VocabularySpec, words: Words, variant: str, *,
               landings: Mapping[str, Landings], airports: Sequence[str] | None = None,
               limit: int | None = None) -> Split:
    """Every labelled flight of ``split`` under the variant ``variant`` (the first ``limit`` when given — a SMOKE
    option, stated by the caller). ``landings``: `airport_landings`."""
    geometries = load_candidates(directory)
    airports = tuple(sorted(geometries)) if airports is None else tuple(airports)
    slots = max(len(g.candidates) for g in geometries.values())
    signals = load_signals(directory, split)
    sentences = load_sentences(directory, split, spec)
    context = VARIANTS[variant].landing_context
    flights = []
    for k in range(len(sentences["signal_index"][:limit])):
        flight = signals[int(sentences["signal_index"][k])]
        grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]
        flights.append(flight_record(flight, grid, geometries[flight.airport],
                                     landings[flight.airport] if context else None, airports.index(flight.airport),
                                     int(sentences["capture_row"][k])))
    identities = [f.dataset_id for f in flights]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{directory} {split}: a flight is labelled twice")
    return Split(flights, airports, candidate_table(geometries, airports, slots), *runway_names(geometries, airports),
                 column_classes(words, slots), variant)


def batches(flights: Sequence[Flight], tokens: int, rng: np.random.Generator | None) -> Iterator[list[int]]:
    """Indices of flights in batches of similar length, each at most ``tokens`` padded steps (a flight longer than
    that is a batch of its own); shuffled when ``rng`` is given, else in length order."""
    order = sorted(range(len(flights)), key=lambda i: flights[i].rows)
    groups, current, longest = [], [], 0
    for i in order:
        longest_after = max(longest, flights[i].rows)
        if current and longest_after * (len(current) + 1) > tokens:
            groups.append(current)
            current, longest_after = [], flights[i].rows
        current.append(i)
        longest = longest_after
    if current:
        groups.append(current)
    if rng is not None:
        rng.shuffle(groups)
    return iter(groups)
