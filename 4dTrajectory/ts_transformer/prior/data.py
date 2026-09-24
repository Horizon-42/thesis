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
sentence's words as written.
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
    targets: np.ndarray        # [N, 6] int16; rows before N_LOOK are never asked
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


def _past_motion(signals: FlightSignals, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(ground speed, direction of motion (compass degrees), vertical rate, first row)`` of rows ``0..n-1`` from
    the displacement since the row before; row 0 has none (zeros, flag 1)."""
    dt = np.diff(signals.time_s[:n])
    de, dn = np.diff(signals.e_m[:n]), np.diff(signals.n_m[:n])
    speed = np.concatenate(([0.0], np.hypot(de, dn) / dt))
    direction = np.concatenate(([0.0], np.degrees(np.arctan2(de, dn))))
    climb = np.concatenate(([0.0], np.diff(signals.altitude_m[:n]) / dt))
    first = np.zeros(n)
    first[0] = 1.0
    return speed, direction, climb, first


def step_inputs(signals: FlightSignals, n: int, geometry: AirportGeometry, landings: Landings | None
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(features [n, step], relative [n, K, relative])`` of the first ``n`` rows of ``signals``; the landing context
    when ``landings`` is given (the variant has it) — the airport's landings less the flight's own."""
    e, north, height = signals.e_m[:n], signals.n_m[:n], signals.altitude_m[:n]
    speed, motion, climb, first = _past_motion(signals, n)
    radians = np.radians(motion)
    features = np.column_stack([e / POSITION_SCALE_M, north / POSITION_SCALE_M, height / HEIGHT_SCALE_M,
                                (signals.time_s[:n] - signals.time_s[0]) / TIME_SCALE_S,
                                speed / SPEED_SCALE_MPS, np.sin(radians) * (1.0 - first),
                                np.cos(radians) * (1.0 - first), climb / VERTICAL_RATE_SCALE_MPS, first])
    clock = utc_s(signals.entry_time_utc) + signals.time_s[:n]
    if landings is not None:
        landings = landings.without(utc_s(signals.landing_time_utc), signals.runway)
    width = len(RELATIVE_FEATURES) + (len(CONTEXT_FEATURES) if landings is not None else 0)
    relative = np.zeros((n, len(geometry.candidates), width))
    for k, candidate in enumerate(geometry.candidates):
        place = relative_to_runway(e, north, motion, height, candidate)
        off = np.radians(place.track_minus_course_deg)
        parts = [np.arcsinh(place.before_threshold_m / RELATIVE_SCALE_M),
                 np.arcsinh(place.right_of_course_m / RELATIVE_SCALE_M),
                 place.height_above_threshold_m / HEIGHT_SCALE_M,
                 np.sin(off) * (1.0 - first), np.cos(off) * (1.0 - first), first]
        if landings is not None:
            since = landings.since_last(clock, candidate.ident)
            none = np.isnan(since)
            parts += [np.log1p(landings.count_before(clock, CONTEXT_WINDOW_S, candidate.ident)),
                      np.where(none, 0.0, np.log1p(np.nan_to_num(since) / 60.0) / LANDING_SINCE_SCALE),
                      none.astype(np.float64)]
        relative[:, k] = np.column_stack(parts)
    return features.astype(np.float32), relative.astype(np.float32)


def sentence_steps(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(in_force, since, targets)`` of one sentence (``[N, 6]``, the artefact's words; N > `N_LOOK`)."""
    n = len(grid)
    if n <= N_LOOK:
        raise ValueError(f"a sentence of {n} rows has no predicted step (N_LOOK = {N_LOOK})")
    grid = grid.astype(np.int64)
    if (grid[0] == UNCHANGED).any():
        raise ValueError("step 0 does not write every column")
    written = grid != UNCHANGED
    rows = np.arange(n)[:, None]
    issued = np.maximum.accumulate(np.where(written, rows, 0), axis=0)   # the row each row's word in force was said
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
    first ``len(grid)`` rows."""
    if signals.airport != geometry.code:
        raise ValueError(f"{signals.dataset_id} lands at {signals.airport}, not {geometry.code}")
    features, relative = step_inputs(signals, len(grid), geometry, landings)
    return (features, relative, np.zeros(len(STATIC_FEATURES), dtype=np.float32), *sentence_steps(grid))


def flight_record(signals: FlightSignals, grid: np.ndarray, geometry: AirportGeometry, landings: Landings | None,
                  airport: int, capture_row: int) -> Flight:
    """One labelled flight as the prior reads it: its steps (`flight_steps`), and what the runway rules and the
    readout need — the first predicted step's time, the entry sector, the past direction of motion there, whether the
    labeller's capture row (the final corridor to the end) is at or before it."""
    steps = flight_steps(signals, grid, geometry, landings)
    _, motion, _, _ = _past_motion(signals, N_LOOK + 1)
    return Flight(signals.dataset_id, airport, *steps,
                  first_step_s=utc_s(signals.entry_time_utc) + float(signals.time_s[N_LOOK]),
                  sector=entry_sector(signals, geometry), course_deg=float(motion[N_LOOK]) % 360.0,
                  established=bool(capture_row <= N_LOOK))


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
