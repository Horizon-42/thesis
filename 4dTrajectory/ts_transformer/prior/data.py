"""The sentence artefact as the prior's training data (prior design §1–§2; the second version §8).

A flight's step ``t`` (one row of its sentence, 2 s) gives the model:

- the state at row ``t`` from POSITIONS (`STATE_FEATURES`, fixed SI scales): airport-frame E, N, MSL height, time
  from row 0; and, by the input set (`INPUT_SETS`), a velocity block (`VELOCITY_FEATURES`): none; ``past`` — the
  displacement from row ``t − 1`` to row ``t`` over its time (ground speed, direction of motion as sin / cos,
  vertical rate), zero with a flag at row 0, which has no row before it; or ``fitted`` — the signals' track (sin,
  cos), ground speed and vertical rate, a least-squares fit over a window CENTRED on the row, so 7.5 s of the future
  (variant 0 only: it measures what that future is worth, it never flies);
- for every candidate runway of the airport, where the aircraft is relative to it (`RELATIVE_FEATURES`, from
  positions only, `instructions.airport.relative_to_runway`): before the threshold along the course, right of the
  centreline (both `asinh` of kilometres), height above the threshold; and, by the input set, a direction block
  (`DIRECTION_FEATURES`): none, the past direction of motion − the course (zero with a flag at row 0), or the fitted
  track − the course. Every candidate is measured, never only the landed one, so nothing gives the answer away;
- the words in force at the step (``in_force``: each column's word + 1, 0 = none yet): at row 0 the hand-over's
  words in `GIVEN_AT_HANDOVER` — every column but the runway, the sentence's own step 0 there, a condition the
  prior is given, not a word it says — and no runway; from row 1 the words said up to the row before. Per column the
  steps since that word (``since``, `log1p` / `SINCE_SCALE`; 0 at row 0);
- the context: the airport, and its candidate runways' geometry (`CANDIDATE_FEATURES`, one table per airport).

and asks for the step's six words (``targets``: 0 = unchanged, else the word + 1). At row 0 only the runway is
asked (`prior.model.predicted_entries`); the hand-over columns' targets there equal their inputs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, COLUMNS, HEADING, SPEED, UNCHANGED, Words

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

#: The columns known at the hand-over (prior design §8.2 item 2, the user's split 2026-09-24): given at row 0, not
#: predicted there. The runway is the one decision step 0 asks for.
GIVEN_AT_HANDOVER = (APPROACH, HEADING, ALTITUDE, ANGLE, SPEED)

STATE_FEATURES = ("e", "n", "height", "time")
VELOCITY_FEATURES = {
    "none": (),
    "past": ("ground_speed", "motion_sin", "motion_cos", "vertical_rate", "first_row"),
    "fitted": ("track_sin", "track_cos", "ground_speed", "vertical_rate"),
}
RELATIVE_FEATURES = ("before_threshold", "right_of_centreline", "height_above_threshold")
DIRECTION_FEATURES = {
    "none": (),
    "past": ("motion_minus_course_sin", "motion_minus_course_cos", "first_row"),
    "fitted": ("track_minus_course_sin", "track_minus_course_cos"),
}
CANDIDATE_FEATURES = ("threshold_e", "threshold_n", "elevation", "course_sin", "course_cos", "length", "valid")


@dataclass(frozen=True)
class InputSet:
    """Which velocity block goes into the step and which direction block into every candidate (§8.3)."""

    velocity: str
    direction: str
    #: How many quantities the set adds to positions (§8.3's "fewer inputs"): the velocity block counts three
    #: (speed, direction, vertical rate), the direction term one.
    added: int

    @property
    def step_features(self) -> tuple[str, ...]:
        return STATE_FEATURES + VELOCITY_FEATURES[self.velocity]

    @property
    def relative_features(self) -> tuple[str, ...]:
        return RELATIVE_FEATURES + DIRECTION_FEATURES[self.direction]


#: §8.3's variants. V0 is the first version's fitted inputs, kept only to measure the future they carry; the others
#: are the choice (`CHOOSABLE`).
INPUT_SETS = {
    "V0": InputSet("fitted", "fitted", 4),
    "V1": InputSet("none", "none", 0),
    "V1d": InputSet("none", "past", 1),
    "V2": InputSet("past", "none", 3),
    "V2d": InputSet("past", "past", 4),
}
CHOOSABLE = ("V1", "V1d", "V2", "V2d")


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
    features: np.ndarray       # [N, len(step_features)] float32
    relative: np.ndarray       # [N, K, len(relative_features)] float32, K = the airport's candidates
    in_force: np.ndarray       # [N, 6] int16
    since: np.ndarray          # [N, 6] float32
    targets: np.ndarray        # [N, 6] int16

    @property
    def rows(self) -> int:
        return len(self.targets)


@dataclass(frozen=True)
class Split:
    flights: list[Flight]
    airports: tuple[str, ...]
    candidates: np.ndarray     # [A, slots, len(CANDIDATE_FEATURES)] float32
    classes: tuple[int, ...]
    inputs: str                # an `INPUT_SETS` name

    def subset(self, indices: Sequence[int]) -> Split:
        return Split([self.flights[i] for i in indices], self.airports, self.candidates, self.classes, self.inputs)


def candidate_table(geometries: dict[str, AirportGeometry], airports: Sequence[str], slots: int) -> np.ndarray:
    """Each airport's candidates in its own order (the sentence's pointer), padded to ``slots`` with ``valid`` 0."""
    table = np.zeros((len(airports), slots, len(CANDIDATE_FEATURES)), dtype=np.float32)
    for a, code in enumerate(airports):
        for c, candidate in enumerate(geometries[code].candidates):
            course = math.radians(candidate.course_deg)
            table[a, c] = (candidate.threshold_e_m / POSITION_SCALE_M, candidate.threshold_n_m / POSITION_SCALE_M,
                           candidate.elevation_m / HEIGHT_SCALE_M, math.sin(course), math.cos(course),
                           candidate.length_m / LENGTH_SCALE_M, 1.0)
    return table


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


def step_inputs(signals: FlightSignals, n: int, geometry: AirportGeometry, inputs: InputSet) -> tuple[np.ndarray, np.ndarray]:
    """``(features [n, step], relative [n, K, relative])`` of the first ``n`` rows of ``signals``."""
    e, north, height = signals.e_m[:n], signals.n_m[:n], signals.altitude_m[:n]
    columns = [e / POSITION_SCALE_M, north / POSITION_SCALE_M, height / HEIGHT_SCALE_M,
               (signals.time_s[:n] - signals.time_s[0]) / TIME_SCALE_S]
    speed, motion, climb, first = _past_motion(signals, n)
    if inputs.velocity == "past":
        radians = np.radians(motion)
        columns += [speed / SPEED_SCALE_MPS, np.sin(radians) * (1.0 - first), np.cos(radians) * (1.0 - first),
                    climb / VERTICAL_RATE_SCALE_MPS, first]
    elif inputs.velocity == "fitted":
        track = np.radians(signals.track_deg[:n])
        columns += [np.sin(track), np.cos(track), signals.ground_speed_mps[:n] / SPEED_SCALE_MPS,
                    signals.vertical_rate_mps[:n] / VERTICAL_RATE_SCALE_MPS]
    direction = {"none": np.zeros(n), "past": motion, "fitted": signals.track_deg[:n]}[inputs.direction]
    relative = np.zeros((n, len(geometry.candidates), len(inputs.relative_features)))
    for k, candidate in enumerate(geometry.candidates):
        place = relative_to_runway(e, north, direction, height, candidate)
        parts = [np.arcsinh(place.before_threshold_m / RELATIVE_SCALE_M),
                 np.arcsinh(place.right_of_course_m / RELATIVE_SCALE_M),
                 place.height_above_threshold_m / HEIGHT_SCALE_M]
        off = np.radians(place.track_minus_course_deg)
        if inputs.direction == "past":
            parts += [np.sin(off) * (1.0 - first), np.cos(off) * (1.0 - first), first]
        elif inputs.direction == "fitted":
            parts += [np.sin(off), np.cos(off)]
        relative[:, k] = np.column_stack(parts)
    return np.column_stack(columns).astype(np.float32), relative.astype(np.float32)


def sentence_steps(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(in_force, since, targets)`` of one sentence (``[N, 6]``, the artefact's words)."""
    n = len(grid)
    grid = grid.astype(np.int64)
    if (grid[0] == UNCHANGED).any():
        raise ValueError("step 0 does not write every column")
    written = grid != UNCHANGED
    steps = np.where(written, np.arange(n)[:, None], 0)
    issued = np.maximum.accumulate(steps, axis=0)
    filled = np.take_along_axis(grid, issued, axis=0)
    in_force = np.zeros((n, 6), dtype=np.int16)
    in_force[1:] = filled[:-1] + 1
    in_force[0, list(GIVEN_AT_HANDOVER)] = grid[0, list(GIVEN_AT_HANDOVER)] + 1   # the hand-over, given
    since = np.zeros((n, 6), dtype=np.float32)
    since[1:] = np.log1p(np.arange(1, n)[:, None] - issued[:-1]) / SINCE_SCALE
    targets = np.where(written, grid + 1, 0).astype(np.int16)
    return in_force, since, targets


def flight_steps(signals: FlightSignals, grid: np.ndarray, geometry: AirportGeometry,
                 inputs: InputSet) -> tuple[np.ndarray, ...]:
    """``(features, relative, in_force, since, targets)`` of one flight's sentence ``grid`` over its signals' first
    ``len(grid)`` rows."""
    if signals.airport != geometry.code:
        raise ValueError(f"{signals.dataset_id} lands at {signals.airport}, not {geometry.code}")
    features, relative = step_inputs(signals, len(grid), geometry, inputs)
    return (features, relative, *sentence_steps(grid))


def load_split(directory: Path, split: str, spec: VocabularySpec, words: Words, inputs: str, *,
               airports: Sequence[str] | None = None, limit: int | None = None) -> Split:
    """Every labelled flight of ``split`` under the input set ``inputs`` (the first ``limit`` when given — a SMOKE
    option, stated by the caller)."""
    geometries = load_candidates(directory)
    airports = tuple(sorted(geometries)) if airports is None else tuple(airports)
    slots = max(len(g.candidates) for g in geometries.values())
    signals = load_signals(directory, split)
    sentences = load_sentences(directory, split, spec)
    input_set = INPUT_SETS[inputs]
    flights = []
    for k in range(len(sentences["signal_index"][:limit])):
        flight = signals[int(sentences["signal_index"][k])]
        grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]
        flights.append(Flight(flight.dataset_id, airports.index(flight.airport),
                              *flight_steps(flight, grid, geometries[flight.airport], input_set)))
    identities = [f.dataset_id for f in flights]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{directory} {split}: a flight is labelled twice")
    return Split(flights, airports, candidate_table(geometries, airports, slots), column_classes(words, slots), inputs)


def selection_ids(split: Split, share: float, seed: int) -> tuple[str, ...]:
    """The train-internal selection set (§8.3): per airport (in name order), ``round(share · n)`` of its flights,
    drawn by one seeded generator over each airport's dataset ids in sorted order; returned sorted."""
    by_airport: dict[int, list[str]] = defaultdict(list)
    for flight in split.flights:
        by_airport[flight.airport].append(flight.dataset_id)
    rng = np.random.default_rng(seed)
    chosen = []
    for airport in sorted(by_airport):
        ids = sorted(by_airport[airport])
        chosen += [ids[i] for i in rng.permutation(len(ids))[: round(share * len(ids))]]
    return tuple(sorted(chosen))


def partition(split: Split, selected: Sequence[str]) -> tuple[Split, Split]:
    """``(the rest, the selected flights)`` of ``split``, each in ``split``'s order; every selected id must be in it."""
    chosen = set(selected)
    inside = [i for i, f in enumerate(split.flights) if f.dataset_id in chosen]
    if len(inside) != len(chosen):
        raise ValueError(f"{len(chosen) - len(inside)} selected flights are not in the split")
    rest = [i for i, f in enumerate(split.flights) if f.dataset_id not in chosen]
    return split.subset(rest), split.subset(inside)


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
