"""The sentence artefact as the prior's training data (prior design §1–§2).

A flight's step ``t`` (one row of its sentence, 2 s) gives the model:

- the state at row ``t`` (`STATE_FEATURES`, fixed SI scales): airport-frame position, MSL height, track (sin, cos),
  ground speed, vertical rate, time from row 0;
- the state relative to the runway IN FORCE (`RUNWAY_FEATURES`): before the threshold, right of the centreline,
  height above the threshold, track − course (sin, cos) and a flag — zero, flag down, at row 0, where the runway
  pointer is still to be said (the pointed runway there would give the answer away);
- the words in force before the step (``in_force``: each column's word + 1, 0 = no word yet) and, per column, the
  steps since that word (``since``, `log1p` / `SINCE_SCALE`);
- the context: the airport and its candidate runways (`MAX_CANDIDATES` slots of `CANDIDATE_FEATURES`, a mask).

and asks for the step's six words (``targets``: 0 = unchanged, else the word + 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import COLUMNS, RUNWAY, UNCHANGED, Words

POSITION_SCALE_M = 10_000.0
HEIGHT_SCALE_M = 1_000.0
SPEED_SCALE_MPS = 100.0
VERTICAL_RATE_SCALE_MPS = 10.0
TIME_SCALE_S = 600.0
OFFSET_SCALE_M = 1_000.0
SINCE_SCALE = 5.0
STATE_FEATURES = ("e", "n", "height", "track_sin", "track_cos", "ground_speed", "vertical_rate", "time")
RUNWAY_FEATURES = ("before_threshold", "right_of_centreline", "height_above_threshold", "off_course_sin",
                   "off_course_cos", "runway_known")
CANDIDATE_FEATURES = ("threshold_e", "threshold_n", "elevation", "course_sin", "course_cos", "valid")
N_FEATURES = len(STATE_FEATURES) + len(RUNWAY_FEATURES)


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
    features: np.ndarray       # [N, N_FEATURES] float32
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
    candidates: np.ndarray     # [A, MAX_CANDIDATES, len(CANDIDATE_FEATURES)] float32
    classes: tuple[int, ...]


def candidate_table(geometries: dict[str, AirportGeometry], airports: Sequence[str], slots: int) -> np.ndarray:
    table = np.zeros((len(airports), slots, len(CANDIDATE_FEATURES)), dtype=np.float32)
    for a, code in enumerate(airports):
        for c, candidate in enumerate(geometries[code].candidates):
            course = math.radians(candidate.course_deg)
            table[a, c] = (candidate.threshold_e_m / POSITION_SCALE_M, candidate.threshold_n_m / POSITION_SCALE_M,
                           candidate.elevation_m / HEIGHT_SCALE_M, math.sin(course), math.cos(course), 1.0)
    return table


def flight_steps(signals: FlightSignals, grid: np.ndarray, geometry: AirportGeometry) -> tuple[np.ndarray, ...]:
    """``(features, in_force, since, targets)`` of one flight's ``grid`` (``[N, 6]``, the artefact's words) over its
    signals' first ``N`` rows."""
    n = len(grid)
    grid = grid.astype(np.int64)
    if (grid[0] == UNCHANGED).any():
        raise ValueError(f"{signals.dataset_id}: step 0 does not write every column")
    written = grid != UNCHANGED
    steps = np.where(written, np.arange(n)[:, None], 0)
    issued = np.maximum.accumulate(steps, axis=0)
    filled = np.take_along_axis(grid, issued, axis=0)
    in_force = np.zeros((n, 6), dtype=np.int16)
    in_force[1:] = filled[:-1] + 1
    since = np.zeros((n, 6), dtype=np.float32)
    since[1:] = np.log1p(np.arange(1, n)[:, None] - issued[:-1]) / SINCE_SCALE
    targets = np.where(written, grid + 1, 0).astype(np.int16)

    track = np.radians(signals.track_deg[:n])
    state = np.column_stack((signals.e_m[:n] / POSITION_SCALE_M, signals.n_m[:n] / POSITION_SCALE_M,
                             signals.altitude_m[:n] / HEIGHT_SCALE_M, np.sin(track), np.cos(track),
                             signals.ground_speed_mps[:n] / SPEED_SCALE_MPS,
                             signals.vertical_rate_mps[:n] / VERTICAL_RATE_SCALE_MPS,
                             (signals.time_s[:n] - signals.time_s[0]) / TIME_SCALE_S))
    runway = np.zeros((n, len(RUNWAY_FEATURES)))
    pointer = filled[:-1, RUNWAY]                  # the runway in force before each step from 1 on
    for index in np.unique(pointer):
        rows = np.nonzero(pointer == index)[0] + 1
        candidate = geometry.candidates[index]
        relative = relative_to_runway(signals.e_m[rows], signals.n_m[rows], signals.track_deg[rows],
                                      signals.altitude_m[rows], candidate)
        off = np.radians(relative.track_minus_course_deg)
        runway[rows] = np.column_stack((relative.before_threshold_m / POSITION_SCALE_M,
                                        relative.right_of_course_m / OFFSET_SCALE_M,
                                        relative.height_above_threshold_m / HEIGHT_SCALE_M,
                                        np.sin(off), np.cos(off), np.ones(len(rows))))
    return np.hstack((state, runway)).astype(np.float32), in_force, since, targets


def load_split(directory: Path, split: str, spec: VocabularySpec, words: Words, *,
               airports: Sequence[str] | None = None, limit: int | None = None) -> Split:
    """Every labelled flight of ``split`` (the first ``limit`` when given — a SMOKE option, stated by the caller)."""
    geometries = load_candidates(directory)
    airports = tuple(sorted(geometries)) if airports is None else tuple(airports)
    slots = max(len(g.candidates) for g in geometries.values())
    signals = load_signals(directory, split)
    sentences = load_sentences(directory, split, spec)
    flights = []
    count = len(sentences["signal_index"]) if limit is None else limit
    for k in range(count):
        flight = signals[int(sentences["signal_index"][k])]
        grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]
        features, in_force, since, targets = flight_steps(flight, grid, geometries[flight.airport])
        flights.append(Flight(flight.dataset_id, airports.index(flight.airport), features, in_force, since, targets))
    return Split(flights, airports, candidate_table(geometries, airports, slots), column_classes(words, slots))


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
