"""Scenes and the airport's landing context (prior design §3.2, §4.2): which aircraft are in the air at one
airport at one time, and which landings an observer at time t has already seen.

Torch-free. Built from one instruction artefact's flights and the harvest's tracks rosters under the artefact's
day split: a test day's landing is never counted, and its flights are never in the artefact to begin with.

Times are UTC epoch seconds. A flight's row r happens at ``entry_time_utc + time_s[r]`` (`FlightSignals`).

Several aircraft share the scene's steps (multi-aircraft design §2.1, user 2026-09-27): the UTC times that are whole
multiples of the rows' step (even seconds for 2 s). A flight's rows hang on the nearest step (`hang`, at most half a step
away); a flight keeps its own row numbers. A segment is cut into teacher-forcing samples by §2.3's rule (`samples`).
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ts_transformer.data.day_split import DaySplit, landing_day, parse_utc
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.signals import FlightSignals

#: The rows at the head of every sentence that are only observed, never predicted (§10: 8 rows, 16 s).
N_LOOK = 8
#: The landing context's window, ``T_cfg`` (§4.2, §10: 30 minutes).
CONTEXT_WINDOW_S = 1800.0
#: A LEADER is an aircraft landing earlier on the same runway that is at most this much closer to the threshold
#: (§8: "同一条跑道前方 10 km 内").
LEADER_RANGE_M = 10_000.0
#: Multi-aircraft design §2.3: the steps a sample's loss is on span at most this long; a longer segment is cut at the
#: step, from `SAMPLE_SEARCH_FROM_S` to `SAMPLE_MAX_S` after the sample's loss starts, with the fewest aircraft in the
#: scene (background aircraft counted; the earliest of equals). The step at a cut is the later sample's.
SAMPLE_MAX_S = 1_200.0
SAMPLE_SEARCH_FROM_S = 900.0


def utc_s(value: str) -> float:
    """An ISO UTC time as epoch seconds."""
    return parse_utc(value).timestamp()


@dataclass(frozen=True)
class Landings:
    """One airport's context landings (epoch seconds, sorted): all of them, and by candidate runway (every candidate,
    one with no landing empty)."""

    times_s: np.ndarray
    by_runway: Mapping[str, np.ndarray]

    def _on(self, runway: str | None) -> np.ndarray:
        return self.times_s if runway is None else self.by_runway[runway]

    def count_before(self, t_s: np.ndarray, window_s: float, runway: str | None = None) -> np.ndarray:
        """Landings in ``[t - window, t)`` — strictly before ``t`` — on ``runway`` (any runway when None)."""
        times = self._on(runway)
        t = np.asarray(t_s, dtype=np.float64)
        return np.searchsorted(times, t, side="left") - np.searchsorted(times, t - window_s, side="left")

    def without(self, time_s: float, runway: str) -> Landings:
        """These landings less one: a flight's own, which must be here (a flight never counts its own landing — the
        roster's landing time is whole seconds, and a sentence's last rows can fall after it)."""
        on_runway = self.by_runway[runway]
        where = np.flatnonzero(on_runway == time_s)
        overall = np.flatnonzero(self.times_s == time_s)
        if not len(where):
            raise ValueError(f"no landing on {runway} at {time_s} in the context pool: the flight's own must be there")
        return Landings(np.delete(self.times_s, overall[0]),
                        {**self.by_runway, runway: np.delete(on_runway, where[0])})

    def since_last(self, t_s: np.ndarray, runway: str) -> np.ndarray:
        """Seconds from the last landing on ``runway`` strictly before ``t``; NaN where there is none."""
        times = self._on(runway)
        t = np.asarray(t_s, dtype=np.float64)
        index = np.searchsorted(times, t, side="left") - 1
        if not len(times):
            return np.full(t.shape, np.nan)
        return np.where(index >= 0, t - times[np.maximum(index, 0)], np.nan)


@dataclass(frozen=True)
class LandingRecord:
    """One context landing: when (epoch seconds), on which runway, which flight, on a day of which split."""

    time_s: float
    runway: str
    flight_key: str
    split: str


@dataclass(frozen=True)
class LandingPool:
    """One airport's context landings on its candidate runways, sorted by time, and how many sealed test-day landings
    were left out."""

    runways: tuple[str, ...]
    records: tuple[LandingRecord, ...]
    sealed: int

    def landings(self) -> Landings:
        return Landings(np.array([r.time_s for r in self.records], dtype=np.float64),
                        {runway: np.array([r.time_s for r in self.records if r.runway == runway], dtype=np.float64)
                         for runway in self.runways})


def context_landings(tracks_manifest: Path, runways: Iterable[str], days: DaySplit) -> LandingPool:
    """The tracks roster's assigned landings on ``runways`` (the airport's candidates), minus every landing on a
    sealed test day. The same pool as runway intent's (`data.runway_context.build_airport_context`): each landing's
    runway and time, whether or not the flight is an eligible arrival."""
    wanted = tuple(runways)
    kept: list[LandingRecord] = []
    sealed = 0
    for row in json.loads(Path(tracks_manifest).read_text(encoding="utf-8"))["records"]:
        if row["outcome"] != "assigned" or row["runway"] not in wanted:
            continue
        split = days.split_of(landing_day(row["landing_time_utc"]))
        if split == "test":
            sealed += 1
            continue
        kept.append(LandingRecord(utc_s(row["landing_time_utc"]), row["runway"], row["flight_key"], split))
    kept.sort(key=lambda record: (record.time_s, record.flight_key))
    return LandingPool(wanted, tuple(kept), sealed)


@dataclass(frozen=True)
class Presence:
    """One flight's time in the scene: from its first row to its last sentence row (the rows before the
    threshold crossing), or to its last signal row when it has no sentence (a background aircraft)."""

    dataset_id: str
    airport: str
    runway: str                  # the runway it landed on (the observed one)
    landing_s: float
    speaking: bool               # it has a sentence: the prior speaks to it
    times_s: np.ndarray          # [R] its rows in the scene, epoch seconds
    to_threshold_m: np.ndarray   # [R] straight-line horizontal distance to its runway's threshold

    @property
    def start_s(self) -> float:
        return float(self.times_s[0])

    @property
    def end_s(self) -> float:
        return float(self.times_s[-1])

    def to_threshold_at(self, t_s: np.ndarray) -> np.ndarray:
        """Its distance to the threshold at times inside its span (linear between rows)."""
        return np.interp(t_s, self.times_s, self.to_threshold_m)


def presence(flight: FlightSignals, sentence_rows: int | None, geometry: AirportGeometry) -> Presence:
    """``sentence_rows``: the length of its sentence, or None for a flight without one."""
    rows = flight.n_rows if sentence_rows is None else sentence_rows
    threshold = geometry.candidates[geometry.candidate_index(flight.runway)]
    distance = np.hypot(flight.e_m[:rows] - threshold.threshold_e_m, flight.n_m[:rows] - threshold.threshold_n_m)
    return Presence(flight.dataset_id, flight.airport, flight.runway, utc_s(flight.landing_time_utc),
                    sentence_rows is not None, utc_s(flight.entry_time_utc) + flight.time_s[:rows], distance)


def leader_gap_m(ego: Presence, near: Sequence[Presence], rows_s: np.ndarray, present: np.ndarray) -> np.ndarray:
    """At each of ``ego``'s rows at ``rows_s`` (epoch seconds), how far ahead its LEADER is (§8): among the ``near``
    aircraft present there (``present``: ``[len(near), len(rows_s)]`` bool), one landing earlier on the same observed
    runway and at most `LEADER_RANGE_M` closer to its threshold (straight-line distances), the nearest; inf without one."""
    ego_to_go = np.interp(rows_s, ego.times_s, ego.to_threshold_m)
    gap = np.full(len(rows_s), np.inf)
    for other, here in zip(near, present):
        if other.runway == ego.runway and other.landing_s < ego.landing_s and here.any():
            ahead = np.where(here, ego_to_go - other.to_threshold_at(rows_s), np.inf)
            gap = np.minimum(gap, np.where((ahead > 0.0) & (ahead <= LEADER_RANGE_M), ahead, np.inf))
    return gap


class SceneIndex:
    """One airport's flights by time: who is in the scene at time t."""

    def __init__(self, flights: Iterable[Presence]) -> None:
        self.flights = sorted(flights, key=lambda item: item.start_s)
        self._starts = [item.start_s for item in self.flights]
        self._longest_s = max((item.end_s - item.start_s for item in self.flights), default=0.0)

    def overlapping(self, start_s: float, end_s: float) -> list[Presence]:
        """The flights whose time in the scene overlaps ``[start, end]``."""
        lo = bisect.bisect_left(self._starts, start_s - self._longest_s)
        hi = bisect.bisect_right(self._starts, end_s)
        return [item for item in self.flights[lo:hi] if item.end_s >= start_s]

    def segments(self, step_s: float) -> list[list[Presence]]:
        """The flights chained into segments on the steps their rows hang on (§3.2; multi-aircraft design §2.1): a flight
        whose first step is no later than the segment's last step joins it, so no step is in two segments."""
        out: list[list[Presence]] = []
        end = -np.inf
        for item in self.flights:
            first, last = hung_span(item, step_s)
            if first > end:
                out.append([])
            out[-1].append(item)
            end = max(end, last)
        return out


# ---- several aircraft on the scene's steps (multi-aircraft design §2.1, §2.3)
def hang(t_s: np.ndarray | float, step_s: float) -> np.ndarray:
    """The scene step each time hangs on: the nearest whole multiple of ``step_s`` (half a step rounds up)."""
    return np.floor(np.asarray(t_s, dtype=np.float64) / step_s + 0.5) * step_s


def scene_steps(start_s: float, end_s: float, step_s: float) -> np.ndarray:
    """The scene's steps from ``start`` to ``end``, both included."""
    return np.arange(math.ceil(start_s / step_s), math.floor(end_s / step_s) + 1) * step_s


def hung_span(flight: Presence, step_s: float) -> tuple[float, float]:
    """The first and the last step a flight's rows hang on."""
    return float(hang(flight.start_s, step_s)), float(hang(flight.end_s, step_s))


def in_scene(flights: Sequence[Presence], steps_s: np.ndarray, step_s: float) -> np.ndarray:
    """How many of ``flights`` (background aircraft too) are in the scene at each step: from the step their first row
    hangs on to the step their last row hangs on."""
    steps = np.asarray(steps_s, dtype=np.float64)
    count = np.zeros(steps.shape, dtype=np.int64)
    for flight in flights:
        first, last = hung_span(flight, step_s)
        count += (steps >= first) & (steps <= last)
    return count


@dataclass(frozen=True)
class Sample:
    """One teacher-forcing sample of a segment (§2.3): the steps its loss is on, ``[loss_start, loss_end)`` (the last
    sample's end is the segment's last step, included), every flight with a step there (one place each on the model's
    aircraft axis), and which of them entered before ``loss_start`` (their earlier rows are only context here)."""

    loss_start_s: float
    loss_end_s: float
    flights: tuple[Presence, ...]
    carried: tuple[Presence, ...]


def samples(segment: Sequence[Presence], step_s: float) -> list[Sample]:
    """A segment cut by §2.3's rule: while what is left runs past `SAMPLE_MAX_S`, cut at the step from
    `SAMPLE_SEARCH_FROM_S` to `SAMPLE_MAX_S` after the sample's loss starts with the fewest aircraft in the scene, the
    earliest of equals."""
    spans = [hung_span(flight, step_s) for flight in segment]
    start, end = min(first for first, _ in spans), max(last for _, last in spans)
    cuts = []
    loss_start = start
    while end - loss_start > SAMPLE_MAX_S:
        steps = scene_steps(loss_start + SAMPLE_SEARCH_FROM_S, loss_start + SAMPLE_MAX_S, step_s)
        loss_start = float(steps[int(np.argmin(in_scene(segment, steps, step_s)))])
        cuts.append(loss_start)
    bounds = [start, *cuts, math.inf]
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        inside = [(flight, first) for flight, (first, last) in zip(segment, spans) if last >= lo and first < hi]
        out.append(Sample(lo, min(hi, end), tuple(flight for flight, _ in inside),
                          tuple(flight for flight, first in inside if first < lo)))
    return out
