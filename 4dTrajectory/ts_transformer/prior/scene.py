"""Scenes and the airport's landing context (prior design §3.2, §4.2): which aircraft are in the air at one
airport at one time, and which landings an observer at time t has already seen.

Torch-free. Built from one instruction artefact's flights and the harvest's tracks rosters under the artefact's
day split: a test day's landing is never counted, and its flights are never in the artefact to begin with.

Times are UTC epoch seconds. A flight's row r happens at ``entry_time_utc + time_s[r]`` (`FlightSignals`).
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

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

    def segments(self) -> list[list[Presence]]:
        """The flights chained into segments: a flight joins the segment whose time it overlaps (§3.2)."""
        out: list[list[Presence]] = []
        end = -np.inf
        for item in self.flights:
            if item.start_s > end:
                out.append([])
            out[-1].append(item)
            end = max(end, item.end_s)
        return out
