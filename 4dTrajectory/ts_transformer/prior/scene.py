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
    """One airport's context landings (epoch seconds, sorted): all of them, and by runway."""

    times_s: np.ndarray
    by_runway: Mapping[str, np.ndarray]

    def count_before(self, t_s: np.ndarray, window_s: float, runway: str | None = None) -> np.ndarray:
        """Landings in ``[t - window, t)`` — strictly before ``t`` — on ``runway`` (any runway when None)."""
        times = self.times_s if runway is None else self.by_runway[runway]
        t = np.asarray(t_s, dtype=np.float64)
        return np.searchsorted(times, t, side="left") - np.searchsorted(times, t - window_s, side="left")


def context_landings(tracks_manifest: Path, runways: Iterable[str], days: DaySplit) -> tuple[Landings, int]:
    """The tracks roster's assigned landings on ``runways`` (the airport's candidates), minus every landing on a
    sealed test day — returned with how many were left out. The same pool as runway intent's
    (`data.runway_context.build_airport_context`): each landing's runway and time, whether or not the flight is
    an eligible arrival."""
    kept: dict[str, list[float]] = {runway: [] for runway in runways}
    sealed = 0
    for row in json.loads(Path(tracks_manifest).read_text(encoding="utf-8"))["records"]:
        if row["outcome"] != "assigned" or row["runway"] not in kept:
            continue
        if days.split_of(landing_day(row["landing_time_utc"])) == "test":
            sealed += 1
            continue
        kept[row["runway"]].append(utc_s(row["landing_time_utc"]))
    by_runway = {runway: np.sort(np.array(times, dtype=np.float64)) for runway, times in kept.items()}
    return Landings(np.sort(np.concatenate(list(by_runway.values()))), by_runway), sealed


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
