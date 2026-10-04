"""The airport's landings before a step (prior design §2 "Candidates", §6): an input of each candidate vector is the
number of landings on that candidate in the 30 min before the step.

Offline, the landings are the harvest's tracks roster of the airport (`repo_layout.tracks_manifest_path`): every
record the harvest assigned to a runway, whether or not its flight has a sentence — less every landing on a sealed test
day (contract C32: a test day is never counted as context). In a loop the caller gives the landings the loop knows.
A flight never counts its own landing (`LandingIndex.counts_before`): a closed-loop sentence flies on to the threshold and its
last rows can fall after the time the observed aircraft landed, and the runway it landed on is the answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ts_transformer.data.day_split import DaySplit, landing_day, parse_utc

#: The window of the landings input, s (§2).
LANDINGS_WINDOW_S = 30.0 * 60.0


@dataclass(frozen=True)
class Landing:
    time_s: float          # UTC epoch seconds
    runway: str            # the candidate's ident
    flight_key: str


@dataclass(frozen=True)
class LandingIndex:
    """One airport's landings on its candidates, sorted by time; ``sealed``: the test-day landings left out."""

    runways: tuple[str, ...]
    landings: tuple[Landing, ...]
    sealed: int

    def __post_init__(self) -> None:
        times = [landing.time_s for landing in self.landings]
        if times != sorted(times):
            raise ValueError("the landings are not in time order")
        unknown = {landing.runway for landing in self.landings} - set(self.runways)
        if unknown:
            raise ValueError(f"landings on {sorted(unknown)}, not candidates {self.runways}")
        times = {runway: np.array([landing.time_s for landing in self.landings if landing.runway == runway],
                                  dtype=np.float64) for runway in self.runways}
        keys = [landing.flight_key for landing in self.landings]
        if len(set(keys)) != len(keys):
            raise ValueError("a flight lands twice in the index")
        object.__setattr__(self, "_times", times)
        object.__setattr__(self, "_own", {landing.flight_key: landing for landing in self.landings})

    def digest(self) -> str:
        """The identity of these landings (D63; D21: data by their flights, never by the roster's bytes): the sha256 of
        each landing's flight, runway and time, in time order, and of the number left out on the sealed test days."""
        rows = [[landing.flight_key, landing.runway, landing.time_s] for landing in self.landings]
        payload = json.dumps({"landings": rows, "sealed": self.sealed}, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def counts_before(self, time_s: np.ndarray, *, without: str) -> np.ndarray:
        """``[T, len(runways)]``: the landings on each candidate in ``[t − 30 min, t)`` at each UTC time ``time_s``, less
        the landing of the flight ``without`` (its own, which must be in the index)."""
        if without not in self._own:
            raise ValueError(f"{without} does not land in the index: a flight's own landing must be there")
        t = np.asarray(time_s, dtype=np.float64)
        out = np.zeros((len(t), len(self.runways)), dtype=np.int64)
        for k, runway in enumerate(self.runways):
            on = self._times[runway]
            out[:, k] = np.searchsorted(on, t, side="left") - np.searchsorted(on, t - LANDINGS_WINDOW_S, side="left")
        own = self._own[without]
        out[:, self.runways.index(own.runway)] -= (own.time_s < t) & (own.time_s >= t - LANDINGS_WINDOW_S)
        return out


def utc_s(value: str) -> float:
    return parse_utc(value).timestamp()


def roster_landings(records: Iterable[Mapping], runways: Iterable[str], days: DaySplit) -> LandingIndex:
    """The landings of a tracks roster's ``records`` (`tracks/manifest.json`'s) on ``runways`` (the airport's candidates,
    in their order), less the sealed test days (module docstring)."""
    wanted = tuple(runways)
    kept, sealed = [], 0
    for record in records:
        if record["outcome"] != "assigned" or record["runway"] not in wanted:
            continue
        if days.split_of(landing_day(record["landing_time_utc"])) == "test":
            sealed += 1
            continue
        kept.append(Landing(utc_s(record["landing_time_utc"]), record["runway"], record["flight_key"]))
    kept.sort(key=lambda landing: (landing.time_s, landing.flight_key))
    return LandingIndex(wanted, tuple(kept), sealed)


def read_roster_landings(tracks_manifest: Path, runways: Iterable[str], days: DaySplit) -> LandingIndex:
    """`roster_landings` of the tracks roster at ``tracks_manifest``."""
    return roster_landings(json.loads(Path(tracks_manifest).read_text(encoding="utf-8"))["records"], runways, days)
