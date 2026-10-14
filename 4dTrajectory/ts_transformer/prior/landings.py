"""The airport's landings before a step (prior design §6): the number of landings on each candidate runway in the 30 min
before the step. The prior reads no landings (D206); the post-training reward reads them for the airport's present
landing direction, and the prior's data identity binds them (D63).

Offline, the landings are the harvest's tracks roster of the airport (`repo_layout.tracks_manifest_path`): every
record the harvest assigned to a runway, whether or not its flight has a sentence — less every landing on a sealed test
day (contract C32: a test day is never counted as context). In a loop the caller gives each aircraft the landings the
loop knows (D105: a window's scene, built from the roster's `Landing`s); the index refuses a landing on a sealed test
day by itself, whoever builds it.
A flight never counts its own landing (`LandingIndex.counts_before`): the runway it landed on is the answer, and a
closed-loop sentence's last rows can fall after the time the observed aircraft landed.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ts_transformer.data.day_split import DaySplit, landing_day, operational_day, parse_utc

#: The window of the landings count, s.
LANDINGS_WINDOW_S = 30.0 * 60.0


@dataclass(frozen=True)
class Landing:
    time_s: float          # UTC epoch seconds
    runway: str            # the candidate's ident
    flight_key: str


@dataclass(frozen=True)
class LandingIndex:
    """One airport's landings on its candidates, sorted by time, then flight; ``sealed``: the test-day landings left out;
    ``days``: the day split, by which a landing on a sealed test day (or on a day outside the split) is refused (C32,
    D105)."""

    runways: tuple[str, ...]
    landings: tuple[Landing, ...]
    sealed: int
    days: DaySplit

    def __post_init__(self) -> None:
        order = [(landing.time_s, landing.flight_key) for landing in self.landings]
        if order != sorted(order):
            raise ValueError("the landings are not in time order, then flight order")
        for landing in self.landings:       # `SealedDay` on a test day, `KeyError` on a day outside the split
            self.days.development_split(operational_day(datetime.fromtimestamp(landing.time_s, timezone.utc)))
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

    def with_landing(self, landing: Landing) -> LandingIndex:
        """These landings and ``landing`` (a landing that a loop adds, multi-aircraft control D150, §6.3 item 5), in time
        order. A landing on a sealed test day is left out and counted with the sealed landings, as a window's shifted
        landing is (`post.landings.window_landings`; the user, 2026-10-07, stage D's requests item 6); the index's other
        checks hold: a landing on a day outside the split, a runway that is no candidate and a second landing of one
        flight are refused."""
        if landing.runway not in self.runways:
            raise ValueError(f"a landing on {landing.runway}, not a candidate of {self.runways}")
        if landing.flight_key in self._own:
            raise ValueError("a flight lands twice in the index")
        day = operational_day(datetime.fromtimestamp(landing.time_s, timezone.utc))
        if self.days.split_of(day) == "test":
            return self._built(self.landings, self.sealed + 1, self._times, self._own)
        # only the new landing is checked and put in its place: the others were checked when this index was built
        # (multi-aircraft control D186 (1); the same index as one built whole)
        at = bisect.bisect_left(self.landings, (landing.time_s, landing.flight_key),
                                key=lambda item: (item.time_s, item.flight_key))
        on = self._times[landing.runway]
        times = {**self._times, landing.runway: np.insert(on, np.searchsorted(on, landing.time_s, side="right"),
                                                          landing.time_s)}
        return self._built((*self.landings[:at], landing, *self.landings[at:]), self.sealed, times,
                           {**self._own, landing.flight_key: landing})

    def counted(self, first_s: float, end_s: float, keep: Iterable[str]) -> LandingIndex:
        """The landings that rows from ``first_s`` to before ``end_s`` count — from `LANDINGS_WINDOW_S` before
        ``first_s`` to before ``end_s`` (a row counts the 30 min before it) — and the landings of the flights ``keep``
        (each a flight's own, which `counts_before` asks for), with the same runways, days and sealed landings: at those
        rows, the same counts (multi-aircraft control D189 (0)). Its landings were checked when this index was built."""
        time = lambda item: item.time_s                                          # noqa: E731
        start = bisect.bisect_left(self.landings, first_s - LANDINGS_WINDOW_S, key=time)
        kept = {landing.flight_key: landing
                for landing in self.landings[start: bisect.bisect_left(self.landings, end_s, key=time)]}
        kept.update((key, self._own[key]) for key in keep)
        landings = tuple(sorted(kept.values(), key=lambda item: (item.time_s, item.flight_key)))
        times = {runway: np.array([landing.time_s for landing in landings if landing.runway == runway],
                                  dtype=np.float64) for runway in self.runways}
        return self._built(landings, self.sealed, times, {landing.flight_key: landing for landing in landings})

    def _built(self, landings: tuple[Landing, ...], sealed: int, times: dict[str, np.ndarray],
               own: dict[str, Landing]) -> LandingIndex:
        """An index of these runways and days with ``landings``, checked by the caller, and their lookups."""
        out = object.__new__(LandingIndex)
        for name, value in (("runways", self.runways), ("landings", landings), ("sealed", sealed), ("days", self.days),
                            ("_times", times), ("_own", own)):
            object.__setattr__(out, name, value)
        return out

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
    return LandingIndex(wanted, tuple(kept), sealed, days)


def read_roster_landings(tracks_manifest: Path, runways: Iterable[str], days: DaySplit) -> LandingIndex:
    """`roster_landings` of the tracks roster at ``tracks_manifest``."""
    return roster_landings(json.loads(Path(tracks_manifest).read_text(encoding="utf-8"))["records"], runways, days)
