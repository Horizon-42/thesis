"""The airport's landings on its candidate runways. No input of the prior (D206) and no loop (post-training D207, D208)
reads them. The prior's data identity binds their digest (D63), and the diagnostic readout of stage C reads a window's
(post-training D175, `post.landings.window_landings`).

The landings are the harvest's tracks roster of the airport (`repo_layout.tracks_manifest_path`): every record the
harvest assigned to a runway, whether or not its flight has a sentence — less every landing on a sealed test day
(contract C32). The index refuses a landing on a sealed test day by itself, whoever builds it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from ts_transformer.data.day_split import DaySplit, landing_day, operational_day, parse_utc


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
        keys = [landing.flight_key for landing in self.landings]
        if len(set(keys)) != len(keys):
            raise ValueError("a flight lands twice in the index")

    def digest(self) -> str:
        """The identity of these landings (D63; D21: data by their flights, never by the roster's bytes): the sha256 of
        each landing's flight, runway and time, in time order, and of the number left out on the sealed test days."""
        rows = [[landing.flight_key, landing.runway, landing.time_s] for landing in self.landings]
        payload = json.dumps({"landings": rows, "sealed": self.sealed}, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
