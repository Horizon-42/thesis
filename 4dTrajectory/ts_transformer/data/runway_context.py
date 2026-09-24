"""Causal runway context: what an observer at time ``t`` knows about the airport's runway use.

Runway-intent plan R0 (`docs/2026-09-13_runway_intent_plan.zh.md` §5, §7): the causal baseline
rules a learned runway head has to beat. Every answer is built from events strictly BEFORE ``t``
— landings that have already happened, METAR reports already issued — so no rule here reads the
flight's own future, and the caller decides which flights may be context at all (the runner
keeps every flight whose split hash is outer-test out of the pool, labels included).

Two levels, because they are two different decisions (plan §2): the DIRECTION GROUP (the
airport's configuration — which way it lands; runways whose inbound courses are within
`DIRECTION_GROUP_DEG` are one group) and the SIDE (which runway inside the group — the
per-aircraft assignment). `direction_groups` is that partition, once.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from flight_scenarios.identity import flight_key
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from ts_transformer.data.dataset import load_flight_dicts
from ts_transformer.data.day_split import parse_utc

#: Inbound courses this close are one landing direction (the parallel-sibling rule of
#: `experiments.runway_hypotheses.parallel_sibling`).
DIRECTION_GROUP_DEG = 30.0
#: A runway whose inbound course is further than this from the aircraft's track is not being
#: flown to (the 2026-09-03 course gate).
COURSE_GATE_DEG = 90.0
#: The entry ring is cut into this many bearing sectors (45 deg each).
ENTRY_SECTORS = 8
#: B4 chooses only among direction groups that carry at least this share of the airport's
#: majority landings: the runway with the most headwind is often one the airport barely uses
#: (KMSY 20: 2 of 4,147 arrivals), and a wind rule blind to that measures the runway's
#: existence, not the configuration (first KMSY readout, 2026-09-13: 55 % against B1's 97 %).
WIND_MIN_GROUP_SHARE = 0.05

RULES = (
    "B0_majority",
    "B1_active_config",
    "B2_active_config_gated",
    "B3_same_sector_last",
    "B4_wind",
)


def wrap_deg(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def direction_groups(courses: Mapping[str, float]) -> dict[str, int]:
    """Runway -> direction group: connected components of "inbound courses within
    `DIRECTION_GROUP_DEG`", numbered in sorted-runway order so the ids are stable."""
    runways = sorted(courses)
    group: dict[str, int] = {}
    for runway in runways:
        if runway in group:
            continue
        group[runway] = len(set(group.values()))
        stack = [runway]
        while stack:
            current = stack.pop()
            for other in runways:
                if other not in group and abs(wrap_deg(courses[other] - courses[current])) <= DIRECTION_GROUP_DEG:
                    group[other] = group[runway]
                    stack.append(other)
    return group


def course_deg(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    """Track course from one position to the next, degrees true in [0, 360)."""
    east = (lon1 - lon0) * metres_per_deg_lon(0.5 * (lat0 + lat1))
    north = (lat1 - lat0) * METRES_PER_DEG_LAT
    return math.degrees(math.atan2(east, north)) % 360.0


def bearing_sector(lon: float, lat: float, ref_lon: float, ref_lat: float) -> int:
    """Which of the `ENTRY_SECTORS` bearing sectors (from the reference point) a position is in."""
    bearing = course_deg(ref_lon, ref_lat, lon, lat)
    return int(bearing // (360.0 / ENTRY_SECTORS)) % ENTRY_SECTORS


@dataclass(frozen=True)
class ContextLanding:
    time: datetime
    runway: str
    #: The entry sector of the landing flight, where its track is known; None otherwise
    #: (a landing seen only in the tracks roster has a time and a runway, no arrival slice).
    sector: int | None = None


@dataclass(frozen=True)
class WindReport:
    valid: datetime
    direction_deg: float | None     # true; None when missing or variable
    speed_kt: float | None


def load_metar(paths: Iterable[Path]) -> list[WindReport]:
    """IEM ASOS rows (``station, valid, drct, sknt, gust``; UTC, degrees true, knots; ``M`` =
    missing), sorted by ``valid``. A missing field stays None — never read as calm."""
    reports: list[WindReport] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                valid = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                direction = None if row["drct"] in ("M", "") else float(row["drct"])
                speed = None if row["sknt"] in ("M", "") else float(row["sknt"])
                reports.append(WindReport(valid, direction, speed))
    return sorted(reports, key=lambda report: report.valid)


@dataclass(frozen=True)
class Pick:
    runway: str
    #: The rule's own evidence was missing (no landing in its window, calm or missing wind, no
    #: runway inside the course gate) and it answered with its stated fallback instead.
    fallback: bool


class RunwayContext:
    """The causal baseline rules over one airport's context pool.

    ``landings`` — every context landing (the caller has already removed what may not be read);
    ``majority`` — the per-runway counts the static majority is taken from (the runner passes
    TRAIN-split landings only, so the majority carries no validation label);
    ``winds`` — METAR reports; a report counts once ``valid + metar_delay <= t``.
    """

    def __init__(
        self,
        landings: Sequence[ContextLanding],
        courses: Mapping[str, float],
        majority: Counter,
        winds: Sequence[WindReport],
        *,
        window: timedelta,
        sector_window: timedelta,
        metar_delay: timedelta,
        calm_kt: float,
    ) -> None:
        self.courses = dict(courses)
        self.groups = direction_groups(courses)
        ordered = sorted(landings, key=lambda landing: landing.time)
        self._landings = ordered
        self._times = [landing.time for landing in ordered]
        self._majority = Counter({runway: majority.get(runway, 0) for runway in courses})
        self._winds = list(winds)
        self._wind_times = [report.valid for report in self._winds]
        self.window = window
        self.sector_window = sector_window
        self.metar_delay = metar_delay
        self.calm_kt = calm_kt

    # ── the evidence ──────────────────────────────────────────────────────────────────────
    def recent(self, t: datetime, window: timedelta) -> list[ContextLanding]:
        """Landings in ``[t - window, t)`` — strictly before ``t``."""
        lo = bisect.bisect_left(self._times, t - window)
        hi = bisect.bisect_left(self._times, t)
        return self._landings[lo:hi]

    def wind_at(self, t: datetime) -> WindReport | None:
        """The latest report already issued at ``t`` (``valid + metar_delay <= t``)."""
        index = bisect.bisect_right(self._wind_times, t - self.metar_delay) - 1
        return self._winds[index] if index >= 0 else None

    def _most_common(self, counts: Counter, among: Iterable[str]) -> str | None:
        pool = [runway for runway in among if counts.get(runway, 0) > 0]
        # Deterministic: count first, then runway name.
        return min(pool, key=lambda runway: (-counts[runway], runway)) if pool else None

    def with_majority(self, majority: Counter) -> "RunwayContext":
        """The same pool read with another static majority — a split whose training days differ
        must not take its majority from the others' labels (R1 review, 2026-09-13)."""
        return RunwayContext(
            self._landings, self.courses, majority, self._winds, window=self.window,
            sector_window=self.sector_window, metar_delay=self.metar_delay, calm_kt=self.calm_kt,
        )

    @property
    def majority_counts(self) -> Counter:
        """The per-runway counts the static majority is taken from (a copy)."""
        return Counter(self._majority)

    def majority_among(self, among: Iterable[str]) -> str:
        pool = sorted(among)
        return min(pool, key=lambda runway: (-self._majority[runway], runway))

    # ── the rules ─────────────────────────────────────────────────────────────────────────
    def picks(self, t: datetime, *, sector: int, track_course_deg: float) -> dict[str, Pick]:
        runways = sorted(self.courses)
        majority = self.majority_among(runways)
        recent = Counter(landing.runway for landing in self.recent(t, self.window))
        active = self._most_common(recent, runways)
        b1 = Pick(active, False) if active else Pick(majority, True)

        gated = [r for r in runways if abs(wrap_deg(self.courses[r] - track_course_deg)) <= COURSE_GATE_DEG]
        if not gated:
            b2 = Pick(b1.runway, True)
        else:
            gated_active = self._most_common(recent, gated)
            b2 = Pick(gated_active, False) if gated_active else Pick(self.majority_among(gated), True)

        same_sector = [
            landing for landing in self.recent(t, self.sector_window) if landing.sector == sector
        ]
        b3 = Pick(same_sector[-1].runway, False) if same_sector else Pick(b1.runway, True)

        wind = self.wind_at(t)
        if wind is None or wind.direction_deg is None or wind.speed_kt is None or wind.speed_kt < self.calm_kt:
            b4 = Pick(b1.runway, True)
        else:
            def headwind(group: int) -> float:
                members = [r for r in runways if self.groups[r] == group]
                course = self.courses[members[0]]
                return wind.speed_kt * math.cos(math.radians(wind.direction_deg - course))

            total = sum(self._majority.values())
            share = Counter()
            for runway, count in self._majority.items():
                share[self.groups[runway]] += count / total if total else 0.0
            in_use = [g for g in sorted(set(self.groups.values())) if share[g] >= WIND_MIN_GROUP_SHARE]
            best = max(in_use or sorted(set(self.groups.values())), key=headwind)
            members = [r for r in runways if self.groups[r] == best]
            in_group = self._most_common(recent, members)
            b4 = Pick(in_group or self.majority_among(members), in_group is None)

        return {
            "B0_majority": Pick(majority, False),
            "B1_active_config": b1,
            "B2_active_config_gated": b2,
            "B3_same_sector_last": b3,
            "B4_wind": b4,
        }


# ── one airport's context pool: what may be read, once ─────────────────────────────────────

@dataclass(frozen=True)
class AirportContext:
    rules: RunwayContext
    #: flight_key -> every arrival whose split hash is not outer-test (the runners take their
    #: validation flights from here, so the pool and the scored flights come from one load).
    flights: dict[str, dict[str, Any]]
    sectors: dict[str, int]
    landings: tuple[ContextLanding, ...]
    excluded_outer_test: int
    metar_paths: tuple[Path, ...]


def airport_reference(targets: Mapping[str, Mapping[str, Any]]) -> tuple[float, float]:
    """The point entry sectors are measured from: the centroid of the runway thresholds."""
    return (
        sum(float(t["lon"]) for t in targets.values()) / len(targets),
        sum(float(t["lat"]) for t in targets.values()) / len(targets),
    )


def build_airport_context(
    manifest_path: Path,
    tracks_path: Path,
    metar_paths: Sequence[Path],
    courses: Mapping[str, float],
    split_of: Callable[[str], str],
    *,
    window: timedelta,
    sector_window: timedelta,
    metar_delay: timedelta,
    calm_kt: float,
) -> AirportContext:
    """The context pool of runway-intent R0: the tracks roster's assigned landings on a
    candidate runway (time + runway) and the arrivals roster's entry sectors, minus EVERY
    flight ``split_of`` calls outer-test — its label is never counted and its track never
    loaded, whether or not it is in the eligible roster. The static majority counts
    train-hash landings only."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    airport = str(manifest["airport"]).upper()
    ref_lon, ref_lat = airport_reference(manifest["runway_targets"])
    readable = [r["flight_key"] for r in manifest["records"] if split_of(r["flight_key"]) != "test"]
    loaded = load_flight_dicts(
        manifest_path, include_flight_keys={f"{airport}:{key}" for key in readable}, verbose=False,
    )
    flights = {flight_key(flight, 0): flight for flight in loaded}
    sectors = {
        key: bearing_sector(flight["waypoints"][0][1], flight["waypoints"][0][2], ref_lon, ref_lat)
        for key, flight in flights.items()
    }
    landings: list[ContextLanding] = []
    majority: Counter = Counter()
    excluded = 0
    for row in json.loads(Path(tracks_path).read_text(encoding="utf-8"))["records"]:
        if row.get("outcome") != "assigned" or row.get("runway") not in courses:
            continue
        split = split_of(row["flight_key"])
        if split == "test":
            excluded += 1
            continue
        landings.append(ContextLanding(
            parse_utc(row["landing_time_utc"]), row["runway"], sectors.get(row["flight_key"]),
        ))
        majority[row["runway"]] += split == "train"
    rules = RunwayContext(
        landings, courses, majority, load_metar(metar_paths),
        window=window, sector_window=sector_window, metar_delay=metar_delay, calm_kt=calm_kt,
    )
    return AirportContext(rules, flights, sectors, tuple(landings), excluded, tuple(metar_paths))
