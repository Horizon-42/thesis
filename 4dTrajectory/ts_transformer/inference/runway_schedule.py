"""Arrival scheduling across runways (runway-intent R3, plan `docs/2026-09-13_runway_intent_plan.zh.md` §17).

Each arrival brings a belief over its candidate runways and an ETA under each; the scheduler gives it a
(runway, landing time) first-come-first-served by ETA, scoring a runway ``log p - lambda * delay`` and
placing it at the earliest time every separation rule allows against the arrivals already placed.
The separation rules live here once (`Separation`), read by the scheduler and by the checks of the
schedule and of the flown references.

Separation is a minimum TIME between a leader and a follower on the APPROACH CLOCK, derived from a
distance at a nominal approach speed: on one runway — or a parallel pair close enough to count as one —
the larger of the radar minimum and the wake minimum for the leader/follower categories; on dependent
parallels, the along-track stagger that keeps the diagonal separation (``sqrt(D^2 - s^2)`` for a diagonal
minimum D and a centerline spacing s); independent parallels and runways of another direction impose
none (crossing-runway operations are not modelled). `faa_separation` builds it from the FAA order, every
value cited to its paragraph. Pure numpy / stdlib: no torch, no data plane.

The approach clock (`Separation.approach_time_s`) is a landing's threshold time less its threshold's
position along the course flown at the approach speed: the time the aircraft passes abeam a common
point. Two parallel thresholds are not level — KRDU 23L sits 0.67 NM further along than 23R, KSTL 11
1.5-2 NM from 12L / 12R — so two threshold times say nothing about where the two aircraft were
relative to each other; on the approach clock their difference times the speed IS their along-track
separation, and every minimum is applied there.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from geokit import FT_M, METRES_PER_DEG_LAT, NM_M, metres_per_deg_lon

#: How far under a minimum two times may be and still keep it: the times are wall clocks (~1.8e9 s,
#: float64 resolution ~2e-7 s), so a slot placed exactly at its minimum can read a few 1e-7 s short.
TIME_EPS_S = 1e-6

SAME = "same"                  # one runway
SINGLE = "single"              # a parallel pair close enough to be separated as one runway
DEPENDENT = "dependent"        # dependent (staggered) parallel approaches
INDEPENDENT = "independent"    # independent parallel approaches: no mutual minimum
UNRELATED = "unrelated"        # another direction (crossing / converging operations are not modelled)


@dataclass(frozen=True)
class ParallelRegime:
    """What a centerline spacing band means for two parallel finals: the relation, and the diagonal
    minimum on dependent approaches."""

    below_ft: float
    relation: str
    diagonal_nm: float = 0.0


@dataclass(frozen=True)
class Separation:
    """The one definition of the minimum time between two landings (`gap_s`).

    ``relations`` maps an unordered runway pair to its relation and ``spacing_nm`` to its centerline
    spacing; ``same_nm`` is the radar minimum on one runway; ``wake_nm`` the in-trail wake minimum by
    (leader category, follower category), applied only on one runway or a pair separated as one;
    ``speed_mps`` converts a distance to the time it takes the follower to close it; ``along_nm`` is
    each runway's threshold position along its course from a common origin (a runway not in it sits
    at 0), which puts every landing on the approach clock."""

    same_nm: float
    speed_mps: float
    relations: Mapping[frozenset, str] = field(default_factory=dict)
    spacing_nm: Mapping[frozenset, float] = field(default_factory=dict)
    diagonal_nm: Mapping[frozenset, float] = field(default_factory=dict)
    wake_nm: Mapping[tuple[str, str], float] = field(default_factory=dict)
    along_nm: Mapping[str, float] = field(default_factory=dict)

    def approach_time_s(self, runway: str, time_s: float) -> float:
        """A threshold time on the approach clock: when the aircraft passed abeam the common origin."""
        return time_s - self.along_nm.get(runway, 0.0) * NM_M / self.speed_mps

    def approach_s(self, slot: "Slot") -> float:
        return self.approach_time_s(slot.runway, slot.time_s)

    def relation(self, a: str, b: str) -> str:
        return SAME if a == b else self.relations.get(frozenset((a, b)), UNRELATED)

    def one_runway(self, a: str, b: str) -> bool:
        """Separated as one runway: the same runway, or a parallel pair too close to be two."""
        return self.relation(a, b) in (SAME, SINGLE)

    @property
    def max_gap_s(self) -> float:
        """The largest minimum any pair can need: no landing further apart ON THE APPROACH CLOCK
        constrains another."""
        staggers = [math.sqrt(max(d * d - self.spacing_nm[pair] ** 2, 0.0)) for pair, d in self.diagonal_nm.items()]
        return max([self.same_nm, *self.wake_nm.values(), *staggers]) * NM_M / self.speed_mps

    def distance_nm(self, leader_runway: str, leader_category: str, follower_runway: str, follower_category: str) -> float:
        relation = self.relation(leader_runway, follower_runway)
        if relation in (SAME, SINGLE):
            return max(self.same_nm, self.wake_nm.get((leader_category, follower_category), 0.0))
        if relation == DEPENDENT:
            pair = frozenset((leader_runway, follower_runway))
            d, s = self.diagonal_nm[pair], self.spacing_nm[pair]
            return math.sqrt(max(d * d - s * s, 0.0))
        return 0.0

    def gap_s(self, leader_runway: str, leader_category: str, follower_runway: str, follower_category: str) -> float:
        return self.distance_nm(leader_runway, leader_category, follower_runway, follower_category) * NM_M / self.speed_mps


def parallel_relations(
    targets: Mapping[str, Mapping[str, float]], regimes: Sequence[ParallelRegime], *, max_course_diff_deg: float = 10.0,
) -> tuple[dict[frozenset, str], dict[frozenset, float], dict[frozenset, float], dict[str, float]]:
    """``(relations, spacing_nm, diagonal_nm, along_nm)``: every pair of runways with (nearly) the same
    inbound course, its centerline spacing and its relation under ``regimes`` (sorted by ``below_ft``; a
    spacing at or above the last band is independent), and every runway's threshold position along its
    own course from the first runway's threshold. ``targets``: runway -> ``{lat, lon, course_deg}``, the
    arrivals manifest's (the course a COMPASS bearing: 0 = north, clockwise)."""
    relations, spacing, diagonal = {}, {}, {}
    names = sorted(targets)
    bands = sorted(regimes, key=lambda band: band.below_ft)
    origin = targets[names[0]]
    along = {}
    for name in names:
        t = targets[name]
        east = (float(t["lon"]) - float(origin["lon"])) * metres_per_deg_lon(float(origin["lat"]))
        north = (float(t["lat"]) - float(origin["lat"])) * METRES_PER_DEG_LAT
        course = math.radians(float(t["course_deg"]))
        along[name] = (east * math.sin(course) + north * math.cos(course)) / NM_M
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ta, tb = targets[a], targets[b]
            if abs((float(ta["course_deg"]) - float(tb["course_deg"]) + 180.0) % 360.0 - 180.0) > max_course_diff_deg:
                continue
            east = (float(tb["lon"]) - float(ta["lon"])) * metres_per_deg_lon(float(ta["lat"]))
            north = (float(tb["lat"]) - float(ta["lat"])) * METRES_PER_DEG_LAT
            course = math.radians(float(ta["course_deg"]))
            spacing_m = abs(east * math.cos(course) - north * math.sin(course))
            pair = frozenset((a, b))
            spacing[pair] = spacing_m / NM_M
            band = next((band for band in bands if spacing_m / FT_M < band.below_ft), None)
            relations[pair] = INDEPENDENT if band is None else band.relation
            if band is not None and band.relation == DEPENDENT:
                diagonal[pair] = band.diagonal_nm
    return relations, spacing, diagonal, along


# ── The FAA arrival minima ──────────────────────────────────────────────────────────────────────────
# FAA Order JO 7110.65BB Change 3 (2026-07-09), quoted with its paragraph numbers in the repo's
# `docs/literature/arrival_separation/README.md` (§2 the text, §7 the constraint table this block
# encodes). What is the text and what is our reading of it:
# - one runway: the terminal radar minimum, 3 NM (5-5-4 a/b), and the wake minimum of TBL 5-5-2 that
#   must exist when the leader is over the threshold (5-5-4 h); a blank cell of the table sets none.
#   The 2.5 NM of 5-5-4 j needs an authorization (documented ROT <= 50 s, CTRDs) none of our airports
#   was checked for, so it is an option (``radar_nm``), not the default.
# - parallels under 2,500 ft are one runway (5-5-4 h NOTE, wake; applying the RADAR minimum across the
#   pair too is our reading, the AIM's "single runway separation", 5-4-14 e). JO 7110.308E lets KSTL's
#   12/30 pairs reduce to a 1.0 NM diagonal behind an F-I leader; its current use at STL is unverified
#   and it is not applied.
# - 2,500-3,600 ft: dependent, 1.0 NM diagonal (5-9-6 a2); 3,600-4,300 ft: dependent, 1.5 NM (a3) —
#   5-9-7 a2 allows independent approaches from 3,600 ft, but under 4,300 ft only with FMA and PRM
#   approaches (5-9-7 c1, 5-9-8 b), which we do not assume; from 4,300 ft independent (5-9-7 a2; FMA
#   not required at a field elevation of 2,000 ft or less, c NOTE) — assuming the charts authorize
#   simultaneous approaches, which was not checked.
# - intersecting runways (3-10-4: the threshold gated on the other arrival passing the intersection)
#   are not modelled: they read as unrelated.
# - a distance becomes a time at the airport's approach ground speed (the minimum is a distance to go
#   when the leader crosses, 5-5-4 h, so it is the follower's time to fly it; Erzberger & Itoh §2.2),
#   between the two landings on the approach clock (the thresholds' along-course stagger taken out).
#   The order's 3,600 ft is inclusive ("no more than 3,600 feet"); the bands are half-open, which only
#   differs at a spacing of exactly 3,600 ft.
FAA_RADAR_NM = 3.0
FAA_REDUCED_RADAR_NM = 2.5      # 5-5-4 j, by authorization only
FAA_PARALLEL_REGIMES = (
    ParallelRegime(2500.0, SINGLE),              # 5-5-4 h NOTE
    ParallelRegime(3600.0, DEPENDENT, 1.0),      # 5-9-6 a2
    ParallelRegime(4300.0, DEPENDENT, 1.5),      # 5-9-6 a3 (independent needs FMA + PRM below 4,300 ft)
)                                                # >= 4,300 ft: independent, 5-9-7 a2
#: TBL 5-5-2 "Wake Turbulence Separation for On Approach" (5-5-4 h), NM, (leader, follower) CWT category.
CWT_ON_APPROACH_NM: dict[tuple[str, str], float] = {
    ("A", "B"): 5.0, ("A", "C"): 6.0, ("A", "D"): 6.0, ("A", "E"): 7.0, ("A", "F"): 7.0, ("A", "G"): 7.0,
    ("A", "H"): 8.0, ("A", "I"): 8.0,
    ("B", "B"): 3.0, ("B", "C"): 4.0, ("B", "D"): 4.0, ("B", "E"): 5.0, ("B", "F"): 5.0, ("B", "G"): 5.0,
    ("B", "H"): 5.0, ("B", "I"): 6.0,
    ("C", "E"): 3.5, ("C", "F"): 3.5, ("C", "G"): 3.5, ("C", "H"): 5.0, ("C", "I"): 6.0,
    ("D", "B"): 3.0, ("D", "C"): 4.0, ("D", "D"): 4.0, ("D", "E"): 5.0, ("D", "F"): 5.0, ("D", "G"): 5.0,
    ("D", "H"): 6.0, ("D", "I"): 6.0,
    ("E", "I"): 4.0,
    ("F", "I"): 4.0,
}
#: The CWT column of JO 7360.1K Appendix A for every type in the R2b rosters (the full parsed table:
#: `docs/literature/arrival_separation/papers/FAA_JO_7360.1K_AppendixA_categories_parsed.csv`).
CWT_BY_TYPECODE: dict[str, str] = {
    "A20N": "F", "A21N": "F", "A319": "F", "A320": "F", "A321": "F", "A332": "B", "A333": "B", "A359": "B",
    "B734": "F", "B737": "F", "B738": "F", "B739": "F", "B38M": "F", "B39M": "F", "B752": "E", "B763": "C",
    "B772": "B", "B788": "B", "B789": "B", "C550": "I", "CRJ9": "G", "E170": "G", "E190": "F", "E75L": "G",
    "GLF6": "F",
}


def wake_category(typecode: str | None) -> str:
    """The CWT category of an ICAO type designator (JO 7360.1K Appendix A); a type not listed raises."""
    if typecode not in CWT_BY_TYPECODE:
        raise KeyError(f"no CWT category for {typecode!r}: add it from JO 7360.1K Appendix A "
                       "(docs/literature/arrival_separation/papers/FAA_JO_7360.1K_AppendixA_categories_parsed.csv)")
    return CWT_BY_TYPECODE[typecode]


def faa_separation(
    targets: Mapping[str, Mapping[str, float]], *, speed_mps: float, radar_nm: float = FAA_RADAR_NM,
    visual_parallels: bool = False,
) -> Separation:
    """The FAA IFR arrival minima for one airport's runways at one approach speed (the block above).
    ``visual_parallels`` is the visual-approach reading instead: no minimum between two parallel
    runways at all (7-4-4 c, pilots maintain visual separation once the leader is on its centreline);
    each runway keeps its own radar and wake minima."""
    relations, spacing, diagonal, along = parallel_relations(targets, FAA_PARALLEL_REGIMES)
    if visual_parallels:
        relations = {pair: INDEPENDENT for pair in relations}
        diagonal = {}
    return Separation(same_nm=radar_nm, speed_mps=speed_mps, relations=relations, spacing_nm=spacing,
                      diagonal_nm=diagonal, wake_nm=CWT_ON_APPROACH_NM, along_nm=along)


@dataclass(frozen=True)
class Arrival:
    """One arrival as the scheduler sees it: an ETA (wall clock, s) and a log probability per candidate
    runway (the same runways in both), and its wake category."""

    key: str
    etas: Mapping[str, float]
    logp: Mapping[str, float]
    category: str

    def __post_init__(self) -> None:
        if set(self.etas) != set(self.logp):
            raise ValueError(f"{self.key}: ETAs for {sorted(self.etas)} but probabilities for {sorted(self.logp)}")


@dataclass(frozen=True)
class Slot:
    key: str
    runway: str
    time_s: float
    eta_s: float
    category: str

    @property
    def delay_s(self) -> float:
        return self.time_s - self.eta_s


def earliest_time(placed: Sequence[Slot], runway: str, category: str, eta_s: float, separation: Separation) -> float:
    """The earliest threshold time at or after ``eta_s`` a landing on ``runway`` keeps every minimum
    against the ``placed`` landings (sorted by `Separation.approach_s`), as leader and as follower. On
    the approach clock the feasible set is ``[eta, inf)`` minus one open interval per placed landing, so
    its earliest point is ``eta`` or the end of some interval; only landings within
    `Separation.max_gap_s` of a time can exclude it."""
    horizon = separation.max_gap_s
    offset = separation.approach_time_s(runway, 0.0)         # approach clock = threshold time + offset

    def keeps(tau: float) -> bool:
        lo = bisect.bisect_left(placed, tau - horizon, key=separation.approach_s)
        hi = bisect.bisect_right(placed, tau + horizon, key=separation.approach_s)
        for s in placed[lo:hi]:
            other = separation.approach_s(s)
            if tau >= other:
                need = separation.gap_s(s.runway, s.category, runway, category)
                if need > 0.0 and tau - other < need - TIME_EPS_S:
                    return False
            else:
                need = separation.gap_s(runway, category, s.runway, s.category)
                if need > 0.0 and other - tau < need - TIME_EPS_S:
                    return False
        return True

    tau_eta = eta_s + offset
    start = bisect.bisect_left(placed, tau_eta - horizon, key=separation.approach_s)
    candidates = [tau_eta] + [separation.approach_s(s) + separation.gap_s(s.runway, s.category, runway, category)
                              for s in placed[start:]]
    for tau in sorted(c for c in candidates if c >= tau_eta):
        if keeps(tau):
            return tau - offset
    raise AssertionError("the end of the latest interval is always feasible")


def eligible_runways(arrival: Arrival, min_probability: float) -> list[str]:
    """The runways the scheduler tries for an arrival: its most probable one, and every other whose
    probability reaches ``min_probability``."""
    if not 0.0 < min_probability <= 1.0:
        raise ValueError(f"min_probability must be in (0, 1], got {min_probability!r}")
    top = max(arrival.logp, key=lambda r: (arrival.logp[r], r))
    floor = math.log(min_probability)
    return sorted(r for r in arrival.logp if r == top or arrival.logp[r] >= floor)


def fcfs_by_eta(arrivals: Sequence[Arrival], min_probability: float) -> list[Arrival]:
    """First come, first served: the arrivals in the order of their earliest ETA over the runways the
    scheduler will try (`eligible_runways`; the key breaks ties)."""
    return sorted(arrivals, key=lambda a: (min(a.etas[r] for r in eligible_runways(a, min_probability)), a.key))


def schedule(
    arrivals: Sequence[Arrival], separation: Separation, *, delay_weight_per_s: float, min_probability: float,
) -> list[Slot]:
    """Place the arrivals one at a time IN THE ORDER GIVEN (`fcfs_by_eta` is the plan's order), each
    frozen once placed: every eligible runway (`eligible_runways`) is placed at its earliest feasible
    time, scored ``log p - delay_weight_per_s * delay``; the best score wins, a tie going to the more
    probable runway. Returns one slot per arrival, in the order they were placed."""
    placed: list[Slot] = []
    timeline: list[Slot] = []          # the placed slots sorted on the approach clock, for `earliest_time`
    for arrival in arrivals:
        best: tuple[float, float, Slot] | None = None
        for runway in eligible_runways(arrival, min_probability):
            eta = arrival.etas[runway]
            t = earliest_time(timeline, runway, arrival.category, eta, separation)
            score = arrival.logp[runway] - delay_weight_per_s * (t - eta)
            slot = Slot(arrival.key, runway, t, eta, arrival.category)
            if best is None or (score, arrival.logp[runway]) > (best[0], best[1]):
                best = (score, arrival.logp[runway], slot)
        placed.append(best[2])
        bisect.insort(timeline, best[2], key=lambda s: (separation.approach_s(s), s.key))
    return placed


def violations(slots: Sequence[Slot], separation: Separation, *, tolerance_s: float = 0.0) -> list[tuple[Slot, Slot, float]]:
    """Every (leader, follower) pair — in order on the approach clock — closer than its minimum less
    ``tolerance_s`` (and `TIME_EPS_S`): ``(leader, follower, shortfall_s)``. A plan the scheduler made
    has none; flown or unscheduled landings may."""
    ordered = sorted(slots, key=lambda s: (separation.approach_s(s), s.key))
    clock = [separation.approach_s(s) for s in ordered]
    horizon = separation.max_gap_s
    out = []
    for j, follower in enumerate(ordered):
        i = j - 1
        while i >= 0 and clock[j] - clock[i] < horizon:
            leader = ordered[i]
            need = separation.gap_s(leader.runway, leader.category, follower.runway, follower.category)
            short = need - (clock[j] - clock[i])
            if need > 0.0 and short > tolerance_s + TIME_EPS_S:
                out.append((leader, follower, short))
            i -= 1
    return out
