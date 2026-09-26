"""The procedure's altitudes as hard masks on the prior's altitude and descent-angle columns (post-training design §3).

For the runway in force (post-training design §3.2–§3.3):

- **the glidepath lower edge** — inside the FAF and inside the LPV cone (`flight_scenarios.fas_geometry`; `inside`) the
  floor is the published glidepath less `GLIDEPATH_BELOW_M`, the optimizer's lower edge, which also binds only inside the
  FAF (the RNAV altitudes coded outside it disagree with 10–14 % of the recorded tracks — design §7, readouts §8.1);
- **the join** is the first row inside that region; before it,
- **the decision altitude** — the runway's published DA (the plate's, `Runway.decision_height_above_threshold_m`) is a
  floor; and
- **no climbing back** — once a row (the observed ones included) has been lower than the entry height (the glidepath at
  the FAF, `entry_m`) by more than the word tolerance, the aircraft climbs no more before the join, unless a go-around is
  in force (`pre_join`, `climb_barred`).

The word rules (§3.4; the word tolerance is half an altitude step, so the step holding a bound stays sayable):

1. inside the region an altitude target T is allowed where ``T ≥ floor − word tolerance``. The executor holds T less the
   tube's margin during and after the descent, and the glidepath only falls toward the runway, so a target allowed here
   stays allowed inbound;
2. "descend to land" is not allowed where the aircraft is inside the region and already more than the word tolerance
   below the floor (no look-ahead: a straight line to the threshold forbade 4.2 % of the labelled words whose aircraft
   never went below);
3. before the join a level is allowed where ``T ≥ DA − word tolerance``;
4. where the climb is barred a level is allowed where ``T ≤ height + word tolerance``, and the descent-angle column may
   not say the climb class ("descend to land" stays: the executor levels under the glidepath's extension, never climbs);
5. "unchanged" (the altitude column) is not allowed where the word in force fails a rule that applies here.

A flown row more than the track tolerance (the word tolerance plus the tube's margin) below the glidepath lower edge ends
the flight (§3.5); rules 3 and 4 need no such check (the executor holds a level to its tube, and under "descend to land"
it keeps above the glidepath's extension less the edge), and are read out instead (`pre_join_readout`).

The FAF is the coded RNAV(GPS) approach's (`flight_scenarios.procedure_final.procedure_skeleton`); the glidepath is the
runway's published one (the harvest's runway data — threshold crossing height and glidepath angle, the executor's
crossing point, `autopilot.runway_data`); the DA is the harvest's too. Positions are the airport frame
(`instructions.airport`), heights m MSL. Torch-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from flight_scenarios.fas_geometry import FasCourseGeometry, course_halfwidth_m, fas_course_geometry
from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT, procedure_skeleton
from trajectory_data_process.harvest.airports import Runway, load_airport
from ts_transformer.instructions.airport import AirportGeometry, RunwayCandidate, relative_to_runway
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import APPROACH_GO_AROUND, APPROACH_NOT_CLEARED, Words

# MIRRORS of the optimizer's constants (its packages are not on this one's import path);
# `tests/test_prior_procedure.py` asserts each equal to its owner.
#: How far below the glidepath the final may go, m: `approach_constraints.segments.DEFAULT_GLIDEPATH_BELOW_M`.
GLIDEPATH_BELOW_M = 60.0
#: How far a document's threshold may sit from the candidate's: `scenario_optimization._FRAME_ANCHOR_TOLERANCE_M`.
THRESHOLD_TOLERANCE_M = 150.0


def word_tolerance_m(spec: VocabularySpec) -> float:
    """Half an altitude step: the step whose rounding bin holds the floor stays sayable."""
    return spec.altitude_step_m / 2.0


def track_tolerance_m(spec: VocabularySpec) -> float:
    """The word tolerance plus the tube's margin: how far below its floor a flown row may be."""
    return word_tolerance_m(spec) + spec.altitude_tolerance_m


@dataclass(frozen=True)
class RunwayProcedure:
    """One candidate runway's final: where the glidepath binds and how high it is."""

    candidate: RunwayCandidate      # the threshold and the course the final is measured from
    crossing_m: float               # the glidepath at the threshold: its elevation + the published TCH, m MSL
    glidepath_tan: float
    faf_d_m: float                  # the FAF's distance to go along the course
    cone: FasCourseGeometry
    decision_m: float               # the published decision altitude, m MSL

    @property
    def ident(self) -> str:
        return self.candidate.ident

    def axes(self, e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(d, xt)``: the distance to go along the course (+ before the threshold) and the distance off it — the
        words' own reading (`instructions.airport.relative_to_runway`; its track and height parts unused)."""
        relative = relative_to_runway(e, n, 0.0, 0.0, self.candidate)
        return relative.before_threshold_m, np.abs(relative.right_of_course_m)

    @property
    def entry_m(self) -> float:
        """The entry height: the glidepath at the FAF, m MSL."""
        return self.crossing_m + self.faf_d_m * self.glidepath_tan

    def inside(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Whether positions ``(e, n)`` are inside the FAF and the LPV cone: where the glidepath lower edge binds, and
        where an aircraft has joined the final."""
        d, xt = self.axes(e, n)
        return (d >= 0.0) & (d <= self.faf_d_m) & (xt <= course_halfwidth_m(d, self.cone))

    def floor_m(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        """The floor at positions ``(e, n)``, m MSL: the glidepath less `GLIDEPATH_BELOW_M` inside the FAF and the cone,
        NaN elsewhere."""
        d, _ = self.axes(e, n)
        return np.where(self.inside(e, n), self.crossing_m + d * self.glidepath_tan - GLIDEPATH_BELOW_M, np.nan)


def pre_join(procedure: RunwayProcedure, e: np.ndarray, n: np.ndarray, h: np.ndarray, spec: VocabularySpec
             ) -> tuple[np.ndarray, np.ndarray]:
    """``(joined, dipped)`` at every row of one flight's rows so far (row 0 first): joined — this row or an earlier one
    is inside the region (`RunwayProcedure.inside`); dipped — this row or an earlier one before the join is lower than
    the entry height less the word tolerance."""
    joined = np.logical_or.accumulate(procedure.inside(e, n))
    low = (np.asarray(h, dtype=np.float64) < procedure.entry_m - word_tolerance_m(spec)) & ~joined
    return joined, np.logical_or.accumulate(low)


def climb_barred(joined: np.ndarray, dipped: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Where rule 4 applies: dipped, not yet joined, and the approach word in force (`Words` index) not a go-around."""
    return np.asarray(dipped) & ~np.asarray(joined) & (np.asarray(approach) != APPROACH_GO_AROUND)


def glidepath_allowed(procedure: RunwayProcedure, word: np.ndarray, e: np.ndarray, n: np.ndarray, h: np.ndarray,
                      words: Words) -> np.ndarray:
    """Rules 1 and 2 for altitude words (`Words` indices: a level, or `Words.altitude_land`) at positions ``(e, n)``,
    height ``h``."""
    word = np.asarray(word)
    limit = procedure.floor_m(e, n) - word_tolerance_m(words.spec)
    # a level's plane is its index × the step (`Words.altitude_m`, one word at a time); "descend to land" is judged by
    # where the aircraft is
    height = np.where(word == words.altitude_land, np.asarray(h, dtype=np.float64), word * words.spec.altitude_step_m)
    return np.isnan(limit) | (height >= limit)


def decision_allowed(procedure: RunwayProcedure, word: np.ndarray, joined: np.ndarray, words: Words) -> np.ndarray:
    """Rule 3 for altitude words: before the join a level no lower than the DA less the word tolerance."""
    word = np.asarray(word)
    return ((word == words.altitude_land) | np.asarray(joined)
            | (word * words.spec.altitude_step_m >= procedure.decision_m - word_tolerance_m(words.spec)))


def climb_allowed(word: np.ndarray, h: np.ndarray, barred: np.ndarray, words: Words) -> np.ndarray:
    """Rule 4 for altitude words: where the climb is barred a level no higher than the height plus the word
    tolerance."""
    word = np.asarray(word)
    return ((word == words.altitude_land) | ~np.asarray(barred)
            | (word * words.spec.altitude_step_m <= np.asarray(h, dtype=np.float64) + word_tolerance_m(words.spec)))


def altitude_word_allowed(procedure: RunwayProcedure, word: np.ndarray, e: np.ndarray, n: np.ndarray, h: np.ndarray,
                          words: Words, *, joined: np.ndarray, barred: np.ndarray) -> np.ndarray:
    """Rules 1–4 for altitude words at positions ``(e, n)``, height ``h``, ``joined`` (`pre_join`) and the climb
    ``barred`` (`climb_barred`) there: the check a word said there must pass, and the one a word in force must still
    pass (rule 5). Broadcast together."""
    return (glidepath_allowed(procedure, word, e, n, h, words) & decision_allowed(procedure, word, joined, words)
            & climb_allowed(word, h, barred, words))


def angle_word_allowed(word: np.ndarray, barred: np.ndarray, words: Words) -> np.ndarray:
    """Rule 4 for descent-angle words (`Words` indices): the climb class is not said where the climb is barred."""
    return (np.asarray(word) != words.angle_climb) | ~np.asarray(barred)


def below_floor(procedure: RunwayProcedure, e: np.ndarray, n: np.ndarray, h: np.ndarray, spec: VocabularySpec
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(below, floor)``: rows more than the track tolerance below their floor (§3.5), and the floor."""
    floor = procedure.floor_m(e, n)
    return ~np.isnan(floor) & (np.asarray(h, dtype=np.float64) < floor - track_tolerance_m(spec)), floor


def _stretch_lowest(h: np.ndarray, barred: np.ndarray) -> np.ndarray:
    """At each ``barred`` row the lowest height since its stretch of consecutive barred rows began (inf elsewhere)."""
    out = np.full(len(h), np.inf)
    lowest = np.inf
    for row in range(len(h)):
        lowest = min(lowest, h[row]) if barred[row] else np.inf
        out[row] = lowest
    return out


def pre_join_readout(finals: Sequence[RunwayProcedure], runway: np.ndarray, approach: np.ndarray, e: np.ndarray,
                     n: np.ndarray, h: np.ndarray, mva: np.ndarray, first: int, spec: VocabularySpec
                     ) -> dict[str, float | bool | None]:
    """One flight's readouts before the join (design §3.5, §3.7) over its rows ``first`` … (the earlier ones only set
    where it joined and dipped): at each row the runway (``runway``: pointers into ``finals``) and the approach word in
    force, the position, height and MVA (NaN off the chart). The most a row fell under the DA and under the MVA (the
    MVA only where not cleared), and the most a row rose above the lowest row of its stretch of barred rows (rows from
    ``first`` where rule 4 applies; a stretch restarts after a go-around, and nothing flown before ``first`` counts);
    each None without such a row, and each against its line — the track tolerance under the DA and the MVA, an
    altitude step up after the dip."""
    runway, approach, h = np.asarray(runway), np.asarray(approach), np.asarray(h, dtype=np.float64)
    count = len(h)
    decision = np.full(count, np.nan)
    climb = np.full(count, np.nan)
    before = np.zeros(count, dtype=bool)
    for pointer in np.unique(runway[first:]):
        final = finals[int(pointer)]
        joined, dipped = pre_join(final, e, n, h, spec)
        rows = (runway == pointer) & (np.arange(count) >= first)
        before |= rows & ~joined
        decision = np.where(rows & ~joined, final.decision_m - h, decision)
        barred = rows & climb_barred(joined, dipped, approach)
        climb = np.where(barred, h - _stretch_lowest(h, barred), climb)
    vectored = before & (approach == APPROACH_NOT_CLEARED) & np.isfinite(np.asarray(mva, dtype=np.float64))
    under_mva = np.where(vectored, np.asarray(mva, dtype=np.float64) - h, np.nan)

    def most(values: np.ndarray) -> float | None:
        return float(np.nanmax(values)) if np.isfinite(values).any() else None

    depth = track_tolerance_m(spec)
    out = {"decision_under_m": most(decision), "climb_after_dip_m": most(climb), "mva_under_m": most(under_mva)}
    return {**out,
            "under_decision": out["decision_under_m"] is not None and out["decision_under_m"] > depth,
            "climbed_after_dip": out["climb_after_dip_m"] is not None and out["climb_after_dip_m"] > spec.altitude_step_m,
            "under_mva": out["mva_under_m"] is not None and out["mva_under_m"] > depth}


def runway_procedure(geometry: AirportGeometry, candidate: RunwayCandidate, runway: Runway, *,
                     root=DEFAULT_PROCEDURE_ROOT) -> RunwayProcedure:
    """``candidate``'s final in the airport frame. Refused: a document whose threshold is not the candidate's (farther
    than `THRESHOLD_TOLERANCE_M`), a runway with no published TCH or glidepath, or no vertically guided minima (the
    harvest's `Runway.decision_height_above_threshold_m` raises)."""
    skeleton = procedure_skeleton(geometry.code, candidate.ident, root=root)
    e0, n0 = geometry.frame.horizontal_from_latlon(skeleton.threshold_lat_deg, skeleton.threshold_lon_deg)
    offset = math.hypot(float(e0) - candidate.threshold_e_m, float(n0) - candidate.threshold_n_m)
    if offset > THRESHOLD_TOLERANCE_M:
        raise ValueError(f"{geometry.code} {candidate.ident}: {skeleton.procedure_uid}'s threshold is {offset:.0f} m "
                         f"from the candidate's")
    if runway.threshold_crossing_height_m is None or runway.published_glidepath_deg is None:
        raise ValueError(f"{geometry.code} {candidate.ident} publishes no threshold crossing height or glidepath")
    faf_e, faf_n = geometry.frame.horizontal_from_latlon(skeleton.faf.lat_deg, skeleton.faf.lon_deg)
    faf_d = relative_to_runway(faf_e, faf_n, 0.0, 0.0, candidate).before_threshold_m
    return RunwayProcedure(candidate=candidate, crossing_m=candidate.elevation_m + runway.threshold_crossing_height_m,
                           glidepath_tan=math.tan(math.radians(runway.published_glidepath_deg)), faf_d_m=float(faf_d),
                           cone=fas_course_geometry(candidate.length_m),
                           decision_m=candidate.elevation_m + runway.decision_height_above_threshold_m)


def airport_procedures(geometry: AirportGeometry, runways: Sequence[Runway], *, root=DEFAULT_PROCEDURE_ROOT
                       ) -> tuple[RunwayProcedure, ...]:
    """Every candidate's final, in the candidates' (the runway pointer's) order."""
    by_ident = {runway.ident: runway for runway in runways}
    return tuple(runway_procedure(geometry, candidate, by_ident[candidate.ident], root=root)
                 for candidate in geometry.candidates)


def published_procedures(geometries: dict[str, AirportGeometry]) -> dict[str, tuple[RunwayProcedure, ...]]:
    """`airport_procedures` for each airport from the harvest's runway data at the default configuration and CIFP (the
    runway data the executor and the evaluator read)."""
    return {code: airport_procedures(geometry, load_airport(code, config_file=DEFAULT_CONFIG,
                                                            cifp_file=DEFAULT_CIFP).runways)
            for code, geometry in geometries.items()}
