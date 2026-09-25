"""The final approach's glidepath lower edge as a hard mask on the prior's altitude column (post-training design §3).

For the runway in force, inside the FAF and inside the LPV cone (`flight_scenarios.fas_geometry`) the floor is the
published glidepath less `GLIDEPATH_BELOW_M` — the optimizer's lower edge, which also binds only inside the FAF;
elsewhere there is none (the RNAV altitudes coded outside the FAF disagree with 10–14 % of the recorded tracks, radar
vectored at MVA — design §7, readouts §8.1). Three word rules (§3.4) and a check of the flown track (§3.5):

1. an altitude target T is allowed where ``T ≥ floor − word tolerance`` (half an altitude step: the step holding the
   floor stays sayable). The executor holds T less the tube's margin during and after the descent, and the glidepath
   only falls toward the runway, so a target allowed here stays allowed inbound;
2. "descend to land" is not allowed where the aircraft is already more than the word tolerance below the floor (no
   look-ahead: a straight line to the threshold forbade 4.2 % of the labelled words whose aircraft never went below);
3. "unchanged" is not allowed where the word in force fails rule 1 or 2 here;
4. a flown row more than the track tolerance (the word tolerance plus the tube's margin) below its floor ends the flight.

The FAF is the coded RNAV(GPS) approach's (`flight_scenarios.procedure_final.procedure_skeleton`); the glidepath is the
runway's published one (the harvest's runway data — threshold crossing height and glidepath angle, the executor's
crossing point, `autopilot.runway_data`). Positions are the airport frame (`instructions.airport`), heights m MSL.
Torch-free.
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
from ts_transformer.instructions.words import Words

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

    @property
    def ident(self) -> str:
        return self.candidate.ident

    def axes(self, e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(d, xt)``: the distance to go along the course (+ before the threshold) and the distance off it — the
        words' own reading (`instructions.airport.relative_to_runway`; its track and height parts unused)."""
        relative = relative_to_runway(e, n, 0.0, 0.0, self.candidate)
        return relative.before_threshold_m, np.abs(relative.right_of_course_m)

    def floor_m(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        """The floor at positions ``(e, n)``, m MSL: the glidepath less `GLIDEPATH_BELOW_M` inside the FAF and the cone,
        NaN elsewhere."""
        d, xt = self.axes(e, n)
        inside = (d >= 0.0) & (d <= self.faf_d_m) & (xt <= course_halfwidth_m(d, self.cone))
        return np.where(inside, self.crossing_m + d * self.glidepath_tan - GLIDEPATH_BELOW_M, np.nan)


def altitude_word_allowed(procedure: RunwayProcedure, word: np.ndarray, e: np.ndarray, n: np.ndarray, h: np.ndarray,
                          words: Words) -> np.ndarray:
    """Rules 1 and 2 for altitude words (`Words` indices: a level, or `Words.altitude_land`) at positions ``(e, n)``,
    height ``h``: the check a word said there must pass, and the one a word in force must still pass (rule 3)."""
    word = np.asarray(word)
    limit = procedure.floor_m(e, n) - word_tolerance_m(words.spec)
    # a level's plane is its index × the step (`Words.altitude_m`, one word at a time); "descend to land" is judged by
    # where the aircraft is
    height = np.where(word == words.altitude_land, np.asarray(h, dtype=np.float64), word * words.spec.altitude_step_m)
    return np.isnan(limit) | (height >= limit)


def below_floor(procedure: RunwayProcedure, e: np.ndarray, n: np.ndarray, h: np.ndarray, spec: VocabularySpec
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(below, floor)``: rows more than the track tolerance below their floor (§3.5), and the floor."""
    floor = procedure.floor_m(e, n)
    return ~np.isnan(floor) & (np.asarray(h, dtype=np.float64) < floor - track_tolerance_m(spec)), floor


def runway_procedure(geometry: AirportGeometry, candidate: RunwayCandidate, runway: Runway, *,
                     root=DEFAULT_PROCEDURE_ROOT) -> RunwayProcedure:
    """``candidate``'s final in the airport frame. Refused: a document whose threshold is not the candidate's (farther
    than `THRESHOLD_TOLERANCE_M`), a runway with no published TCH or glidepath."""
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
                           cone=fas_course_geometry(candidate.length_m))


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
