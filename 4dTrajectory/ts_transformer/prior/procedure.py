"""The procedure masks (prior design §4; D14, D52): three masks from the procedure of the runway R on the altitude and
angle words a prior says. Torch-free.

For each candidate (its final, `Final`):

- **the glidepath lower edge**: inside the FAF and the LPV cone, the published glidepath − `GLIDEPATH_BELOW_M`
  (nowhere else: the RNAV floors outside the FAF disagree with 10–14 % of the recorded tracks);
- **the join** is the first row inside that region; before it, **the published DA**;
- **no climb**: once the aircraft is below the entry height (the glidepath at the FAF) before the join, no climb.

The word rules (a level word is checked against a limit with the band ε of its level, `Words.altitude_tolerance_m`,
D52; Claude's reading of §4, carried from the instruction-v3 prior's rules with ε in place of half a step):

1. inside the region, a level T only where T ≥ edge − ε(T); "no level-off" not where the aircraft is already more than
   ε("no level-off") below the edge;
2. before the join, a level T only where T ≥ DA − ε(T);
3. where the climb is barred, a level T only where T ≤ the aircraft's height + ε(T), and the angle column not the climb;
4. "unchanged" in the altitude column only where the word in force passes the rules that apply (at the first predicted
   step the model already blocks it).

**While G is true (D14)** rule 3 does not apply; the stretch of the rule starts again after the go-around: a go-around
clears where the aircraft joined and dipped (Claude's reading of "its stretch starts again": the next approach is read
as a new one). Rules 1 and 2 are lower limits and apply.

Heights are above the airport elevation E, as the level words are (D58). The vertical path (TCH, glidepath angle, DA)
is the artefact's (`candidates.json`, D61); the FAF and the LPV cone are the FAA CIFP procedure's (prior §6:
`flight_scenarios.procedure_final`, `flight_scenarios.fas_geometry`), and their documents' sha256 are part of a prior's
identity of its masks (§8 item 2, `procedure_digests`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from flight_scenarios.fas_geometry import FasCourseGeometry, course_halfwidth_m, fas_course_geometry
from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT, procedure_skeleton, rnav_gps_procedure_path
from ts_transformer.instructions.airport import AirportGeometry, published_glidepath_height_m, relative_to_runway
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.words import ALTITUDE, ANGLE, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256

#: The glidepath lower edge below the published glidepath, m (prior §4, §9).
GLIDEPATH_BELOW_M = 60.0
#: How far a CIFP document's threshold may lie from the candidate's, m. MIRROR of the optimizer's
#: `scenario_optimization._FRAME_ANCHOR_TOLERANCE_M` (not on this package's import path); `tests/test_prior_speaker.py`
#: checks the two agree.
THRESHOLD_TOLERANCE_M = 150.0
#: The name of this set of masks, recorded beside a prior (§8 item 2).
PROCEDURE_MASKS = "procedure-masks-v4"


@dataclass(frozen=True)
class Final:
    """One candidate's final, heights above E: the glidepath, the FAF, the LPV cone and the DA."""

    geometry: AirportGeometry
    index: int                      # the candidate's index (the runway word's pointer)
    faf_m: float                    # the FAF's distance before the threshold along the course
    cone: FasCourseGeometry

    @property
    def threshold_m(self) -> float:
        """The threshold's height above E."""
        return self.geometry.candidates[self.index].elevation_m - self.geometry.elevation_m

    @property
    def decision_m(self) -> float:
        return self.threshold_m + self.geometry.candidates[self.index].vertical_path.decision_height_m

    @property
    def entry_m(self) -> float:
        """The entry height: the glidepath at the FAF."""
        return float(self.glidepath_m(np.array(self.faf_m)))

    def glidepath_m(self, before_threshold_m: np.ndarray) -> np.ndarray:
        return self.threshold_m + published_glidepath_height_m(self.geometry, self.index, before_threshold_m)

    def axes(self, e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The distance before the threshold along the course and the distance off it."""
        relative = relative_to_runway(e, n, 0.0, 0.0, self.geometry.candidates[self.index])
        return relative.before_threshold_m, np.abs(relative.right_of_course_m)

    def inside(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Inside the FAF and the LPV cone: where the edge binds and where an aircraft has joined the final."""
        d, off = self.axes(e, n)
        return (d >= 0.0) & (d <= self.faf_m) & (off <= course_halfwidth_m(d, self.cone))

    def edge_m(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        """The glidepath lower edge at ``(e, n)``, NaN outside the region."""
        d, _ = self.axes(e, n)
        return np.where(self.inside(e, n), self.glidepath_m(d) - GLIDEPATH_BELOW_M, np.nan)


def airport_finals(geometry: AirportGeometry, *, root: Path = DEFAULT_PROCEDURE_ROOT) -> tuple[Final, ...]:
    """Every candidate's final, in the candidates' order; refused where the CIFP document's threshold is farther than
    `THRESHOLD_TOLERANCE_M` from the candidate's."""
    finals = []
    for index, candidate in enumerate(geometry.candidates):
        skeleton = procedure_skeleton(geometry.code, candidate.ident, root=root)
        e0, n0 = geometry.frame.horizontal_from_latlon(skeleton.threshold_lat_deg, skeleton.threshold_lon_deg)
        offset = math.hypot(float(e0) - candidate.threshold_e_m, float(n0) - candidate.threshold_n_m)
        if offset > THRESHOLD_TOLERANCE_M:
            raise ValueError(f"{geometry.code} {candidate.ident}: {skeleton.procedure_uid}'s threshold is {offset:.0f} m "
                             f"from the candidate's")
        faf_e, faf_n = geometry.frame.horizontal_from_latlon(skeleton.faf.lat_deg, skeleton.faf.lon_deg)
        faf = float(relative_to_runway(faf_e, faf_n, 0.0, 0.0, candidate).before_threshold_m)
        finals.append(Final(geometry, index, faf, fas_course_geometry(candidate.length_m)))
    return tuple(finals)


def procedure_digests(geometries: Mapping[str, AirportGeometry], *, root: Path = DEFAULT_PROCEDURE_ROOT
                      ) -> dict[str, dict[str, str]]:
    """The sha256 of each candidate's CIFP procedure document: the data of the masks (§8 item 2)."""
    return {code: {c.ident: file_sha256(rnav_gps_procedure_path(code, c.ident, root=root)) for c in geometry.candidates}
            for code, geometry in sorted(geometries.items())}


class ProcedureMasks:
    """The rules on a batch of aircraft the prior speaks to (module docstring): each aircraft's finals (its airport's
    candidates, in the pointer's order) and, as of its newest row, where it has joined each and dipped under its entry
    height before that (`track`, a row at a time)."""

    columns = (ALTITUDE, ANGLE)

    def __init__(self, finals: Sequence[Sequence[Final]], words: Words) -> None:
        self.finals, self.words = [tuple(f) for f in finals], words
        width = max(len(f) for f in self.finals)
        self.joined = np.zeros((len(self.finals), width), dtype=bool)
        self.dipped = np.zeros((len(self.finals), width), dtype=bool)
        levels = np.arange(words.n_altitude_levels + 1)
        #: each altitude word's level (NaN: "no level-off") and band ε (D52)
        self.levels = np.array([np.nan if v == words.altitude_no_level_off else words.altitude_level_m(int(v))
                                for v in levels])
        self.bands = np.array([words.altitude_tolerance_m(int(v)) for v in levels])

    def track(self, e: np.ndarray, n: np.ndarray, height_m: np.ndarray, go_around: np.ndarray) -> None:
        """Each aircraft's joined and dipped taken on to its newest row at ``(e, n)``, ``height_m`` above E, with G
        ``go_around`` in force before the row (while G, nothing is kept: the stretch starts again after it)."""
        for b, finals in enumerate(self.finals):
            if go_around[b]:
                self.joined[b], self.dipped[b] = False, False
                continue
            for k, final in enumerate(finals):
                self.joined[b, k] |= bool(final.inside(e[b], n[b]))
                self.dipped[b, k] |= bool(height_m[b] < final.entry_m) and not self.joined[b, k]

    def permitted(self, column: int, runway: np.ndarray, go_around: np.ndarray, altitude_in_force: np.ndarray,
                  e: np.ndarray, n: np.ndarray, height_m: np.ndarray) -> np.ndarray:
        """``[B, words]`` (`instructions.grammar.column_words` order) of ``column``: the words each aircraft may say at its
        newest row, under the runway ``runway`` and G ``go_around`` in force after this row's runway word, the altitude
        word in force before the row (``altitude_in_force``, −1 none yet), at ``(e, n)``, ``height_m`` above E."""
        count = len(self.finals)
        out = np.ones((count, len(column_words(column, self.words, 1))), dtype=bool)
        for b in range(count):
            k = int(runway[b])
            barred = bool(self.dipped[b, k] and not self.joined[b, k] and not go_around[b])
            if column == ANGLE:
                out[b, 1 + self.words.angle_climb] = not barred
                continue
            ok = self._altitude(self.finals[b][k], bool(self.joined[b, k]), barred, e[b], n[b], height_m[b])
            out[b, 1:] = ok
            out[b, 0] = altitude_in_force[b] == UNCHANGED or bool(ok[altitude_in_force[b]])
        return out

    def _altitude(self, final: Final, joined: bool, barred: bool, e: float, n: float, height_m: float) -> np.ndarray:
        """Rules 1–3 for every altitude word (levels, then "no level-off")."""
        edge = float(final.edge_m(np.array(e), np.array(n)))
        level = ~np.isnan(self.levels)
        ok = np.ones(len(self.levels), dtype=bool)
        if not math.isnan(edge):
            ok[level] &= self.levels[level] >= edge - self.bands[level]
            ok[~level] = height_m >= edge - self.bands[~level][0]
        if not joined:
            ok[level] &= self.levels[level] >= final.decision_m - self.bands[level]
        if barred:
            ok[level] &= self.levels[level] <= height_m + self.bands[level]
        return ok
