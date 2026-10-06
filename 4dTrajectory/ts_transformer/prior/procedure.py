"""The procedure masks (prior design §4; D14, D52, D64): three masks from the procedure of the runway R on the altitude
and angle words a prior says. Torch-free.

For each candidate (its final, `Final`), the region is inside its FAF and its LPV cone; the join is the first row inside
it. The word rules (D64), each level T checked with the band ε of its level (`Words.altitude_tolerance_m`, D52):

1. **The glidepath lower edge**, inside the region: the published glidepath − `GLIDEPATH_BELOW_M` (nowhere else: the
   RNAV floors outside the FAF disagree with 10–14 % of the recorded tracks). A level T only where T ≥ edge − ε(T);
   "no level-off" not where the aircraft is already more than ε("no level-off") below the edge.
2. **The DA**, wherever the aircraft is not inside the region: before the join, and after it when the aircraft has left
   the region. A level T only where T ≥ DA − ε(T).
3. **No climb**, before the join, once the aircraft has been below the entry height (the glidepath at the FAF) by more
   than the band ε of the level nearest the entry height: no level above the aircraft's height + ε(T), no climb class.

A mask blocks a word when it is said, never "unchanged" (D64). **While G is true (D14)** rule 3 does not apply; a
go-around clears the join and the passage below the entry height, and while G is true neither is kept. Rules 1 and 2
are lower limits and apply. At the row whose runway word ends G, G is false after the word: the join and the passage
start again at that row (its masks read the row's own state, `permitted`, and keep it, `after_row`), not one row later.

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
from ts_transformer.instructions.words import ALTITUDE, ANGLE, Words
from ts_transformer.io_utils import file_sha256

#: The glidepath lower edge below the published glidepath, m (prior §4, §9).
GLIDEPATH_BELOW_M = 60.0
#: How far a CIFP document's threshold may lie from the candidate's, m. MIRROR of the optimizer's
#: `scenario_optimization._FRAME_ANCHOR_TOLERANCE_M` (not on this package's import path); `tests/test_prior_speaker.py`
#: checks the two agree.
THRESHOLD_TOLERANCE_M = 150.0
#: The name of this set of masks, recorded beside a prior (§8 item 2). v5 (B10, D64): the row whose runway word ends G
#: reads and keeps its own state.
PROCEDURE_MASKS = "procedure-masks-v5"


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
        #: each aircraft's newest row was taken on under G (its joined and dipped cleared, not kept)
        self.cleared = np.zeros(len(self.finals), dtype=bool)
        #: each aircraft's entry heights less the band of the level nearest them: below this, it passed under (rule 3)
        self.entry_low = np.full((len(self.finals), width), np.nan)
        for b, finals_b in enumerate(self.finals):
            for k, final in enumerate(finals_b):
                self.entry_low[b, k] = final.entry_m - words.altitude_tolerance_m(words.altitude_index(final.entry_m))
        levels = np.arange(words.n_altitude_levels + 1)
        #: each altitude word's level (NaN: "no level-off") and band ε (D52)
        self.levels = np.array([np.nan if v == words.altitude_no_level_off else words.altitude_level_m(int(v))
                                for v in levels])
        self.bands = np.array([words.altitude_tolerance_m(int(v)) for v in levels])

    def select(self, indices: Sequence[int]) -> ProcedureMasks:
        """The masks of the aircraft ``indices`` (repeats permitted), in that order, with their state (a speaker's copy,
        D96 item 5)."""
        indices = list(indices)
        out = object.__new__(ProcedureMasks)
        out.finals, out.words = [self.finals[i] for i in indices], self.words
        out.joined, out.dipped = self.joined[indices].copy(), self.dipped[indices].copy()
        out.cleared = self.cleared[indices].copy()
        out.entry_low = self.entry_low[indices].copy()
        out.levels, out.bands = self.levels, self.bands
        return out

    def track(self, e: np.ndarray, n: np.ndarray, height_m: np.ndarray, go_around: np.ndarray) -> None:
        """Each aircraft's joined and dipped taken on to its newest row at ``(e, n)``, ``height_m`` above E, with G
        ``go_around`` in force before the row (while G, nothing is kept: the stretch starts again after it)."""
        self.cleared = np.asarray(go_around, dtype=bool).copy()
        for b, finals in enumerate(self.finals):
            if go_around[b]:
                self.joined[b], self.dipped[b] = False, False
                continue
            self._take(b, e[b], n[b], height_m[b])

    def after_row(self, e: np.ndarray, n: np.ndarray, height_m: np.ndarray, go_around: np.ndarray) -> None:
        """The newest row said, with G ``go_around`` in force after its runway word: an aircraft whose word ended G keeps
        the row's own joined and dipped (D64: the stretch starts again at that row)."""
        for b in np.flatnonzero(self.cleared & ~np.asarray(go_around, dtype=bool)):
            self._take(b, e[b], n[b], height_m[b])
        self.cleared = self.cleared & np.asarray(go_around, dtype=bool)

    def _take(self, b: int, e: float, n: float, height_m: float) -> None:
        for k, final in enumerate(self.finals[b]):
            self.joined[b, k] |= bool(final.inside(e, n))
            self.dipped[b, k] |= bool(height_m < self.entry_low[b, k]) and not self.joined[b, k]

    def permitted(self, column: int, runway: np.ndarray, go_around: np.ndarray, e: np.ndarray, n: np.ndarray,
                  height_m: np.ndarray) -> np.ndarray:
        """``[B, words]`` (`instructions.grammar.column_words` order) of ``column``: the words each aircraft may say at its
        newest row, under the runway ``runway`` and G ``go_around`` in force after this row's runway word, at ``(e, n)``,
        ``height_m`` above E. "Unchanged" is always permitted (D64)."""
        count = len(self.finals)
        out = np.ones((count, len(column_words(column, self.words, 1))), dtype=bool)
        for b in range(count):
            k = int(runway[b])
            joined, dipped = self.joined[b, k], self.dipped[b, k]
            if self.cleared[b] and not go_around[b]:       # the row's word ends G: the row's own state (D64)
                joined = bool(self.finals[b][k].inside(e[b], n[b]))
                dipped = bool(height_m[b] < self.entry_low[b, k]) and not joined
            barred = bool(dipped and not joined and not go_around[b])
            if column == ANGLE:
                out[b, 1 + self.words.angle_climb] = not barred
                continue
            out[b, 1:] = self._altitude(self.finals[b][k], barred, e[b], n[b], height_m[b])
        return out

    def _altitude(self, final: Final, barred: bool, e: float, n: float, height_m: float) -> np.ndarray:
        """Rules 1–3 for every altitude word (levels, then "no level-off")."""
        edge = float(final.edge_m(np.array(e), np.array(n)))
        level = ~np.isnan(self.levels)
        ok = np.ones(len(self.levels), dtype=bool)
        if not math.isnan(edge):
            ok[level] &= self.levels[level] >= edge - self.bands[level]
            ok[~level] = height_m >= edge - self.bands[~level][0]
        else:                           # not inside the region: before the join, or out of it after (rule 2)
            ok[level] &= self.levels[level] >= final.decision_m - self.bands[level]
        if barred:
            ok[level] &= self.levels[level] <= height_m + self.bands[level]
        return ok
