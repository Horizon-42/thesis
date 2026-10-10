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

THE MASKS ARE COMPUTED FOR EVERY AIRCRAFT AT ONCE (B14; outline D187: the one form). Each aircraft's finals are held
as arrays by aircraft and candidate (`_FinalArrays`), each value computed as `Final` and
`instructions.airport.relative_to_runway` compute it (the course's unit vector with `math`, the glidepath's tangent and
the earth's radius of the vertical path), and the rules read them with the same arithmetic in the same order as `Final`'s
methods: `Final.inside`, `Final.edge_m` and `entry_low_m` are the definition, which the rules' tests read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from flight_scenarios.fas_geometry import FasCourseGeometry, course_halfwidth_m, fas_course_geometry
from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT, procedure_skeleton, rnav_gps_procedure_path
from ts_transformer.instructions.airport import (
    AirportGeometry, curvature_radius_m, published_glidepath_height_m, relative_to_runway,
)
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


def entry_low_m(final: Final, words: Words) -> float:
    """The entry height less the band of the level nearest it: below this, an aircraft passed under (rule 3)."""
    return final.entry_m - words.altitude_tolerance_m(words.altitude_index(final.entry_m))


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


@dataclass(frozen=True)
class _FinalArrays:
    """Each aircraft's finals as arrays ``[B, width]`` by candidate (module docstring): what `Final` and
    `relative_to_runway` read, each value computed as they compute it; a candidate an aircraft does not have is NaN
    (never inside, never below its entry)."""

    threshold_e: np.ndarray
    threshold_n: np.ndarray
    along_e: np.ndarray             # the unit vector along the course
    along_n: np.ndarray
    faf_m: np.ndarray
    course_width_m: np.ndarray      # the LPV cone (`course_halfwidth_m`)
    garp_m: np.ndarray
    threshold_m: np.ndarray
    crossing_m: np.ndarray          # the glidepath (`glidepath_height_m`): TCH, tan of its angle, 2 R
    tan_angle: np.ndarray
    two_radius_m: np.ndarray
    decision_m: np.ndarray

    @staticmethod
    def of(finals: Sequence[Sequence[Final]], width: int) -> _FinalArrays:
        values = {name: np.full((len(finals), width), np.nan) for name in _FinalArrays.__dataclass_fields__}
        for b, finals_b in enumerate(finals):
            for k, final in enumerate(finals_b):
                candidate = final.geometry.candidates[final.index]
                course = math.radians(candidate.course_deg)
                path = candidate.vertical_path
                for name, value in (
                        ("threshold_e", candidate.threshold_e_m), ("threshold_n", candidate.threshold_n_m),
                        ("along_e", math.sin(course)), ("along_n", math.cos(course)), ("faf_m", final.faf_m),
                        ("course_width_m", final.cone.course_width_m), ("garp_m", final.cone.d_garp_m),
                        ("threshold_m", final.threshold_m), ("crossing_m", path.crossing_height_m),
                        ("tan_angle", math.tan(math.radians(path.glidepath_deg))),
                        ("two_radius_m", 2.0 * curvature_radius_m(final.geometry.frame.lat0, candidate.course_deg)),
                        ("decision_m", final.decision_m)):
                    values[name][b, k] = value
        return _FinalArrays(**values)

    def take(self, rows: np.ndarray, columns: np.ndarray | slice = slice(None)) -> _FinalArrays:
        """The values of the aircraft ``rows`` (and, with ``columns``, one candidate each)."""
        return _FinalArrays(**{name: getattr(self, name)[rows, columns] for name in self.__dataclass_fields__})

    def inside(self, e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`Final.inside` and the distance before the threshold, at ``(e, n)`` (broadcast against the arrays)."""
        de, dn = e - self.threshold_e, n - self.threshold_n
        d = -(de * self.along_e + dn * self.along_n)
        off = np.abs(de * self.along_n - dn * self.along_e)
        return (d >= 0.0) & (d <= self.faf_m) & (off <= self.course_width_m * (d + self.garp_m) / self.garp_m), d

    def edge_m(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        """`Final.edge_m`: the glidepath lower edge, NaN outside the region."""
        inside, d = self.inside(e, n)
        glidepath = self.threshold_m + (self.crossing_m + d * self.tan_angle + d ** 2 / self.two_radius_m)
        return np.where(inside, glidepath - GLIDEPATH_BELOW_M, np.nan)


def require_float64(*positions: np.ndarray) -> None:
    """Positions and heights in float64, the type the masks compute in (`Final`'s methods compute in their argument's
    type: the two agree in float64 only)."""
    if any(np.asarray(v).dtype != np.float64 for v in positions):
        raise TypeError(f"the procedure masks read float64 positions, got {[np.asarray(v).dtype for v in positions]}")


class ProcedureMasks:
    """The rules on a batch of aircraft the prior speaks to (module docstring): each aircraft's finals (its airport's
    candidates, in the pointer's order) and, as of its newest row, where it has joined each and dipped under its entry
    height before that (`track`, a row at a time), every aircraft at once."""

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
                self.entry_low[b, k] = entry_low_m(final, words)
        levels = np.arange(words.n_altitude_levels + 1)
        #: each altitude word's level (NaN: "no level-off") and band ε (D52)
        self.levels = np.array([np.nan if v == words.altitude_no_level_off else words.altitude_level_m(int(v))
                                for v in levels])
        self.bands = np.array([words.altitude_tolerance_m(int(v)) for v in levels])
        self.n_candidates = np.array([len(f) for f in self.finals], dtype=np.int64)
        self.arrays = _FinalArrays.of(self.finals, width)

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
        out.n_candidates = self.n_candidates[indices].copy()
        out.arrays = self.arrays.take(np.asarray(indices, dtype=np.int64))
        return out

    @staticmethod
    def together(parts: Sequence[tuple[ProcedureMasks, Sequence[int]]]) -> ProcedureMasks:
        """The masks of aircraft of several copies of one batch's as one (a speaker's aircraft together, multi-aircraft
        control D188): ``parts`` each masks and their aircraft (repeats permitted), in that order, with their state."""
        index = [np.asarray(list(aircraft), dtype=np.int64) for _, aircraft in parts]

        def rows(get: Callable[[ProcedureMasks], np.ndarray]) -> np.ndarray:
            return np.concatenate([get(masks)[i] for (masks, _), i in zip(parts, index)])

        first = parts[0][0]
        out = object.__new__(ProcedureMasks)
        out.finals = [masks.finals[b] for (masks, _), i in zip(parts, index) for b in i]
        out.words, out.levels, out.bands = first.words, first.levels, first.bands
        for name in ("joined", "dipped", "cleared", "entry_low", "n_candidates"):
            setattr(out, name, rows(lambda masks, name=name: getattr(masks, name)))
        out.arrays = _FinalArrays(**{name: rows(lambda masks, name=name: getattr(masks.arrays, name))
                                     for name in _FinalArrays.__dataclass_fields__})
        return out

    def keep(self, earlier: ProcedureMasks, aircraft: np.ndarray) -> None:
        """The state of the aircraft ``aircraft`` (``[B]`` bool) set back to ``earlier``'s (a copy of these masks, the
        same aircraft): a row that is not one of theirs takes nothing on (an aircraft that has not joined its loop,
        multi-aircraft control D150)."""
        for name in ("joined", "dipped", "cleared"):
            value = getattr(self, name).copy()
            value[aircraft] = getattr(earlier, name)[aircraft]
            setattr(self, name, value)

    def track(self, e: np.ndarray, n: np.ndarray, height_m: np.ndarray, go_around: np.ndarray) -> None:
        """Each aircraft's joined and dipped taken on to its newest row at ``(e, n)``, ``height_m`` above E, with G
        ``go_around`` in force before the row (while G, nothing is kept: the stretch starts again after it)."""
        require_float64(e, n, height_m)
        self.cleared = np.asarray(go_around, dtype=bool).copy()
        self._take_on(self.cleared, ~self.cleared, e, n, height_m)

    def after_row(self, e: np.ndarray, n: np.ndarray, height_m: np.ndarray, go_around: np.ndarray) -> None:
        """The newest row said, with G ``go_around`` in force after its runway word: an aircraft whose word ended G keeps
        the row's own joined and dipped (D64: the stretch starts again at that row)."""
        require_float64(e, n, height_m)
        ends = self.cleared & ~np.asarray(go_around, dtype=bool)
        self._take_on(np.zeros_like(ends), ends, e, n, height_m)
        self.cleared = self.cleared & np.asarray(go_around, dtype=bool)

    def _take_on(self, clear: np.ndarray, take: np.ndarray, e: np.ndarray, n: np.ndarray, height_m: np.ndarray
                 ) -> None:
        """`track`'s and `after_row`'s step: the aircraft ``clear`` cleared; the aircraft ``take`` taken on to the row at
        ``(e, n)``, ``height_m`` — on each final, joined once inside its region, dipped once under its entry height
        before joining it; the others unchanged."""
        e, n, height = (np.asarray(v, dtype=np.float64)[:, None] for v in (e, n, height_m))
        inside, _ = self.arrays.inside(e, n)
        joined = self.joined | inside
        dipped = self.dipped | ((height < self.entry_low) & ~joined)
        keep = ~(clear | take)[:, None]
        self.joined = np.where(take[:, None], joined, self.joined & keep)
        self.dipped = np.where(take[:, None], dipped, self.dipped & keep)

    def permitted(self, column: int, runway: np.ndarray, go_around: np.ndarray, e: np.ndarray, n: np.ndarray,
                  height_m: np.ndarray) -> np.ndarray:
        """``[B, words]`` (`instructions.grammar.column_words` order) of ``column``: the words each aircraft may say at its
        newest row, under the runway ``runway`` and G ``go_around`` in force after this row's runway word, at ``(e, n)``,
        ``height_m`` above E. "Unchanged" is always permitted (D64). Each aircraft's rules on its runway's final: the
        climb barred (rule 3) after it dipped under the entry height before joining, out of G; the levels and "no
        level-off" at or above the glidepath lower edge inside the region (rule 1), the levels at or above the DA
        outside it (rule 2), each within its band."""
        require_float64(e, n, height_m)
        out = np.ones((len(self.finals), len(column_words(column, self.words, 1))), dtype=bool)
        k = np.asarray(runway, dtype=np.int64)
        if ((k < 0) | (k >= self.n_candidates)).any():
            raise IndexError(f"a runway word beyond an aircraft's candidates: {k.tolist()[:5]}")
        rows = np.arange(len(k))
        go_around = np.asarray(go_around, dtype=bool)
        e, n, height = (np.asarray(v, dtype=np.float64) for v in (e, n, height_m))
        final = self.arrays.take(rows, k)
        entry_low = self.entry_low[rows, k]
        joined, dipped = self.joined[rows, k], self.dipped[rows, k]
        own = self.cleared & ~go_around                    # the row's word ends G: the row's own state (D64)
        inside, _ = final.inside(e, n)
        joined = np.where(own, inside, joined)
        dipped = np.where(own, (height < entry_low) & ~joined, dipped)
        barred = dipped & ~joined & ~go_around
        if column == ANGLE:
            out[:, 1 + self.words.angle_climb] = ~barred
            return out
        edge = final.edge_m(e, n)
        level = ~np.isnan(self.levels)
        has_edge = ~np.isnan(edge)
        levels, bands = self.levels[level][None, :], self.bands[level][None, :]
        ok = np.where(has_edge[:, None], levels >= edge[:, None] - bands, levels >= final.decision_m[:, None] - bands)
        ok &= ~barred[:, None] | (levels <= height[:, None] + bands)
        no_level = np.where(has_edge, height >= edge - self.bands[~level][0], True)
        altitude = out[:, 1:]
        altitude[:, level] = ok
        altitude[:, ~level] = no_level[:, None]
        return out
