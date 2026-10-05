"""The closed-loop reading (vocabulary §4.9, D32): the labeller's second pass. It flies each open-loop sentence (the observed
words, `instructions.labeller`) with the executor and adds CORRECTION WORDS where the flown path leaves the observed
one; the prior trains on the flown states of these sentences (prior §2).

THE FLIGHT. A batch of sentences on one row interval Δ (`replay.batch_of`) is flown from the FIRST PREDICTED STEP, Δ row
``OBSERVATION_S / Δ`` (16 s after the sentence's first Δ row), from its start state there (`flights.start_state`: the
row's position and height, its velocity by the executor spec's start rule from the rows at or before it, D77); the rows
before it stay observed and say nothing, their velocity by the same rule (`flights.observed_rows`). The words are said a step at a time, as in free generation (`sentence.Spoken`): at each Δ row
the reading finds the matched point, decides the row's words — the observed words the matched point has reached, with
the corrections — and the executor hears them on the cycle that starts the row (the sentence's own clock: a row is Δ
seconds). The flight ends at the end of the cycle in which the executor is done (a crossing, the ground, the dynamics,
or the replay's time limit: the remaining observed time × the timeout factor, `replay.time_limits_s`, and each
go-around's extra time), not at the open-loop sentence's last row: the closed-loop sentence has the flown rows (§4.9
item 6). A sentence that is in a go-around at its first predicted step, or has fewer than two rows from there, is
refused.

WHEN THE OBSERVED WORDS ARE SAID (D42, D45, D46). The observed words are the open-loop reading's at the data's 2 s rows
(not the Δ grid of `labeller.interval`), each with its 2 s time. The observed time of the matched point is the time at
which the observed aircraft was there (`ObservedPath.match`). At each Δ row the reading says every observed word whose
time is less than Δ/2 after the matched point's observed time and that was not said before (the Δ grid's own rule,
`labeller.interval.last_heard_row`: a word exactly Δ/2 after waits for the next row): a word comes at the Δ row nearest
the place where the observed aircraft heard it, and its mean lateness is zero (D45). Of
several words of a column, the last (`last_words`). While the flown aircraft is behind the observed one the observed
words wait; ahead of it, they come sooner. The first predicted step says every column (rule 1): the observed words in
force before Δ/2 after its observed time. A heading word is said in the frame where it is heard (D46, §3.3): the observed word
says an absolute track (its class under the course of the runway in force at its 2 s row), and the row that says it
gives the class nearest that track minus the course of the runway in force at that row — the observed runway words up
to there are said with it, the runway column first. A heading word is said when the observed word in force or the
correction changes, and its track differs from the one the executor holds; a change of runway alone says none (the
executor keeps its absolute track).

THE COMPARISON (`ObservedPath.match`). The observed path is the observed flight at the data's 2 s rows, its positions
as observed and its height and track as the labeller reads them (`labeller.read.smooth`). The matched point is the point
of the path nearest the flown position, searched forward from the row before's: the matched segment advances while the
next segment is no farther from the flown position, so a path that crosses itself does not jump. At the matched point
the lateral error e_y is the flown position's distance from the path — from the matched segment, not from its line
beyond it, so a position on the outside of a turn is as far from the path as it is from the turn's vertex (D83) — on
the right positive, the vertical error e_h the flown height minus the observed height. A difference along the path (in time) is
not corrected, so the speed words stay the observed ones. PAST THE END of the observed path (D44: an observed slice stops
short of the threshold, the flown aircraft flies on to it) the path goes on along its last segment's line and e_y is
measured against that line, for the readouts only; e_h is NaN there (no observed height: the slope of one 2 s segment
is no measurement), the reading says no correction, a correction in force ends, and the rows count as rows without
correction (D34).

THE CORRECTIONS (`Corrector`), with Y = `closed_loop_lateral_m` of the spec and H the vertical tolerance in force
(`vertical_tolerance_m`, D66): `closed_loop_final_vertical_m` (H_final) while "no level-off" is in force — the final
descent, told by the words — and `closed_loop_vertical_m` elsewhere:

- lateral: when |e_y| > Y and the row says no new observed heading word, the heading class one step (5°)
  from the observed word in force, toward the path; the observed word again when |e_y| < Y / 2 or e_y changes sign —
  except an OVERSHOOT (D53): e_y changed its sign beyond Y, and the opposite correction is said in the same row; a
  new observed heading word ends a correction (it is said, and the comparison goes on from the next row);
- vertical: only while a descent class of the observed words is in force — when e_h > H the next steeper descent class,
  when e_h < −H the next shallower (none beyond descent 4 or descent 1); the observed class again when |e_h| < H / 2 or
  e_h changes sign — an overshoot beyond H takes the opposite class in the same row where it exists (D53); a new
  observed altitude or angle word ends a correction (so a correction never runs across a change of the tolerance: the
  altitude word changes there). A level hold and a climb get none. A
  level reached by a descent says no angle word (the descent class stays in force, `labeller.vertical`), so a LEVEL
  HOLD is the executor's: the level in force captured (`vertical.Vertical.captured`, the level-off begun) — there no
  correction starts and one in force ends (Claude's reading of §4.9 "during a level hold"; the rounding of the level to
  the grid is not corrected).

A correction word is an ordinary word of its column: the grammar (`instructions.grammar`) reads every row at the flown
height above the airport elevation E (D58), and a row it refuses refuses the flight (counted by reason). A word equal to the one in force is not said (the
sentence's own rule, `labeller.sentence`). The capture, the runway words and the go-around rows are the open-loop
reading's; the executor reads only words.

THE RESULT (`ClosedLoopSentence`): the said words from the first predicted step (each word the reading added marked a
correction: every word that is not an observed word said at its row), the states on the data's 2 s rows from the
sentence's first row to its last said row, the Δ rows marked (D51) — observed before the first predicted step, flown
from it (position in the airport frame, MSL height, track, ground speed, vertical rate; a flown 2 s row between two Δ
rows is the executor's state at the end of its cycle there) — e_y, e_h at each said row, the matched point's observed time and the last 2 s row whose
observed words have been said at each row, whether its outcome is a timeout (D90), and its OUTCOME (D74: the
judge's, `judge.outcome_of`, on what the reading's executor flew from the first predicted step to its end). Flown
again from the same state on its own rows (the replay: `replay_batch`, under the same time limit), a
closed-loop sentence gives the same states on every 2 s row.
"""

from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.conformance import DEVICE, ROUNDOFF, STATE_BOUND_M
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs, observed_rows
from ts_transformer.autopilot.judge import TIMEOUT
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.replay import Batch, subset
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.autopilot.start import Loop, start_row
from ts_transformer.instructions.artefact import (
    CLOSED_LOOP_DIRECTORY, STATE_COLUMNS, ClosedLoopSentence, SentenceRows, Withheld,
)
from ts_transformer.instructions.grammar import InForce, Ungrammatical, apply
from ts_transformer.instructions.labeller.interval import (
    in_force, interval_rows, last_heard_row, on_interval_rows,
)
from ts_transformer.instructions.labeller.read import Reading, smooth, truncated
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.readout import stratum
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words, same_track, wrap180,
)
from ts_transformer.io_utils import utc_now

class Match(NamedTuple):
    """A flown position against the observed path (`ObservedPath.match`)."""

    lateral_m: float            # e_y, right of the observed track positive
    vertical_m: float           # e_h, the flown height minus the observed one
    row: float                  # the matched point's observed time, in the path's rows from its first (D42)
    along_m: float              # the matched point's distance along the path from its first row
    past_end: bool              # past the end of the path (D44: e_h NaN, no correction)


#: A turn of the observed path at a vertex above this is a reversal (D83, `ObservedPath._at_vertex`): there the two
#: normals nearly cancel and their sum gives no side, so the matched segment's line gives it. No aircraft turns this
#: much between two observed positions seconds apart; such a vertex is a fault of the observed track (repo
#: docs/open-items.md, 2026-10-05). Claude's value, on the user's order to change the rule (2026-10-05).
REVERSAL_TURN_DEG = 170.0


class ObservedPath:
    """An observed flight's path at the data's rows, and the forward search of the matched point (module docstring)."""

    def __init__(self, e_m: np.ndarray, n_m: np.ndarray, height_m: np.ndarray, first: int) -> None:
        e_m, n_m, height_m = (np.asarray(a, dtype=np.float64) for a in (e_m, n_m, height_m))
        # a row at the position of the one before is no segment: dropped (the matched segment counts the rows kept)
        moved = np.concatenate(([True], (np.diff(e_m) != 0.0) | (np.diff(n_m) != 0.0)))
        if moved.sum() < 2:
            raise ValueError("a path needs two positions")
        self.e, self.n, self.height = e_m[moved], n_m[moved], height_m[moved]
        # the observed times: a segment runs from the last row at its first point (when the observed aircraft left it)
        # to the first row at its second (when it arrived); past the path's end, its last row
        self.arrived = np.flatnonzero(moved)
        self.left = np.append(self.arrived[1:] - 1, len(e_m) - 1)
        #: the distance along the path at every row (a held position adds nothing)
        self.along_rows = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(e_m), np.diff(n_m)))))
        self.segment = min(int(moved[: first + 1].sum()) - 1, len(self.e) - 2)

    def _nearest(self, i: int, e_m: float, n_m: float) -> tuple[float, float]:
        """``(t, distance)``: the point of segment ``i`` nearest the position (``t`` along it, 0 to 1)."""
        de, dn = self.e[i + 1] - self.e[i], self.n[i + 1] - self.n[i]
        t = min(max(((e_m - self.e[i]) * de + (n_m - self.n[i]) * dn) / (de * de + dn * dn), 0.0), 1.0)
        return t, math.hypot(e_m - self.e[i] - t * de, n_m - self.n[i] - t * dn)

    def _across(self, i: int, e_m: float, n_m: float) -> float:
        """The signed distance of a position from segment ``i``'s line, right of the track positive."""
        de, dn = self.e[i + 1] - self.e[i], self.n[i + 1] - self.n[i]
        return ((e_m - self.e[i]) * dn - (n_m - self.n[i]) * de) / math.hypot(de, dn)

    def _at_vertex(self, vertex: int, matched: int, other: int, e_m: float, n_m: float, distance: float) -> float:
        """e_y where the matched segment's nearest point is ``vertex`` (between segments ``vertex`` − 1 and ``vertex``)
        and ``other`` is the segment on its far side (D83): the smaller distance to the two, on the nearer one's side —
        the other segment's when its own nearest point is not the shared vertex; else the vertex is the nearest point of
        both (the outside of the turn) and the side is that of the sum of their normals, or the matched segment's where
        the path turns more than `REVERSAL_TURN_DEG` there (a reversal). Claude's reading of "the side from the sum of
        their normals" (A37)."""
        t_other, d_other = self._nearest(other, e_m, n_m)
        if t_other != (1.0 if other < vertex else 0.0):          # the other's nearest point is not the shared vertex
            return math.copysign(d_other, self._across(other, e_m, n_m))
        units = []
        for k in (vertex - 1, vertex):
            de, dn = self.e[k + 1] - self.e[k], self.n[k + 1] - self.n[k]
            units.append((de / math.hypot(de, dn), dn / math.hypot(de, dn)))
        (e0, n0), (e1, n1) = units
        if math.degrees(math.acos(min(max(e0 * e1 + n0 * n1, -1.0), 1.0))) > REVERSAL_TURN_DEG:
            return math.copysign(distance, self._across(matched, e_m, n_m))
        normal_e, normal_n = 0.0 + n0 + n1, 0.0 - e0 - e1        # their normals' sum (right), from 0.0: a zero's sign kept
        return math.copysign(distance, (e_m - self.e[vertex]) * normal_e + (n_m - self.n[vertex]) * normal_n)

    def match(self, e_m: float, n_m: float, height_m: float) -> Match:
        """A flown position and height against the path (module docstring): the matched segment moves forward while the
        next one is no farther from the position, never back."""
        t, distance = self._nearest(self.segment, e_m, n_m)
        while self.segment + 2 < len(self.e):
            t_next, next_distance = self._nearest(self.segment + 1, e_m, n_m)
            if next_distance > distance:
                break
            self.segment, t, distance = self.segment + 1, t_next, next_distance
        i = self.segment
        de, dn = self.e[i + 1] - self.e[i], self.n[i + 1] - self.n[i]
        across = self._across(i, e_m, n_m)
        last = i == len(self.e) - 2 and t >= 1.0
        # the distance from the path's segments (D83), on the side of the path the position is; at a vertex the two
        # segments around it (`_at_vertex`; the matched point stays where it is: it never moves back); past the end,
        # from the last segment's line (D44)
        if last:
            lateral = across
        elif t == 0.0 and i > 0:
            lateral = self._at_vertex(i, i, i - 1, e_m, n_m, distance)
        elif t == 1.0:                                           # a segment's end that is not the path's
            lateral = self._at_vertex(i + 1, i, i + 1, e_m, n_m, distance)
        else:
            lateral = math.copysign(distance, across)
        row = float(self.left[-1]) if last else float(self.left[i] + t * (self.arrived[i + 1] - self.left[i]))
        if last:                       # past the end: on along the last segment's line, no height (module docstring)
            t = ((e_m - self.e[i]) * de + (n_m - self.n[i]) * dn) / (de * de + dn * dn)
        along = float(self.along_rows[self.arrived[i]]) + t * math.hypot(de, dn)
        vertical = math.nan if last else height_m - float(self.height[i] + t * (self.height[i + 1] - self.height[i]))
        return Match(lateral, vertical, row, along, last)


def vertical_tolerance_m(words: Words, altitude: np.ndarray | int) -> np.ndarray:
    """The vertical tolerance in force under each altitude word in force (D66, module docstring): H_final under "no
    level-off", H under a level. The one definition the corrections and the rule of D50 read."""
    spec = words.spec
    return np.where(np.asarray(altitude) == words.altitude_no_level_off, spec.closed_loop_final_vertical_m,
                    spec.closed_loop_vertical_m)


def uncorrected_m(errors: np.ndarray, uncorrectable: np.ndarray) -> float:
    """The largest |error| on the rows where §4.9 makes no correction (D34; 0 where there is none; a NaN error, past the
    end of the observed path, is no error)."""
    return float(np.nanmax(np.abs(errors[uncorrectable]), initial=0.0))


def heading_lateness_rows(sentence: ClosedLoopSentence, observed: np.ndarray) -> np.ndarray:
    """The lateness of each observed heading word a closed-loop sentence says after its first row (vocabulary §12.1 A12): the
    matched point's observed time at the row that says it minus the word's 2 s time, in 2 s rows; ``observed`` the
    open-loop reading's words from the sentence's first row (a correction is no observed word)."""
    rows = np.arange(len(observed))
    word_row = np.maximum.accumulate(np.where(observed[:, HEADING] != UNCHANGED, rows, 0))
    rows, withheld = sentence.rows, sentence.withheld
    says = (rows.grid[:, HEADING] != UNCHANGED) & ~rows.correction[:, HEADING]
    says[0] = False
    return withheld.matched_row[says] - word_row[withheld.observed_row[says]]


def observed_tracks(observed: np.ndarray, words: Words, courses_deg: Sequence[float]) -> tuple[np.ndarray, ...]:
    """Each 2 s row of an open-loop reading (``observed``, from its row 0): its runway (its last runway word, the
    go-around aside), the row its heading word in force was said at, and the absolute track that word says — its class
    under the course of the runway there (§3.3, D46)."""
    rows = np.arange(len(observed))
    runway = observed[np.maximum.accumulate(np.where(observed[:, RUNWAY] >= 0, rows, 0)), RUNWAY]
    heading_row = np.maximum.accumulate(np.where(observed[:, HEADING] != UNCHANGED, rows, 0))
    track = np.asarray(courses_deg, dtype=np.float64)[runway[heading_row]] + np.array(
        [words.heading_relative_deg(int(c)) for c in in_force(observed)[:, HEADING]])
    return runway, heading_row, track


def outside_rows(sentence: ClosedLoopSentence, observed: np.ndarray, first_row: int, words: Words,
                 courses_deg: Sequence[float]) -> dict[str, np.ndarray]:
    """Read from the stored sentence and its open-loop reading (``observed``, from its row 0; the sentence starts at its
    2 s row ``first_row``): the rows where §4.9 permits a correction, those where the flown path is outside the tolerance
    (|e_y| > Y, |e_h| > the vertical tolerance in force at the row, D66; their share is a reading of the ablation,
    D34), and of those the rows after which no
    correction TOWARD the path is in force (the rule of D50, vocabulary §12.2): the heading word in force says a track on the
    path's side of the observed word's (said in the frame of the word in force), the angle in force is steeper than the observed class when too high and
    shallower when too low. By column, ``lateral`` and ``vertical``, each ``[3, M]``."""
    _, _, track = observed_tracks(observed, words, courses_deg)
    grid, withheld = sentence.rows.grid, sentence.withheld
    seen = withheld.observed_row + first_row
    rows = np.arange(len(grid))
    runway = grid[np.maximum.accumulate(np.where(grid[:, RUNWAY] >= 0, rows, 0)), RUNWAY]
    heading_row = np.maximum.accumulate(np.where(grid[:, HEADING] != UNCHANGED, rows, 0))
    course = np.asarray(courses_deg, dtype=np.float64)[runway[heading_row]]    # the frame of the word in force
    said = course + np.array([words.heading_relative_deg(int(c)) for c in in_force(grid)[:, HEADING]])
    # the observed word as that frame says it (D46: a word moved across a runway word is up to half a class off its
    # observed track, which is no correction)
    plain = np.array([words.heading_track_deg(words.heading_class(float(a), float(c)), float(c))
                      for a, c in zip(track[seen], course)])
    # right of the path (e_y > 0): a track left of the observed word's; too high (e_h > 0): a steeper class
    toward = (np.sign(np.round(wrap180(said - plain), 9)) == -np.sign(withheld.lateral_m),
              np.sign(in_force(grid)[:, ANGLE].astype(int) - in_force(observed)[seen, ANGLE].astype(int))
              == np.sign(np.nan_to_num(withheld.vertical_m)))
    tolerances = (words.spec.closed_loop_lateral_m, vertical_tolerance_m(words, in_force(grid)[:, ALTITUDE]))
    out = {}
    for k, (name, errors) in enumerate((("lateral", withheld.lateral_m), ("vertical", withheld.vertical_m))):
        correctable = ~withheld.uncorrectable[:, k]
        outside = correctable & (np.nan_to_num(np.abs(errors)) > tolerances[k])
        out[name] = np.stack((correctable, outside, outside & ~toward[k]))
    return out


def largest_m(errors: np.ndarray) -> float:
    """The largest |error| of a sentence (a NaN error, past the end of the observed path, is no error)."""
    return float(np.nanmax(np.abs(errors), initial=0.0))


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1


def last_words(rows: np.ndarray) -> np.ndarray:
    """Each column's last word in ``rows`` (``[n, 5]``; `UNCHANGED` where a column says none, or ``n`` is 0): what one
    said row says of the open-loop rows the matched point passed in it (D42)."""
    out = np.full(len(COLUMNS), UNCHANGED, dtype=np.int64)
    written = rows != UNCHANGED
    for column in np.flatnonzero(written.any(axis=0)):
        out[column] = rows[np.flatnonzero(written[:, column])[-1], column]
    return out


class Corrector:
    """One flight's words, row by row (module docstring): the open-loop reading's words on the 2 s rows (``observed``,
    from its row 0) from the sentence's first Δ row (``first_row``, a 2 s row), the first predicted step ``start`` 2 s
    rows after it and Δ = ``every`` 2 s rows; how far they have been said, the corrections in force, the track the
    executor holds and the grammar's state of the said sentence; ``courses_deg`` the candidates' courses."""

    def __init__(self, observed: np.ndarray, first_row: int, start: int, every: int, words: Words,
                 courses_deg: Sequence[float]) -> None:
        held = in_force(observed)
        runway, heading_row, track = observed_tracks(observed, words, courses_deg)
        self.courses_deg = np.asarray(courses_deg, dtype=np.float64)
        self.observed, self.held, self.runway = observed[first_row:], held[first_row:], runway[first_row:]
        self.heading_row, self.track = heading_row[first_row:], track[first_row:]
        self.start, self.every, self.words = start, every, words
        spec = words.spec
        self.lateral_m = spec.closed_loop_lateral_m
        self.next = start        # the first 2 s row not said yet
        self.rows_said = 0
        self.turn = 0            # a heading correction in force: the sign of e_y it answers (0: none)
        self.slope = 0           # an angle correction in force: the sign of e_h it answers (0: none)
        self.said: np.ndarray | None = None     # the said sentence's words in force (the heading column aside)
        self.heading_key: tuple[int, int] | None = None   # the observed heading word and the correction last said
        self.target_deg: float | None = None     # the absolute track the executor holds
        self.grammar: InForce | None = None

    def row(self, matched_row: float, lateral_m: float, vertical_m: float, height_m: float, *, holding: bool,
            past_end: bool) -> tuple[np.ndarray, np.ndarray]:
        """The next said row's words and which are corrections: the observed words not said before up to (not at) Δ/2 after the
        matched point's observed time ``matched_row`` (the first predicted step: after its own), and the corrections from
        the errors, the flown height above the airport elevation E there (D58), whether the executor holds the level in force (``holding``) and whether the
        matched point is past the end of the observed path (``past_end``, module docstring); refused when the grammar
        refuses the row."""
        first = self.said is None
        self.matched_row = float(self.start) if first else matched_row
        last = min(last_heard_row(self.matched_row, self.every), len(self.observed) - 1)
        observed = last_words(self.observed[self.next: last + 1])
        self.next = max(self.next, last + 1)
        self.observed_row = self.next - 1
        held = self.held[self.observed_row]
        wanted = held.astype(np.int64).copy()
        #: rows where §4.9 makes no correction of each column (the readings of D34): the first predicted step, a row
        #: that says a new observed word, past the end of the observed path; vertically also a level hold, no descent
        #: class, no class beyond it
        self.uncorrectable = np.ones(2, dtype=bool)
        if not first:
            self.uncorrectable[0] = observed[HEADING] != UNCHANGED or past_end
            if observed[HEADING] != UNCHANGED or past_end:
                self.turn = 0
            elif self.turn and (abs(lateral_m) < self.lateral_m / 2 or _sign(lateral_m) != self.turn):
                # ended; an overshoot beyond the tolerance takes the opposite correction in the same row (D53)
                self.turn = _sign(lateral_m) if abs(lateral_m) > self.lateral_m else 0
            elif not self.turn and abs(lateral_m) > self.lateral_m:
                self.turn = _sign(lateral_m)
            angle = int(held[ANGLE])
            tolerance = float(vertical_tolerance_m(self.words, int(held[ALTITUDE])))     # in force at the row (D66)
            ends = observed[ALTITUDE] != UNCHANGED or observed[ANGLE] != UNCHANGED or holding or past_end
            self.uncorrectable[1] = (ends or not self.words.is_descent(angle)
                                     or (abs(vertical_m) > tolerance
                                         and not self.words.is_descent(angle + _sign(vertical_m))))
            if ends:
                self.slope = 0
            elif self.slope and (abs(vertical_m) < tolerance / 2 or _sign(vertical_m) != self.slope):
                self.slope = (_sign(vertical_m) if abs(vertical_m) > tolerance      # an overshoot (D53)
                              and self.words.is_descent(angle + _sign(vertical_m)) else 0)
            elif (not self.slope and self.words.is_descent(angle) and abs(vertical_m) > tolerance
                  and self.words.is_descent(angle + _sign(vertical_m))):
                self.slope = _sign(vertical_m)
            if self.slope:                                 # too high: the next steeper class
                wanted[ANGLE] = angle + self.slope
        said = wanted.copy() if first else np.where(wanted != self.said, wanted, UNCHANGED)
        said[HEADING] = self._heading()
        k, self.rows_said = self.rows_said, self.rows_said + 1
        try:
            self.grammar = apply(self.grammar, said, height_m, self.words, len(self.courses_deg))
        except Ungrammatical as error:
            raise Refused(f"closed loop: {error.reason}", f"said row {k}: {error.detail}") from None
        self.said = wanted
        added = (said != UNCHANGED) & ~((observed != UNCHANGED) & (observed == said))
        added[HEADING] = said[HEADING] != UNCHANGED and observed[HEADING] == UNCHANGED
        return said.astype(np.int16), np.zeros(len(COLUMNS), dtype=bool) if first else added

    def _heading(self) -> int:
        """The row's heading word (module docstring, D46): when the observed word in force or the correction changed,
        the class nearest the observed word's track under the course of the runway in force at the row, one class
        toward the path while a correction is in force — said if its track is not the one the executor holds."""
        key = (int(self.heading_row[self.observed_row]), self.turn)
        if key == self.heading_key:
            return UNCHANGED
        self.heading_key = key
        course = float(self.courses_deg[self.runway[self.observed_row]])
        # right of the path (turn +1): one class to the left
        word = (self.words.heading_class(float(self.track[self.observed_row]), course) - self.turn) % self.words.n_heading
        target = course + self.words.heading_relative_deg(word)
        if self.target_deg is not None and same_track(target, self.target_deg):
            return UNCHANGED
        self.target_deg = target
        return word


def first_predicted_rows(batch: Batch, step_s: float) -> list[int]:
    """Each flight's first predicted step: the 2 s row of its observed flight (from row 0) the closed loop starts at."""
    every, start = interval_rows(batch.row_interval_s, step_s), start_row(batch.row_interval_s)
    return [s.first_row + start * every for s in batch.sentences]


def start_inputs(batch: Batch, params: ExecutorParams, step_s: float, *, device: torch.device) -> FlightInputs:
    """Every flight's physical context at its first predicted step (`flights.flight_inputs` at that 2 s row,
    `first_predicted_rows`; its start state by the executor spec's start rule, D77)."""
    return flight_inputs(batch.series, batch.observed, first_predicted_rows(batch, step_s), batch.geometries,
                         params.start_rule, device=device)


def _rows(inputs: FlightInputs, flights: list[int]) -> FlightInputs:
    return FlightInputs(*(getattr(inputs, name)[flights] for name in
                          ("initial_state", "aero_params", "frame_params", "max_thrust_n")))


def refusal(sentence: replay.Sentence, reading: Reading, row_interval_s: float, step_s: float) -> Refused | None:
    """Why a sentence cannot be read in closed loop before it is flown, or None (module docstring): fewer than two Δ
    rows from the first predicted step, or a go-around in force in the words the first predicted step says."""
    start = start_row(row_interval_s)
    if len(sentence.grid) - start < 2:
        return Refused("too short for the closed loop", f"{len(sentence.grid)} rows of {row_interval_s:g} s")
    every = interval_rows(row_interval_s, step_s)
    first = min(sentence.first_row + last_heard_row(start * every, every), len(reading.words) - 1)
    if in_force(reading.words)[first, RUNWAY] == RUNWAY_GO_AROUND:
        return Refused("go-around at the first predicted step", f"row {start} of {row_interval_s:g} s")
    return None


def refusals(batch: Batch, step_s: float) -> list[Refused | None]:
    """`refusal` of every flight of ``batch``, in its order."""
    return [refusal(s, r, batch.row_interval_s, step_s) for s, r in zip(batch.sentences, batch.readings, strict=True)]


def read(batch: Batch, inputs: FlightInputs, params: ExecutorParams, words: Words, *,
         device: torch.device) -> list[ClosedLoopSentence | Refused]:
    """Every flight of ``batch`` read in closed loop (module docstring), in the batch's order: its sentence, or why it
    has none; ``inputs`` every flight's physical context at its first predicted step (`start_inputs`; any rows for the
    flights `refusal` refuses, which are not flown)."""
    spec, interval = words.spec, batch.row_interval_s
    every, start = interval_rows(interval, spec.step_s), start_row(interval)
    out: list[ClosedLoopSentence | Refused | None] = refusals(batch, spec.step_s)
    flying = [j for j, result in enumerate(out) if result is None]
    if not flying:
        return out  # type: ignore[return-value]
    paths, correctors, said, added, blocked, reached, matched, errors, states, limits = ([] for _ in range(10))
    between: list[list[np.ndarray]] = [[] for _ in flying]   # the flown 2 s rows since the last Δ row (D51)
    for j in flying:
        signals, reading, sentence = batch.signals[j], batch.readings[j], batch.sentences[j]
        span = len(reading.words) - sentence.first_row          # the observed rows the sentence covers
        smoothed = smooth(truncated(signals, span), spec)
        paths.append(ObservedPath(signals.e_m[:span], signals.n_m[:span], smoothed.altitude_m, start * every))
        correctors.append(Corrector(reading.words, sentence.first_row, start * every, every, words,
                                    [candidate.course_deg for candidate in batch.geometries[j].candidates]))
        limits.append(replay.time_limit_s(len(reading.words), sentence.first_row + start * every, params, spec.step_s))
        # the observed 2 s rows (D51), their velocity by the start rule (D77)
        states.append([*observed_rows(batch.observed[j], sentence.first_row + np.arange(start * every),
                                      params.start_rule, batch.geometries[j])])
        said.append([])
        added.append([])
        blocked.append([])
        reached.append([])
        matched.append([])
        errors.append([])
    # the most go-arounds a flight can say: its reading's from the first predicted step (the cycles' layout)
    go_arounds = max(int((batch.readings[j].words[batch.sentences[j].first_row + start * every:, RUNWAY]
                          == RUNWAY_GO_AROUND).sum()) for j in flying)
    loop = Loop(_rows(inputs, flying), [batch.geometries[j] for j in flying],
                [batch.approach_ias_mps[j] for j in flying], limits, params, words, interval_s=interval,
                most_go_arounds=go_arounds, device=device)       # the start of a closed loop (D67)
    live = np.ones(len(flying), dtype=bool)
    s = 0
    now = loop.rows()
    while live.any():
        captured = loop.executor.vertical.captured.cpu().numpy()
        step = np.full((len(flying), len(COLUMNS)), UNCHANGED, dtype=np.int64)
        for f, j in enumerate(flying):
            if not live[f]:
                continue
            flown = now[f]
            match = paths[f].match(float(flown[0]), float(flown[1]), float(flown[2]))
            lateral, vertical = match.lateral_m, match.vertical_m
            try:
                height = float(flown[2]) - batch.geometries[j].elevation_m       # above E, as the grammar reads it (D58)
                words_row, mask = correctors[f].row(match.row, lateral, vertical, height, holding=bool(captured[f]),
                                                    past_end=match.past_end)
            except Refused as refused:
                out[j] = refused
                live[f] = False
                loop.halt(~live)
                if s == 0:                                  # the cycle starts with every column said
                    held = correctors[f].held
                    step[f] = held[min(last_heard_row(start * every, every), len(held) - 1)]
                continue
            step[f] = words_row
            states[f] += between[f]
            said[f].append(words_row)
            added[f].append(mask)
            blocked[f].append(correctors[f].uncorrectable)
            reached[f].append(correctors[f].observed_row)
            matched[f].append(correctors[f].matched_row)
            errors[f].append((lateral, vertical))
            states[f].append(flown)
        if not live.any():
            break
        rows, done = loop.step(step)
        between = [[*rows[f, :-1]] if live[f] else [] for f in range(len(flying))]   # the 2 s rows between two Δ rows
        now = rows[:, -1]
        live &= ~done
        s += 1
    for f, j in enumerate(flying):
        if out[j] is not None:
            continue
        lateral, vertical = (np.array([e[c] for e in errors[f]]) for c in (0, 1))
        reading, signals = batch.readings[j], batch.observed[j]
        out[j] = ClosedLoopSentence(
            rows=SentenceRows(first_row=batch.sentences[j].first_row, start=start,
                              grid=np.array(said[f], dtype=np.int16), correction=np.array(added[f], dtype=bool),
                              states=np.array(states[f], dtype=np.float64),
                              on_interval=on_interval_rows(len(states[f]), every)),
            withheld=Withheld(runway=signals.runway, runway_index=reading.runway_index,
                              landing_time_utc=signals.landing_time_utc, capture_row=reading.capture_row,
                              go_around_rows=np.asarray(reading.go_around_rows, dtype=np.int64),
                              stratum=stratum(reading),
                              outcome=loop.outcome(f).outcome,      # the judge's, on what it flew (D74)
                              timed_out=loop.outcome(f).outcome == TIMEOUT,     # the judge's too (D90)
                              lateral_m=lateral, vertical_m=vertical,
                              uncorrectable=np.array(blocked[f], dtype=bool),
                              observed_row=np.array(reached[f], dtype=np.int64),
                              matched_row=np.array(matched[f], dtype=np.float64)))
    return out  # type: ignore[return-value]


def read_chunked(batch: Batch, params: ExecutorParams, words: Words, *, chunk: int,
                 device: torch.device) -> list[ClosedLoopSentence | Refused]:
    """`read` over ``batch`` in chunks of ``chunk`` flights (each flown on its own executor), in the batch's order."""
    out: list[ClosedLoopSentence | Refused | None] = refusals(batch, words.spec.step_s)
    flying = [j for j, result in enumerate(out) if result is None]     # only these have a first predicted step
    for first in range(0, len(flying), chunk):
        members = flying[first: first + chunk]
        part = subset(batch, members)
        for j, result in zip(members, read(part, start_inputs(part, params, words.spec.step_s, device=device), params,
                                           words, device=device)):
            out[j] = result
    return out  # type: ignore[return-value]


# ---- the conformance (vocabulary §7.2 #2): the closed-loop reading checked by what it reads, as the labeller's and the
# executor's are. A fixed REFERENCE sample — the train split's first `REFERENCE_PER_AIRPORT` flown flights of each
# airport in a permutation seeded by `REFERENCE_SEED`, at every row interval the closed-loop sentences were written at — is read by the
# code that wrote the artefact's closed-loop sentences, in the same run, and stored; the CHECK reads it again with the code
# in the process and compares: the same flights, the same refusals, the same words and corrections, and states and errors
# no further apart than the executor's conformance bound. It runs in every process that reads closed-loop sentences or
# flies them again, before its work (`require_conforming_closed_loop`, D69, D73), with the labeller's and the executor's
# checks; a difference refuses by name. There is no passed record and no digest of code.
#: v5: A20 (D58), the words above E; v6 (A29, D73): no digest of code; v7 (A31, D74): each flight's outcome compared;
#: v8 (A32): the closed-loop format v8 (D77–D83), and the train flights with a labelled go-around added (§7.2 #2).
REFERENCE_SCHEMA = "ts-closed-loop-conformance-reference-v8"
REFERENCE_SPLIT, REFERENCE_PER_AIRPORT, REFERENCE_SEED = "train", 10, 1337
#: Besides them, up to this many of each airport with a labelled go-around (§7.2 #2).
REFERENCE_GO_AROUND_PER_AIRPORT = 10
CONFORMANCE = "conformance"


def _reference_results(instructions: Path, params: ExecutorParams, words: Words, intervals: Sequence[float], *,
                       device: torch.device) -> dict[float, tuple[list[str], list[ClosedLoopSentence | Refused]]]:
    drawn, readings = replay.draw_readings(instructions, REFERENCE_SPLIT, words.spec, words,
                                           per_airport=REFERENCE_PER_AIRPORT, seed=REFERENCE_SEED,
                                           groups=(replay.OWN, replay.STAND_IN),
                                           go_around_per_airport=REFERENCE_GO_AROUND_PER_AIRPORT)
    out = {}
    for interval in intervals:
        batch = replay.batch_of(drawn, list(range(len(readings))), readings, interval, words)
        out[interval] = ([s.dataset_id for s in batch.signals],
                         read_chunked(batch, params, words, chunk=len(batch.sentences), device=device))
    return out


def _result_json(result: ClosedLoopSentence | Refused) -> dict[str, Any]:
    if isinstance(result, Refused):
        return {"refused": result.reason}
    return {"rows": len(result.rows.grid), "timed_out": result.withheld.timed_out, "outcome": result.withheld.outcome}


def write_reference(instructions: Path, params: ExecutorParams, words: Words, intervals: Sequence[float], *,
                    git: dict[str, Any], target: Path) -> Path:
    """Write the reference into ``target`` (`CONFORMANCE` inside the closed-loop directory being written), at every row
    interval of ``intervals`` (the closed-loop sentences')."""
    target.mkdir()
    payload: dict[str, Any] = {"schema": REFERENCE_SCHEMA, "written_utc": utc_now(), "git": git,
                               "python": platform.python_version(),
                               "spec_sha256": words.spec.sha256, "executor_params_sha256": params_sha256(params),
                               "split": REFERENCE_SPLIT, "per_airport": REFERENCE_PER_AIRPORT,
                               "go_around_per_airport": REFERENCE_GO_AROUND_PER_AIRPORT, "seed": REFERENCE_SEED,
                               "intervals": {}}
    for interval, (flights, results) in _reference_results(instructions, params, words, intervals,
                                                           device=DEVICE).items():
        payload["intervals"][f"{interval:g}"] = {"flights": flights, "results": [_result_json(r) for r in results]}
        read_ok = [r for r in results if isinstance(r, ClosedLoopSentence)]
        with (target / f"reference_{interval:g}s.npz").open("xb") as handle:
            np.savez_compressed(handle, **_stacked(read_ok))
    (target / "reference.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _stacked(results: Sequence[ClosedLoopSentence]) -> dict[str, np.ndarray]:
    def cat(name: str, width: int, dtype: Any) -> np.ndarray:
        rows = [getattr(r.rows if hasattr(r.rows, name) else r.withheld, name) for r in results]
        return np.concatenate(rows).astype(dtype) if rows else np.zeros((0, width) if width else 0, dtype=dtype)

    return {"words": cat("grid", 5, np.int16), "correction": cat("correction", 5, bool),
            "uncorrectable": cat("uncorrectable", 2, bool), "observed_row": cat("observed_row", 0, np.int64),
            "matched_row": cat("matched_row", 0, np.float64),
            "states": cat("states", len(STATE_COLUMNS), np.float64), "on_interval": cat("on_interval", 0, bool),
            "lateral_m": cat("lateral_m", 0, np.float64),
            "vertical_m": cat("vertical_m", 0, np.float64)}


@dataclass
class Checked:
    flights: int
    largest_state_difference_m: float = 0.0
    mismatches: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.mismatches


def check(instructions: Path, params: ExecutorParams, words: Words) -> Checked:
    """Read the reference again with the code in this process and compare (the section's comment)."""
    directory = instructions / CLOSED_LOOP_DIRECTORY / CONFORMANCE
    payload = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
    if payload["schema"] != REFERENCE_SCHEMA:
        raise ValueError(f"{directory / 'reference.json'} is not a {REFERENCE_SCHEMA} file")
    if payload["spec_sha256"] != words.spec.sha256 or payload["executor_params_sha256"] != params_sha256(params):
        raise ValueError(f"{directory} was read with another vocabulary spec or executor spec")
    checked = Checked(flights=0)
    intervals = [float(name) for name in payload["intervals"]]
    for interval, (flights, results) in _reference_results(instructions, params, words, intervals,
                                                           device=DEVICE).items():
        stored = payload["intervals"][f"{interval:g}"]
        if flights != stored["flights"]:
            checked.mismatches.setdefault(f"{interval:g} s", []).append(
                "other flights than the reference's: the data moved, or the row interval refuses others")
            continue
        with np.load(directory / f"reference_{interval:g}s.npz") as data:
            reference = {name: data[name] for name in data.files}
        again = _stacked([r for r in results if isinstance(r, ClosedLoopSentence)])
        checked.flights += len(flights)
        for flight, result, expected in zip(flights, results, stored["results"], strict=True):
            if _result_json(result) != expected:
                checked.mismatches.setdefault(f"{flight} at {interval:g} s", []).append(
                    f"{_result_json(result)}, the reference {expected}")
        if checked.mismatches:
            continue
        for name in ("words", "correction", "uncorrectable", "observed_row", "on_interval"):
            if not np.array_equal(again[name], reference[name]):
                checked.mismatches.setdefault(f"{interval:g} s", []).append(f"the {name} differ")
        if any(again[n].shape != reference[n].shape for n in ("states", "lateral_m", "vertical_m", "matched_row")):
            checked.mismatches.setdefault(f"{interval:g} s", []).append("the states have another shape")
            continue

        def apart(a: np.ndarray, b: np.ndarray) -> float:
            """The largest difference, infinite where one is NaN and the other not (both NaN: equal)."""
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                return math.inf
            return float(np.abs(np.nan_to_num(a) - np.nan_to_num(b)).max(initial=0.0))

        difference = apart(again["states"][:, :3], reference["states"][:, :3])
        others = max(apart(again[n], reference[n]) for n in ("lateral_m", "vertical_m"))
        # the matched point's observed time, in 2 s rows: a state difference of the bound moves it by far less than a row
        later = apart(again["matched_row"], reference["matched_row"])
        rest = apart(again["states"][:, 3:], reference["states"][:, 3:])
        checked.largest_state_difference_m = max(checked.largest_state_difference_m, difference, others)
        if not (difference <= STATE_BOUND_M and others <= STATE_BOUND_M and rest <= ROUNDOFF and later <= ROUNDOFF):
            checked.mismatches.setdefault(f"{interval:g} s", []).append(
                f"states {difference:.3g} m, errors {others:.3g} m, other columns {rest:.3g}, "
                f"matched times {later:.3g} rows")
    return checked


def require_conforming_closed_loop(instructions: Path, executor: Path) -> tuple[ExecutorParams, dict[str, Any], Words]:
    """The executor spec in ``executor`` opened for ``instructions`` (`replay.open_executor`: the labeller's and the
    executor's checks), refused by name unless the closed-loop reading in this process reads the artefact's closed-loop
    reference as it was read (the section's comment; D69: the one call before closed-loop sentences are read or flown
    again; once a process, as `replay.open_executor`'s). Returns what `replay.open_executor` returns, its record's
    ``checks`` with the closed loop's, so the caller runs no check twice."""
    params, record, words = replay.open_executor(executor, instructions)
    key = ("closed_loop", executor.resolve(), instructions.resolve())
    if key not in replay.CHECKED:
        checked = check(instructions, params, words)
        if not checked.passed:
            shown = "; ".join(f"{name}: {', '.join(problems[:3])}"
                              for name, problems in list(checked.mismatches.items())[:5])
            raise ValueError(f"the closed-loop reading reads {instructions.name}'s closed-loop reference otherwise: {shown}")
        replay.CHECKED[key] = {"flights": checked.flights,
                               "largest_state_difference_m": checked.largest_state_difference_m}
        print(f"checks: the closed-loop reading reads {checked.flights} reference flights as they were read, largest "
              f"state difference {checked.largest_state_difference_m:.2g} m", flush=True)
    return params, {**record, "checks": {**record["checks"], "closed_loop": replay.CHECKED[key]}}, words


def replay_batch(batch: Batch, stored: dict[int, ClosedLoopSentence], words: Words) -> tuple[Batch, int]:
    """``batch``'s flights that have a closed-loop sentence in ``stored`` (`artefact.closed_loop_sentences`), each to
    be flown on it from its first predicted step (its sentence's first row and its observed flight moved there), and how many had none."""
    every, start = interval_rows(batch.row_interval_s, words.spec.step_s), start_row(batch.row_interval_s)
    kept = [j for j, index in enumerate(batch.indices) if index in stored]
    out = subset(batch, kept)
    for position, j in enumerate(kept):
        rows = stored[batch.indices[j]].rows
        grid = rows.grid
        if rows.first_row != batch.sentences[j].first_row:
            raise ValueError(f"{batch.signals[j].dataset_id}: the closed-loop sentence starts at 2 s row "
                             f"{rows.first_row}, the labelled one at {batch.sentences[j].first_row}")
        out.sentences[position] = replay.Sentence(
            grid=grid, instructions=replay.instructions_of(grid, batch.geometries[j], words),
            first_row=batch.sentences[j].first_row + start * every)
        out.signals[position] = replay.from_row(batch.signals[j], start * every)
    return out, len(batch.indices) - len(kept)
