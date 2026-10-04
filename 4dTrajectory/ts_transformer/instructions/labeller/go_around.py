"""Go-arounds inside a sentence (vocabulary §4.6, D18, D19): the rule of R40 (`go_around_census`, archived in
`archive/two_tier_v3_2026_10/`) on the sentence's own rows.

R40 read the harvest's raw samples; the labeller reads the data plane's 2 s rows — already resampled and repaired, so
R40's stray-sample filter has nothing to do here — with the smoothed altitude the altitude words are read from. On
those rows:

- a LOW PASS is a run of rows (gaps of at most `go_around_max_gap_rows` rows) on one candidate's final — at most
  `go_around_max_cross_m` off its centreline, between `go_around_along_m` of its threshold along the landing direction
  (past it positive) and at most `go_around_max_height_m` above it — that moves at least `go_around_min_progress_m`
  along the landing direction between its first and last row (flying the final, not crossing it). Passes of two
  candidates whose rows overlap (close parallels) are one pass, kept on the candidate with the smaller median
  |cross-track|. Its lowest row is the GO-AROUND POINT;
- a pass is a GO-AROUND when the aircraft held a level at least `go_around_min_drop_m` above that point for
  `go_around_hold_s` since the previous pass (it came down to it: a takeoff starts on the runway), and held one at least
  `go_around_min_climb_m` above it after it, before the next pass begins or the sentence ends: a dip and a shallow
  level-off on one final are not two go-arounds. Held, not reached;
- a go-around point past the threshold and at most `go_around_on_runway_height_m` above it is on the runway — a
  touch-and-go, or a landing balked in the flare: the data has a landing there the vocabulary cannot say, so
  `read.read_flight` refuses the flight.

The sentence's last pass (the landing approach) has nothing after it, so it is never a go-around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ts_transformer.instructions.airport import RunwayRelative
from ts_transformer.instructions.spec import VocabularySpec


@dataclass(frozen=True)
class Pass:
    """A low pass: rows ``first`` … ``last`` on candidate ``candidate``'s final, lowest at ``lowest``."""

    candidate: int
    first: int
    last: int
    lowest: int
    median_abs_cross_m: float


@dataclass(frozen=True)
class GoAround:
    """A go-around: its pass, and the go-around point (the pass's lowest row) relative to the pass's candidate. The
    go-around ROW, where "go-around" is said, is the row its climb is said at (`vertical.read_vertical`, D26)."""

    low_pass: Pass
    point: int
    height_m: float         # above the threshold
    along_m: float          # along the landing direction from the threshold (negative before it)
    cross_m: float          # right of the centreline
    drop_m: float           # the highest level held before the point (since the previous pass), above it
    climb_m: float          # the highest level held after it (before the next pass or the end), above it
    on_runway: bool         # past the threshold and under `go_around_on_runway_height_m`: a touch-and-go


def low_passes(relatives: Sequence[RunwayRelative], spec: VocabularySpec) -> list[Pass]:
    """Every low pass over the rows, one candidate's frame each in ``relatives`` (the candidates' order), in row order;
    passes whose rows overlap are one pass (module docstring)."""
    low_along, high_along = spec.go_around_along_m
    found: list[Pass] = []
    for candidate, relative in enumerate(relatives):
        along = -np.asarray(relative.before_threshold_m)
        cross = np.asarray(relative.right_of_course_m)
        height = np.asarray(relative.height_above_threshold_m)
        low = np.nonzero((np.abs(cross) <= spec.go_around_max_cross_m) & (along >= low_along) & (along <= high_along)
                         & (height <= spec.go_around_max_height_m))[0]
        runs: list[list[int]] = []
        for row in low.tolist():
            if runs and row - runs[-1][-1] - 1 <= spec.go_around_max_gap_rows:
                runs[-1].append(row)
            else:
                runs.append([row])
        for run in runs:
            if along[run[-1]] - along[run[0]] < spec.go_around_min_progress_m:
                continue
            found.append(Pass(candidate, run[0], run[-1], min(run, key=lambda row: height[row]),
                              float(np.median(np.abs(cross[run])))))
    found.sort(key=lambda item: item.first)
    clusters: list[list[Pass]] = []
    for item in found:
        if clusters and item.first <= max(p.last for p in clusters[-1]):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return [min(cluster, key=lambda p: p.median_abs_cross_m) for cluster in clusters]


def held_level(time_s: np.ndarray, altitude_m: np.ndarray, first: int, end: int, hold_s: float) -> float:
    """The highest altitude rows ``first`` … ``end - 1`` stay at or above for ``hold_s``: over every stretch of them
    spanning at least ``hold_s``, its lowest row, at its highest (``-inf`` when no stretch qualifies)."""
    best = -np.inf
    last = first
    for start in range(first, end):
        last = max(last, start)
        while last < end and time_s[last] - time_s[start] < hold_s:
            last += 1
        if last >= end:
            break
        best = max(best, float(np.min(altitude_m[start: last + 1])))
    return best


def go_arounds(time_s: np.ndarray, altitude_m: np.ndarray, relatives: Sequence[RunwayRelative],
               spec: VocabularySpec) -> list[GoAround]:
    """The go-arounds over the rows (module docstring): ``altitude_m`` the smoothed altitude the words are read from,
    ``relatives`` every candidate's frame on the same rows (heights from that altitude)."""
    passes = low_passes(relatives, spec)
    found = []
    for n, low in enumerate(passes):
        point = low.lowest
        end = passes[n + 1].first if n + 1 < len(passes) else len(time_s)
        # came down to it since the pass before: a pass split by the ceiling (a level-off just above it after a
        # go-around) has nothing to come down from, and is not a second go-around (R40's review of 2026-10-02)
        since = passes[n - 1].last + 1 if n else 0
        drop = held_level(time_s, altitude_m, since, point, spec.go_around_hold_s) - float(altitude_m[point])
        climb = held_level(time_s, altitude_m, point + 1, end, spec.go_around_hold_s) - float(altitude_m[point])
        if drop < spec.go_around_min_drop_m or climb < spec.go_around_min_climb_m:
            continue
        relative = relatives[low.candidate]
        along, height = -float(relative.before_threshold_m[point]), float(relative.height_above_threshold_m[point])
        found.append(GoAround(low, point, height, along, float(relative.right_of_course_m[point]), drop, climb,
                              along > 0.0 and height <= spec.go_around_on_runway_height_m))
    return found
