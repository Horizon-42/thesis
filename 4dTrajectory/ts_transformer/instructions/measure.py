"""The measurements that set the vocabulary's measured values (vocabulary §8), and the
rule that turns them into a spec.

Every measurement runs on the TRAIN split's signals only, and only on flights the labeller
admits (`labeller.read.admit`: the rows before the landing, ground-speed refusals applied).
`SUGGESTED` holds the values the design fixes by choice (with the reason in the design
document); `MeasuredValues` holds what the data sets; `build_spec` combines them. Per-flight
work is in `measure_flight` (pass A) and `measure_final` (pass B, once the course tolerance is
known) — pure functions, so the runner can fan them out.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from final_approach.assign import LandingScreen
from ts_transformer.instructions import envelope
from ts_transformer.instructions.labeller.lateral import per_step_words, turn_runs
from ts_transformer.instructions.labeller.read import Admitted
from ts_transformer.instructions.labeller.vertical import MOVE, vertical_pieces
from ts_transformer.instructions.piecewise import fit_pieces
from ts_transformer.instructions.spec import (
    ATC_MAX_INTERCEPT_DEG, ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M, VocabularySpec,
)
from ts_transformer.instructions.words import Words, grid_levels

#: The heading tolerance is half the grid step plus this allowance for the track's own wander in
#: straight flight — a choice: the wander measured inside a free hold (no grid) grows with the band it is
#: measured in, so `measurements.json` lists it for the bands in `FREE_HOLD_HALF_RANGES_DEG`.
HEADING_WANDER_ALLOWANCE_DEG = 2.0
FREE_HOLD_HALF_RANGES_DEG = (1.0, 2.0, 3.0, 4.0)
#: A free hold (straight flight inside a band, for the wander above) is at least this long.
FREE_HOLD_MIN_S = 10.0
#: Likewise the altitude tolerance is half the step plus the fit tolerance (a level piece's rows
#: lie within it of their line); the level wander is listed for these fit tolerances.
LEVEL_FIT_TOLERANCES_M = (5.0, 10.0, 20.0)

#: The values fixed by choice (vocabulary §8 and the labeller's reading parameters).
SUGGESTED: dict[str, Any] = {
    "step_s": 2.0,
    "track_smoothing_s": 6.0,
    "altitude_smoothing_s": 10.0,
    "speed_smoothing_s": 10.0,
    # vocabulary §4.3 (user 2026-09-24): the per-step reading on a 5° grid, each row labelled with the track 4 s later
    "heading_lead_s": 4.0,
    "heading_step_deg": 5.0,
    "turn_onset_rate_deg_s": 0.2,
    "turn_rate_min_from_deg": 10.0,
    "lined_up_deg": ATC_MAX_INTERCEPT_DEG,
    "landing_cross_limit_m": LandingScreen().threshold_radius_m,
    "landing_max_height_m": LandingScreen().max_crossing_height_m,
    # MIRROR of trajectory_data_process.harvest.threshold_event.MAX_PARALLEL_COURSE_DELTA_DEG
    # (checked equal in tests/test_instruction_vocabulary.py)
    "parallel_course_delta_deg": 5.0,
    "altitude_fit_tolerance_m": 10.0,
    "level_min_s": 20.0,
    # §4.4: a level is a piece whose rows stay within this of its own median (the instruction-v3 band, 25 m)
    "level_band_m": 25.0,
    "climb_angle_max_deg": 15.0,
    "ground_speed_floor_mps": 15.0,
    "ground_speed_ceiling_mps": 350.0,
    "speed_step_mps": 5.0,
    "speed_min_mps": 20.0,
    "speed_max_mps": 250.0,
    "speed_tolerance_mps": 5.0,
    "speed_fit_tolerance_mps": 1.5,
    "speed_flat_accel_mps2": 0.1,
    "speed_min_hold_s": 20.0,
    "unspecified_plateau_s": 30.0,
    "unspecified_distance_m": ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M,
    # §4.6 (D18): the go-around rule of R40 `go_around_census` (archive/two_tier_v3_2026_10/), its values unchanged —
    # a low pass 500 m about a candidate's centreline, from 10 km before its threshold to 3 km past it, at most 600 m
    # up, moving 1 km along it (gaps of 3 rows), and levels held 20 s at least 150 m above its lowest point before and
    # after it; a point past the threshold under 15 m is on the runway (under every published TCH, harvest TD9)
    "go_around_max_cross_m": 500.0,
    "go_around_along_m": (-10_000.0, 3_000.0),
    "go_around_max_height_m": 600.0,
    "go_around_max_gap_rows": 3,
    "go_around_min_progress_m": 1_000.0,
    "go_around_hold_s": 20.0,
    "go_around_min_drop_m": 150.0,
    "go_around_min_climb_m": 150.0,
    "go_around_on_runway_height_m": 15.0,
    # the closed-loop reading's tolerances (vocabulary §4.9, D32: the user, 2026-10-03)
    "closed_loop_lateral_m": 30.0,
    "closed_loop_vertical_m": 15.0,
}
SUGGESTED["heading_tolerance_deg"] = SUGGESTED["heading_step_deg"] / 2 + HEADING_WANDER_ALLOWANCE_DEG
#: Descent classes: how many, and the outer edges (a slightly negative floor so a flat stretch
#: inside a descent keeps a descent class; the steepest descent an airliner could fly).
DESCENT_CLASSES = 4
DESCENT_FLOOR_DEG = -0.5
DESCENT_CEILING_DEG = 10.0
#: A level piece for the altitude wander: flatter than this.
LEVEL_ANGLE_DEG = 0.3
#: The rows the course tolerance is measured on: the last stretch of path flown before the
#: threshold, short enough that every stabilised approach is aligned there (on the fleet 5 % of
#: KRDU flights are still turning onto the final within the last 5 km flown; within 1.5 km the
#: pooled p99 course error is 1.8°). An along-course window would also catch a downwind abeam.
FINAL_MEASURE_M = 1500.0
#: The aligned final (the last run of rows within the course tolerance) is binned by distance
#: before the threshold; the corridor is the line through the bins' p99 offsets.
CORRIDOR_BIN_EDGES_M = (0.0, 3000.0, 6000.0, 9000.0, 12000.0, 15000.0, 20000.0, 30000.0)
CORRIDOR_BIN_MIN_ROWS = 200
#: Heading grids compared in the readout.
HEADING_GRIDS_DEG = (1.0, 2.0, 5.0)
#: Descent class counts compared in the readout.
DESCENT_CLASS_COUNTS = (3, 4, 5, 6)
PERCENTILES = (50, 90, 95, 99, 99.9)
#: D15: every value the measurement fits from data (the descent centres and inner edges, the climb centre) is written with
#: these rounder candidates and the end-of-piece height error each leaves; the user chooses.
ROUNDING_STEPS_DEG = (0.5, 0.25, 0.1)
#: The rows of `rounding_candidates`, by name: the user chooses one (D15, D56; `instruction_spec --candidate`).
CANDIDATE_NAMES = ("fitted", *(f"{step:g}" for step in ROUNDING_STEPS_DEG))
#: O12: the climb angles' distribution, in bins this wide.
CLIMB_BIN_DEG = 0.5
#: D22 (vocabulary §3.4, the altitude-grid proposal's "optimal 40 levels", fitted on MSL level-offs): 60 m to 1,260 m, 120 m
#: to 2,700 m, 450 m to 5,400 m. Since D58 the levels are heights above E, and the spec measurement fits the grid again
#: on the level-offs above E (`fit_altitude_grid`) and writes it beside this one; the user chooses (`GRID_NAMES`).
D22_GRID = {"altitude_segment_steps_m": (60.0, 120.0, 450.0), "altitude_segment_tops_m": (1260.0, 2700.0, 5400.0)}
#: The fit of vocabulary §3.4 (§9.5): at most `GRID_SEGMENTS_MAX` uniform segments from 0 m to `GRID_TOP_M` (D22's range),
#: the break points on a `GRID_BREAK_M` grid, each segment's step one of `GRID_STEPS_M` and its length a whole number of
#: steps, `GRID_LEVELS` levels (0 m included), the smallest sum of squared rounding errors of the level-offs.
GRID_SEGMENTS_MAX = 3
GRID_TOP_M = D22_GRID["altitude_segment_tops_m"][-1]
GRID_BREAK_M = 15.0
GRID_STEPS_M = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0, 150.0, 180.0, 240.0, 300.0, 450.0, 600.0)
GRID_LEVELS = len(grid_levels(*D22_GRID.values()))
#: The rows of `grid_candidates`, by name: the user chooses one (D55, D58; `instruction_spec --grid`).
GRID_NAMES = ("fitted", "d22")


@dataclass(frozen=True)
class MeasuredValues:
    turn_rate_max_deg_s: float
    turn_bank_max_deg: float
    corridor_half_width_m: float
    corridor_widening_deg: float
    corridor_course_tolerance_deg: float
    descent_angle_edges_deg: tuple[float, ...]
    descent_angle_centres_deg: tuple[float, ...]
    climb_angle_centre_deg: float
    speed_accel_max_mps2: float
    altitude_segment_steps_m: tuple[float, ...]
    altitude_segment_tops_m: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in ("descent_angle_edges_deg", "descent_angle_centres_deg", "altitude_segment_steps_m",
                     "altitude_segment_tops_m"):
            data[name] = list(getattr(self, name))
        return data


def build_spec(measured: MeasuredValues) -> VocabularySpec:
    return VocabularySpec(**SUGGESTED, **asdict(measured))


def provisional_spec() -> VocabularySpec:
    """The spec the first measurements read with, before anything is measured: the measured
    fields at stand-in values that satisfy the spec's own invariants. Only `measure_flight`
    uses it, and only for the fields that do not depend on what it measures."""
    return build_spec(MeasuredValues(
        turn_rate_max_deg_s=10.0, turn_bank_max_deg=45.0,
        corridor_half_width_m=500.0, corridor_widening_deg=0.0, corridor_course_tolerance_deg=10.0,
        descent_angle_edges_deg=(DESCENT_FLOOR_DEG, 2.0, 2.75, 3.5, DESCENT_CEILING_DEG),
        descent_angle_centres_deg=(1.5, 2.4, 3.1, 4.0),
        climb_angle_centre_deg=3.0, speed_accel_max_mps2=2.0, **D22_GRID,
    ))


def _free_holds(track: np.ndarray, min_rows: int, half_range_deg: float) -> list[tuple[int, int]]:
    holds, row, n = [], 0, len(track)
    while row < n:
        stop = row + 1
        low = high = track[row]
        while stop < n and max(high, track[stop]) - min(low, track[stop]) <= 2 * half_range_deg:
            low, high = min(low, track[stop]), max(high, track[stop])
            stop += 1
        if stop - row >= min_rows:
            holds.append((row, stop))
            row = stop
        else:
            row += 1
    return holds


def measure_flight(flight: Admitted, spec: VocabularySpec, elevation_m: float) -> dict[str, np.ndarray]:
    """Pass A: one admitted flight's contribution to every measurement that needs no measured
    value, as flat arrays; ``elevation_m`` its airport's elevation E (the level-offs are heights above it, D58)."""
    smoothed, relative = flight.smoothed, flight.relative
    track, speed, distance = smoothed.track_deg, smoothed.ground_speed_mps, smoothed.distance_m
    out: dict[str, list[float]] = {name: [] for name in (
        *(f"heading_wander_deg_band{h:g}" for h in FREE_HOLD_HALF_RANGES_DEG),
        *(f"level_wander_m_fit{t:g}" for t in LEVEL_FIT_TOLERANCES_M),
        "turn_mean_rate_deg_s", "turn_row_rate_deg_s", "turn_row_bank_deg", "speed_wander_mps", "transition_accel_mps2",
        "final_course_error_deg", "move_angle_deg", "move_length_m", "level_height_m",
    )}
    for half_range in FREE_HOLD_HALF_RANGES_DEG:
        for start, stop in _free_holds(track, spec.rows(FREE_HOLD_MIN_S), half_range):
            segment = track[start:stop]
            out[f"heading_wander_deg_band{half_range:g}"] += list(np.abs(segment - np.median(segment)))
    rate = np.diff(track) / spec.step_s
    bank = envelope.bank_deg_from_turn_rate(rate, speed[1:])
    for start, stop in turn_runs(rate, spec.turn_onset_rate_deg_s):
        if abs(track[stop] - track[start]) >= spec.turn_rate_min_from_deg:
            out["turn_mean_rate_deg_s"].append(float(abs(rate[start:stop].mean())))
            out["turn_row_rate_deg_s"] += list(np.abs(rate[start:stop]))
            out["turn_row_bank_deg"] += list(bank[start:stop])
    for fit_tolerance in LEVEL_FIT_TOLERANCES_M:
        for piece in fit_pieces(distance, smoothed.altitude_m, fit_tolerance):
            length = distance[piece.stop - 1] - distance[piece.start]
            if (piece.rows >= spec.rows(spec.level_min_s) and length > 0
                    and abs(math.degrees(math.atan(piece.slope))) < LEVEL_ANGLE_DEG):
                rows = smoothed.altitude_m[piece.start: piece.stop]
                out[f"level_wander_m_fit{fit_tolerance:g}"] += list(np.abs(rows - np.median(rows)))
    for piece in fit_pieces(flight.signals.time_s, speed, spec.speed_fit_tolerance_mps):
        rows = speed[piece.start: piece.stop]
        if piece.rows >= spec.rows(spec.speed_min_hold_s) and abs(piece.slope) <= spec.speed_flat_accel_mps2:
            out["speed_wander_mps"] += list(np.abs(rows - np.median(rows)))
        else:
            out["transition_accel_mps2"] += list(np.abs(np.diff(rows)) / spec.step_s)
    final = distance[-1] - distance <= FINAL_MEASURE_M
    out["final_course_error_deg"] += list(np.abs(relative.track_minus_course_deg[final]))
    for piece in vertical_pieces(distance, smoothed.altitude_m - elevation_m, spec, Words(spec)):
        if piece.kind == MOVE:
            out["move_angle_deg"].append(piece.angle_deg)
            out["move_length_m"].append(float(distance[piece.stop - 1] - distance[piece.start]))
        elif piece.start > 0:               # a level word other than row 0's: the height above E it is held at (§9.5)
            out["level_height_m"].append(piece.median_m)
    return {name: np.asarray(values, dtype=np.float64) for name, values in out.items()}


def aligned_final_row(flight: Admitted, course_tolerance_deg: float) -> int | None:
    """First row of the last run of rows aligned with the course within the tolerance, to the
    end; ``None`` when the flight does not end aligned."""
    aligned = np.abs(flight.relative.track_minus_course_deg) <= course_tolerance_deg
    if not aligned[-1]:
        return None
    first = len(aligned) - 1
    while first > 0 and aligned[first - 1]:
        first -= 1
    return first


def measure_final(flight: Admitted, spec: VocabularySpec, course_tolerance_deg: float,
                  grids: dict[float, float]) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Pass B, with the measured course tolerance: the aligned final's offsets by distance (the
    corridor), and for each heading grid (step → tolerance) the heading words the per-step reading says
    before the aligned final (`labeller.lateral.per_step_words`, at the spec's lead, relative to the course) — the
    trade the grid choice is made on."""
    first = aligned_final_row(flight, course_tolerance_deg)
    relative, smoothed = flight.relative, flight.smoothed
    arrays = {"aligned_offset_m": np.zeros(0), "aligned_distance_m": np.zeros(0)}
    if first is not None:
        arrays = {"aligned_offset_m": np.abs(relative.right_of_course_m[first:]),
                  "aligned_distance_m": np.asarray(relative.before_threshold_m[first:], dtype=np.float64)}
    stop = len(smoothed.track_deg) - 1 if first is None else first
    lead = spec.rows_exact(spec.heading_lead_s)
    # the rows before the aligned final, each led into it by the track up to its first row
    before = max(stop, 1)
    track = smoothed.track_deg[: before + 1]
    course = np.full(len(track), flight.candidate.course_deg)
    rows = {f"{step:g}": [float(sum(row < before for row, _ in per_step_words(track, course, step, lead)) - 1)]
            for step in grids}
    return arrays, rows


def percentiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"n": 0}
    return {"n": int(len(values)), **{f"p{p:g}": float(np.percentile(values, p)) for p in PERCENTILES},
            "max": float(np.max(values))}


def fit_descent_classes(angle_deg: np.ndarray, length_m: np.ndarray, classes: int) -> dict[str, Any]:
    """Descent classes by weighted 1-D k-means on tan(angle), each piece weighted by its length
    squared — the squared height error a piece flown at its class centre ends with. Edges are
    the midpoints between centres (in tan), the outer edges fixed; returns the classes and the
    end-of-piece height error they leave."""
    keep = (angle_deg >= DESCENT_FLOOR_DEG) & (angle_deg <= DESCENT_CEILING_DEG) & (length_m > 0)
    x = np.tan(np.radians(angle_deg[keep]))
    weight = length_m[keep] ** 2
    order = np.argsort(x)
    cumulative = np.cumsum(weight[order]) / weight.sum()
    centres = np.array([x[order][np.searchsorted(cumulative, (k + 0.5) / classes)] for k in range(classes)])
    for _ in range(200):
        edges = np.concatenate(([-np.inf], (centres[1:] + centres[:-1]) / 2, [np.inf]))
        member = np.searchsorted(edges, x, side="right") - 1
        updated = np.array([np.average(x[member == k], weights=weight[member == k]) if np.any(member == k) else centres[k]
                            for k in range(classes)])
        if np.allclose(updated, centres, atol=1e-10):
            break
        centres = updated
    edges_tan = (centres[1:] + centres[:-1]) / 2
    member = np.searchsorted(edges_tan, x, side="right")
    error = length_m[keep] * np.abs(x - centres[member])
    return {
        "classes": classes,
        "centres_deg": [float(math.degrees(math.atan(c))) for c in centres],
        "edges_deg": [DESCENT_FLOOR_DEG, *(float(math.degrees(math.atan(e))) for e in edges_tan), DESCENT_CEILING_DEG],
        "pieces": int(keep.sum()), "outside": int((~keep & (length_m > 0) & (angle_deg >= DESCENT_FLOOR_DEG)).sum()),
        "end_height_error_m": percentiles(error),
        "share_per_class": [float(np.mean(member == k)) for k in range(classes)],
    }


def _descent_kept(angle_deg: np.ndarray, length_m: np.ndarray) -> np.ndarray:
    return (angle_deg >= DESCENT_FLOOR_DEG) & (angle_deg <= DESCENT_CEILING_DEG) & (length_m > 0)


def descent_end_error(angle_deg: np.ndarray, length_m: np.ndarray, centres_deg: Sequence[float],
                      edges_deg: Sequence[float]) -> dict[str, float]:
    """The end-of-piece height error descent classes of these centres and edges leave on the move pieces
    (`fit_descent_classes`'s error: each piece flown at its class's centre, ``length · |tan a − tan c|``)."""
    keep = _descent_kept(angle_deg, length_m)
    member = np.clip(np.searchsorted(np.asarray(edges_deg[1:-1]), angle_deg[keep], side="right"), 0, len(centres_deg) - 1)
    centre_tan = np.tan(np.radians(np.asarray(centres_deg)))[member]
    return percentiles(length_m[keep] * np.abs(np.tan(np.radians(angle_deg[keep])) - centre_tan))


def climb_end_error(climb_deg: np.ndarray, length_m: np.ndarray, centre_deg: float) -> dict[str, float]:
    """The same error for the climb pieces (``climb_deg`` climbing positive) flown at one climb angle."""
    return percentiles(length_m * np.abs(np.tan(np.radians(climb_deg)) - math.tan(math.radians(centre_deg))))


def rounding_candidates(angle_deg: np.ndarray, length_m: np.ndarray, centres_deg: Sequence[float],
                        edges_deg: Sequence[float], climb_centre_deg: float) -> dict[str, Any]:
    """D15: the fitted descent centres and inner edges and the fitted climb centre, and each rounded to every step of
    `ROUNDING_STEPS_DEG` (the outer edges are choices and stay), with the end-of-piece height error each leaves."""
    climbs = angle_deg < DESCENT_FLOOR_DEG
    climb_deg, climb_length = -angle_deg[climbs], length_m[climbs]
    rows = {"fitted": (list(centres_deg), list(edges_deg), climb_centre_deg)}
    for name, step in zip(CANDIDATE_NAMES[1:], ROUNDING_STEPS_DEG):
        rows[name] = ([round(round(c / step) * step, 9) for c in centres_deg],
                             [edges_deg[0], *(round(round(e / step) * step, 9) for e in edges_deg[1:-1]), edges_deg[-1]],
                             round(round(climb_centre_deg / step) * step, 9))
    return {name: {"descent_centres_deg": centres, "descent_edges_deg": edges,
                   "descent_end_height_error_m": descent_end_error(angle_deg, length_m, centres, edges),
                   "climb_centre_deg": climb, "climb_end_height_error_m": climb_end_error(climb_deg, climb_length, climb)}
            for name, (centres, edges, climb) in rows.items()}


def candidate_values(candidates: dict[str, Any], name: str) -> dict[str, Any]:
    """The spec's descent edges and nominals and its climb nominal from the row ``name`` of `rounding_candidates`
    (to 0.01°, as the fitted values always were)."""
    if name not in candidates:
        raise ValueError(f"no candidate {name!r}: the rows are {sorted(candidates)}")
    row = candidates[name]
    return {"descent_angle_edges_deg": tuple(round(e, 2) for e in row["descent_edges_deg"]),
            "descent_angle_centres_deg": tuple(round(c, 2) for c in row["descent_centres_deg"]),
            "climb_angle_centre_deg": round(row["climb_centre_deg"], 2)}


def climb_distribution(angle_deg: np.ndarray, length_m: np.ndarray) -> dict[str, Any]:
    """O12: the climb pieces' angles (climbing positive): count, length-weighted percentiles, and the length in each
    `CLIMB_BIN_DEG` bin up to the climb class's 15° (whether they form two groups)."""
    climbs = angle_deg < DESCENT_FLOOR_DEG
    climb, length = -angle_deg[climbs], length_m[climbs]
    order = np.argsort(climb)
    cumulative = np.cumsum(length[order]) / length.sum()
    edges = np.arange(0.0, SUGGESTED["climb_angle_max_deg"] + CLIMB_BIN_DEG, CLIMB_BIN_DEG)
    counts, _ = np.histogram(climb, bins=edges, weights=length)
    return {"pieces": int(climbs.sum()), "angle_deg": percentiles(climb),
            "length_weighted_deg": {f"p{p:g}": float(climb[order][np.searchsorted(cumulative, p / 100.0)])
                                    for p in (10, 25, 50, 75, 90)},
            "length_m_by_bin": {f"{low:g}-{high:g}": float(c) for low, high, c in zip(edges, edges[1:], counts)}}


def rounding_error_m(heights_m: np.ndarray, levels_m: np.ndarray) -> np.ndarray:
    """Each height's distance to its nearest level."""
    return np.abs(heights_m[:, None] - levels_m[None, :]).min(axis=1) if len(heights_m) else heights_m


def fit_altitude_grid(heights_m: np.ndarray) -> dict[str, tuple[float, ...]]:
    """The grid of vocabulary §3.4 (`GRID_SEGMENTS_MAX` ... `GRID_LEVELS`) with the smallest sum of squared rounding errors
    of ``heights_m``, by exact dynamic programming over the break points: a segment [a, b] of step s rounds the heights in
    it to a, a + s, ..., b, so the error is additive over segments, and the state is (segments, break point, levels
    used). Every step is a whole number of `GRID_BREAK_M`, so a segment's error depends on its start only through the
    phase a mod s: one prefix sum over the sorted heights for each step and phase gives every segment's error. A height
    outside [0, `GRID_TOP_M`] rounds to the end level of every grid alike and does not move the choice. Ties go to the
    first found (fewer segments; for a segment, the later start and then the smaller step); no heights raise."""
    heights = np.sort(np.asarray(heights_m, dtype=np.float64))
    if not len(heights):
        raise ValueError("no level-off to fit the altitude grid on")
    unit = GRID_BREAK_M
    points = int(round(GRID_TOP_M / unit)) + 1                       # break points 0, 15, ..., GRID_TOP_M
    at = np.searchsorted(heights, np.arange(points) * unit, side="left")
    # error[s][phase][p]: the squared rounding error of the heights below break point p, to the levels phase + k s
    error: dict[float, np.ndarray] = {}
    for step in GRID_STEPS_M:
        phases = int(round(step / unit))
        table = np.empty((phases, points))
        for phase in range(phases):
            r = np.mod(heights - phase * unit, step)
            table[phase] = np.concatenate(([0.0], np.cumsum(np.minimum(r, step - r) ** 2)))[at]
        error[step] = table
    counts = GRID_LEVELS - 1                                          # the levels above 0 m
    best = np.full((GRID_SEGMENTS_MAX + 1, points, counts + 1), np.inf)
    best[0, 0, 0] = 0.0
    parent: dict[tuple[int, int, int], tuple[int, float]] = {}
    for k in range(1, GRID_SEGMENTS_MAX + 1):
        for end in range(1, points):
            for step in GRID_STEPS_M:
                stride = int(round(step / unit))
                starts = np.arange(end - stride, -1, -stride)          # a whole number of steps before ``end``
                if not len(starts):
                    continue
                levels = (end - starts) // stride
                cost = error[step][starts % stride, end] - error[step][starts % stride, starts]
                used = np.arange(counts + 1)[None, :] - levels[:, None]
                previous = np.where(used >= 0, best[k - 1, starts[:, None], np.clip(used, 0, None)], np.inf)
                total = previous + cost[:, None]
                pick = np.argmin(total, axis=0)
                value = total[pick, np.arange(counts + 1)]
                better = value < best[k, end]
                for n in np.nonzero(better)[0]:
                    parent[(k, end, int(n))] = (int(starts[pick[n]]), step)
                best[k, end] = np.where(better, value, best[k, end])
    k = int(np.argmin(best[1:, points - 1, counts])) + 1
    if not np.isfinite(best[k, points - 1, counts]):
        raise ValueError(f"no grid of {GRID_LEVELS} levels fits {GRID_TOP_M:g} m")
    steps, tops, end, n = [], [], points - 1, counts
    for segment in range(k, 0, -1):
        start, step = parent[(segment, end, n)]
        steps.append(step)
        tops.append(end * unit)
        n -= (end - start) // int(round(step / unit))
        end = start
    return {"altitude_segment_steps_m": tuple(steps[::-1]), "altitude_segment_tops_m": tuple(tops[::-1])}


def grid_candidates(heights_m: np.ndarray) -> dict[str, Any]:
    """D58: the grid fitted on the level-offs above E (`fit_altitude_grid`) and the grid of D22, each with its levels
    and the rounding error of the level words (`GRID_NAMES`)."""
    rows = {"fitted": fit_altitude_grid(heights_m), "d22": D22_GRID}
    return {name: {"altitude_segment_steps_m": list(grid["altitude_segment_steps_m"]),
                   "altitude_segment_tops_m": list(grid["altitude_segment_tops_m"]),
                   "levels": len(grid_levels(grid["altitude_segment_steps_m"], grid["altitude_segment_tops_m"])),
                   "rounding_error_m": percentiles(rounding_error_m(heights_m, grid_levels(
                       grid["altitude_segment_steps_m"], grid["altitude_segment_tops_m"])))}
            for name, grid in rows.items()}


def grid_values(candidates: dict[str, Any], name: str) -> dict[str, tuple[float, ...]]:
    """The spec's altitude grid from the row ``name`` of `grid_candidates`."""
    if name not in candidates:
        raise ValueError(f"no grid {name!r}: the rows are {sorted(candidates)}")
    row = candidates[name]
    return {"altitude_segment_steps_m": tuple(float(v) for v in row["altitude_segment_steps_m"]),
            "altitude_segment_tops_m": tuple(float(v) for v in row["altitude_segment_tops_m"])}


def round_up(value: float, step: float) -> float:
    """Up to the next multiple of ``step``, as a clean decimal (0.1 × 17 is 1.7, not 1.7000000000000002)."""
    return round(math.ceil(value / step - 1e-9) * step, 9)


def round_down(value: float, step: float) -> float:
    return round(math.floor(value / step + 1e-9) * step, 9)


def fit_corridor(offset_m: np.ndarray, distance_m: np.ndarray) -> dict[str, Any]:
    """The widening corridor: the half width at the threshold is the nearest bin's p99 offset,
    and the widening angle the smallest that keeps every bin's p99 inside the corridor at the
    bin's middle (bins with fewer than `CORRIDOR_BIN_MIN_ROWS` rows left out)."""
    bins = []
    for low, high in zip(CORRIDOR_BIN_EDGES_M, CORRIDOR_BIN_EDGES_M[1:]):
        inside = (distance_m >= low) & (distance_m < high)
        if inside.sum() >= CORRIDOR_BIN_MIN_ROWS:
            bins.append({"from_m": low, "to_m": high, "rows": int(inside.sum()),
                         "p99_m": float(np.percentile(offset_m[inside], 99)),
                         "p95_m": float(np.percentile(offset_m[inside], 95))})
    middle = np.array([(b["from_m"] + b["to_m"]) / 2 for b in bins])
    p99 = np.array([b["p99_m"] for b in bins])
    half_width = float(p99[0])
    slope = float(np.max(np.maximum(p99 - half_width, 0.0) / middle))
    return {"bins": bins, "half_width_m": half_width, "slope": slope}
