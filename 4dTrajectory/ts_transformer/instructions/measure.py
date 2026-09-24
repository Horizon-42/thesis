"""The measurements that set the vocabulary's measured values (vocabulary design §8), and the
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
from typing import Any

import numpy as np

from final_approach.assign import LandingScreen
from flight_scenarios.start_state import DEFAULT_WINDOW_S as VELOCITY_FIT_WINDOW_S
from ts_transformer.instructions import envelope
from ts_transformer.instructions.labeller.lateral import find_holds
from ts_transformer.instructions.labeller.read import Admitted
from ts_transformer.instructions.labeller.vertical import MOVE, vertical_pieces
from ts_transformer.instructions.piecewise import fit_pieces
from ts_transformer.instructions.spec import (
    ATC_MAX_INTERCEPT_DEG, ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M, VocabularySpec,
)
from ts_transformer.instructions.words import Words

#: The heading tolerance is half the grid step plus this allowance for the track's own wander in
#: a hold — a choice: the wander measured inside a free hold (no grid) grows with the band it is
#: measured in, so `measurements.json` lists it for the bands in `FREE_HOLD_HALF_RANGES_DEG`.
HEADING_WANDER_ALLOWANCE_DEG = 2.0
FREE_HOLD_HALF_RANGES_DEG = (1.0, 2.0, 3.0, 4.0)
#: Likewise the altitude tolerance is half the step plus the fit tolerance (a level piece's rows
#: lie within it of their line); the level wander is listed for these fit tolerances.
LEVEL_FIT_TOLERANCES_M = (5.0, 10.0, 20.0)

#: The values fixed by choice (design §8 and the labeller's reading parameters).
SUGGESTED: dict[str, Any] = {
    "step_s": 2.0,
    "track_smoothing_s": 6.0,
    "altitude_smoothing_s": 10.0,
    "speed_smoothing_s": 10.0,
    # §10.1 compares the two heading readings (`experiments/heading_reading_compare.py` sets its own); until the
    # user picks one, a formal spec keeps the reading of instruction-v2
    "heading_reading": "holds",
    "heading_lead_s": 0.0,
    "heading_band_deg": 2.5,
    "heading_step_deg": 5.0,
    "heading_min_hold_s": 10.0,
    "heading_max_turn_deg": 150.0,
    "heading_split_part_deg": 140.0,
    "heading_continue_lead_deg": 10.0,
    "turn_onset_rate_deg_s": 0.2,
    "turn_rate_min_from_deg": 10.0,
    "heading_hold_max_rate_deg_s": 0.2,
    "intercept_angle_deg": ATC_MAX_INTERCEPT_DEG,
    "landing_cross_limit_m": LandingScreen().threshold_radius_m,
    "landing_max_height_m": LandingScreen().max_crossing_height_m,
    # MIRROR of trajectory_data_process.harvest.threshold_event.MAX_PARALLEL_COURSE_DELTA_DEG
    # (checked equal in tests/test_instruction_vocabulary.py)
    "parallel_course_delta_deg": 5.0,
    "altitude_step_m": 30.0,
    "altitude_max_m": 5400.0,
    "altitude_fit_tolerance_m": 10.0,
    "level_min_s": 20.0,
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
}
SUGGESTED["heading_tolerance_deg"] = SUGGESTED["heading_step_deg"] / 2 + HEADING_WANDER_ALLOWANCE_DEG
SUGGESTED["turn_start_delay_max_s"] = (VELOCITY_FIT_WINDOW_S + SUGGESTED["track_smoothing_s"]) / 2
SUGGESTED["altitude_tolerance_m"] = SUGGESTED["altitude_step_m"] / 2 + SUGGESTED["altitude_fit_tolerance_m"]
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


@dataclass(frozen=True)
class MeasuredValues:
    turn_rate_min_deg_s: float
    turn_rate_max_deg_s: float
    turn_bank_max_deg: float
    corridor_half_width_m: float
    corridor_widening_deg: float
    corridor_course_tolerance_deg: float
    descent_angle_edges_deg: tuple[float, ...]
    descent_angle_centres_deg: tuple[float, ...]
    climb_angle_centre_deg: float
    speed_accel_max_mps2: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["descent_angle_edges_deg"] = list(self.descent_angle_edges_deg)
        data["descent_angle_centres_deg"] = list(self.descent_angle_centres_deg)
        return data


def build_spec(measured: MeasuredValues) -> VocabularySpec:
    return VocabularySpec(**SUGGESTED, **asdict(measured))


def provisional_spec() -> VocabularySpec:
    """The spec the first measurements read with, before anything is measured: the measured
    fields at stand-in values that satisfy the spec's own invariants. Only `measure_flight`
    uses it, and only for the fields that do not depend on what it measures."""
    return build_spec(MeasuredValues(
        turn_rate_min_deg_s=0.1, turn_rate_max_deg_s=10.0, turn_bank_max_deg=45.0,
        corridor_half_width_m=500.0, corridor_widening_deg=0.0, corridor_course_tolerance_deg=10.0,
        descent_angle_edges_deg=(DESCENT_FLOOR_DEG, 2.0, 2.75, 3.5, DESCENT_CEILING_DEG),
        descent_angle_centres_deg=(1.5, 2.4, 3.1, 4.0),
        climb_angle_centre_deg=3.0, speed_accel_max_mps2=2.0,
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


def _turn_runs(rate: np.ndarray, onset_deg_s: float) -> list[tuple[int, int]]:
    """Runs of rows turning one way faster than the labeller's turn onset rate."""
    turning = np.abs(rate) > onset_deg_s
    runs, row = [], 0
    while row < len(rate):
        if not turning[row]:
            row += 1
            continue
        stop, sign = row + 1, np.sign(rate[row])
        while stop < len(rate) and turning[stop] and np.sign(rate[stop]) == sign:
            stop += 1
        runs.append((row, stop))
        row = stop
    return runs


def measure_flight(flight: Admitted, spec: VocabularySpec) -> dict[str, np.ndarray]:
    """Pass A: one admitted flight's contribution to every measurement that needs no measured
    value, as flat arrays."""
    smoothed, relative = flight.smoothed, flight.relative
    track, speed, distance = smoothed.track_deg, smoothed.ground_speed_mps, smoothed.distance_m
    out: dict[str, list[float]] = {name: [] for name in (
        *(f"heading_wander_deg_band{h:g}" for h in FREE_HOLD_HALF_RANGES_DEG),
        *(f"level_wander_m_fit{t:g}" for t in LEVEL_FIT_TOLERANCES_M),
        "turn_mean_rate_deg_s", "turn_row_rate_deg_s", "turn_row_bank_deg", "speed_wander_mps", "transition_accel_mps2",
        "final_course_error_deg", "move_angle_deg", "move_length_m", "turns_over_max",
    )}
    for half_range in FREE_HOLD_HALF_RANGES_DEG:
        for start, stop in _free_holds(track, spec.rows(spec.heading_min_hold_s), half_range):
            segment = track[start:stop]
            out[f"heading_wander_deg_band{half_range:g}"] += list(np.abs(segment - np.median(segment)))
    rate = np.diff(track) / spec.step_s
    bank = envelope.bank_deg_from_turn_rate(rate, speed[1:])
    for start, stop in _turn_runs(rate, spec.turn_onset_rate_deg_s):
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
    for piece in vertical_pieces(distance, smoothed.altitude_m, spec, Words(spec)):
        if piece.kind == MOVE:
            out["move_angle_deg"].append(piece.angle_deg)
            out["move_length_m"].append(float(distance[piece.stop - 1] - distance[piece.start]))
    holds = find_holds(track, len(track), spec)
    for previous, hold in zip(holds, holds[1:]):
        if abs(hold.target_unwrapped - previous.target_unwrapped) > spec.heading_max_turn_deg:
            out["turns_over_max"].append(abs(hold.target_unwrapped - previous.target_unwrapped))
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
    corridor), and for each heading grid (step → tolerance) the holds before the aligned final,
    their changes of target and each hold's funnel half width at its end — the trade the grid
    choice is made on."""
    first = aligned_final_row(flight, course_tolerance_deg)
    relative, smoothed = flight.relative, flight.smoothed
    arrays = {"aligned_offset_m": np.zeros(0), "aligned_distance_m": np.zeros(0)}
    if first is not None:
        arrays = {"aligned_offset_m": np.abs(relative.right_of_course_m[first:]),
                  "aligned_distance_m": np.asarray(relative.before_threshold_m[first:], dtype=np.float64)}
    stop = len(smoothed.track_deg) if first is None else first
    rows: dict[str, list[float]] = {}
    for step, tolerance in grids.items():
        grid_spec = _with(spec, heading_step_deg=step, heading_tolerance_deg=tolerance)
        holds = find_holds(smoothed.track_deg, stop, grid_spec)
        changes = sum(1 for a, b in zip(holds, holds[1:]) if a.target_unwrapped != b.target_unwrapped)
        widths = [float(envelope.funnel_half_width_m(smoothed.distance_m[h.stop - 1] - smoothed.distance_m[h.start],
                                                     tolerance)) for h in holds]
        rows[f"{step:g}"] = [float(len(holds)), float(changes), *widths]
    return arrays, rows


def _with(spec: VocabularySpec, **changes: Any) -> VocabularySpec:
    data = spec.to_dict()
    data.update(changes)
    return VocabularySpec.from_dict(data)


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
