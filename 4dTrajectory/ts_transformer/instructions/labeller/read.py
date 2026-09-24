"""One flight: signals → sentence, or a refusal with its reason (vocabulary design §3)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from final_approach.crossing import bracket_fraction
from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import (
    AirportGeometry, RunwayCandidate, RunwayRelative, landing_cross_limit_m, relative_to_runway,
)
from ts_transformer.instructions.labeller.lateral import read_lateral
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.labeller.sentence import assemble
from ts_transformer.instructions.labeller.speed import read_speed, span_checks
from ts_transformer.instructions.labeller.vertical import read_vertical, tube_checks
from ts_transformer.instructions.piecewise import moving_average
from ts_transformer.instructions.signals import ROW_FIELDS, FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import HEADING, RUNWAY, Words


@dataclass
class Smoothed:
    track_deg: np.ndarray
    altitude_m: np.ndarray
    ground_speed_mps: np.ndarray
    distance_m: np.ndarray


def smooth(signals: FlightSignals, spec: VocabularySpec) -> Smoothed:
    speed = moving_average(signals.ground_speed_mps, spec.rows(spec.speed_smoothing_s))
    dt = np.diff(signals.time_s)
    distance = np.concatenate(([0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * dt)))
    return Smoothed(
        track_deg=moving_average(signals.track_deg, spec.rows(spec.track_smoothing_s)),
        altitude_m=moving_average(signals.altitude_m, spec.rows(spec.altitude_smoothing_s)),
        ground_speed_mps=speed,
        distance_m=distance,
    )


@dataclass
class Reading:
    dataset_id: str
    airport: str
    runway_index: int
    words: np.ndarray                     # [N, 6] int16, UNCHANGED = -1
    instructions: list[Instruction]
    capture_row: int
    join_row: int
    unspecified_row: int
    #: The sentence covers the signal rows before the threshold crossing: ``len(words)`` rows.
    cut_at_crossing: bool
    checks: dict[str, Any] = field(default_factory=dict)


def landing_passages(relative: RunwayRelative, cross_limit_m: float, spec: VocabularySpec) -> list[int]:
    """Rows at which the flight lands on this runway (§2.2), as the harvest judges a landing
    (`trajectory_data_process.harvest.threshold_event`): the previous row is ahead of the threshold
    plane and this one on or past it, and at the crossing — interpolated between the two rows with
    `final_approach.crossing.bracket_fraction` — the flight is within ``cross_limit_m`` of the
    centreline and within `landing_max_height_m` of the threshold's height
    (`final_approach.assign.LandingScreen`)."""
    before, right, height = (relative.before_threshold_m, relative.right_of_course_m,
                             relative.height_above_threshold_m)
    rows = []
    for row in np.nonzero((before[:-1] > 0.0) & (before[1:] <= 0.0))[0]:
        fraction = bracket_fraction(-float(before[row]), -float(before[row + 1]))
        cross = float(right[row] + fraction * (right[row + 1] - right[row]))
        above = float(height[row] + fraction * (height[row + 1] - height[row]))
        if abs(cross) <= cross_limit_m and abs(above) <= spec.landing_max_height_m:
            rows.append(int(row) + 1)
    return rows


@dataclass(frozen=True)
class Admitted:
    """A flight the labeller reads: its rows before the landing, smoothed, relative to its runway."""

    signals: FlightSignals
    runway_index: int
    candidate: RunwayCandidate
    smoothed: Smoothed
    relative: RunwayRelative       # on the smoothed track and altitude
    cut_at_crossing: bool


def admit(signals: FlightSignals, geometry: AirportGeometry, spec: VocabularySpec) -> Admitted:
    """The one gate in front of both the labeller and the spec's measurements: the step, the
    runway, the cut before the landing, and the ground-speed refusals (§3.1, §3.6).

    The data plane cuts a flight at its landing, so a series normally ends short of the threshold.
    A landing as the harvest judges one (`landing_passages`) that the flight comes back from is not
    its landing (a low pass, or an approach flown before the one that landed): refused, since the
    landing it would be read up to is not in the series. A crossing of the plane too high or too far
    off the centreline to be a landing (an overflight, a downwind abeam) is not a landing at all."""
    if not np.allclose(np.diff(signals.time_s), spec.step_s, atol=1e-6):
        raise Refused("step mismatch", f"rows are not {spec.step_s:g} s apart")
    try:
        runway_index = geometry.candidate_index(signals.runway)
    except KeyError as error:
        raise Refused("runway not a candidate", str(error)) from None
    candidate = geometry.candidates[runway_index]
    raw = relative_to_runway(signals.e_m, signals.n_m, signals.track_deg, signals.altitude_m, candidate)
    limit = landing_cross_limit_m(geometry, runway_index, spec.landing_cross_limit_m, spec.parallel_course_delta_deg)
    passages = landing_passages(raw, limit, spec)
    crossing = passages[0] if passages else None
    if crossing is not None:
        if (raw.before_threshold_m[crossing:] > 0.0).any():
            raise Refused("threshold passed before the landing",
                          f"lands at row {crossing} (as the harvest judges a landing) and comes back ahead of it")
        signals = truncated(signals, crossing)
    if signals.n_rows < 2:
        raise Refused("too short", f"{signals.n_rows} rows before the threshold")
    speed = signals.ground_speed_mps
    if speed.min() < spec.ground_speed_floor_mps or speed.max() > spec.ground_speed_ceiling_mps:
        raise Refused("impossible ground speed", f"raw rows {speed.min():.1f}–{speed.max():.1f} m/s")
    smoothed = smooth(signals, spec)
    speed = smoothed.ground_speed_mps
    low, high = spec.speed_min_mps - spec.speed_tolerance_mps, spec.speed_max_mps + spec.speed_tolerance_mps
    if speed.min() < low or speed.max() > high:
        raise Refused("ground speed outside the speed words", f"smoothed {speed.min():.1f}–{speed.max():.1f} m/s")
    relative = relative_to_runway(signals.e_m, signals.n_m, smoothed.track_deg, smoothed.altitude_m, candidate)
    return Admitted(signals=signals, runway_index=runway_index, candidate=candidate, smoothed=smoothed,
                    relative=relative, cut_at_crossing=crossing is not None)


def read_flight(signals: FlightSignals, geometry: AirportGeometry, spec: VocabularySpec,
                words: Words | None = None) -> Reading:
    words = words or Words(spec)
    flight = admit(signals, geometry, spec)
    signals, smoothed, relative, candidate = flight.signals, flight.smoothed, flight.relative, flight.candidate
    lateral = read_lateral(smoothed.track_deg, smoothed.ground_speed_mps, relative, candidate.course_deg, spec, words)
    vertical = read_vertical(smoothed.distance_m, smoothed.altitude_m, spec, words)
    speeds = read_speed(signals.time_s, smoothed.ground_speed_mps, relative.before_threshold_m, lateral.join_row,
                        spec, words)
    instructions = [Instruction(RUNWAY, flight.runway_index, 0, "initial", {"ident": candidate.ident}),
                    *lateral.instructions, *vertical.instructions, *speeds.instructions]
    grid, kept = assemble(signals.n_rows, instructions, smoothed.altitude_m, spec, words)

    # every check runs on the sentence as kept: a word the assembly dropped is not judged
    heading = [(item.row, float(item.info["target_deg"])) for item in kept if item.column == HEADING]
    checks = {
        "heading": envelope.heading_words_inside(smoothed.track_deg, heading, spec.rows_exact(spec.heading_lead_s),
                                                 lateral.join_row, spec.heading_tolerance_deg),
        "capture_turn": lateral.capture_turn, "turning_deg": lateral.turning_deg,
        "capture_before_threshold_m": float(relative.before_threshold_m[lateral.capture_row]),
        "vertical": tube_checks(kept, smoothed.distance_m, smoothed.altitude_m, spec, words),
        "speed": span_checks(kept, smoothed.ground_speed_mps, spec, words),
    }
    return Reading(dataset_id=signals.dataset_id, airport=signals.airport, runway_index=flight.runway_index,
                   words=grid, instructions=kept, capture_row=lateral.capture_row, join_row=lateral.join_row,
                   unspecified_row=speeds.unspecified_row, cut_at_crossing=flight.cut_at_crossing, checks=checks)


def truncated(signals: FlightSignals, rows: int) -> FlightSignals:
    """The first ``rows`` rows of a flight."""
    return replace(signals, **{name: getattr(signals, name)[:rows] for name in ROW_FIELDS})
