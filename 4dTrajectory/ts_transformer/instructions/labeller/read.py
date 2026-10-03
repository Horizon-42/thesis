"""One flight: signals → sentence, or a refusal with its reason (design §4).

The runway column says the landed runway at row 0 (the harvest's assignment; the data cannot show a change of runway,
so a labelled sentence never changes it, §4.2). Each go-around inside the sentence (`go_around.go_arounds`, D18) says
"go-around" at its point, and the landed runway again where the approach is taken up (D19): at the first level-off
after the go-around (the first level piece of the altitude reading that starts after the point), and not later than
the row of the next "no level-off" (rule 5: the runway word, said first in its row, ends the go-around before it).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from final_approach.crossing import bracket_fraction
from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import (
    AirportGeometry, RunwayCandidate, RunwayRelative, landing_cross_limit_m, relative_to_runway,
)
from ts_transformer.instructions.labeller.go_around import GoAround, go_arounds
from ts_transformer.instructions.labeller.lateral import read_lateral
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.labeller.sentence import assemble
from ts_transformer.instructions.labeller.speed import read_speed, span_checks
from ts_transformer.instructions.labeller.vertical import LEVEL, VerticalReading, held_altitude, read_vertical, tube_checks
from ts_transformer.instructions.piecewise import moving_average
from ts_transformer.instructions.signals import ROW_FIELDS, FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import ALTITUDE, HEADING, RUNWAY, RUNWAY_GO_AROUND, Words


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
    words: np.ndarray                     # [N, 5] int16, UNCHANGED = -1
    instructions: list[Instruction]
    capture_row: int
    unspecified_row: int
    #: Each go-around's row ("go-around" said) and the row the landed runway is said again (D19), in order.
    go_around_rows: list[int]
    runway_again_rows: list[int]
    #: The sentence covers the signal rows before the threshold crossing: ``len(words)`` rows.
    cut_at_crossing: bool
    #: The height each row's words were checked at (`vertical.held_altitude`): what the grammar is read at again.
    held_altitude_m: np.ndarray
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
    #: The rows of the landing passages BEFORE the landing (low crossings the flight came back from): each must lie in
    #: a go-around's low pass (`read_flight`).
    passages_before: tuple[int, ...] = ()


def admit(signals: FlightSignals, geometry: AirportGeometry, spec: VocabularySpec) -> Admitted:
    """The one gate in front of both the labeller and the spec's measurements: the step, the
    runway, the cut before the landing, and the ground-speed refusals (§3.1, §3.6).

    The data plane cuts a flight at its landing, so a series normally ends short of the threshold and
    holds no landing passage at all. A landing passage (`landing_passages`, as the harvest judges a
    landing) the flight never comes back ahead of the threshold from is its landing: the sentence is cut
    there. Every other passage is a low pass the flight came back from, which `read_flight` reads as a
    go-around or refuses (``passages_before``). A crossing of the plane too high or too far off the
    centreline to be a landing (an overflight, a downwind abeam) is not a landing at all."""
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
    crossing = passages[-1] if passages and not (raw.before_threshold_m[passages[-1]:] > 0.0).any() else None
    passages_before = tuple(passages[:-1] if crossing is not None else passages)
    if crossing is not None:
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
                    relative=relative, cut_at_crossing=crossing is not None, passages_before=passages_before)


def flight_go_arounds(flight: Admitted, geometry: AirportGeometry, spec: VocabularySpec) -> list[GoAround]:
    """The admitted flight's go-arounds (`go_around.go_arounds`) on its smoothed altitude, every candidate's frame;
    refused when one is on the runway (a touch-and-go) or when a landing passage before the landing is not inside a
    go-around's low pass."""
    smoothed, signals = flight.smoothed, flight.signals
    relatives = [relative_to_runway(signals.e_m, signals.n_m, smoothed.track_deg, smoothed.altitude_m, candidate)
                 for candidate in geometry.candidates]
    found = go_arounds(signals.time_s, smoothed.altitude_m, relatives, spec)
    for item in found:
        if item.on_runway:
            raise Refused("touch-and-go", f"row {item.row}: {item.along_m:.0f} m past the threshold, "
                                          f"{item.height_m:.0f} m up")
    for row in flight.passages_before:
        if not any(item.low_pass.first <= row <= item.low_pass.last + 1 for item in found):
            raise Refused("threshold passed before the landing",
                          f"a landing passage at row {row} that no go-around explains")
    return found


def runway_words(found: list[GoAround], vertical: VerticalReading, altitude_m: np.ndarray, runway_index: int,
                 ident: str, n_rows: int, spec: VocabularySpec, words: Words) -> list[Instruction]:
    """The runway column's words (module docstring): the landed runway at row 0, and per go-around "go-around" at its
    point and the landed runway again at the first level-off after its climb — the first level piece after the point
    held at least `go_around_min_climb_m` above it (R40's climb; a level flown at the low point before the climb is not
    it) — at the latest at the next "no level-off"."""
    out = [Instruction(RUNWAY, runway_index, 0, "initial", {"ident": ident})]
    no_level_off = [item.row for item in vertical.instructions
                    if item.column == ALTITUDE and item.value == words.altitude_no_level_off]
    rows = [item.row for item in found]
    for position, row in enumerate(rows):
        climbed = float(altitude_m[row]) + spec.go_around_min_climb_m
        levels = [piece.start for piece in vertical.pieces
                  if piece.kind == LEVEL and piece.start > row and piece.median_m >= climbed]
        candidates = levels[:1] + [start for start in no_level_off if start >= row]
        if not candidates:
            raise Refused("go-around not ended", f"row {row}: no level-off and no final descent after it")
        again = min(candidates)
        following = rows[position + 1] if position + 1 < len(rows) else n_rows
        if again >= following:
            raise Refused("go-around not ended", f"row {row}: the next go-around at row {following} comes first")
        out += [Instruction(RUNWAY, RUNWAY_GO_AROUND, row, "go-around"),
                Instruction(RUNWAY, runway_index, again, "runway again", {"ident": ident})]
    return out


def read_flight(signals: FlightSignals, geometry: AirportGeometry, spec: VocabularySpec,
                words: Words | None = None) -> Reading:
    words = words or Words(spec)
    flight = admit(signals, geometry, spec)
    signals, smoothed, relative, candidate = flight.signals, flight.smoothed, flight.relative, flight.candidate
    found = flight_go_arounds(flight, geometry, spec)
    lateral = read_lateral(smoothed.track_deg, relative, candidate.course_deg, spec, words)
    vertical = read_vertical(smoothed.distance_m, smoothed.altitude_m, spec, words)
    runway = runway_words(found, vertical, smoothed.altitude_m, flight.runway_index, candidate.ident, signals.n_rows,
                          spec, words)
    speeds = read_speed(signals.time_s, smoothed.ground_speed_mps, relative.before_threshold_m, lateral.capture_row,
                        spec, words)
    instructions = [*runway, *lateral.instructions, *vertical.instructions, *speeds.instructions]
    held = held_altitude(vertical.pieces, smoothed.altitude_m)
    grid, kept = assemble(signals.n_rows, instructions, held, words, len(geometry.candidates))

    # every check runs on the sentence as kept: a word the assembly dropped is not judged
    heading = [(item.row, float(item.info["target_deg"])) for item in kept if item.column == HEADING]
    checks = {
        "heading": envelope.heading_words_inside(smoothed.track_deg, heading, spec.rows_exact(spec.heading_lead_s),
                                                 signals.n_rows, spec.heading_tolerance_deg),
        "turning_deg": lateral.turning_deg,
        "capture_before_threshold_m": float(relative.before_threshold_m[lateral.capture_row]),
        "go_arounds": [{"row": item.row, "candidate": item.low_pass.candidate, "height_m": item.height_m,
                        "along_m": item.along_m, "drop_m": item.drop_m, "climb_m": item.climb_m} for item in found],
        "vertical": tube_checks(kept, smoothed.distance_m, smoothed.altitude_m, spec, words),
        "speed": span_checks(kept, smoothed.ground_speed_mps, spec, words),
    }
    return Reading(dataset_id=signals.dataset_id, airport=signals.airport, runway_index=flight.runway_index,
                   words=grid, instructions=kept, capture_row=lateral.capture_row,
                   unspecified_row=speeds.unspecified_row,
                   go_around_rows=[item.row for item in runway if item.kind == "go-around"],
                   runway_again_rows=[item.row for item in runway if item.kind == "runway again"],
                   cut_at_crossing=flight.cut_at_crossing, held_altitude_m=held, checks=checks)


def truncated(signals: FlightSignals, rows: int) -> FlightSignals:
    """The first ``rows`` rows of a flight."""
    return replace(signals, **{name: getattr(signals, name)[:rows] for name in ROW_FIELDS})
