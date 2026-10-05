"""One flight: signals → sentence, or a refusal with its reason (vocabulary §4).

A flight is read APPROACH BY APPROACH (D26): an approach runs from row 0, or from the runway word that ends a go-around,
to the landing or to the next go-around row. Each go-around inside the sentence (`go_around.go_arounds`, D18) ends one:
its row is the row its climb is said at (`vertical.read_vertical`), where the runway column says "go-around". The
approach before it flies to the runway of its low pass; the last approach to the landed runway (the harvest's
assignment). Row 0 says the first approach's runway (§4.2), and the runway word that ends each go-around the next
approach's (D19): at the first level-off after the climb (the first level piece of the altitude reading after the
go-around row held at least `go_around_min_climb_m` above the lowest point), and not later than the row of the next "no
level-off" (rule 5: the runway word, said first in its row, ends the go-around before it). The heading words read the
course of the runway in force at each row; the capture row, "unspecified" and the speed words are each approach's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from final_approach.crossing import bracket_fraction
from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import (
    AirportGeometry, RunwayCandidate, RunwayRelative, landing_cross_limit_m, relative_to_runway,
)
from ts_transformer.instructions.labeller.go_around import GoAround, go_arounds
from ts_transformer.instructions.labeller.lateral import Approach, read_lateral
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.labeller.sentence import assemble
from ts_transformer.instructions.labeller.speed import read_speed, span_checks
from ts_transformer.instructions.labeller.vertical import LEVEL, VerticalReading, held_height, read_vertical, tube_checks
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


@dataclass(frozen=True)
class ApproachReading:
    """One approach (`lateral.Approach`) as read: its capture row and the row "unspecified" starts (both None where it
    ends at a go-around row outside the corridor)."""

    first: int
    end: int
    runway_index: int
    capture_row: int | None
    unspecified_row: int | None


@dataclass
class Reading:
    dataset_id: str
    airport: str
    runway_index: int                     # the landed runway: the last approach's
    words: np.ndarray                     # [N, 5] int16, UNCHANGED = -1
    instructions: list[Instruction]
    #: The landing approach's capture row and "unspecified" row (the last of ``approaches``).
    capture_row: int
    unspecified_row: int
    #: Each go-around's row ("go-around" said) and the row the next approach's runway is said (D19), in order.
    go_around_rows: list[int]
    runway_again_rows: list[int]
    approaches: list[ApproachReading]
    #: The sentence covers the signal rows before the threshold crossing: ``len(words)`` rows.
    cut_at_crossing: bool
    #: The height above the airport elevation E each row's words were checked at (`vertical.held_height`, D58): what
    #: the grammar is read at again.
    held_height_m: np.ndarray
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
    #: Every candidate's frame on the same rows (``relatives[runway_index]`` is ``relative``).
    relatives: tuple[RunwayRelative, ...]
    cut_at_crossing: bool
    #: The go-arounds on its rows (`flight_go_arounds`): every landing passage it came back from lies in one's low pass.
    go_arounds: tuple[GoAround, ...]


def admit(signals: FlightSignals, geometry: AirportGeometry, spec: VocabularySpec) -> Admitted:
    """The one gate in front of both the labeller and the spec's measurements: the step, the
    runway, the cut before the landing, the go-arounds (and the passages they must explain), and the
    ground-speed refusals (§4.1, §4.6).

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
    passages_before = passages[:-1] if crossing is not None else passages
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
    relatives = tuple(relative_to_runway(signals.e_m, signals.n_m, smoothed.track_deg, smoothed.altitude_m, c)
                      for c in geometry.candidates)
    return Admitted(signals=signals, runway_index=runway_index, candidate=candidate, smoothed=smoothed,
                    relative=relatives[runway_index], relatives=relatives, cut_at_crossing=crossing is not None,
                    go_arounds=tuple(flight_go_arounds(signals, smoothed, relatives, passages_before, spec)))


def flight_go_arounds(signals: FlightSignals, smoothed: Smoothed, relatives: Sequence[RunwayRelative],
                      passages_before: list[int], spec: VocabularySpec) -> list[GoAround]:
    """A flight's go-arounds (`go_around.go_arounds`) on its smoothed altitude, every candidate's frame
    (``relatives``); refused when one is on the runway (a touch-and-go) or when a landing passage it came back from
    (``passages_before``) is not inside a go-around's low pass."""
    found = go_arounds(signals.time_s, smoothed.altitude_m, relatives, spec)
    for item in found:
        if item.on_runway:
            raise Refused("touch-and-go", f"row {item.point}: {item.along_m:.0f} m past the threshold, "
                                          f"{item.height_m:.0f} m up")
    for row in passages_before:
        if not any(item.low_pass.first <= row <= item.low_pass.last + 1 for item in found):
            raise Refused("threshold passed before the landing",
                          f"a landing passage at row {row} that no go-around explains")
    return found


def runway_again_rows(found: Sequence[GoAround], go_around_rows: list[int], vertical: VerticalReading,
                      height_m: np.ndarray, n_rows: int, spec: VocabularySpec, words: Words) -> list[int]:
    """The row of the runway word that ends each go-around (module docstring): the first level piece after its row held
    at least `go_around_min_climb_m` above its lowest point (R40's climb; a level flown at the low point before the climb
    is not it), at the latest the next "no level-off"; refused when it comes at or after the next go-around row."""
    no_level_off = [item.row for item in vertical.instructions
                    if item.column == ALTITUDE and item.value == words.altitude_no_level_off]
    out = []
    for position, (item, row) in enumerate(zip(found, go_around_rows)):
        climbed = float(height_m[item.point]) + spec.go_around_min_climb_m
        levels = [piece.start for piece in vertical.pieces
                  if piece.kind == LEVEL and piece.start > row and piece.median_m >= climbed]
        candidates = levels[:1] + [start for start in no_level_off if start >= row]
        if not candidates:
            raise Refused("go-around not ended", f"row {row}: no level-off and no final descent after it")
        again = min(candidates)
        following = go_around_rows[position + 1] if position + 1 < len(go_around_rows) else n_rows
        if again >= following:
            raise Refused("go-around not ended", f"row {row}: the next go-around at row {following} comes first")
        out.append(again)
    return out


def read_heights(flight: Admitted, height_m: np.ndarray, spec: VocabularySpec,
                 words: Words) -> tuple[VerticalReading, list[int]]:
    """An admitted flight's vertical reading (`vertical.read_vertical`) on ``height_m``, its smoothed heights above E,
    and the row of the runway word that ends each go-around (`runway_again_rows`): G is true from each go-around row to
    its runway word (D19). Refused as `read_flight` refuses them."""
    found = list(flight.go_arounds)
    vertical = read_vertical(flight.smoothed.distance_m, height_m, spec, words, [item.point for item in found])
    again = runway_again_rows(found, vertical.go_around_rows, vertical, height_m, flight.signals.n_rows, spec, words)
    return vertical, again


def read_flight(signals: FlightSignals, geometry: AirportGeometry, spec: VocabularySpec,
                words: Words | None = None) -> Reading:
    words = words or Words(spec)
    flight = admit(signals, geometry, spec)
    signals, smoothed = flight.signals, flight.smoothed
    found = list(flight.go_arounds)
    n_rows = signals.n_rows
    height = smoothed.altitude_m - geometry.elevation_m          # the altitude words: heights above E (D58)
    vertical, again = read_heights(flight, height, spec, words)
    rows = vertical.go_around_rows
    runways = [item.low_pass.candidate for item in found] + [flight.runway_index]
    approaches = [Approach(first, end, runway) for first, end, runway in zip([0, *again], [*rows, n_rows], runways)]
    runway_rows = np.full(n_rows, runways[0], dtype=np.int64)
    for row, runway in zip(again, runways[1:]):
        runway_rows[row:] = runway
    courses = [candidate.course_deg for candidate in geometry.candidates]
    lateral = read_lateral(smoothed.track_deg, list(flight.relatives), courses, approaches, runway_rows, spec, words)

    # the runway column (module docstring), and each approach's speed reading on its own rows
    idents = [geometry.candidates[runway].ident for runway in runways]
    instructions = [Instruction(RUNWAY, runways[0], 0, "initial", {"ident": idents[0]}),
                    *[Instruction(RUNWAY, RUNWAY_GO_AROUND, row, "go-around") for row in rows],
                    *[Instruction(RUNWAY, runway, row, "runway again", {"ident": ident})
                      for row, runway, ident in zip(again, runways[1:], idents[1:])]]
    approach_readings = []
    for position, (approach, capture) in enumerate(zip(approaches, lateral.capture_rows)):
        start = rows[position - 1] if position else 0
        span = slice(start, approach.end)
        reading = read_speed(signals.time_s[span], smoothed.ground_speed_mps[span],
                             flight.relatives[approach.runway_index].before_threshold_m[span],
                             None if capture is None else capture - start, spec, words)
        for item in reading.instructions:          # a later approach's reading starts at its go-around row
            moved = replace(item, row=item.row + start)
            instructions.append(replace(moved, kind="approach") if position and item.row == 0 else moved)
        unspecified = None if reading.unspecified_row is None else reading.unspecified_row + start
        approach_readings.append(ApproachReading(approach.first, approach.end, approach.runway_index, capture,
                                                 unspecified))
    instructions += [*lateral.instructions, *vertical.instructions]
    held = held_height(vertical.pieces, height)
    grid, kept = assemble(n_rows, instructions, held, words, courses)

    # every check runs on the sentence as kept: a word the assembly dropped is not judged
    landing = approach_readings[-1]
    heading = [(item.row, float(item.info["target_deg"])) for item in kept if item.column == HEADING]
    checks = {
        "heading": envelope.heading_words_inside(smoothed.track_deg, heading, spec.rows_exact(spec.heading_lead_s),
                                                 n_rows, spec.heading_tolerance_deg),
        "turning_deg": lateral.turning_deg,
        "capture_before_threshold_m": float(flight.relative.before_threshold_m[landing.capture_row]),
        "go_arounds": [{"row": row, "point": item.point, "candidate": item.low_pass.candidate,
                        "height_m": item.height_m, "along_m": item.along_m, "drop_m": item.drop_m,
                        "climb_m": item.climb_m} for item, row in zip(found, rows)],
        "vertical": tube_checks(kept, smoothed.distance_m, height, spec, words),
        "speed": span_checks(kept, smoothed.ground_speed_mps, spec, words),
    }
    return Reading(dataset_id=signals.dataset_id, airport=signals.airport, runway_index=flight.runway_index,
                   words=grid, instructions=kept, capture_row=landing.capture_row,
                   unspecified_row=landing.unspecified_row, go_around_rows=list(rows), runway_again_rows=again,
                   approaches=approach_readings, cut_at_crossing=flight.cut_at_crossing, held_height_m=held,
                   checks=checks)


def truncated(signals: FlightSignals, rows: int) -> FlightSignals:
    """The first ``rows`` rows of a flight."""
    return replace(signals, **{name: getattr(signals, name)[:rows] for name in ROW_FIELDS})
