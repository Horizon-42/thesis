"""What a read sentence's envelopes look like, as geometry to draw (vocabulary design §2).

The design's rule is ONE implementation of the envelopes for the labeller's checks, the executor's
limits and the display (§1 principle 4). This module is the display side, built only from the
public functions of `envelope.py` and `labeller/*`:

- a turn's radius is the inverse of `envelope.bank_deg_from_turn_rate` (and checked against it:
  a turn at the radius's rate has exactly that bank);
- a hold's funnel widens by `envelope.funnel_half_width_m`, the capture corridor by
  `envelope.corridor_half_width_m`, and its rows are judged by `envelope.corridor`;
- the altitude tubes ARE `labeller.vertical.tube_bounds`; the speed spans' verdicts are
  `labeller.speed.span_checks` and their band rows `envelope.speed_band`;
- every verdict shown is the labeller's own (`Reading.checks`), never recomputed.

It is outside the labeller's source hash (`artefact.LABELLER_MODULES`): changing how a sentence is
drawn does not change the sentence. Everything is in the airport frame (metres east / north of the
airport reference point, geometric MSL, compass degrees true, seconds); the exporter adds the
geodesy (latitude / longitude, the ellipsoid heights Cesium draws in).

Two readings of the design that this module makes, both stated where they are made:

- the TURN REGION is swept by the arcs of every bank in [`turn_bank_min_deg`, `turn_bank_max_deg`]
  at the issue ground speed, turning the shorter way to θ. The labeller applies the lowest bank
  only to turns of at least `turn_bank_min_from_deg` (a smaller change of the ground track is
  mostly wind drift); the region is still drawn with the full range, and the word's verdict says
  whether the lowest bank applied (`bank_min_applies`).
- the HOLD FUNNEL starts where the turn ends. The turn ends somewhere on the segment between the
  tightest and the widest arc's end (which one depends on the bank flown), so the funnel starts as
  that segment's extent across θ, its nominal line along θ through the segment's midpoint, and
  widens by the distance flown along θ × tan(`heading_tolerance_deg`). A word the flight was
  already holding when the slice began has no turn: its funnel starts at the issue point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from aerodynamic_model.common import GRAVITY_MPS2
from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import RunwayCandidate
from ts_transformer.instructions.labeller.read import Admitted, Reading
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.labeller.speed import span_checks
from ts_transformer.instructions.labeller.vertical import tube_bounds
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import ANGLE, ANGLE_LEVEL, HEADING, SPEED, Words, wrap180

#: One outline point per this many degrees of an arc. A drawing resolution, not an envelope value.
ARC_STEP_DEG = 2.0


# ---- plane geometry (compass bearings: 0 = north, clockwise; unit vector (sin b, cos b) in (E, N))
def _unit(bearing_deg: float) -> np.ndarray:
    radians = math.radians(bearing_deg)
    return np.array([math.sin(radians), math.cos(radians)])


def _right(bearing_deg: float) -> np.ndarray:
    """The unit vector 90° clockwise of a bearing: 'right of the track'."""
    return _unit(bearing_deg + 90.0)


@dataclass(frozen=True)
class Line:
    """Points in the airport frame, in drawing order."""

    e_m: np.ndarray
    n_m: np.ndarray

    @classmethod
    def of(cls, points: list[np.ndarray] | np.ndarray) -> Line:
        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return cls(e_m=array[:, 0].copy(), n_m=array[:, 1].copy())


# ---- turns
def turn_radius_m(ground_speed_mps: float, bank_deg: float) -> float:
    """The radius of a coordinated turn at this bank and ground speed: R = V² / (g·tan φ).

    The inverse of `envelope.bank_deg_from_turn_rate` — a turn flown at rate V / R has bank φ —
    and checked against it, so the radius drawn and the bank the labeller judges cannot part."""
    radius = ground_speed_mps ** 2 / (GRAVITY_MPS2 * math.tan(math.radians(bank_deg)))
    back = float(envelope.bank_deg_from_turn_rate(math.degrees(ground_speed_mps / radius), ground_speed_mps))
    if not math.isclose(back, bank_deg, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(f"the turn radius for {bank_deg}° at {ground_speed_mps} m/s reads back as {back}° "
                           f"through envelope.bank_deg_from_turn_rate: the display and the envelope disagree")
    return radius


def arc(start: np.ndarray, track_deg: float, turn_deg: float, radius_m: float) -> np.ndarray:
    """``[points, 2]``: the path of a constant-radius turn from ``start`` on ``track_deg``, through
    ``turn_deg`` (positive = right, clockwise)."""
    side = 1.0 if turn_deg >= 0.0 else -1.0
    centre = start + radius_m * _right(track_deg) * side
    count = max(2, math.ceil(abs(turn_deg) / ARC_STEP_DEG) + 1)
    return np.array([centre + radius_m * _unit(track_deg + side * (progress - 90.0))
                     for progress in np.linspace(0.0, abs(turn_deg), count)])


@dataclass(frozen=True)
class TurnRegion:
    """Where a turn from ``start`` may take the aircraft: the region the arcs of every allowed
    bank sweep (§2.3). ``turn_deg`` is the shorter way to the target (positive = right)."""

    from_track_deg: float
    turn_deg: float
    ground_speed_mps: float
    radius_min_m: float          # at the highest bank
    radius_max_m: float          # at the lowest bank
    inner: Line                  # the tightest arc, start → its end
    outer: Line                  # the widest arc, start → its end
    outline: Line                # inner arc, then the widest arc back to the start
    end: Line                    # the turn's end: the tightest arc's end → the widest arc's end


def turn_region(start: np.ndarray, from_track_deg: float, target_deg: float, ground_speed_mps: float,
                spec: VocabularySpec) -> TurnRegion:
    turn = float(wrap180(target_deg - from_track_deg))
    radius_min = turn_radius_m(ground_speed_mps, spec.turn_bank_max_deg)
    radius_max = turn_radius_m(ground_speed_mps, spec.turn_bank_min_deg)
    inner = arc(start, from_track_deg, turn, radius_min)
    outer = arc(start, from_track_deg, turn, radius_max)
    return TurnRegion(
        from_track_deg=float(from_track_deg % 360.0), turn_deg=turn, ground_speed_mps=float(ground_speed_mps),
        radius_min_m=radius_min, radius_max_m=radius_max, inner=Line.of(inner), outer=Line.of(outer),
        outline=Line.of(np.concatenate((inner, outer[::-1][:-1]))), end=Line.of([inner[-1], outer[-1]]))


# ---- holds
@dataclass(frozen=True)
class Funnel:
    """A hold's allowed positions along θ (§2.3): the nominal line and its widening half width."""

    target_deg: float
    length_m: float
    start_half_width_m: float
    end_half_width_m: float
    axis: Line                   # the nominal line, start → end
    outline: Line                # start cap left, end cap left, end cap right, start cap right


def funnel(turn_end: Line, target_deg: float, length_m: float, spec: VocabularySpec) -> Funnel:
    """From ``turn_end`` (one point when no turn was flown): the nominal line along θ through its
    midpoint, half width = its extent across θ + `envelope.funnel_half_width_m` of the distance
    flown along θ."""
    ends = np.column_stack((turn_end.e_m, turn_end.n_m))
    middle = ends.mean(axis=0)
    along, right = _unit(target_deg), _right(target_deg)
    across = (ends - middle) @ right
    start_width = float((across.max() - across.min()) / 2.0)
    end_width = start_width + float(envelope.funnel_half_width_m(length_m, spec.heading_tolerance_deg))
    far = middle + length_m * along
    return Funnel(
        target_deg=float(target_deg % 360.0), length_m=float(length_m), start_half_width_m=start_width,
        end_half_width_m=end_width, axis=Line.of([middle, far]),
        outline=Line.of([middle - start_width * right, far - end_width * right,
                         far + end_width * right, middle + start_width * right]))


def on_branch(reference_deg: float, target_deg: float) -> float:
    """``target_deg`` on the branch of an unwrapped track at ``reference_deg`` (the shorter way)."""
    return float(reference_deg + wrap180(target_deg - reference_deg))


@dataclass(frozen=True)
class HeadingEnvelope:
    """One heading word's envelope, from its issue point (§2.3)."""

    word: Instruction
    target_deg: float            # θ, the word's own grid value
    #: Where its turn ends: the labeller's turn end for a single turn or a split turn's last part,
    #: the next part's issue row for an earlier part; ``None`` for a word with no turn.
    turn_end_row: int | None
    #: Where its hold begins: its turn's end as the labeller read it (the issue row for a word the
    #: flight was already holding at entry); ``None`` for a split part the next part supersedes
    #: mid-turn.
    hold_start_row: int | None
    #: Where its hold ends: the next heading word's row, or — for the last one — where the
    #: labeller's capture turn begins (the capture itself when there is no capture turn).
    hold_end_row: int
    from_track_deg: float        # the smoothed track at the issue row, unwrapped
    target_on_track_deg: float   # θ on that track's branch
    #: The chart bands. The turn: from the track at issue to θ, each side widened by the tolerance
    #: (the bound `envelope.turn_progress_ok` implies); ``None`` for a word with no turn. The hold:
    #: θ ± the tolerance; ``None`` when the word has no hold.
    turn_band_deg: tuple[float, float] | None
    hold_band_deg: tuple[float, float] | None
    #: The labeller's check of the turn this word belongs to (`Reading.checks["turns"]`) — shared by
    #: the parts of a split turn; ``None`` for a word the flight was already holding.
    turn_check: dict[str, Any] | None
    turn: TurnRegion | None
    funnel: Funnel | None


def _turn_check(word: Instruction, turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The turn check a heading word belongs to: the one it DEPARTS (a single turn, or a split
    turn's first part), or the split turn it continues (a later part, issued mid-turn)."""
    if word.kind == "initial":
        return None
    kind = word.kind.removesuffix("-split")
    if word.info["part"] == 1:
        found = [t for t in turns if t["kind"] == kind and t["departure_row"] == word.row]
    else:
        found = [t for t in turns if t["kind"] == kind and t["departure_row"] < word.row <= t["arrival_row"]]
    if len(found) != 1:
        raise ValueError(f"heading word {word.kind} at row {word.row}: {len(found)} turn checks cover it")
    return found[0]


def heading_envelopes(flight: Admitted, reading: Reading, spec: VocabularySpec, words: Words) -> list[HeadingEnvelope]:
    smoothed = flight.smoothed
    track, speed, distance = smoothed.track_deg, smoothed.ground_speed_mps, smoothed.distance_m
    positions = np.column_stack((flight.signals.e_m, flight.signals.n_m))
    heading = sorted((i for i in reading.instructions if i.column == HEADING), key=lambda item: item.row)
    capture = reading.checks["capture_turn"]
    last_end = reading.capture_row if capture is None else int(capture["start_row"])
    ends = [w.row for w in heading[1:]] + [last_end]
    tolerance = spec.heading_tolerance_deg
    result = []
    for word, hold_end in zip(heading, ends):
        check = _turn_check(word, reading.checks["turns"])
        target = words.heading_deg(word.value)
        start_track = float(track[word.row])
        target_on_track = on_branch(start_track, target)
        if check is None:
            hold_start: int | None = word.row
            turn_end_row: int | None = None
            region, turn_band = None, None
            turn_end = Line.of([positions[word.row]])
        else:
            hold_start = int(check["arrival_row"]) if word.info["part"] == word.info["parts"] else None
            turn_end_row = int(hold_end) if hold_start is None else hold_start
            region = turn_region(positions[word.row], start_track, target, float(speed[word.row]), spec)
            low, high = sorted((start_track, target_on_track))
            turn_band = (low - tolerance, high + tolerance)
            turn_end = region.end
        held = hold_start is not None and hold_end > hold_start
        drawn = funnel(turn_end, target, float(distance[hold_end] - distance[hold_start]), spec) if held else None
        result.append(HeadingEnvelope(
            word=word, target_deg=target, turn_end_row=turn_end_row, hold_start_row=hold_start if held else None, hold_end_row=int(hold_end),
            from_track_deg=start_track, target_on_track_deg=target_on_track, turn_band_deg=turn_band,
            hold_band_deg=(target_on_track - tolerance, target_on_track + tolerance) if held else None,
            turn_check=check, turn=region, funnel=drawn))
    return result


# ---- the approach
def centreline(candidate: RunwayCandidate, length_m: float) -> Line:
    """The extended centreline, from the threshold out along the approach side."""
    threshold = np.array([candidate.threshold_e_m, candidate.threshold_n_m])
    return Line.of([threshold, threshold - length_m * _unit(candidate.course_deg)])


def runway(candidate: RunwayCandidate) -> Line:
    """The runway itself: the threshold to its far end along the course."""
    threshold = np.array([candidate.threshold_e_m, candidate.threshold_n_m])
    return Line.of([threshold, threshold + candidate.length_m * _unit(candidate.course_deg)])


@dataclass(frozen=True)
class Corridor:
    """After the capture (§2.2): within `envelope.corridor_half_width_m` of the extended centreline
    and `corridor_course_tolerance_deg` of the course, from the capture to the threshold."""

    before_threshold_m: float     # the capture point's distance before the threshold
    half_width_at_capture_m: float
    half_width_at_threshold_m: float
    axis: Line                    # capture distance → threshold, on the centreline
    outline: Line
    #: The rows from the capture to the sentence's end — every one inside `envelope.corridor`, by the
    #: capture's definition (checked).
    rows: int


def corridor(flight: Admitted, reading: Reading, spec: VocabularySpec) -> Corridor:
    candidate, relative = flight.candidate, flight.relative
    rows = slice(reading.capture_row, len(reading.words))
    inside = envelope.corridor(relative.right_of_course_m[rows], relative.track_minus_course_deg[rows],
                               relative.before_threshold_m[rows], spec.corridor_half_width_m,
                               spec.corridor_widening_deg, spec.corridor_course_tolerance_deg)
    if not inside.all():
        raise ValueError(f"{reading.dataset_id}: {int(np.count_nonzero(~inside))} rows after the capture leave the "
                         f"corridor, which the capture's definition excludes")
    before = float(relative.before_threshold_m[reading.capture_row])
    far_width, near_width = (float(envelope.corridor_half_width_m(d, spec.corridor_half_width_m,
                                                                   spec.corridor_widening_deg)) for d in (before, 0.0))
    threshold = np.array([candidate.threshold_e_m, candidate.threshold_n_m])
    along, right = _unit(candidate.course_deg), _right(candidate.course_deg)
    far = threshold - before * along
    return Corridor(
        before_threshold_m=before, half_width_at_capture_m=far_width, half_width_at_threshold_m=near_width,
        axis=Line.of([far, threshold]),
        outline=Line.of([far - far_width * right, threshold - near_width * right,
                         threshold + near_width * right, far + far_width * right]),
        rows=int(len(inside)))


@dataclass(frozen=True)
class CaptureTurn:
    """From the end of the last hold (or of an inserted intercept) onto the course (§2.2)."""

    start_row: int
    course_on_track_deg: float    # the course on the branch of the smoothed track at the start row
    #: The chart band: from the track at the start row to the course, each side widened by the
    #: heading tolerance (the bound the labeller's progress check implies).
    band_deg: tuple[float, float]
    check: dict[str, Any]         # `Reading.checks["capture_turn"]`
    region: TurnRegion


def capture_turn(flight: Admitted, reading: Reading, spec: VocabularySpec) -> CaptureTurn | None:
    check = reading.checks["capture_turn"]
    if check is None:
        return None
    start = int(check["start_row"])
    track = float(flight.smoothed.track_deg[start])
    course = on_branch(track, flight.candidate.course_deg)
    low, high = sorted((track, course))
    position = np.array([flight.signals.e_m[start], flight.signals.n_m[start]])
    return CaptureTurn(start_row=start, course_on_track_deg=course, check=check,
                       band_deg=(low - spec.heading_tolerance_deg, high + spec.heading_tolerance_deg),
                       region=turn_region(position, track, flight.candidate.course_deg,
                                          float(flight.smoothed.ground_speed_mps[start]), spec))


def course_band_deg(flight: Admitted, reading: Reading, spec: VocabularySpec) -> tuple[float, float]:
    """The corridor's course tolerance on the heading chart, on the branch of the track at the capture."""
    course = on_branch(float(flight.smoothed.track_deg[reading.capture_row]), flight.candidate.course_deg)
    return course - spec.corridor_course_tolerance_deg, course + spec.corridor_course_tolerance_deg


# ---- vertical
@dataclass(frozen=True)
class AltitudeTube:
    """One altitude word's tube (§2.5), exactly `labeller.vertical.tube_bounds`, with its rows'
    containment and the labeller's own verdict."""

    word: Instruction
    end_row: int                  # one past the last row it covers
    lower_m: np.ndarray
    upper_m: np.ndarray
    inside: np.ndarray            # per row, bool
    check: dict[str, Any]         # `Reading.checks["vertical"]` for this word


def altitude_tubes(flight: Admitted, reading: Reading, spec: VocabularySpec, words: Words) -> list[AltitudeTube]:
    distance, altitude = flight.smoothed.distance_m, flight.smoothed.altitude_m
    checks = {check["row"]: check for check in reading.checks["vertical"]}
    tubes = []
    for word, end, low, high in tube_bounds(reading.instructions, distance, altitude, spec, words):
        span = altitude[word.row: end]
        inside = (span >= low) & (span <= high)
        check = checks[word.row]
        if int(np.count_nonzero(inside)) != check["inside"] or end - word.row != check["rows"]:
            raise ValueError(f"altitude word at row {word.row}: the tube's rows disagree with the labeller's check")
        tubes.append(AltitudeTube(word=word, end_row=int(end), lower_m=low, upper_m=high, inside=inside, check=check))
    return tubes


@dataclass(frozen=True)
class AngleWord:
    word: Instruction
    #: The straight piece's angle the class was read from; ``None`` for the level class, which is
    #: a target reached, not a measured slope.
    measured_deg: float | None


def angle_words(reading: Reading) -> list[AngleWord]:
    result = []
    for word in sorted((i for i in reading.instructions if i.column == ANGLE), key=lambda item: item.row):
        if word.value == ANGLE_LEVEL:
            if "angle_deg" in word.info:
                raise ValueError(f"the level angle word at row {word.row} carries a measured angle")
            result.append(AngleWord(word=word, measured_deg=None))
        else:
            result.append(AngleWord(word=word, measured_deg=float(word.info["angle_deg"])))
    return result


# ---- speed
@dataclass(frozen=True)
class SpeedSpan:
    """One speed word's span (§2.6). A target: the monotone transition from the issue row to the
    row the band is first met, then the band V ± tolerance. "Unspecified": the global range only."""

    word: Instruction
    end_row: int                          # one past the last row
    target_mps: float | None
    #: A target's transition bound, rows ``word.row`` .. ``arrival_row`` (inclusive; to the span's
    #: last row when the next word cuts it before the band is met): the speed stays between the
    #: start and the target, each widened by the tolerance (`envelope.speed_transition_ok`'s bound),
    #: and within `speed_accel_max_mps2` × elapsed time of the start (the acceleration bound).
    arrival_row: int | None
    transition_lower_mps: np.ndarray | None
    transition_upper_mps: np.ndarray | None
    band_mps: tuple[float, float] | None
    band_inside: np.ndarray | None        # per band row (arrival .. end), `envelope.speed_band`
    check: dict[str, Any] | None          # `labeller.speed.span_checks` for this word
    #: "unspecified": the only bound left is the speed words' range, as `read.admit` applies it.
    range_mps: tuple[float, float] | None


def speed_spans(flight: Admitted, reading: Reading, spec: VocabularySpec, words: Words) -> list[SpeedSpan]:
    speed, time = flight.smoothed.ground_speed_mps, flight.signals.time_s
    ordered = sorted((i for i in reading.instructions if i.column == SPEED), key=lambda item: item.row)
    ends = [item.row for item in ordered[1:]] + [len(speed)]
    checks = {check["row"]: check for check in span_checks(reading.instructions, speed, spec, words)}
    tolerance = spec.speed_tolerance_mps
    spans = []
    for word, end in zip(ordered, ends):
        target = words.speed_mps(word.value)
        if target is None:
            spans.append(SpeedSpan(word=word, end_row=int(end), target_mps=None, arrival_row=None,
                                   transition_lower_mps=None, transition_upper_mps=None, band_mps=None,
                                   band_inside=None, check=None,
                                   range_mps=(spec.speed_min_mps - tolerance, spec.speed_max_mps + tolerance)))
            continue
        check = checks[word.row]
        cut = bool(check["cut_before_arrival"])
        arrival = None if cut else word.row + int(check["arrival_rows"])
        last = end - 1 if cut else arrival
        rows = slice(word.row, last + 1)
        start = float(speed[word.row])
        elapsed = time[rows] - time[word.row]
        low = np.maximum(min(start, target) - tolerance, start - spec.speed_accel_max_mps2 * elapsed)
        high = np.minimum(max(start, target) + tolerance, start + spec.speed_accel_max_mps2 * elapsed)
        band_inside = None if cut else envelope.speed_band(speed[arrival:end], target, tolerance)
        if band_inside is not None and (int(np.count_nonzero(band_inside)) != check["band_inside"]
                                        or len(band_inside) != check["band_rows"]):
            raise ValueError(f"speed word at row {word.row}: the band's rows disagree with the labeller's check")
        spans.append(SpeedSpan(word=word, end_row=int(end), target_mps=float(target), arrival_row=arrival,
                               transition_lower_mps=low, transition_upper_mps=high,
                               band_mps=(target - tolerance, target + tolerance), band_inside=band_inside,
                               check=check, range_mps=None))
    return spans


@dataclass(frozen=True)
class FlightEnvelopes:
    heading: list[HeadingEnvelope]
    capture_turn: CaptureTurn | None
    course_band_deg: tuple[float, float]
    corridor: Corridor
    altitude: list[AltitudeTube]
    angle: list[AngleWord]
    speed: list[SpeedSpan]


def flight_envelopes(flight: Admitted, reading: Reading, spec: VocabularySpec, words: Words) -> FlightEnvelopes:
    """Every envelope of one read flight. ``flight`` is `labeller.read.admit` of the signals the
    ``reading`` was read from (the rows the sentence covers, smoothed, relative to its runway)."""
    if flight.signals.n_rows != len(reading.words):
        raise ValueError(f"{reading.dataset_id}: {flight.signals.n_rows} admitted rows but a sentence of "
                         f"{len(reading.words)} steps")
    return FlightEnvelopes(
        heading=heading_envelopes(flight, reading, spec, words), capture_turn=capture_turn(flight, reading, spec),
        course_band_deg=course_band_deg(flight, reading, spec), corridor=corridor(flight, reading, spec),
        altitude=altitude_tubes(flight, reading, spec, words), angle=angle_words(reading),
        speed=speed_spans(flight, reading, spec, words))
