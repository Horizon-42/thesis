"""What a read sentence's envelopes look like, as data to draw (vocabulary design §2; the heading word's, §10.1).

The design's rule is ONE implementation of the envelopes for the labeller's checks, the executor's
judge and the display (§1 principle 4). This module is the display side, built only from the public
functions of `envelope.py` and `labeller/*`, and it draws only what is judged:

- a heading word (instruction-v3) bounds no position: it says where the track is `heading_lead_s`
  after it is said, so its envelope is the band θ ± `heading_tolerance_deg` over the rows it is
  judged on — `envelope.heading_word_rows` (from its row plus the lead to the next heading word's
  row plus the lead, never at or past the clearance) — with each row's verdict asked of
  `envelope.heading_words_inside` one row at a time (`rows_inside`), and the count checked against
  the labeller's own (`Reading.checks["heading"]`), so what is drawn is what was judged;
- the capture turn is the labeller's `checks["capture_turn"]` over its rows, from the clearance to
  the capture, toward the course (`labeller.lateral.turn_check`);
- the capture corridor widens by `envelope.corridor_half_width_m`, and its rows are judged by
  `envelope.corridor`;
- the altitude tubes ARE `labeller.vertical.tube_bounds`; the speed spans' band rows are judged by
  `envelope.speed_band`, and counted against the labeller's own `labeller.speed.span_checks`;
- every verdict shown is the labeller's own (`Reading.checks`), never recomputed.

It is outside the labeller's source hash (`artefact.LABELLER_MODULES`): changing how a sentence is
drawn does not change the sentence. Everything is in the airport frame (metres east / north of the
airport reference point, geometric MSL, compass degrees true, seconds); the exporter adds the
geodesy (latitude / longitude, the ellipsoid heights Cesium draws in).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import RunwayCandidate
from ts_transformer.instructions.labeller.read import Admitted, Reading
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.labeller.vertical import tube_bounds
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import ANGLE, ANGLE_LEVEL, HEADING, SPEED, Words, wrap180

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


def on_branch(reference_deg: float, target_deg: float) -> float:
    """``target_deg`` on the branch of an unwrapped track at ``reference_deg`` (the shorter way)."""
    return float(reference_deg + wrap180(target_deg - reference_deg))


# ---- heading words (instruction-v3, §10.1)
def rows_inside(track_deg, target_deg: float, first: int, stop: int, tolerance_deg: float) -> np.ndarray:
    """Row by row over ``first..stop - 1``: is the track within the tolerance of the target? The labeller's own check
    (`envelope.heading_words_inside`) asked of one row at a time — a word at every row, no lead, each ending at the
    next — so each flag is the check's, and their sum is its count."""
    rows = list(range(first, stop))
    counted = envelope.heading_words_inside(track_deg, [(row, target_deg) for row in rows], 0, stop, tolerance_deg)
    return np.array([item["inside"] == 1 for item in counted], dtype=bool)


@dataclass(frozen=True)
class HeadingBand:
    """One heading word's envelope as a chart draws it: the band θ ± the tolerance over the rows it is judged on
    (``first_row .. stop_row - 1``; none when the lead carries them to the clearance), and each row's verdict."""

    first_row: int
    stop_row: int
    #: θ on the branch of the track at the first judged row (at the word's own row when it has none).
    target_on_track_deg: float
    band_deg: tuple[float, float]
    inside: np.ndarray


def heading_band(track_deg, word_row: int, target_deg: float, first: int, stop: int, tolerance_deg: float,
                 check: dict[str, Any]) -> HeadingBand:
    """The band of a heading word judged over rows ``first..stop - 1`` of ``track_deg`` (unwrapped), refused unless
    its rows and its rows inside are ``check``'s — the labeller's, or the executor's judge's, count of this word."""
    inside = rows_inside(track_deg, target_deg, first, stop, tolerance_deg)
    if (len(inside), int(inside.sum())) != (check["rows"], check["inside"]):
        raise ValueError(f"the heading word at row {word_row}: {int(inside.sum())} of {len(inside)} rows inside its band, "
                         f"its check counts {check['inside']} of {check['rows']}")
    target = on_branch(float(np.asarray(track_deg)[first if stop > first else word_row]), target_deg)
    return HeadingBand(first_row=int(first), stop_row=int(stop), target_on_track_deg=target,
                       band_deg=(target - tolerance_deg, target + tolerance_deg), inside=inside)


@dataclass(frozen=True)
class HeadingEnvelope:
    """One heading word's envelope (§10.1): its band over the rows it is judged on, and the labeller's check of it
    (`Reading.checks["heading"]`: its row, its rows and how many are inside)."""

    word: Instruction
    target_deg: float            # θ, the word's own grid value
    band: HeadingBand
    check: dict[str, Any]


def heading_envelopes(flight: Admitted, reading: Reading, spec: VocabularySpec, words: Words) -> list[HeadingEnvelope]:
    """Every heading word of the sentence, in order: its rows from `envelope.heading_word_rows` (to the clearance,
    `Reading.join_row`), each row judged on the smoothed track the labeller read."""
    said = sorted((item for item in reading.instructions if item.column == HEADING), key=lambda item: item.row)
    checks = reading.checks["heading"]
    if [check["row"] for check in checks] != [word.row for word in said]:
        raise ValueError(f"{reading.dataset_id}: the labeller's heading checks are not the heading words said")
    spans = envelope.heading_word_rows([word.row for word in said], spec.rows_exact(spec.heading_lead_s),
                                       reading.join_row)
    result = []
    for word, (first, stop), check in zip(said, spans, checks):
        # judged against the target the labeller judged it against (its `target_deg`, the class's value mod 360)
        band = heading_band(flight.smoothed.track_deg, word.row, float(word.info["target_deg"]), first, stop,
                            spec.heading_tolerance_deg, check)
        result.append(HeadingEnvelope(word=word, target_deg=words.heading_deg(word.value), band=band, check=check))
    return result


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
    """The capture turn (§2.2, §10.1): from the clearance onto the course, judged over its rows — the clearance's
    row to the capture's — by the labeller's `turn_check` (monotone toward the course; its rate and its bank)."""

    start_row: int
    end_row: int                  # the capture: the first row of the final run inside the corridor
    course_on_track_deg: float    # the course on the branch of the smoothed track at the start row
    check: dict[str, Any]         # `Reading.checks["capture_turn"]`


def capture_turn(flight: Admitted, reading: Reading) -> CaptureTurn | None:
    """``None`` for a flight already on the final at row 0 (the labeller reads no capture turn)."""
    check = reading.checks["capture_turn"]
    if check is None:
        return None
    start = int(check["start_row"])
    if start != reading.join_row:
        raise ValueError(f"{reading.dataset_id}: the capture turn starts at row {start}, the clearance is said at "
                         f"{reading.join_row}")
    course = on_branch(float(flight.smoothed.track_deg[start]), flight.candidate.course_deg)
    return CaptureTurn(start_row=start, end_row=reading.capture_row, course_on_track_deg=course, check=check)


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
    check: dict[str, Any] | None          # `Reading.checks["speed"]` for this word
    #: "unspecified": the only bound left is the speed words' range, as `read.admit` applies it.
    range_mps: tuple[float, float] | None


def speed_spans(flight: Admitted, reading: Reading, spec: VocabularySpec, words: Words) -> list[SpeedSpan]:
    speed, time = flight.smoothed.ground_speed_mps, flight.signals.time_s
    ordered = sorted((i for i in reading.instructions if i.column == SPEED), key=lambda item: item.row)
    ends = [item.row for item in ordered[1:]] + [len(speed)]
    checks = {check["row"]: check for check in reading.checks["speed"]}
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
        heading=heading_envelopes(flight, reading, spec, words), capture_turn=capture_turn(flight, reading),
        course_band_deg=course_band_deg(flight, reading, spec), corridor=corridor(flight, reading, spec),
        altitude=altitude_tubes(flight, reading, spec, words), angle=angle_words(reading),
        speed=speed_spans(flight, reading, spec, words))
