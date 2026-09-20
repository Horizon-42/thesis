"""Instruction words read from an arrival's track (two-tier v3 stage B, plan §5.2.1; decisions D50–D54).

The second layer's vocabulary is the controller's: an instruction changes ONE target and the
target is an absolute value — a heading, an altitude, a speed, or "intercept the final approach
course" — never a rate (D50: rates are the executor's business; a rate read off a 2 s ADS-B slope
is not reliable; the 09-18 "change class" vocabulary carried no anchor and the executor ignored
it). The words (D51; the bins were settled on the 2026-09-20 read-back of the B cohort — the
starting ceilings of 8 000 ft and 230 kt clamped 7 % of the altitude and 30 % of the speed words,
because entry heights reach 8 000 ft above the threshold and entry ground speeds 300 kt):

    heading    the ground-track course RELATIVE TO THE FINAL APPROACH COURSE, `HEADING_BIN_DEG`
               bins over (−180°, 180°] — bin 0 is the course itself, +180° the downwind, ±90° a base
    altitude   height ABOVE THE THRESHOLD in `ALTITUDE_BIN_M` bins up to `ALTITUDE_MAX_M` (the
               ceiling is the top word's centre)
    speed      ground speed in `SPEED_BIN_MPS` bins over [`SPEED_MIN_MPS`, `SPEED_MAX_MPS`] (ADS-B
               carries no airspeed: a ground-speed word is what the data can say — the wind is in it)
    intercept  the turn that captures the course, by its intercept angle (`INTERCEPT_ANGLE_BINS_DEG`)

**Reading them off the track** (D53, settled after the first hand check of 2026-09-20 — the
rate-threshold reading it replaced started gentle manoeuvres late, sliced long ramps into
rolling targets and missed heading changes slower than 1°/s; 300 pages, 88 % agreement): a
TARGET is a PLATEAU — a stretch where the smoothed signal (`COURSE_SMOOTHING_S` for the course,
`SMOOTHING_S` for height and speed: ADS-B altitude is quantised at 25 ft) stays within the
kind's tolerance (`COURSE_TOLERANCE_DEG`, `HEIGHT_TOLERANCE_M`, `SPEED_TOLERANCE_MPS`) of the
value it settled at, for at least `PLATEAU_MIN_S` (two sentence positions: a target in force for
less is not a word of its own). Everything between two plateaus is the manoeuvre that takes
the aircraft from one to the next, whatever its rate: the instruction's TARGET is the plateau
reached (its median), it is ISSUED where the previous plateau ends (the controller spoke a few
seconds earlier; that gap is not recoverable and is stated) and ``settled_s`` is where the new
plateau begins. A record that opens mid-manoeuvre carries that manoeuvre's target as its word
at t = 0; a manoeuvre that runs to the end of the record targets the final value (the last
descent targets the threshold, bin 0); a record with no plateau at all is one manoeuvre to its
end. A manoeuvre whose plateau reads as the word already in force is NOT an instruction — the
previous one is still being executed (a step inside one bin, an orbit back onto the same
heading); it is recorded as `Absorbed` so the hand check can see it and the summary count it.
The intercept is the LAST heading instruction (a manoeuvre that changed the word — an absorbed
wiggle after the capture is no turn) after which the aircraft is established under the
package's one rule (`approach_difficulty.course_frame_rows`: within `ESTABLISHED_CROSS_TRACK_M` of
the course, ahead of the threshold, heading within `ESTABLISHED_TRACK_TOLERANCE_DEG`); its angle
is the relative course held before it — the previous heading word's target, or the first row's
course when the record opens inside the capture. The reading rule's version (`READING_RULE`) is
part of the vocabulary's spec and sha.

**The sentence** (D52): one position every `TOKEN_STEP_S` from the record's start, each position
the four words IN FORCE (the latest issued instruction of each kind); an instruction is the
position where a word changes, "hold" is a position where none does. The fixed positions are the
prior's grid; the executor reads the words in force at its own anchor + k·τ (`Reading.words_at`),
so its segment need not sit on the grid; "when" is expressed by the position a word changes at,
not by a duration word.

Leaf module (`tests/test_architecture.py` `MANOEUVRE_LEAVES`): the data plane, geokit and numpy
only — the executor reads these words as its condition, the prior predicts them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from geokit.constants import FT_M, KT_MS
from ts_transformer.data.approach_difficulty import (
    ESTABLISHED_CROSS_TRACK_M,
    ESTABLISHED_TRACK_TOLERANCE_DEG,
    course_frame_rows,
)
from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
from ts_transformer.data.runway_context import wrap_deg
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.manoeuvre.segments import MINIMUM_GROUND_SPEED_MPS

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

#: The three kinds every position carries a word of, then the intercept (absent before a capture).
MANDATORY_KINDS = ("heading", "altitude", "speed")
INSTRUCTION_KINDS = MANDATORY_KINDS + ("intercept",)
#: An empty intercept word (no capture turn in force).
NO_INTERCEPT = -1

#: D51 bins (settled 2026-09-20, module docstring). Heading: 10° over a full circle (36 words;
#: bin 0 = the final approach course). Altitude: 1000 ft above the threshold, 0 … 10 000 ft
#: (11 words). Speed: 10 kt of ground speed, 120 … 320 kt (21 words). A value outside a range is
#: clamped to the edge word and COUNTED. Intercept: ≤ 30° / 30–45° / beyond (3 words; the third
#: holds what a procedure would not clear — 42 % of the cohort's captures read there).
HEADING_BIN_DEG = 10.0
FT, KT = FT_M, KT_MS                      # geokit's exact definitions; every bin below derives from them
ALTITUDE_BIN_M = 1000.0 * FT
ALTITUDE_MAX_M = 10000.0 * FT
SPEED_BIN_MPS = 10.0 * KT
SPEED_MIN_MPS = 120.0 * KT
SPEED_MAX_MPS = 320.0 * KT
INTERCEPT_ANGLE_BINS_DEG = (30.0, 45.0)

#: D53 thresholds (the module docstring says how they bind together): the smoothing, the
#: plateau tolerances (each under half a bin, so a plateau cannot straddle two words), and the
#: least a plateau holds to be a target.
SMOOTHING_S = 10.0
COURSE_SMOOTHING_S = 6.0
COURSE_TOLERANCE_DEG = 4.0
HEIGHT_TOLERANCE_M = 30.0
SPEED_TOLERANCE_MPS = 2.5
PLATEAU_MIN_S = 20.0
#: D54: the sentence's position interval (τ); 5 s is the ablation.
TOKEN_STEP_S = 10.0
#: The reading rule's version — part of the vocabulary's identity, because the words an executor
#: trained on depend on HOW the track was read, not only on the bins. v1 (retired 2026-09-20): rate
#: thresholds. v2: plateaus; the intercept is the last heading INSTRUCTION after which the aircraft is
#: established (not the last plateau boundary — an absorbed wiggle after the capture is no turn).
#: v3: a plateau's window may not drift by more than half the tolerance end to end (a slow
#: descent is not a plateau).
READING_RULE = "plateau-v3"

VOCABULARY_SCHEMA = "ts-instruction-vocabulary-v1"
VOCABULARY_FILE = "instruction_vocabulary.json"


@dataclass(frozen=True)
class Vocabulary:
    """The bins and thresholds that turn a track into words — the artefact every executor and
    prior binds to (`sha256` over `spec`). The established rule's two numbers are part of the
    spec so the sha moves with them, but they are the package's one definition
    (`approach_difficulty`) and cannot be set here."""

    heading_bin_deg: float = HEADING_BIN_DEG
    altitude_bin_m: float = ALTITUDE_BIN_M
    altitude_max_m: float = ALTITUDE_MAX_M
    speed_bin_mps: float = SPEED_BIN_MPS
    speed_min_mps: float = SPEED_MIN_MPS
    speed_max_mps: float = SPEED_MAX_MPS
    intercept_angle_bins_deg: tuple[float, ...] = INTERCEPT_ANGLE_BINS_DEG
    established_cross_track_m: float = ESTABLISHED_CROSS_TRACK_M
    established_track_tolerance_deg: float = ESTABLISHED_TRACK_TOLERANCE_DEG
    smoothing_s: float = SMOOTHING_S
    course_smoothing_s: float = COURSE_SMOOTHING_S
    course_tolerance_deg: float = COURSE_TOLERANCE_DEG
    height_tolerance_m: float = HEIGHT_TOLERANCE_M
    speed_tolerance_mps: float = SPEED_TOLERANCE_MPS
    plateau_min_s: float = PLATEAU_MIN_S
    token_step_s: float = TOKEN_STEP_S
    reading_rule: str = READING_RULE

    def __post_init__(self) -> None:
        if self.reading_rule != READING_RULE:
            raise ValueError(f"reading_rule {self.reading_rule!r} is not this code's ({READING_RULE!r}): the words were read by another rule")
        if 360.0 % self.heading_bin_deg:
            raise ValueError(f"heading_bin_deg={self.heading_bin_deg:g} does not divide 360°")
        if self.altitude_max_m <= 0 or self.altitude_bin_m <= 0 or self.speed_bin_mps <= 0:
            raise ValueError("bins are positive")
        if self.speed_max_mps <= self.speed_min_mps:
            raise ValueError("the speed range is empty")
        # the top word's centre IS the ceiling (the altitude and speed counts follow one rule)
        for name, span, bin_ in (("altitude", self.altitude_max_m, self.altitude_bin_m),
                                 ("speed", self.speed_max_mps - self.speed_min_mps, self.speed_bin_mps)):
            if abs(span / bin_ - round(span / bin_)) > 1e-9:
                raise ValueError(f"the {name} bin does not divide its range ({span:g} / {bin_:g})")
        if list(self.intercept_angle_bins_deg) != sorted(self.intercept_angle_bins_deg) or not self.intercept_angle_bins_deg:
            raise ValueError("intercept angle bins are ascending edges")
        if (self.established_cross_track_m, self.established_track_tolerance_deg) != (ESTABLISHED_CROSS_TRACK_M, ESTABLISHED_TRACK_TOLERANCE_DEG):
            raise ValueError(
                "the established rule is the package's one (approach_difficulty); it is recorded in the spec "
                "so the sha moves with it, not to be set here"
            )
        for name in ("smoothing_s", "course_smoothing_s", "plateau_min_s", "token_step_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} is positive seconds")
        for name, half_bin in (("course_tolerance_deg", self.heading_bin_deg / 2), ("height_tolerance_m", self.altitude_bin_m / 2),
                               ("speed_tolerance_mps", self.speed_bin_mps / 2)):
            if not 0 < getattr(self, name) < half_bin:
                raise ValueError(f"{name} is positive and under half a bin ({half_bin:g}), so a plateau cannot straddle two words")

    # ── word counts ──────────────────────────────────────────────────────
    @property
    def heading_words(self) -> int:
        return int(round(360.0 / self.heading_bin_deg))

    @property
    def altitude_words(self) -> int:
        return int(round(self.altitude_max_m / self.altitude_bin_m)) + 1       # 0 … the ceiling, inclusive

    @property
    def speed_words(self) -> int:
        return int(round((self.speed_max_mps - self.speed_min_mps) / self.speed_bin_mps)) + 1

    @property
    def intercept_words(self) -> int:
        return len(self.intercept_angle_bins_deg) + 1

    @property
    def words(self) -> dict[str, int]:
        return {"heading": self.heading_words, "altitude": self.altitude_words, "speed": self.speed_words,
                "intercept": self.intercept_words}

    # ── binning (each returns the word index; the caller counts clamps) ───
    def heading_bin(self, relative_deg: float) -> int:
        """The word of a course relative to the final approach course; bin 0 is the course,
        bin 18 (at 10°) the downwind; the wrap is at ±180°."""
        return int(math.floor(wrap_deg(relative_deg) / self.heading_bin_deg + 0.5)) % self.heading_words     # half-up, never banker's

    def heading_centre_deg(self, word: int) -> float:
        return wrap_deg(word * self.heading_bin_deg)

    def altitude_bin(self, height_m: float) -> tuple[int, bool]:
        """``(word, clamped)``: the word of a height above the threshold; below 0 reads as 0,
        above the top as the top word."""
        word = int(math.floor(height_m / self.altitude_bin_m + 0.5))
        clamped = word < 0 or word >= self.altitude_words
        return min(max(word, 0), self.altitude_words - 1), clamped

    def altitude_centre_m(self, word: int) -> float:
        return word * self.altitude_bin_m

    def speed_bin(self, speed_mps: float) -> tuple[int, bool]:
        word = int(math.floor((speed_mps - self.speed_min_mps) / self.speed_bin_mps + 0.5))
        clamped = word < 0 or word >= self.speed_words
        return min(max(word, 0), self.speed_words - 1), clamped

    def speed_centre_mps(self, word: int) -> float:
        return self.speed_min_mps + word * self.speed_bin_mps

    def intercept_bin(self, angle_deg: float) -> int:
        angle = abs(angle_deg)
        for word, edge in enumerate(self.intercept_angle_bins_deg):
            if angle <= edge:
                return word
        return len(self.intercept_angle_bins_deg)

    # ── the executor's view of the words ──────────────────────────────────
    #: one position's conditioning: cos, sin of the heading centre; the altitude and speed
    #: centres as fractions of their ceilings; the intercept as (bin + 1) / bins, 0 for none
    CONDITIONING_WIDTH = 5

    def conditioning(self, words: np.ndarray) -> np.ndarray:
        """The bin CENTRES the executor conditions on, ``[..., CONDITIONING_WIDTH]`` float32 for
        words ``[..., 4]`` (plan §5.2.1 "执行器怎么吃"): the heading as cos / sin so the wrap at
        ±180° is continuous, the altitude and speed as fractions of the vocabulary's ceilings,
        the intercept as a graded flag."""
        words = np.asarray(words)
        if words.shape[-1] != len(INSTRUCTION_KINDS):
            raise ValueError(f"words are [..., {len(INSTRUCTION_KINDS)}], got {words.shape}")
        heading = np.radians(words[..., 0] * self.heading_bin_deg)
        out = np.stack((
            np.cos(heading), np.sin(heading),
            words[..., 1] * self.altitude_bin_m / self.altitude_max_m,
            (self.speed_min_mps + words[..., 2] * self.speed_bin_mps) / self.speed_max_mps,
            np.where(words[..., 3] == NO_INTERCEPT, 0.0, (words[..., 3] + 1) / self.intercept_words),
        ), axis=-1)
        return out.astype(np.float32)

    # ── the artefact's identity ───────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        spec = asdict(self)
        spec["intercept_angle_bins_deg"] = list(self.intercept_angle_bins_deg)
        return spec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        data = dict(data)
        data["intercept_angle_bins_deg"] = tuple(float(v) for v in data["intercept_angle_bins_deg"])
        return cls(**data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Instruction:
    """One word issued at one time: ``kind`` ∈ `INSTRUCTION_KINDS`, ``word`` its index, ``target``
    the plateau value it was read from (deg relative to the course / m above the threshold / m/s /
    deg of intercept angle), ``issued_s`` the manoeuvre's start (the previous plateau's end),
    ``settled_s`` where the target plateau begins (None for a manoeuvre running to the end of
    the record), ``clamped`` when the target fell outside the vocabulary's range."""

    kind: str
    word: int
    target: float
    issued_s: float
    settled_s: float | None
    clamped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Why a manoeuvre the reading found carries no word of its own.
ABSORBED_SAME_WORD = "same word"        # its plateau reads as the word already in force
ABSORBED_SHORT_TAIL = "short tail"      # it leaves the last plateau less than a plateau's length before the record ends


@dataclass(frozen=True)
class Absorbed:
    """A manoeuvre the reading found but did not word — its plateau is the word already in force
    (a step inside one bin, an orbit back onto the same heading), or it starts too close to the
    record's end to be read (`ABSORBED_SHORT_TAIL`: the last rows before the threshold swing and
    settle nowhere). Recorded so nothing vanishes silently — the hand check draws it, the
    summary counts it. ``change`` is what the manoeuvre moved the signal by (deg / m / m/s)."""

    kind: str
    start_s: float
    end_s: float
    word: int
    change: float
    reason: str = ABSORBED_SAME_WORD

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Reading:
    """A flight's words: the instructions in issue order, the sentence positions, the absorbed
    manoeuvres, and the diagnostics the hand check reads."""

    dataset_id: str
    flight_id: str
    instructions: tuple[Instruction, ...]
    positions_s: np.ndarray            # [P]
    words: np.ndarray                  # [P, 4] heading / altitude / speed / intercept word per position
    established_from_start: bool
    duration_s: float
    absorbed: tuple[Absorbed, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id, "flight_id": self.flight_id,
            "instructions": [item.to_dict() for item in self.instructions],
            "positions_s": self.positions_s.tolist(), "words": self.words.tolist(),
            "established_from_start": self.established_from_start, "duration_s": self.duration_s,
            "absorbed": [item.to_dict() for item in self.absorbed],
        }

    def words_at(self, times_s: np.ndarray) -> np.ndarray:
        """The four words in force at each time, ``[len(times), 4]``: the latest position at or
        before it (an instruction issued between two positions is in force from the NEXT one,
        as the sentence says it); after the last position the last words stay in force (the
        aircraft lands on them); before the first there is nothing — that is an error."""
        times = np.asarray(times_s, dtype=np.float64)
        if times.min() < self.positions_s[0]:
            raise ValueError(f"{self.flight_id}: no word in force before the record's start ({times.min():g} s)")
        index = np.searchsorted(self.positions_s, times, side="right") - 1
        return self.words[index]


def segment_positions_s(start_s: float, horizon_s: float, step_s: float) -> np.ndarray:
    """The instruction positions one executor segment spans — ``start, start + τ, …`` up to but
    not including ``start + Δ`` (Δ / τ of them; Δ must be a multiple of τ, D54)."""
    count = horizon_s / step_s
    if abs(count - round(count)) > 1e-9 or count < 1:
        raise ValueError(f"the segment ({horizon_s:g} s) is not a whole number of instruction steps ({step_s:g} s)")
    return float(start_s) + np.arange(int(round(count)), dtype=np.float64) * float(step_s)


# ── the course frame ──────────────────────────────────────────────────────

def course_frame(series: FlightSeries) -> dict[str, np.ndarray]:
    """The observed rows in the final approach course's frame (`course_frame_rows`, the one
    definition): ``t``, ``to_go_m`` (positive before the threshold), ``cross_m`` (positive right
    of the course), ``relative_course_deg`` (the ground track against the course, wrapped),
    ``course_unwrapped_deg`` (the same, UNWRAPPED along the rows — what the rates and plateaus
    are read from), ``height_m`` (above the threshold), ``ground_speed_mps`` (the chart's
    horizontal speed: the derivative channels carry the transport factors, ≈ 0.3 % at 25 km —
    inside a word), ``established`` per row. A row slower than `MINIMUM_GROUND_SPEED_MPS` has no
    course and is refused (a padded or corrupt row must not fabricate a turn)."""
    target = series.scenario.target
    if target is None:
        raise ValueError(f"{series.flight_id}: no target, no approach course")
    values = np.asarray(series.values, dtype=np.float64)
    position = values[:, list(POSITION_IDX)] - series.target_chart
    velocity = values[:, list(VELOCITY_IDX)]
    east, north = np.vectorize(series.frame.to_world_horizontal)(position[:, 0], position[:, 1])
    v_east, v_north = np.vectorize(series.frame.to_world_horizontal)(velocity[:, 0], velocity[:, 1])
    speed = np.hypot(v_east, v_north)
    if speed.min() < MINIMUM_GROUND_SPEED_MPS:
        raise ValueError(f"{series.flight_id}: a row moves at {speed.min():.2f} m/s; a padded or corrupt row has no course to read")
    frame = course_frame_rows(east, north, v_east, v_north, float(target.psi))
    return {
        "t": np.asarray(series.times, dtype=np.float64), "to_go_m": frame["to_go_m"], "cross_m": frame["cross_m"],
        "relative_course_deg": frame["relative_course_deg"],
        "course_unwrapped_deg": np.degrees(np.unwrap(np.radians(frame["relative_course_deg"]))),
        "height_m": position[:, 2], "ground_speed_mps": speed, "established": frame["established"],
    }


# ── segmentation ─────────────────────────────────────────────────────────

def smooth(signal: np.ndarray, window_rows: int) -> np.ndarray:
    """A centred moving average over ``window_rows`` rows (edges padded with the edge value)."""
    if window_rows <= 1:
        return np.asarray(signal, dtype=np.float64)
    pad = window_rows // 2
    padded = np.concatenate((np.full(pad, signal[0]), signal, np.full(window_rows - 1 - pad, signal[-1])))
    return np.convolve(padded, np.full(window_rows, 1.0 / window_rows), mode="valid")


def min_rows(seconds: float, dt_s: float) -> int:
    """The rows a stretch must hold to span AT LEAST ``seconds`` (``k`` rows span ``(k − 1) · dt``)."""
    return int(round(seconds / dt_s)) + 1


def plateaus(signal: np.ndarray, tolerance: float, rows: int) -> list[tuple[int, int]]:
    """``[(start, end)]`` (end exclusive) of the signal's plateaus: a plateau opens where the
    next ``rows`` rows all lie within ``tolerance`` of their median AND the window's two ends
    (the medians of its first and last thirds) differ by at most half the tolerance — a slow
    transit that merely crosses the band is not a plateau (the second hand check found a 3 m/s
    descent reading as "level" every 305 m) — and holds while the signal stays within
    ``tolerance`` of that value; it is at least ``rows`` long. Everything outside a plateau is a
    manoeuvre. The edges sit INSIDE the band: a plateau ends where the signal has left its value
    by the tolerance (so an instruction is issued up to tolerance / rate after the aircraft
    began to move), and the next opens as soon as the signal is within the tolerance of where
    it will settle."""
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    out: list[tuple[int, int]] = []
    third = max(rows // 3, 1)
    i = 0
    while i + rows <= n:
        window = signal[i : i + rows]
        reference = float(np.median(window))
        drift = float(np.median(window[-third:])) - float(np.median(window[:third]))
        if np.abs(window - reference).max() > tolerance or abs(drift) > tolerance / 2:
            i += 1
            continue
        j = i + rows
        while j < n and abs(signal[j] - reference) <= tolerance:
            j += 1
        out.append((i, j))
        i = j
    return out


def _plateau_value(signal: np.ndarray, start: int, end: int) -> float:
    """The value a signal holds over ``[start, end)``: its median."""
    if end <= start:
        raise ValueError(f"a plateau needs rows, got [{start}, {end})")
    return float(np.median(signal[start:end]))


def _manoeuvre_words(kind: str, times: np.ndarray, signal: np.ndarray, flats: list[tuple[int, int]], rows: int,
                     to_word: Callable[[float], tuple[int, float, bool]]) -> tuple[list[Instruction], list[Absorbed]]:
    """The instructions of one kind from its plateaus: the first plateau's value is the word at
    t = 0 (the record starts on it, or is already manoeuvring towards it); every later plateau
    is a target issued where the plateau before it ends; a signal that leaves its last plateau
    for good at least ``rows`` before the end targets the final value (a shorter tail is
    unreadable and recorded), and a signal with no plateau is one manoeuvre to its end. A
    target that reads as the word already in force is recorded as `Absorbed`."""
    n = len(times)
    out: list[Instruction] = []
    absorbed: list[Absorbed] = []
    if not flats:
        word, target, clamped = to_word(float(signal[-1]))
        return [Instruction(kind, word, target, float(times[0]), None, clamped)], absorbed
    first_start, first_end = flats[0]
    word, target, clamped = to_word(_plateau_value(signal, first_start, first_end))
    out.append(Instruction(kind, word, target, float(times[0]), float(times[first_start]), clamped))
    previous_value = target
    for index in range(1, len(flats)):
        issued, (start, end) = flats[index - 1][1], flats[index]
        value = _plateau_value(signal, start, end)
        word, target, clamped = to_word(value)
        if out[-1].word == word:            # the word in force already: the earlier instruction continues
            absorbed.append(Absorbed(kind, float(times[issued]), float(times[start]), word, float(value - previous_value)))
        else:
            out.append(Instruction(kind, word, target, float(times[issued]), float(times[start]), clamped))
        previous_value = value
    last_end = flats[-1][1]
    if last_end < n:                        # the signal leaves its last plateau and never settles again
        value = float(signal[-1])
        word, target, clamped = to_word(value)
        if n - last_end < rows:
            absorbed.append(Absorbed(kind, float(times[last_end]), float(times[-1]), out[-1].word, float(value - previous_value), ABSORBED_SHORT_TAIL))
        elif out[-1].word == word:
            absorbed.append(Absorbed(kind, float(times[last_end]), float(times[-1]), word, float(value - previous_value)))
        else:
            out.append(Instruction(kind, word, target, float(times[last_end]), None, clamped))
    return out, absorbed


def read_instructions(series: FlightSeries, vocabulary: Vocabulary) -> Reading:
    """Every instruction of one flight, and its sentence (`Reading`)."""
    frame = course_frame(series)
    times = frame["t"]
    n = len(times)
    if n < 2:
        raise ValueError(f"{series.flight_id}: a track of {n} rows has no manoeuvre to read")
    dt = float(np.median(np.diff(times)))
    course = smooth(frame["course_unwrapped_deg"], max(int(round(vocabulary.course_smoothing_s / dt)), 1))
    window = max(int(round(vocabulary.smoothing_s / dt)), 1)
    height = smooth(frame["height_m"], window)
    speed = smooth(frame["ground_speed_mps"], window)
    rows = min_rows(vocabulary.plateau_min_s, dt)
    flats = {
        "heading": plateaus(course, vocabulary.course_tolerance_deg, rows),
        "altitude": plateaus(height, vocabulary.height_tolerance_m, rows),
        "speed": plateaus(speed, vocabulary.speed_tolerance_mps, rows),
    }

    def heading_word(value: float) -> tuple[int, float, bool]:
        return vocabulary.heading_bin(value), wrap_deg(value), False

    def altitude_word(value: float) -> tuple[int, float, bool]:
        word, clamped = vocabulary.altitude_bin(value)
        return word, value, clamped

    def speed_word(value: float) -> tuple[int, float, bool]:
        word, clamped = vocabulary.speed_bin(value)
        return word, value, clamped

    instructions: list[Instruction] = []
    absorbed: list[Absorbed] = []
    for kind, signal, to_word in (("heading", course, heading_word), ("altitude", height, altitude_word), ("speed", speed, speed_word)):
        words, dropped = _manoeuvre_words(kind, times, signal, flats[kind], rows, to_word)
        instructions.extend(words)
        absorbed.extend(dropped)
    # the intercept: the last heading INSTRUCTION (a manoeuvre that changed the word — never an
    # absorbed wiggle) after which the aircraft is established (the rule per row over the plateau
    # that follows it, until the next heading word; the last row when it runs to the end), by the
    # relative course held before it: the previous heading word's target, or the first row's
    # course when the record opens inside the capture
    established = frame["established"]
    intercept: Instruction | None = None
    headings = [item for item in instructions if item.kind == "heading"]
    for index, word in enumerate(headings):
        if word.issued_s == float(times[0]) and word.settled_s == float(times[0]):
            continue                                                    # the record starts on this plateau: no manoeuvre
        plateau_end = int(np.searchsorted(times, headings[index + 1].issued_s)) if index + 1 < len(headings) else n
        after = established[int(np.searchsorted(times, word.settled_s)):plateau_end] if word.settled_s is not None else established[n - 1:]
        if after.any():
            before = float(course[0]) if index == 0 else headings[index - 1].target
            angle = wrap_deg(before)
            intercept = Instruction("intercept", vocabulary.intercept_bin(angle), abs(angle), word.issued_s, word.settled_s)
    if intercept is not None:
        instructions.append(intercept)
    instructions.sort(key=lambda item: (item.issued_s, INSTRUCTION_KINDS.index(item.kind)))
    positions, words = sentence(instructions, float(times[0]), float(times[-1]), vocabulary.token_step_s)
    return Reading(
        dataset_id=series.dataset_id, flight_id=series.flight_id, instructions=tuple(instructions),
        positions_s=positions, words=words, established_from_start=bool(established[0]),
        duration_s=float(times[-1] - times[0]), absorbed=tuple(sorted(absorbed, key=lambda item: item.start_s)),
    )


def sentence(instructions: Sequence[Instruction], start_s: float, end_s: float, step_s: float) -> tuple[np.ndarray, np.ndarray]:
    """The positions every ``step_s`` from ``start_s`` to ``end_s`` and the words in force at each
    (``[P, 4]``: heading, altitude, speed, intercept — `NO_INTERCEPT` before a capture turn)."""
    positions = start_s + step_s * np.arange(int(math.floor((end_s - start_s) / step_s + 1e-9)) + 1, dtype=np.float64)
    words = np.full((len(positions), len(INSTRUCTION_KINDS)), NO_INTERCEPT, dtype=np.int64)
    for column, kind in enumerate(INSTRUCTION_KINDS):
        issued = sorted((item for item in instructions if item.kind == kind), key=lambda item: item.issued_s)
        for item in issued:
            words[positions >= item.issued_s - 1e-9, column] = item.word
    missing = [kind for column, kind in enumerate(MANDATORY_KINDS) if (words[:, column] == NO_INTERCEPT).any()]
    if missing:
        raise ValueError(f"no word in force for {missing} at some position: every kind but the intercept starts at t = 0")
    return positions, words


# ── the vocabulary artefact ───────────────────────────────────────────────

def write_vocabulary(directory: str | Path, vocabulary: Vocabulary, *, cohort_identity: dict[str, Any],
                     counts: dict[str, Any], source: dict[str, Any]) -> Path:
    """Write ``instruction_vocabulary.json`` (refused if it exists): the spec (under its sha), the
    cohort the counts were read on, the per-word counts, and where it came from."""
    directory = Path(directory)
    path = directory / VOCABULARY_FILE
    if path.exists():
        raise FileExistsError(f"{path} exists; a vocabulary is never overwritten")
    write_json_atomic(path, {
        "schema": VOCABULARY_SCHEMA, "written_utc": utc_now(), "spec": vocabulary.to_dict(), "sha256": vocabulary.sha256,
        "words": vocabulary.words, "cohort_identity": dict(cohort_identity), "counts": dict(counts), "source": dict(source),
    })
    return path


def load_vocabulary(path: str | Path) -> tuple[Vocabulary, dict[str, Any]]:
    """``(vocabulary, the file's payload)``; refuses a payload whose sha is not its spec's."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["schema"] != VOCABULARY_SCHEMA:
        raise ValueError(f"{path}: schema {payload['schema']!r} is not {VOCABULARY_SCHEMA!r}")
    vocabulary = Vocabulary.from_dict(payload["spec"])
    if vocabulary.sha256 != payload["sha256"]:
        raise ValueError(f"{path}: the spec's sha {vocabulary.sha256[:12]}… is not the file's {payload['sha256'][:12]}…")
    return vocabulary, payload


__all__ = [
    "ALTITUDE_BIN_M", "ALTITUDE_MAX_M", "COURSE_SMOOTHING_S", "COURSE_TOLERANCE_DEG", "HEADING_BIN_DEG", "HEIGHT_TOLERANCE_M",
    "INSTRUCTION_KINDS", "INTERCEPT_ANGLE_BINS_DEG", "MANDATORY_KINDS", "NO_INTERCEPT", "PLATEAU_MIN_S", "READING_RULE", "SMOOTHING_S",
    "SPEED_BIN_MPS", "SPEED_MAX_MPS", "SPEED_MIN_MPS", "SPEED_TOLERANCE_MPS", "TOKEN_STEP_S", "VOCABULARY_FILE",
    "VOCABULARY_SCHEMA", "ABSORBED_SAME_WORD", "ABSORBED_SHORT_TAIL", "Absorbed", "Instruction", "Reading", "Vocabulary",
    "course_frame", "load_vocabulary", "min_rows",
    "plateaus", "read_instructions", "segment_positions_s", "sentence", "smooth", "write_vocabulary",
]
