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

**Reading them off the track** (D53): the track's relative course, height and ground speed are
smoothed (`COURSE_SMOOTHING_S` for the course, `SMOOTHING_S` for the other two — ADS-B altitude
is quantised at 25 ft, so a 2 s vertical rate is noise and needs the longer window; a 10°
heading correction lasts 3–5 s and the longer window would erase it), the manoeuvre stretches
are rate thresholds held for a minimum duration (a run of ``round(min_s / dt) + 1`` rows, so it
spans AT LEAST the declared seconds), a turn must also SWEEP at least `TURN_MIN_SWEEP_DEG` (a turn
is a manoeuvre that can change the heading word; the smoothing attenuates anything shorter than
its window, so the rate threshold alone is not the binding rule and the sweep is stated), and
every manoeuvre's TARGET is the plateau value after it — the course the aircraft settles on
after the turn, the height it levels at, the speed it holds — rounded to a bin; a manoeuvre that
runs to the end of the record targets the final value (the last descent targets the threshold,
bin 0). The instruction is ISSUED at the manoeuvre's start (the controller spoke a few seconds
earlier; that gap is not recoverable and is stated). A manoeuvre whose plateau reads as the word
already in force is NOT an instruction — the previous one is still being executed (a
deceleration the rate threshold splits in two, a step-down inside one altitude bin, an orbit
back onto the same heading); it is recorded as `Absorbed` so the hand check can see it and the
summary count it, and the instruction keeps the FIRST plateau's target and ``settled_s``. Before
the first manoeuvre of a kind the word in force is the plateau the record starts on, issued at
t = 0. The intercept is the LAST turn after which the aircraft is established under the
package's one rule (`approach_difficulty.course_frame_rows`: within `ESTABLISHED_CROSS_TRACK_M` of
the course, ahead of the threshold, heading within `ESTABLISHED_TRACK_TOLERANCE_DEG`); its angle
is the relative course held before the turn (a turn's sweep is likewise the change between the
plateaus around it — a turn already under way at the record's first row reads from that row).

**The sentence** (D52): one position every `TOKEN_STEP_S` from the record's start, each position
the four words IN FORCE (the latest issued instruction of each kind); an instruction is the
position where a word changes, "hold" is a position where none does. Fixed positions keep the
executor's segments and the prior's positions aligned, and "when" is expressed by the position a
word changes at, not by a duration word.

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

#: D53 thresholds (the module docstring says how they bind together).
SMOOTHING_S = 10.0
COURSE_SMOOTHING_S = 6.0
TURN_RATE_DEG_S = 1.0
TURN_MIN_S = 4.0
TURN_MIN_SWEEP_DEG = 10.0
DESCENT_RATE_MPS = -1.5
DESCENT_MIN_S = 10.0
DECEL_MPS2 = -0.15
DECEL_MIN_S = 10.0
#: D54: the sentence's position interval (τ); 5 s is the ablation.
TOKEN_STEP_S = 10.0

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
    turn_rate_deg_s: float = TURN_RATE_DEG_S
    turn_min_s: float = TURN_MIN_S
    turn_min_sweep_deg: float = TURN_MIN_SWEEP_DEG
    descent_rate_mps: float = DESCENT_RATE_MPS
    descent_min_s: float = DESCENT_MIN_S
    decel_mps2: float = DECEL_MPS2
    decel_min_s: float = DECEL_MIN_S
    token_step_s: float = TOKEN_STEP_S

    def __post_init__(self) -> None:
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
        for name in ("smoothing_s", "course_smoothing_s", "turn_min_s", "descent_min_s", "decel_min_s", "token_step_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} is positive seconds")
        if self.turn_min_sweep_deg <= 0 or self.turn_rate_deg_s <= 0:
            raise ValueError("a turn's sweep and rate thresholds are positive")

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
    deg of intercept angle), ``issued_s`` the manoeuvre's start, ``settled_s`` where the FIRST
    plateau at this word begins (the manoeuvre's end; None for a manoeuvre running to the end of
    the record), ``clamped`` when the target fell outside the vocabulary's range."""

    kind: str
    word: int
    target: float
    issued_s: float
    settled_s: float | None
    clamped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Absorbed:
    """A manoeuvre the reading found but did not word: its plateau is the word already in force
    (a split deceleration, a step-down inside one altitude bin, an orbit back onto the same
    heading). Recorded so nothing vanishes silently — the hand check draws it, the summary
    counts it. ``change`` is what the manoeuvre moved the signal by (deg swept / m / m/s)."""

    kind: str
    start_s: float
    end_s: float
    word: int
    change: float

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


def runs_where(mask: np.ndarray, min_rows: int) -> list[tuple[int, int]]:
    """``[(start, end)]`` (end exclusive) of every contiguous True run of at least ``min_rows``."""
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(np.concatenate((mask, [False]))):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= min_rows:
                runs.append((start, index))
            start = None
    return runs


def min_rows(seconds: float, dt_s: float) -> int:
    """The rows a run must hold to span AT LEAST ``seconds`` (``k`` rows span ``(k − 1) · dt``)."""
    return int(round(seconds / dt_s)) + 1


def _plateau_value(signal: np.ndarray, start: int, end: int) -> float:
    """The value a signal holds over ``[start, end)``: its median."""
    if end <= start:
        raise ValueError(f"a plateau needs rows, got [{start}, {end})")
    return float(np.median(signal[start:end]))


def _manoeuvre_words(kind: str, times: np.ndarray, signal: np.ndarray, runs: list[tuple[int, int]],
                     to_word: Callable[[float], tuple[int, float, bool]]) -> tuple[list[Instruction], list[Absorbed]]:
    """The instructions of one kind: the plateau the record starts on (issued at t = 0), then one
    per manoeuvre run — target = the plateau after it (until the next run, or the final value) —
    unless that plateau is the word already in force, which is recorded as `Absorbed`."""
    n = len(times)
    out: list[Instruction] = []
    absorbed: list[Absorbed] = []
    first_start = runs[0][0] if runs else n
    if first_start > 0:
        word, target, clamped = to_word(_plateau_value(signal, 0, first_start))
        out.append(Instruction(kind, word, target, float(times[0]), float(times[0]), clamped))
    for index, (start, end) in enumerate(runs):
        plateau_end = runs[index + 1][0] if index + 1 < len(runs) else n
        if end >= n:                       # the manoeuvre runs to the end of the record
            value, settled = float(signal[-1]), None
        else:
            value, settled = _plateau_value(signal, end, plateau_end), float(times[end])
        word, target, clamped = to_word(value)
        if out and out[-1].word == word:    # the word in force already: the earlier instruction continues
            absorbed.append(Absorbed(kind, float(times[start]), float(times[min(end, n - 1)]), word, float(value - signal[start])))
            continue
        out.append(Instruction(kind, word, target, float(times[start]), settled, clamped))
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
    course_rate = np.gradient(course, times)
    vertical_rate = np.gradient(height, times)
    acceleration = np.gradient(speed, times)

    # a turn's sweep is the change between the plateaus around it (the run's own edges sit
    # inside the smoothed ramp and understate it); the plateau before is also the intercept angle
    candidates = runs_where(np.abs(course_rate) > vocabulary.turn_rate_deg_s, min_rows(vocabulary.turn_min_s, dt))
    turns: list[tuple[int, int]] = []
    course_before: dict[tuple[int, int], float] = {}
    for index, (start, end) in enumerate(candidates):
        previous_end = candidates[index - 1][1] if index else 0
        next_start = candidates[index + 1][0] if index + 1 < len(candidates) else n
        # a turn already under way at the record's start has no plateau before it: its first row stands in
        before = _plateau_value(course, previous_end, start) if start > previous_end else float(course[start])
        after = _plateau_value(course, end, next_start) if end < next_start else float(course[-1])
        if abs(after - before) >= vocabulary.turn_min_sweep_deg:
            turns.append((start, end))
            course_before[(start, end)] = before
    descents = runs_where(vertical_rate < vocabulary.descent_rate_mps, min_rows(vocabulary.descent_min_s, dt))
    decels = runs_where(acceleration < vocabulary.decel_mps2, min_rows(vocabulary.decel_min_s, dt))

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
    for kind, signal, runs, to_word in (("heading", course, turns, heading_word), ("altitude", height, descents, altitude_word),
                                        ("speed", speed, decels, speed_word)):
        words, dropped = _manoeuvre_words(kind, times, signal, runs, to_word)
        instructions.extend(words)
        absorbed.extend(dropped)
    # the intercept: the last turn after which the aircraft is established (the rule per row over
    # the plateau that follows the turn — the last row when the turn runs to the end), by the
    # relative course held before the turn
    established = frame["established"]
    intercept: Instruction | None = None
    for index, (start, end) in enumerate(turns):
        plateau_end = turns[index + 1][0] if index + 1 < len(turns) else n
        after = established[end:plateau_end] if end < n else established[n - 1:]
        if after.any():
            angle = wrap_deg(course_before[(start, end)])
            intercept = Instruction("intercept", vocabulary.intercept_bin(angle), abs(angle), float(times[start]),
                                    float(times[end]) if end < n else None)
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
    "ALTITUDE_BIN_M", "ALTITUDE_MAX_M", "COURSE_SMOOTHING_S", "DECEL_MIN_S", "DECEL_MPS2", "DESCENT_MIN_S", "DESCENT_RATE_MPS",
    "HEADING_BIN_DEG", "INSTRUCTION_KINDS", "INTERCEPT_ANGLE_BINS_DEG", "MANDATORY_KINDS", "NO_INTERCEPT", "SMOOTHING_S",
    "SPEED_BIN_MPS", "SPEED_MAX_MPS", "SPEED_MIN_MPS", "TOKEN_STEP_S", "TURN_MIN_S", "TURN_MIN_SWEEP_DEG", "TURN_RATE_DEG_S",
    "VOCABULARY_FILE", "VOCABULARY_SCHEMA", "Absorbed", "Instruction", "Reading", "Vocabulary", "course_frame",
    "load_vocabulary", "min_rows", "read_instructions", "runs_where", "segment_positions_s", "sentence", "smooth",
    "write_vocabulary",
]
