"""Instruction words read from an arrival's track (two-tier v3 stage B, plan §5.2.1).

An arrival is a SENTENCE: a sequence of EVENTS, one per moment something changed, each carrying
six words — heading, altitude and speed as absolute targets, the runway they are measured against,
the gap since the previous event, and whether the approach ends here.

    heading    the ground track against the final approach course, 10° a bin, the whole circle
    altitude   height above the landing threshold, 1000 ft a bin, 0 … 10 000 ft
    speed      ground speed (ADS-B carries no airspeed; the wind is inside this number)
    runway     which threshold the three above are measured against — a per-airport class set,
               carried BESIDE the spec so one vocabulary still serves five airports (D62)
    duration   the gap to the previous event, 2 s a bin (D71): the ADS-B row grid's own resolution
    terminal   continue / landed / go-around (D72) — one question, one answer, no separate head

There is no "hold" word: a moment at which nothing changed is not an event and costs no token.
There is no intercept word (D73): it was the heading word's shadow — every one of them was issued
at the same instant as a heading word — and an intercept ANGLE is a consequence of the heading a
controller assigns, not a target he states.

A level is a stretch that holds within a tolerance for `PLATEAU_MIN_S`, whose fitted slope moves it
by no more than half that tolerance; everything between two levels is a manoeuvre, whatever its
rate, and the instruction's target is the level it reaches, issued where the signal departs the
level held just before it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import numpy as np

from geokit.constants import FT_M, KT_MS
from ts_transformer.data.approach_difficulty import (
    course_frame_rows,
)
from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
from ts_transformer.data.runway_context import wrap_deg
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.manoeuvre.segments import MINIMUM_GROUND_SPEED_MPS

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

#: The three kinds read from the track by the plateau rule. `INSTRUCTION_KINDS` adds the runway,
#: the duration and the terminal word; MANDATORY_KINDS stays a PREFIX of it.
MANDATORY_KINDS = ("heading", "altitude", "speed")
#: The RUNWAY is the fifth word (D62): the other four are angles and heights measured against a
#: runway, so a sentence without it cannot say WHICH runway — and in real control the runway is
#: something the controller says out loud. It is a word of a different kind: its classes are the
#: airport's thresholds (`RunwayVocabulary`, bound to the cohort like the aircraft types), so it
#: is NOT part of the spec and does not move the sha — otherwise one vocabulary could not serve
#: five airports. It is a per-position column, so the sentence CAN say a change mid-approach
#: (D64, which the FAA order allows); this data's labeller writes it constant, because a late
#: change is 1 flight in 44 622 and an early one is not resolvable (results §13).
#: The DURATION is the sixth word (D71): the sentence is an EVENT SEQUENCE, so each event has to
#: say how long since the previous one. It is a word, not a regression output, because generation
#: needs a distribution — D61's free running and D56's decoding both take K candidates a step, and
#: a point estimate gives none; and because the gaps are spread (p5 4 s, p50 44 s, p95 128 s), so a
#: squared error would collapse to the mean and never speak the tail.
#: It must stay LAST: `Vocabulary.CONDITIONED_KINDS` is a PREFIX slice of this tuple.
#: The TERMINAL word (D72, user 2026-09-20) says whether this approach ends at the runway:
#: ``continue`` / ``landed`` / ``go-around``. It replaces the prior's separate ``landed`` BCE head
#: — "does the sentence stop here" is one question with three answers, and a separate head made it
#: two mechanisms. Three consequences, all measured: D61's criterion becomes native (free-run until
#: the model itself says ``landed``); the marker is 17 % of events instead of 2.5 % of grid
#: positions, so it is learnable; and ``go-around`` shares a head that IS well trained on
#: continue-vs-landed, so RL meets a meaningful representation rather than a noise head.
#: ``go-around`` is NEVER emitted by the labeller — the 25 km arrival slice keeps only the final
#: approach (results §13.5) — so it must stay out of every gate.
INSTRUCTION_KINDS = MANDATORY_KINDS + ("runway", "duration", "terminal")

TERMINAL_CONTINUE, TERMINAL_LANDED, TERMINAL_GO_AROUND = 0, 1, 2
TERMINAL_WORDS = 3
#: The fill value while a sentence row is being built; no kind may keep it (see `sentence`).
NO_INTERCEPT = -1
#: The runway word has no continuous value it was binned from — a threshold has a NAME, not a
#: number. `float("nan")` would serialise as `NaN`, which Python reads back and strict JSON
#: readers (jq, JSON.parse) refuse, and the sentences files are read by both.
NO_TARGET = 0.0

#: D51 bins (settled 2026-09-20, module docstring). Heading: 10° over a full circle (36 words;
#: bin 0 = the final approach course). Altitude: 1000 ft above the threshold, 0 … 10 000 ft
#: (11 words). Speed: 10 kt of ground speed, 100 … 320 kt (23 words). A value outside a range is
#: clamped to the edge word and COUNTED. Intercept: ≤ 30° / 30–45° / beyond (3 words; the third
#: holds what a procedure would not clear — 42 % of the cohort's captures read there).
HEADING_BIN_DEG = 10.0
FT, KT = FT_M, KT_MS                      # geokit's exact definitions; every bin below derives from them
ALTITUDE_BIN_M = 1000.0 * FT
ALTITUDE_MAX_M = 10000.0 * FT
SPEED_BIN_MPS = 10.0 * KT
SPEED_MIN_MPS = 100.0 * KT
SPEED_MAX_MPS = 320.0 * KT

#: D53 thresholds (the module docstring says how they bind together): the smoothing, the
#: plateau tolerances (each under half a bin, so a plateau cannot straddle two words), the least
#: a plateau holds to be a target, and the least a plateau must move the signal from the word in
#: force to be a NEW word (half a bin for the heading and the height — a 5° correction or a
#: 500 ft step-down is a real instruction; one bin for the speed — an 8 kt wander is wind).
SMOOTHING_S = 10.0
COURSE_SMOOTHING_S = 6.0
COURSE_TOLERANCE_DEG = 4.0
HEIGHT_TOLERANCE_M = 30.0
SPEED_TOLERANCE_MPS = 2.5
PLATEAU_MIN_S = 20.0
HEADING_MIN_CHANGE_DEG = HEADING_BIN_DEG / 2
HEIGHT_MIN_CHANGE_M = ALTITUDE_BIN_M / 2
SPEED_MIN_CHANGE_MPS = SPEED_BIN_MPS
#: How long a plateau must be held for its own word to stand although the change is under the
#: minimum: two plateaus, four sentence positions. Below it a level is a transient (bin-edge
#: wander: a ±3 m/s oscillation straddling a boundary holds each leg ~20 s).
HOLD_MIN_S = 2 * PLATEAU_MIN_S
#: D54: the sentence's position interval (τ); 5 s is the ablation.
#: The duration word's bins: the issue times come off the ADS-B row grid, so 2 s is the data's own
#: resolution and a 2 s bin is lossless (user, 2026-09-20 — deliberately NOT fitted to the observed
#: distribution). The ceiling exists because the class set must be finite; a longer gap clamps to it
#: and is COUNTED, like every other clamp.
DURATION_BIN_S = 2.0
DURATION_MAX_S = 300.0
TOKEN_STEP_S = 10.0
#: The reading rule's version — part of the vocabulary's identity, because the words an executor
#: trained on depend on HOW the track was read, not only on the bins. v1 (retired 2026-09-20): rate
#: thresholds. v2: plateaus; the intercept is the last heading INSTRUCTION after which the aircraft is
#: established (not the last plateau boundary — an absorbed wiggle after the capture is no turn).
#: v3: a plateau's window may not drift by more than half the tolerance end to end (a slow
#: descent is not a plateau). v4: an instruction is issued where the signal DEPARTS its plateau
#: (the last row within half the tolerance of the plateau's value, not where it leaves the
#: band — half the lag on a gentle change), and a plateau that moves the signal by less than the
#: kind's minimum change from the word in force is absorbed (a wobble is not an instruction,
#: whatever bin edge it crosses).
#: v5: the drift over a plateau's opening window is its fitted slope × span; the intercept needs the
#: aircraft NOT established before the manoeuvre (a straight-in never carries one). v6 (retired the
#: same day): a word issued where the signal left the plateau of the word IN FORCE — wrong when the
#: absorbed step was a long level (the fourth hand check: words 100–200 s before the manoeuvre).
#: v7: a word is issued where the signal departs the plateau it held just before the change,
#: absorbed or not — an aircraft that held an intermediate level got its instruction when it left
#: it; and the speed floor is 100 kt (a 105 kt final clamped to the 120 kt edge word read as a
#: full bin high). v8: the minimum change suppresses a TRANSIENT only — a plateau held
#: `HOLD_MIN_S` or longer whose word differs is an instruction whatever its size (measured on the
#: B cohort: 7 598 speed manoeuvres were being absorbed, median held 42 s, 4 032 of them in a
#: different bin from the word in force — 0.6 real speed steps a flight the sentence never said,
#: which is what made every earlier rule read as early or late around them). v9: that hold also
#: needs the level to be at least one TOLERANCE from the word in force's — two levels closer than
#: the tolerance are the same level measured twice, and wording the second one split a steady
#: stretch in two whenever its median sat on a bin edge (the fifth hand check: 6 of 50 pages).
#: v11 (2026-09-20): the sentence became an EVENT SEQUENCE and gained the duration and terminal
#: words (D52 / D70 / D71 / D72). v10 was the grid with the runway column (D62); v9 the four-kind
#: grid. The version is part of the spec and therefore of the sha, so an artefact read under an
#: older rule is refused by name rather than reinterpreted.
READING_RULE = "plateau-v11"

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
    smoothing_s: float = SMOOTHING_S
    course_smoothing_s: float = COURSE_SMOOTHING_S
    course_tolerance_deg: float = COURSE_TOLERANCE_DEG
    height_tolerance_m: float = HEIGHT_TOLERANCE_M
    speed_tolerance_mps: float = SPEED_TOLERANCE_MPS
    plateau_min_s: float = PLATEAU_MIN_S
    heading_min_change_deg: float = HEADING_MIN_CHANGE_DEG
    height_min_change_m: float = HEIGHT_MIN_CHANGE_M
    speed_min_change_mps: float = SPEED_MIN_CHANGE_MPS
    hold_min_s: float = HOLD_MIN_S
    duration_bin_s: float = DURATION_BIN_S
    duration_max_s: float = DURATION_MAX_S
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
        for name in ("smoothing_s", "course_smoothing_s", "plateau_min_s", "token_step_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} is positive seconds")
        for name, half_bin in (("course_tolerance_deg", self.heading_bin_deg / 2), ("height_tolerance_m", self.altitude_bin_m / 2),
                               ("speed_tolerance_mps", self.speed_bin_mps / 2)):
            if not 0 < getattr(self, name) < half_bin:
                raise ValueError(f"{name} is positive and under half a bin ({half_bin:g}), so a plateau cannot straddle two words")
        for name, tolerance in (("heading_min_change_deg", self.course_tolerance_deg), ("height_min_change_m", self.height_tolerance_m),
                                ("speed_min_change_mps", self.speed_tolerance_mps)):
            if getattr(self, name) < tolerance:
                raise ValueError(f"{name} is at least the kind's tolerance ({tolerance:g}): two plateaus closer than that are one")
        if self.hold_min_s < self.plateau_min_s:
            raise ValueError(f"hold_min_s ({self.hold_min_s:g}) is at least plateau_min_s ({self.plateau_min_s:g}): a shorter level is not a plateau")
        if self.token_step_s > self.plateau_min_s:
            raise ValueError("token_step_s is at most plateau_min_s: an instruction issued after the last position would fall out of the sentence")

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
    def words(self) -> dict[str, int]:
        """The class count of every kind the SPEC fixes. The runway's is per airport and lives in
        `RunwayVocabulary`, so it is not here — `word_counts` composes the two."""
        return {"heading": self.heading_words, "altitude": self.altitude_words,
                "speed": self.speed_words,
                "duration": self.duration_words, "terminal": TERMINAL_WORDS}

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

    @property
    def duration_words(self) -> int:
        return int(round(self.duration_max_s / self.duration_bin_s)) + 1

    def duration_bin(self, seconds: float) -> tuple[int, bool]:
        """The word for a gap, and whether it clamped. Exact for a gap on the ADS-B row grid."""
        word = int(math.floor(seconds / self.duration_bin_s + 0.5))
        clamped = not 0 <= word < self.duration_words
        return min(max(word, 0), self.duration_words - 1), clamped

    def duration_centre_s(self, word: int) -> float:
        return word * self.duration_bin_s



    # ── the executor's view of the words ──────────────────────────────────
    #: one event's conditioning: cos, sin of the heading centre, then the altitude and speed
    #: centres as fractions of their ceilings. The intercept column went with D73.
    #: The runway is NOT here: the executor already works in the runway's own frame, so the
    #: runway is implicit in its coordinates, and handing it an index as well would be redundant
    #: AND would make the executor per-airport. The runway is a word the PRIOR says, not a number
    #: the executor listens to (D62).
    CONDITIONING_WIDTH = 4
    CONDITIONED_KINDS = MANDATORY_KINDS
    if CONDITIONED_KINDS != INSTRUCTION_KINDS[:len(CONDITIONED_KINDS)]:      # the slice below is a PREFIX slice
        raise RuntimeError("the conditioned kinds must be the first kinds: a kind inserted before the "
                           "runway would silently condition the executor on the wrong columns")

    def conditioning(self, words: np.ndarray) -> np.ndarray:
        """The bin CENTRES the executor conditions on, ``[..., CONDITIONING_WIDTH]`` float32 for
        the sentence's words (plan §5.2.1 "执行器怎么吃"): the heading as cos / sin so the wrap at
        ±180° is continuous, the altitude and speed as fractions of the vocabulary's ceilings."""
        words = np.asarray(words)
        if words.shape[-1] != len(INSTRUCTION_KINDS):
            raise ValueError(f"words are [..., {len(INSTRUCTION_KINDS)}], got {words.shape}")
        words = words[..., :len(self.CONDITIONED_KINDS)]      # the runway column is not conditioning
        heading = np.radians(words[..., 0] * self.heading_bin_deg)
        out = np.stack((
            np.cos(heading), np.sin(heading),
            words[..., 1] * self.altitude_bin_m / self.altitude_max_m,
            (self.speed_min_mps + words[..., 2] * self.speed_bin_mps) / self.speed_max_mps,
        ), axis=-1)
        return out.astype(np.float32)

    # ── the artefact's identity ───────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        spec = asdict(self)
        return spec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        data = dict(data)
        return cls(**data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def runway_sha256(runway_vocabulary: RunwayVocabulary) -> str:
    """A digest of the runway CLASSES, to be carried and checked wherever `Vocabulary.sha256` is.

    The classes are deliberately outside the spec so one vocabulary serves five airports — but
    that also means two artefacts with the SAME spec sha can disagree about what word 1 means.
    Without this, a prior trained on ``("05L", "23R")`` and decoded against ``("05L", "14",
    "23R")`` reads word 1 as "14" instead of "23R": both indices are in range, so nothing raises
    and every runway is silently relabelled."""
    return hashlib.sha256(json.dumps(list(runway_vocabulary.idents), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class RunwayVocabulary:
    """The runway word's classes: one airport's thresholds, sorted.

    Deliberately NOT a field of `Vocabulary`: the class set is per airport, and putting it in the
    spec would give every airport a different sha and end the property that ONE vocabulary reads
    every runway. It is bound to the cohort and carried beside the spec, exactly as the aircraft
    type vocabulary is (`manoeuvre.context.TypeVocabulary`)."""

    idents: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.idents)) != len(self.idents) or not self.idents:
            raise ValueError("runway idents are unique and there is at least one")
        if list(self.idents) != sorted(self.idents):
            raise ValueError(f"runway idents are stored sorted, got {self.idents}")

    @classmethod
    def from_idents(cls, idents: Iterable[str]) -> RunwayVocabulary:
        return cls(tuple(sorted(set(idents))))

    def __len__(self) -> int:
        return len(self.idents)

    def index(self, ident: str) -> int:
        """The word for this threshold. An unknown one RAISES: a runway the vocabulary was not
        built on is a different airport, not a fallback."""
        try:
            return self.idents.index(ident)
        except ValueError:
            raise ValueError(f"runway {ident!r} is not in this vocabulary ({', '.join(self.idents)})") from None

    def ident(self, word: int) -> str:
        return self.idents[word]


def word_counts(vocabulary: Vocabulary, runway_vocabulary: RunwayVocabulary) -> dict[str, int]:
    """The word count of EVERY kind, **in `INSTRUCTION_KINDS` order**: the spec's five
    (`Vocabulary.words`) with the cohort's runway classes in their place. One definition — the
    prior's head sizes (`PriorConfig.words`) and the readouts that print "words used" read this,
    never a dict built beside it. The ORDER is load-bearing: the prior embeds and scores by column
    index, so a dict in a different order would silently size the wrong head."""
    counts = {**vocabulary.words, "runway": len(runway_vocabulary)}
    return {kind: counts[kind] for kind in INSTRUCTION_KINDS}


def flight_runway(series: FlightSeries) -> str:
    """The threshold ident this flight landed on — the runway word's value. It lives in the
    scenario's SOURCE (the manifest's `runway`): `GeodeticState` carries the threshold's
    geometry, not its name. One definition WITHIN the instruction path, so the labeller and the
    cohort that sizes the runway vocabulary read the same field; `approach_clustering/cli.py` and
    `outputs/guidance/skeleton.py` read the same key directly and predate this."""
    return str(series.scenario.source["runway"])


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
ABSORBED_SMALL_CHANGE = "small change"  # its plateau moves the signal by less than the kind's minimum change from the word in force
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
    event_times_s: np.ndarray          # [E] the moments something changed
    words: np.ndarray                  # [E, 6] heading / altitude / speed / intercept / runway / duration
    runway: str                        # the threshold the four geometric kinds are measured against
    established_from_start: bool
    duration_s: float
    duration_clamped: int = 0
    absorbed: tuple[Absorbed, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id, "flight_id": self.flight_id,
            "instructions": [item.to_dict() for item in self.instructions],
            "event_times_s": self.event_times_s.tolist(), "words": self.words.tolist(),
            "runway": self.runway, "duration_clamped": self.duration_clamped,
            "established_from_start": self.established_from_start, "duration_s": self.duration_s,
            "absorbed": [item.to_dict() for item in self.absorbed],
        }

    def words_at(self, times_s: np.ndarray) -> np.ndarray:
        """The words in force at each time, ``[len(times), 6]``: the latest EVENT at or before it.

        This is the one query every consumer outside the prior uses — the executor asks it for the
        words over its segment — and it works over the event sequence exactly as it worked over the
        old grid, which is why nothing under `outputs/control/` changed when D52 replaced one with
        the other. Before the first event there is nothing, and that is an error.
        """
        times = np.asarray(times_s, dtype=np.float64)
        if times.min() < self.event_times_s[0]:
            raise ValueError(f"{self.flight_id}: no word in force before the first event ({times.min():g} s)")
        index = np.searchsorted(self.event_times_s, times, side="right") - 1
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
    if not np.isfinite(values).all():
        raise ValueError(f"{series.flight_id}: a row holds a non-finite value; a corrupt row is not a track to read")
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
    next ``rows`` rows all lie within ``tolerance`` of their median AND their fitted slope moves
    the signal by at most half the tolerance over the window — a slow transit that merely
    crosses the band, or a pause inside one, is not a plateau (the second hand check found a
    3 m/s descent reading as "level" every 305 m) — and holds while the signal stays within
    ``tolerance`` of that value; it is at least ``rows`` long (``rows`` ≥ 3, so a slope exists).
    Everything outside a plateau is a manoeuvre. The edges sit INSIDE the band: a plateau ends
    where the signal has left its value by the tolerance, and the next opens as soon as the
    signal is within the tolerance of where it will settle (`departure_row` reads the issue row
    off the first, half a tolerance in)."""
    if rows < 3:
        raise ValueError(f"a plateau is at least 3 rows (a slope needs them), got {rows}")
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    out: list[tuple[int, int]] = []
    axis = np.arange(rows, dtype=np.float64)
    i = 0
    while i + rows <= n:
        window = signal[i : i + rows]
        reference = float(np.median(window))
        slope = float(np.polyfit(axis, window, 1)[0])
        if np.abs(window - reference).max() > tolerance or abs(slope) * (rows - 1) > tolerance / 2:
            i += 1
            continue
        j = i + rows
        while j < n and abs(signal[j] - reference) <= tolerance:
            j += 1
        out.append((i, j))
        i = j
    return out


def _plateau_value(signal: np.ndarray, start: int, end: int) -> float:
    """The value a signal holds over ``[start, end)`` (a plateau: never empty): its median."""
    return float(np.median(signal[start:end]))


def departure_row(signal: np.ndarray, start: int, end: int, value: float, tolerance: float) -> int:
    """Where the signal DEPARTS its plateau ``[start, end)`` of ``value``: one past the last row
    within half the tolerance of it — the instruction's issue row (the plateau's ``end`` is where
    the signal has left the band by the whole tolerance, up to tolerance / rate later)."""
    inside = np.flatnonzero(np.abs(signal[start:end] - value) <= tolerance / 2)
    return start + int(inside[-1]) + 1 if len(inside) else start


def _manoeuvre_words(kind: str, times: np.ndarray, signal: np.ndarray, flats: list[tuple[int, int]], rows: int,
                     tolerance: float, min_change: float, hold_min_s: float,
                     to_word: Callable[[float], tuple[int, float, bool]]) -> tuple[list[Instruction], list[Absorbed]]:
    """The instructions of one kind from its plateaus: the first plateau's value is the word at
    t = 0 (the record starts on it, or is already manoeuvring towards it); every later plateau
    is a target issued where the signal departs the plateau it held just before (`departure_row`,
    with ``tolerance``); a signal that leaves its last plateau for good at least ``rows`` before
    the end targets the final value (a shorter tail is unreadable and recorded), and a signal
    with no plateau is one manoeuvre to its end. A target reading as the word already in force
    is recorded as `Absorbed`, and so is one moving the signal by less than ``min_change`` —
    unless the aircraft held it for ``hold_min_s`` or longer, which makes it an instruction
    whatever its size."""
    n = len(times)
    out: list[Instruction] = []
    absorbed: list[Absorbed] = []
    if not flats:
        word, target, clamped = to_word(float(signal[-1]))
        return [Instruction(kind, word, target, float(times[0]), None, clamped)], absorbed
    values = [_plateau_value(signal, start, end) for start, end in flats]
    word, target, clamped = to_word(values[0])
    out.append(Instruction(kind, word, target, float(times[0]), float(times[flats[0][0]]), clamped))
    in_force = 0                            # the plateau of the word in force (an absorbed pause does not move it)
    for index in range(1, len(flats)):
        start, end = flats[index]
        left = departure_row(signal, *flats[index - 1], values[index - 1], tolerance)   # the level it held just before
        word, target, clamped = to_word(values[index])
        change = float(values[index] - values[in_force])
        held_s = float(times[min(end, n - 1)] - times[start])
        if out[-1].word == word:            # the word in force already: the earlier instruction continues
            absorbed.append(Absorbed(kind, float(times[left]), float(times[start]), word, change))
            in_force = index
        elif abs(change) < min_change and not (held_s >= hold_min_s and abs(change) >= tolerance):
            # under the minimum, and either a transient or the same level measured twice (a median
            # that moved less than the tolerance is not a level the aircraft changed to)
            absorbed.append(Absorbed(kind, float(times[left]), float(times[start]), out[-1].word, change, ABSORBED_SMALL_CHANGE))
        else:                               # a decisive change, or a level the aircraft held: its own word
            out.append(Instruction(kind, word, target, float(times[left]), float(times[start]), clamped))
            in_force = index
    last_start, last_end = flats[-1]
    if last_end < n:                        # the signal leaves its last plateau and never settles again
        left = departure_row(signal, last_start, last_end, values[-1], tolerance)
        value = float(signal[-1])
        word, target, clamped = to_word(value)
        change = float(value - values[in_force])
        if n - last_end < rows:
            absorbed.append(Absorbed(kind, float(times[left]), float(times[-1]), out[-1].word, change, ABSORBED_SHORT_TAIL))
        elif abs(change) < min_change:
            absorbed.append(Absorbed(kind, float(times[left]), float(times[-1]), out[-1].word, change, ABSORBED_SMALL_CHANGE))
        elif out[-1].word == word:
            absorbed.append(Absorbed(kind, float(times[left]), float(times[-1]), word, change))
        else:
            out.append(Instruction(kind, word, target, float(times[left]), None, clamped))
    return out, absorbed


def read_instructions(series: FlightSeries, vocabulary: Vocabulary,
                      runway_vocabulary: RunwayVocabulary) -> Reading:
    """Every instruction of one flight, and its sentence (`Reading`)."""
    frame = course_frame(series)
    times = frame["t"]
    n = len(times)
    if n < 2:
        raise ValueError(f"{series.flight_id}: a track of {n} rows has no manoeuvre to read")
    dt = float(np.median(np.diff(times)))
    course = smooth(frame["course_unwrapped_deg"], min_rows(vocabulary.course_smoothing_s, dt))
    window = min_rows(vocabulary.smoothing_s, dt)
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
    tolerances = {"heading": vocabulary.course_tolerance_deg, "altitude": vocabulary.height_tolerance_m, "speed": vocabulary.speed_tolerance_mps}
    min_changes = {"heading": vocabulary.heading_min_change_deg, "altitude": vocabulary.height_min_change_m, "speed": vocabulary.speed_min_change_mps}
    for kind, signal, to_word in (("heading", course, heading_word), ("altitude", height, altitude_word), ("speed", speed, speed_word)):
        words, dropped = _manoeuvre_words(kind, times, signal, flats[kind], rows, tolerances[kind], min_changes[kind],
                                          vocabulary.hold_min_s, to_word)
        instructions.extend(words)
        absorbed.extend(dropped)
    # `established` decides no WORD any more (the intercept went with D73) but it is still the
    # reading's own diagnostic: whether the record opens already on the final approach course.
    established = frame["established"]
    # the runway word (D62): said at the first position, because the controller assigns it before
    # the arrival. It is a CONSTANT here — the sentence can carry a change (the column exists) but
    # this data cannot resolve one (results §13: a late change is 1 flight in 44 622).
    runway_ident = flight_runway(series)
    instructions.append(Instruction("runway", runway_vocabulary.index(runway_ident), NO_TARGET,
                                    float(times[0]), float(times[0]), False))
    instructions.sort(key=lambda item: (item.issued_s, INSTRUCTION_KINDS.index(item.kind)))
    events_s, words, duration_clamped = sentence(instructions, vocabulary)
    return Reading(
        dataset_id=series.dataset_id, flight_id=series.flight_id, instructions=tuple(instructions),
        event_times_s=events_s, words=words, runway=runway_ident, duration_clamped=duration_clamped,
        established_from_start=bool(established[0]), duration_s=float(times[-1] - times[0]),
        absorbed=tuple(sorted(absorbed, key=lambda item: item.start_s)),
    )


def sentence(instructions: Sequence[Instruction], vocabulary: Vocabulary) -> tuple[np.ndarray, np.ndarray, int]:
    """The EVENT SEQUENCE: ``(times, words, clamped)`` (D52, 2026-09-20).

    One row per moment at which something changed — never a row that repeats the one before, and
    no "hold" word. ``words`` is ``[E, 6]``: heading, altitude, speed, intercept, runway, duration,
    the last being the gap to the PREVIOUS event as a word (the first event's is 0).

    This replaced an even grid of one position every τ. The grid did not lose an instruction — two
    of a kind are never closer than the 20 s a level must be held — but it snapped every issue time
    FORWARD by 0–8 s, mean 4 s, and only ever late: 557 m of extra track at 270 kt on average, and
    a bias that accumulates round by round in a closed loop and reads as model error. The event
    time here is the row the labeller read, at the ADS-B grid's own 2 s.
    """
    moments = sorted({item.issued_s for item in instructions})
    if not moments:
        raise ValueError("a sentence needs at least one instruction")
    columns = len(INSTRUCTION_KINDS)
    duration_column = INSTRUCTION_KINDS.index("duration")
    words = np.full((len(moments), columns), NO_INTERCEPT, dtype=np.int64)
    for column, kind in enumerate(INSTRUCTION_KINDS):
        if kind == "duration":
            continue
        issued = sorted((item for item in instructions if item.kind == kind), key=lambda item: item.issued_s)
        for item in issued:                                    # hold each word forward over the events
            words[np.asarray(moments) >= item.issued_s - 1e-9, column] = item.word
    times = np.asarray(moments, dtype=np.float64)
    clamped = 0
    for row, gap in enumerate(np.concatenate(([0.0], np.diff(times)))):
        word, hit = vocabulary.duration_bin(float(gap))
        words[row, duration_column] = word
        clamped += int(hit)
    # the intercept is the ONE kind that may have no word (before the capture turn); every other
    # kind must be in force from the first event. The runway matters most: its column indexes an
    # embedding, and a −1 there would be read as the LAST runway instead of raising.
    terminal_column = INSTRUCTION_KINDS.index("terminal")
    words[:, terminal_column] = TERMINAL_CONTINUE
    words[-1, terminal_column] = TERMINAL_LANDED         # every cohort flight lands; go-around never fires here
    always = MANDATORY_KINDS + ("runway", "duration", "terminal")
    missing = [kind for kind in always if (words[:, INSTRUCTION_KINDS.index(kind)] == NO_INTERCEPT).any()]
    if missing:
        raise ValueError(f"no word in force for {missing} at some event: only the intercept may be absent")
    free = [c for c in range(columns) if c not in (duration_column, terminal_column)]
    if len(moments) > 1 and (np.diff(words, axis=0)[:, free] == 0).all(axis=1).any():
        raise ValueError("an event repeats the one before it: the sequence keeps only moments that changed")
    return times, words, clamped


# ── the vocabulary artefact ───────────────────────────────────────────────

def write_vocabulary(directory: str | Path, vocabulary: Vocabulary, *, runway_vocabulary: RunwayVocabulary,
                     cohort_identity: dict[str, Any], counts: dict[str, Any], source: dict[str, Any]) -> Path:
    """Write ``instruction_vocabulary.json`` (refused if it exists): the spec (under its sha), the
    cohort's runway classes BESIDE it (``runway_idents`` — in the spec they would give every
    airport a different sha), the cohort the counts were read on, the per-word counts, and where
    it came from."""
    directory = Path(directory)
    path = directory / VOCABULARY_FILE
    if path.exists():
        raise FileExistsError(f"{path} exists; a vocabulary is never overwritten")
    write_json_atomic(path, {
        "schema": VOCABULARY_SCHEMA, "written_utc": utc_now(), "spec": vocabulary.to_dict(), "sha256": vocabulary.sha256,
        "runway_idents": list(runway_vocabulary.idents), "words": vocabulary.words,
        "cohort_identity": dict(cohort_identity), "counts": dict(counts), "source": dict(source),
    })
    return path


def load_vocabulary(path: str | Path) -> tuple[Vocabulary, RunwayVocabulary, dict[str, Any]]:
    """``(vocabulary, runway vocabulary, the file's payload)``; refuses a payload whose sha is not
    its spec's. The runway classes are the file's own (`write_vocabulary`), not the spec's."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["schema"] != VOCABULARY_SCHEMA:
        raise ValueError(f"{path}: schema {payload['schema']!r} is not {VOCABULARY_SCHEMA!r}")
    vocabulary = Vocabulary.from_dict(payload["spec"])
    if vocabulary.sha256 != payload["sha256"]:
        raise ValueError(f"{path}: the spec's sha {vocabulary.sha256[:12]}… is not the file's {payload['sha256'][:12]}…")
    return vocabulary, RunwayVocabulary(tuple(payload["runway_idents"])), payload


__all__ = [
    "ALTITUDE_BIN_M", "ALTITUDE_MAX_M", "COURSE_SMOOTHING_S", "COURSE_TOLERANCE_DEG", "HEADING_BIN_DEG", "HEIGHT_TOLERANCE_M",
    "DURATION_BIN_S", "DURATION_MAX_S", "HOLD_MIN_S", "INSTRUCTION_KINDS", "MANDATORY_KINDS", "NO_INTERCEPT", "PLATEAU_MIN_S",
    "READING_RULE", "SMOOTHING_S", "TERMINAL_CONTINUE", "TERMINAL_GO_AROUND", "TERMINAL_LANDED",
    "TERMINAL_WORDS", "NO_TARGET",
    "SPEED_BIN_MPS", "SPEED_MAX_MPS", "SPEED_MIN_MPS", "SPEED_TOLERANCE_MPS", "TOKEN_STEP_S", "VOCABULARY_FILE",
    "VOCABULARY_SCHEMA", "ABSORBED_SAME_WORD", "ABSORBED_SHORT_TAIL", "ABSORBED_SMALL_CHANGE", "Absorbed", "Instruction",
    "Reading", "RunwayVocabulary", "Vocabulary", "course_frame", "departure_row", "flight_runway", "load_vocabulary", "min_rows",
    "plateaus", "read_instructions", "segment_positions_s", "sentence", "smooth", "word_counts", "write_vocabulary",
]
