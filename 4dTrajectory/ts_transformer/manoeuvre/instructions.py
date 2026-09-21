"""Instruction words read from an arrival's track (two-tier v3 stage B, plan §5.2.1).

An arrival is a SENTENCE: a sequence of EVENTS, one per moment something changed plus one for the
landing, each carrying six words — heading, vertical and speed as targets, the runway they are
measured against, the gap since the previous event, and whether the approach ends here.

    heading    the ground track against the final approach course, 5° a bin, the whole circle;
               word 0 names the COURSE, and flying it means tracking the centreline
    vertical   the FLIGHT PATH ANGLE, six modes, DESCENT POSITIVE — a rate, read by fitting the
               height profile, not a height to hold
    speed      ground speed, 16 fitted centres (ADS-B carries no airspeed; the wind is inside it)
    runway     which threshold the three above are measured against — `AIRPORT:ident`, a
               per-cohort class set carried BESIDE the spec so one vocabulary serves five
               airports (D62)
    duration   the gap to the previous event, 2 s a bin (D71): the ADS-B row grid's own resolution
    terminal   continue / landed / go-around (D72) — one question, one answer, no separate head

There is no "hold" word: a moment at which nothing changed is not an event and costs no token.
There is no intercept word (D73): it was the heading word's shadow — every one of them was issued
at the same instant as a heading word — and an intercept ANGLE is a consequence of the heading a
controller assigns, not a target he states.

TWO KINDS OF QUANTITY, READ TWO WAYS. The heading and the speed are ABSOLUTE TARGETS, so a level —
a stretch that holds within a tolerance for `PLATEAU_MIN_S`, whose fitted slope moves it by no more
than half that tolerance — is what an instruction looks like; everything between two levels is a
manoeuvre, whatever its rate, and the target is the level it reaches, issued where the signal
departs the level held just before it. The vertical is a RATE, which describes the shape of the
whole curve, so its segments must TILE the track: they come from an optimal piecewise-linear fit of
height against cumulative horizontal distance (`_vertical_instructions`).

WHAT AN ABSOLUTE TARGET CANNOT SAY, and how the reading pays for it: a direction cannot say which
way ROUND to turn, so a turn wider than `turn_split_deg` is split into intermediate targets the
aircraft passed through; and a direction cannot hold a LINE, so flying word 0 means tracking the
centreline rather than its bearing (`instruction_kinematics.target_course_deg`). Both were found by
flying the sentences back (`run_ts.py instruction_replay`), where they cost 63 points of landing
rate between them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
MANDATORY_KINDS = ("heading", "vertical", "speed")
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

#: The heading bin (2026-09-21): 5° over a full circle, 72 words, bin 0 = the final approach
#: course. It halved from 10° when the vocabulary was re-measured; each class still carries a
#: median 497 samples on the five-airport cohort. A value outside a kind's range is clamped to the
#: edge word and COUNTED, never silently.
HEADING_BIN_DEG = 5.0
FT, KT = FT_M, KT_MS                      # geokit's exact definitions, kept for the readouts
#: The VERTICAL word is a flight path angle, not a height: on an approach an aircraft does two
#: things, hold level or descend near 3 deg, and that angle barely changes from twenty thousand
#: feet to the threshold. Descent is POSITIVE, so the climb mode is negative. The climb mode is
#: for the go-around that post-training constructs; it occurs zero times in this data, and it is
#: here because the vocabulary has to be fixed before the experiments run.
VERTICAL_MODES_DEG = (-3.0, 0.0, 1.4, 2.4, 3.1, 4.4)
LEVEL_MODE_DEG = 0.0
#: How many straight segments one approach's vertical profile is cut into (§2.3 of the plan).
VERTICAL_SEGMENTS = 5
#: How many points the vertical profile is resampled to before the fit. The optimal-breakpoint
#: search is quadratic in the point count and the profile is smooth, so a few dozen points carry
#: it. It IS in the spec (`Vocabulary.vertical_fit_points`) because it can move a breakpoint, and
#: a number that moves a breakpoint moves the words.
VERTICAL_FIT_POINTS = 70
#: Ground speed centres, fitted to the data and rounded to 1 m/s. Ground speed carries no
#: whole-knot structure — control assigns indicated airspeed and the wind is inside this number —
#: so the centres come from the distribution, not from a round unit.
SPEED_CENTRES_MPS = (44.0, 56.0, 63.0, 68.0, 74.0, 79.0, 86.0, 93.0,
                     99.0, 107.0, 114.0, 121.0, 129.0, 138.0, 147.0, 157.0)
#: Tolerances a word carries into decoding: the level mode an absolute one (a percentage of zero
#: is no tolerance at all), every other mode and the speed a fraction of the value.
VERTICAL_LEVEL_TOLERANCE_DEG = 0.1
VERTICAL_TOLERANCE_FRACTION = 0.07
SPEED_TOLERANCE_FRACTION = 0.03

#: D53 thresholds (the module docstring says how they bind together): the smoothing, the
#: plateau tolerances (each under half a bin, so a plateau cannot straddle two words), the least
#: a plateau holds to be a target, and the least a plateau must move the signal from the word in
#: force to be a NEW word (half a bin for the heading and the height — a 5° correction or a
#: 500 ft step-down is a real instruction; one bin for the speed — an 8 kt wander is wind).
SMOOTHING_S = 10.0
COURSE_SMOOTHING_S = 6.0
COURSE_TOLERANCE_DEG = 2.0          # strictly under half a 5° bin, so a plateau cannot straddle two words
#: A plateau is about the signal being STEADY, so this one stays absolute (it is not the speed
#: word's decode band — that is `speed_tolerance_fraction`). It must be strictly under half the
#: narrowest gap between centres (5.0 m/s at 63→68 and 74→79) or a plateau can straddle two words;
#: 2.5 was exactly half and violated it. 2.0 is the same fraction of the half-gap the heading uses
#: (2.0° of a 2.5° half-bin), which is the only reason to prefer it to any other value under 2.5.
SPEED_TOLERANCE_MPS = 2.0
#: The widest turn one heading word may be asked to express. A heading word is an ABSOLUTE target
#: and "turn to X" does not say which way round — so anything a controller says as "turn LEFT
#: heading 270" loses its direction here. Under this angle the shortest way IS the way the
#: aircraft went (measured: |real turn − the rotation the words imply| p50 1°, p90 3°, max 4°);
#: at a half circle the two are the same distance apart and the reader would be guessing. Measured
#: on the five-airport train split: 13.4 % of the 28,221 heading changes exceed 150° and 9.9 % are
#: exactly a half circle, so 17.1 % of flights carry at least one — and a reconstruction that
#: guesses wrong mirrors the whole track (up to 18 km). A wider turn is SPLIT into equal parts
#: under this angle, each said when the aircraft reaches the one before, which is how a controller
#: says it too.
TURN_SPLIT_DEG = 150.0
PLATEAU_MIN_S = 20.0
HEADING_MIN_CHANGE_DEG = HEADING_BIN_DEG / 2
SPEED_MIN_CHANGE_FRACTION = SPEED_TOLERANCE_FRACTION

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
#: grid.
#: v12 (2026-09-21): the altitude word became the VERTICAL word — a flight path angle read by
#: optimal piecewise-linear fitting of height against horizontal distance, not a plateau — the
#: heading bin halved to 5°, the speed moved to fitted centres, and the runway word's class label
#: was qualified by airport.
#: v13 (2026-09-21), all four from reading the v12 artefacts back:
#:   · the LANDING is an event. v12 wrote the terminal word on the last CHANGE, a median 136 s
#:     (p95 240) before the record ended, and the duration word says the gap to the previous
#:     event — so the final leg's length was in no word and a sentence could not say when it
#:     landed.
#:   · a turn wider than `turn_split_deg` is SPLIT (see there): an absolute heading word cannot
#:     say which way round, and 9.9 % of heading changes were exactly a half circle.
#:   · the minimum change is a fraction of the value IN FORCE (`Vocabulary.min_change`), not of
#:     the flight's median speed, which judged a 70 m/s final by its own 140 m/s cruise.
#:   · `vertical_fit_points` joined the spec (it moves breakpoints, so it moves words) and the
#:     merge of two same-word segments re-angles over the combined span instead of keeping the
#:     first piece's.
#: The version is part of the spec and therefore of the sha, so an artefact read under an older
#: rule is refused by name rather than reinterpreted.
READING_RULE = "segment-v13"

VOCABULARY_SCHEMA = "ts-instruction-vocabulary-v1"
VOCABULARY_FILE = "instruction_vocabulary.json"


@dataclass(frozen=True)
class Vocabulary:
    """The bins and thresholds that turn a track into words — the artefact every executor and
    prior binds to (`sha256` over `spec`). The established rule's two numbers are part of the
    spec so the sha moves with them, but they are the package's one definition
    (`approach_difficulty`) and cannot be set here."""

    heading_bin_deg: float = HEADING_BIN_DEG
    vertical_modes_deg: tuple[float, ...] = VERTICAL_MODES_DEG
    vertical_segments: int = VERTICAL_SEGMENTS
    vertical_fit_points: int = VERTICAL_FIT_POINTS
    speed_centres_mps: tuple[float, ...] = SPEED_CENTRES_MPS
    vertical_level_tolerance_deg: float = VERTICAL_LEVEL_TOLERANCE_DEG
    vertical_tolerance_fraction: float = VERTICAL_TOLERANCE_FRACTION
    speed_tolerance_fraction: float = SPEED_TOLERANCE_FRACTION
    smoothing_s: float = SMOOTHING_S
    course_smoothing_s: float = COURSE_SMOOTHING_S
    course_tolerance_deg: float = COURSE_TOLERANCE_DEG
    speed_tolerance_mps: float = SPEED_TOLERANCE_MPS
    turn_split_deg: float = TURN_SPLIT_DEG
    plateau_min_s: float = PLATEAU_MIN_S
    heading_min_change_deg: float = HEADING_MIN_CHANGE_DEG
    speed_min_change_fraction: float = SPEED_MIN_CHANGE_FRACTION
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
        for name, centres in (("vertical_modes_deg", self.vertical_modes_deg),
                              ("speed_centres_mps", self.speed_centres_mps)):
            if len(centres) < 2 or list(centres) != sorted(centres) or len(set(centres)) != len(centres):
                raise ValueError(f"{name} is at least two distinct centres, stored sorted")
        if LEVEL_MODE_DEG not in self.vertical_modes_deg:
            raise ValueError(f"the vertical modes must contain the level mode ({LEVEL_MODE_DEG:g}°)")
        if self.vertical_segments < 1:
            raise ValueError("vertical_segments is at least 1")
        if self.vertical_fit_points < 2 * self.vertical_segments:
            raise ValueError("vertical_fit_points gives every segment at least two points to fit")
        for name in ("smoothing_s", "course_smoothing_s", "plateau_min_s", "token_step_s",
                     "vertical_level_tolerance_deg", "vertical_tolerance_fraction",
                     "speed_tolerance_fraction"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} is positive")
        if not 0 < self.course_tolerance_deg < self.heading_bin_deg / 2:
            raise ValueError(f"course_tolerance_deg is under half a heading bin ({self.heading_bin_deg / 2:g}°), "
                             "so a plateau cannot straddle two words")
        # the same invariant the heading's tolerance carries, and it used to be stated for the
        # speed too: a plateau tolerance at or above half the narrowest gap lets one plateau
        # straddle two words, so the word read from it depends on where inside the plateau the
        # reader looked. With the fitted centres the narrowest gap is 5 m/s (63→68, 74→79).
        narrowest = min(b - a for a, b in zip(self.speed_centres_mps, self.speed_centres_mps[1:]))
        if not 0 < self.speed_tolerance_mps < narrowest / 2:
            raise ValueError(f"speed_tolerance_mps is under half the narrowest centre gap ({narrowest / 2:g} m/s), "
                             "so a plateau cannot straddle two words")
        if not self.heading_bin_deg < self.turn_split_deg < 180.0:
            raise ValueError("turn_split_deg is between one heading bin and a half circle: at 180° the "
                             "shortest way is a coin toss, and under a bin every change would split")
        if self.heading_min_change_deg < self.course_tolerance_deg:
            raise ValueError("heading_min_change_deg is at least the kind's tolerance: two plateaus closer than that are one")
        if self.hold_min_s < self.plateau_min_s:
            raise ValueError(f"hold_min_s ({self.hold_min_s:g}) is at least plateau_min_s ({self.plateau_min_s:g}): a shorter level is not a plateau")
        if self.token_step_s > self.plateau_min_s:
            raise ValueError("token_step_s is at most plateau_min_s: an instruction issued after the last position would fall out of the sentence")

    # ── word counts ──────────────────────────────────────────────────────
    @property
    def heading_words(self) -> int:
        return int(round(360.0 / self.heading_bin_deg))

    @property
    def vertical_words(self) -> int:
        return len(self.vertical_modes_deg)

    @property
    def speed_words(self) -> int:
        return len(self.speed_centres_mps)


    @property
    def words(self) -> dict[str, int]:
        """The class count of every kind the SPEC fixes. The runway's is per airport and lives in
        `RunwayVocabulary`, so it is not here — `word_counts` composes the two."""
        return {"heading": self.heading_words, "vertical": self.vertical_words,
                "speed": self.speed_words,
                "duration": self.duration_words, "terminal": TERMINAL_WORDS}

    # ── binning (each returns the word index; the caller counts clamps) ───
    def heading_bin(self, relative_deg: float) -> int:
        """The word of a course relative to the final approach course; bin 0 is the course,
        bin 18 (at 10°) the downwind; the wrap is at ±180°."""
        return int(math.floor(wrap_deg(relative_deg) / self.heading_bin_deg + 0.5)) % self.heading_words     # half-up, never banker's

    def heading_centre_deg(self, word: int) -> float:
        return wrap_deg(word * self.heading_bin_deg)

    @staticmethod
    def _nearest(centres: tuple[float, ...], value: float) -> tuple[int, bool]:
        """``(word, outside)``: the nearest centre, and whether the value fell beyond the ends.

        The centres are not evenly spaced, so a word is the nearest centre rather than an index
        computed from a bin width. ``outside`` is what the readouts count: a value past either end
        takes the end word, and a non-zero count has to be stated."""
        word = min(range(len(centres)), key=lambda i: abs(centres[i] - value))
        return word, value < centres[0] or value > centres[-1]

    def vertical_bin(self, angle_deg: float) -> tuple[int, bool]:
        """The word of a flight path angle in degrees, descent POSITIVE."""
        return self._nearest(self.vertical_modes_deg, angle_deg)

    def vertical_centre_deg(self, word: int) -> float:
        return self.vertical_modes_deg[word]

    def vertical_tolerance_deg(self, word: int) -> float:
        """How far the executor may sit from the commanded angle. The level mode gets an absolute
        tolerance because a fraction of zero is no tolerance at all, and level flight is 12 % of
        the segments; every other mode gets a fraction of itself, which keeps the tolerance small
        on the long shallow descents that produce the largest errors."""
        centre = self.vertical_modes_deg[word]
        if centre == LEVEL_MODE_DEG:
            return self.vertical_level_tolerance_deg
        return abs(centre) * self.vertical_tolerance_fraction

    def speed_bin(self, speed_mps: float) -> tuple[int, bool]:
        return self._nearest(self.speed_centres_mps, speed_mps)

    def speed_centre_mps(self, word: int) -> float:
        return self.speed_centres_mps[word]

    def speed_tolerance(self, word: int) -> float:
        """A fraction of the commanded speed: the cost of a speed tolerance is arrival TIME, and
        that cost goes as delta over v squared, so an absolute tolerance is most expensive exactly
        where the aircraft is slowest."""
        return self.speed_centres_mps[word] * self.speed_tolerance_fraction

    def min_change(self, kind: str, in_force: float) -> float:
        """The least a new plateau must move the signal from the value IN FORCE to be a new
        instruction rather than the same one continuing. ONE definition: the reader calls it and
        so does anything checking the reader, because a test that restates this expression is a
        test of the expression, not of the reading.

        The heading's is absolute (half a bin: a 2.5° correction is not an instruction). The
        speed's is a FRACTION of the speed in force, floored at the plateau tolerance — below the
        tolerance two plateaus are not distinguishable in the first place, and a fraction is what
        keeps a 140 m/s flight and a 70 m/s one judged on the same terms."""
        if kind == "heading":
            return self.heading_min_change_deg
        if kind == "speed":
            return max(abs(in_force) * self.speed_min_change_fraction, self.speed_tolerance_mps)
        raise ValueError(f"{kind!r} is not read from plateaus, so it has no minimum change")

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
    #: one event's conditioning: cos, sin of the heading centre, then the vertical and speed
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
        ±180° is continuous, the vertical and speed as fractions of the vocabulary's ceilings."""
        words = np.asarray(words)
        if words.shape[-1] != len(INSTRUCTION_KINDS):
            raise ValueError(f"words are [..., {len(INSTRUCTION_KINDS)}], got {words.shape}")
        words = words[..., :len(self.CONDITIONED_KINDS)]      # the runway column is not conditioning
        if (words < 0).any():
            # `NO_TARGET`/the fill value indexes from the END of the centre arrays, so a −1 would
            # read back as "descend 4.4°, 157 m/s" — a silent wrong answer for the one value the
            # module defines as "nothing said yet"
            raise ValueError("a word column holds the fill value: every conditioned kind is in force from the first event")
        heading = np.radians(words[..., 0] * self.heading_bin_deg)
        modes = np.asarray(self.vertical_modes_deg, dtype=np.float64)
        speeds = np.asarray(self.speed_centres_mps, dtype=np.float64)
        out = np.stack((
            np.cos(heading), np.sin(heading),
            modes[words[..., 1]] / np.abs(modes).max(),
            speeds[words[..., 2]] / speeds.max(),
        ), axis=-1)
        return out.astype(np.float32)

    # ── the artefact's identity ───────────────────────────────────────────
    #: The fields the sha is NOT taken over. A vocabulary's identity is what it makes the words
    #: BE — the bins, the centres, the segmentation, the plateau rules. These three say how wide a
    #: band a word ALLOWS when something flies it, which changes no word anywhere: read the same
    #: track under two tolerances and you get the same sentence. Keeping them in the sha (as this
    #: did until 2026-09-21) means moving a decode band refuses every artefact and checkpoint bound
    #: to the vocabulary, for a change none of them can observe. They are written beside the spec
    #: instead, and `to_dict` still carries them so a vocabulary round-trips exactly.
    DECODE_ONLY_FIELDS = ("vertical_level_tolerance_deg", "vertical_tolerance_fraction", "speed_tolerance_fraction")

    def to_dict(self) -> dict[str, Any]:
        """The whole spec, tolerances included — what `from_dict` round-trips."""
        return asdict(self)

    def identity(self) -> dict[str, Any]:
        """What `sha256` is taken over: the spec WITHOUT the decode-only tolerances."""
        return {name: value for name, value in asdict(self).items() if name not in self.DECODE_ONLY_FIELDS}

    def tolerances(self) -> dict[str, float]:
        """The decode bands, carried beside the sha rather than inside it."""
        return {name: getattr(self, name) for name in self.DECODE_ONLY_FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocabulary:
        """JSON has no tuples, so the centre lists come back as lists. They are coerced here —
        a list field would compare unequal to the same vocabulary built in code, and the sha is
        taken over this dict, so the two must round-trip exactly."""
        data = dict(data)
        for name in ("vertical_modes_deg", "speed_centres_mps"):
            if name in data:
                data[name] = tuple(float(value) for value in data[name])
        return cls(**data)

    @property
    def sha256(self) -> str:
        """The vocabulary's identity — over `identity()`, so a decode tolerance can move without
        refusing every artefact that was read under this reading."""
        return hashlib.sha256(json.dumps(self.identity(), sort_keys=True).encode()).hexdigest()


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
    """The threshold this flight landed on, as ``AIRPORT:ident`` — the runway word's value.

    The ident lives in the scenario's SOURCE (the manifest's `runway`): `GeodeticState` carries
    the threshold's geometry, not its name. One definition WITHIN the instruction path, so the
    labeller and the cohort that sizes the runway vocabulary read the same field;
    `approach_clustering/cli.py` and `outputs/guidance/skeleton.py` read the same key directly and
    predate this.

    QUALIFIED BY AIRPORT, because idents collide across a pooled cohort: KSJC and KSTL both have
    12L / 12R / 30L / 30R, KSTL and KMSY both have 11 and 29 — six idents covering twelve
    different runways whose approach courses point different ways. Unqualified, one embedding
    would have to stand for both, and the word could not say which runway it meant. On a single
    airport the prefix is redundant and harmless; on the five-airport cohort it is the difference
    between 16 classes and 22. (The classes are the COHORT's, not the airports': the five arrival
    manifests hold 23 thresholds — KRDU 4, KSJC 4, KSTL 8, KSMF 3, KMSY 4 — and the pooled
    development cohort carries 22 of them, no KSTL 06 arrival being in it.)"""
    airport = series.airport
    if not airport:
        raise ValueError(
            f"{series.dataset_id!r} carries no arrival airport, so its runway word cannot be qualified"
        )
    return f"{airport}:{series.scenario.source['runway']}"


@dataclass(frozen=True)
class Instruction:
    """One word issued at one time: ``kind`` ∈ `INSTRUCTION_KINDS`, ``word`` its index, ``target``
    the value it was read from (deg relative to the course / deg of flight path angle, descent
    positive / m/s; `NO_TARGET` for the runway, which has a name rather than a number),
    ``issued_s`` the manoeuvre's start (the previous plateau's end),
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
ABSORBED_SHORT_SEGMENT = "short segment"  # the vertical fit cut a piece shorter than a plateau, reading a DIFFERENT word


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
    words: np.ndarray                  # [E, 6] `INSTRUCTION_KINDS`: heading / vertical / speed / runway / duration / terminal
    runway: str                        # the threshold the four geometric kinds are measured against
    established_from_start: bool
    duration_s: float
    duration_clamped: int = 0
    absorbed: tuple[Absorbed, ...] = ()
    #: What the vertical fit's segment CEILING costs on this flight: the RMS height error of the
    #: words' own profile against the track, and how many fitted pieces read as one word and were
    #: folded together. Stated because a cap that cannot be seen is a silent approximation.
    vertical_fit_rms_m: float = 0.0
    vertical_pieces_merged: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id, "flight_id": self.flight_id,
            "instructions": [item.to_dict() for item in self.instructions],
            "event_times_s": self.event_times_s.tolist(), "words": self.words.tolist(),
            "runway": self.runway, "duration_clamped": self.duration_clamped,
            "vertical_fit_rms_m": self.vertical_fit_rms_m, "vertical_pieces_merged": self.vertical_pieces_merged,
            "established_from_start": self.established_from_start, "duration_s": self.duration_s,
            "absorbed": [item.to_dict() for item in self.absorbed],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Reading:
        """Read back what `to_dict` wrote — the artefact's own sentence.

        Anything JUDGING a vocabulary artefact (does its sentence fly to the runway?) must fly
        what the file SAYS, not re-derive it from the track: a gate that re-reads its own input
        cannot see the file drift from the code that wrote it."""
        return cls(
            dataset_id=data["dataset_id"], flight_id=data["flight_id"],
            instructions=tuple(Instruction(**item) for item in data["instructions"]),
            event_times_s=np.asarray(data["event_times_s"], dtype=np.float64),
            words=np.asarray(data["words"], dtype=np.int64),
            runway=data["runway"], established_from_start=bool(data["established_from_start"]),
            duration_s=float(data["duration_s"]), duration_clamped=int(data["duration_clamped"]),
            absorbed=tuple(Absorbed(**item) for item in data["absorbed"]),
            vertical_fit_rms_m=float(data["vertical_fit_rms_m"]),
            vertical_pieces_merged=int(data["vertical_pieces_merged"]),
        )

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
                     tolerance: float, min_change: Callable[[float], float], hold_min_s: float,
                     to_word: Callable[[float], tuple[int, float, bool]]) -> tuple[list[Instruction], list[Absorbed]]:
    """The instructions of one kind from its plateaus: the first plateau's value is the word at
    t = 0 (the record starts on it, or is already manoeuvring towards it); every later plateau
    is a target issued where the signal departs the plateau it held just before (`departure_row`,
    with ``tolerance``); a signal that leaves its last plateau for good at least ``rows`` before
    the end targets the final value (a shorter tail is unreadable and recorded), and a signal
    with no plateau is one manoeuvre to its end. A target reading as the word already in force
    is recorded as `Absorbed`, and so is one moving the signal by less than ``min_change`` of the
    value IN FORCE — unless the aircraft held it for ``hold_min_s`` or longer, which makes it an
    instruction whatever its size.

    ``min_change`` is a function of the value in force rather than a number because for the speed
    it is a FRACTION: one flight decelerating 140 → 70 m/s must not have its slow half judged by a
    threshold taken from its fast half (which a flight-median constant does)."""
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
        elif abs(change) < min_change(values[in_force]) and not (held_s >= hold_min_s and abs(change) >= tolerance):
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
        elif abs(change) < min_change(values[in_force]):
            absorbed.append(Absorbed(kind, float(times[left]), float(times[-1]), out[-1].word, change, ABSORBED_SMALL_CHANGE))
        elif out[-1].word == word:
            absorbed.append(Absorbed(kind, float(times[left]), float(times[-1]), word, change))
        else:
            out.append(Instruction(kind, word, target, float(times[left]), None, clamped))
    return out, absorbed


def _profile(times: np.ndarray, height_m: np.ndarray, ground_speed_mps: np.ndarray,
             points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x, y, rows)``: height against CUMULATIVE HORIZONTAL DISTANCE, resampled.

    Distance rather than time, because the vertical word is an angle and an angle is this curve's
    slope. ``rows`` maps each resampled point back to a row of the track, so a breakpoint can be
    given a time."""
    # no floor on the speed and no check that the track covers ground: `course_frame` REFUSES a
    # row slower than `MINIMUM_GROUND_SPEED_MPS` before this is reached, so both could only ever
    # restate a guarantee — and a bound that cannot bind reads as though it had
    step = ground_speed_mps[1:] * np.diff(times)
    x = np.concatenate([[0.0], np.cumsum(step)])
    take = np.unique(np.searchsorted(x, np.linspace(0.0, x[-1], min(points, len(x)))))
    take = np.clip(take, 0, len(x) - 1)
    return x[take], height_m[take], take


def _segment_costs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``cost[i][j]`` = the squared error of the best straight line through points i…j."""
    n = len(x)
    cost = np.zeros((n, n))
    for i in range(n):
        sx = sy = sxx = sxy = syy = 0.0
        for j in range(i, n):
            sx += x[j]; sy += y[j]; sxx += x[j] * x[j]; sxy += x[j] * y[j]; syy += y[j] * y[j]
            k = j - i + 1
            if k < 2:
                continue
            denominator = k * sxx - sx * sx
            if denominator <= 1e-9:
                continue
            slope = (k * sxy - sx * sy) / denominator
            intercept = (sy - slope * sx) / k
            cost[i][j] = max(syy + k * intercept ** 2 + slope ** 2 * sxx
                             + 2 * intercept * slope * sx - 2 * intercept * sy - 2 * slope * sxy, 0.0)
    return cost


def _breakpoints(x: np.ndarray, y: np.ndarray, segments: int) -> list[int]:
    """The optimal ``segments`` breakpoints, shared endpoints, tiling the whole profile."""
    n = len(x)
    segments = max(1, min(segments, n - 1))
    cost = _segment_costs(x, y)
    best = np.full((segments + 1, n), np.inf)
    came = np.zeros((segments + 1, n), dtype=int)
    best[0][0] = 0.0
    for k in range(1, segments + 1):
        for j in range(1, n):
            for i in range(k - 1, j):
                candidate = best[k - 1][i] + cost[i][j]
                if candidate < best[k][j]:
                    best[k][j] = candidate
                    came[k][j] = i
    out = [n - 1]
    j = n - 1
    for k in range(segments, 0, -1):
        j = came[k][j]
        out.append(j)
    return sorted(set(out))


def _vertical_instructions(times: np.ndarray, height_m: np.ndarray, ground_speed_mps: np.ndarray,
                           vocabulary: Vocabulary) -> tuple[list[Instruction], list[Absorbed]]:
    """The vertical words: one straight segment of the height profile is one instruction.

    A plateau rule cannot read this kind. A plateau is a stretch where a signal HOLDS, which is
    what an absolute target looks like; an angle is a RATE, and a rate describes the shape of the
    whole curve, so the segments have to tile the track. Plateaus leave the transitions between
    them unowned, and on this data those transitions carry most of the vertical movement."""
    x, y, rows = _profile(times, height_m, ground_speed_mps, vocabulary.vertical_fit_points)
    if len(x) < 2:
        raise ValueError("a vertical profile of fewer than two points has no segment to read")
    bounds = _breakpoints(x, y, vocabulary.vertical_segments)
    # One row per fitted segment, BEFORE any rule is applied to it.
    cut: list[tuple[int, float, float, float, bool]] = []       # word, angle, issued, settled, outside
    for a, b in zip(bounds[:-1], bounds[1:]):
        span = x[b] - x[a]
        if span <= 0.0:
            continue
        angle = math.degrees(math.atan2(y[a] - y[b], span))      # descent positive
        word, outside = vocabulary.vertical_bin(angle)
        cut.append((word, angle, float(times[rows[a]]), float(times[rows[b]]), outside))

    absorbed: list[Absorbed] = []

    # The segment count is a CEILING, not a quota. A profile with fewer real segments than that
    # spends the spare ones on transitions a few seconds long, and a target held for less than a
    # plateau is not an instruction — the same rule the heading and the speed are read under.
    # So merge bottom-up until nothing is too short: take the shortest offender and fold it into
    # whichever neighbour its angle is closer to, recomputing that neighbour's angle from the two
    # spans it now covers. Iterating to a fixed point is what a one-pass rule cannot do — a sliver
    # sitting BETWEEN two segments of the same angle has to go before they can be seen as one.
    pieces = [[word, angle, issued, settled, outside, x[b_] - x[a_]]
              for (word, angle, issued, settled, outside), (a_, b_)
              in zip(cut, zip(bounds[:-1], bounds[1:])) if x[b_] - x[a_] > 0.0]
    while len(pieces) > 1:
        short = min(range(len(pieces)), key=lambda i: pieces[i][3] - pieces[i][2])
        if pieces[short][3] - pieces[short][2] >= vocabulary.plateau_min_s:
            break
        left, right = short - 1, short + 1
        if left < 0:
            into = right
        elif right >= len(pieces):
            into = left
        else:
            into = left if abs(pieces[left][1] - pieces[short][1]) <= abs(pieces[right][1] - pieces[short][1]) else right
        host = pieces[into]
        drop = pieces[short]
        if drop[0] != host[0]:
            # A sliver that read as a DIFFERENT word is an instruction the rule refused; one that
            # read as the same word is just the fit having cut one angle in two, and recording it
            # would fill the diagnostics with the fit's own bookkeeping. `change` is what the
            # manoeuvre MOVED the signal by, as it is for every other kind — the sliver's angle
            # against the one it was folded into, not the angle itself.
            absorbed.append(Absorbed("vertical", drop[2], drop[3], drop[0], drop[1] - host[1], ABSORBED_SHORT_SEGMENT))
        rise = math.tan(math.radians(host[1])) * host[5] + math.tan(math.radians(drop[1])) * drop[5]
        host[5] += drop[5]
        host[1] = math.degrees(math.atan2(rise, host[5]))
        host[0], host[4] = vocabulary.vertical_bin(host[1])
        host[2] = min(host[2], drop[2])
        host[3] = max(host[3], drop[3])
        pieces.pop(short)

    # Then two segments that read as the same word are one instruction. This is the fit's own
    # bookkeeping, not a manoeuvre the rule refused: the segment count is a CEILING, so a straight
    # 3° descent is routinely cut into pieces that all read as 3° (measured: a perfectly straight
    # descent produced two "absorbed manoeuvres"). Recording it as absorbed swamped the diagnostic
    # and greyed out most of the hand check's vertical panel, so it is COUNTED instead.
    out: list[Instruction] = []
    spans: list[float] = []
    merged = 0
    for word, angle, issued, settled, outside, span in pieces:
        if out and out[-1].word == word:
            # the merged instruction's target is the angle over the COMBINED span — keeping the
            # first piece's left it claiming an angle up to 0.9° from what was flown over the rest
            # of its own span, and `target_to_bin_centre` reads that field
            rise = math.tan(math.radians(out[-1].target)) * spans[-1] + math.tan(math.radians(angle)) * span
            spans[-1] += span
            out[-1] = replace(out[-1], target=math.degrees(math.atan2(rise, spans[-1])),
                              settled_s=settled, clamped=out[-1].clamped or outside)
            merged += 1
            continue
        out.append(Instruction("vertical", word, angle, issued, settled, outside))
        spans.append(span)

    if out:
        out[0] = replace(out[0], issued_s=float(times[0]))       # the first segment is in force from the start
    # What the segment CEILING costs, stated rather than implied (the repo's rule: a bounded
    # coverage is never silent). The residual is the height error of the words' own profile —
    # each instruction's angle held over its span, chained from the first height — against the
    # resampled track. A profile with more phases than the ceiling allows shows up here.
    return out, absorbed, _fit_residual_m(x, y, out, spans), merged


def _fit_residual_m(x: np.ndarray, y: np.ndarray, out: list[Instruction], spans: list[float]) -> float:
    """RMS height error of the instructions' own profile against the resampled one."""
    if not out:
        return 0.0
    edges = np.concatenate(([0.0], np.cumsum(spans)))
    predicted = np.empty_like(y)
    height = float(y[0])
    for instruction, a, b in zip(out, edges[:-1], edges[1:]):
        inside = (x >= a) & (x <= b)
        predicted[inside] = height - math.tan(math.radians(instruction.target)) * (x[inside] - a)
        height -= math.tan(math.radians(instruction.target)) * (b - a)
    return float(np.sqrt(np.mean((predicted - y) ** 2)))


def _split_long_turns(headings: list[Instruction], times: np.ndarray, course: np.ndarray,
                      vocabulary: Vocabulary) -> list[Instruction]:
    """Turns wider than `turn_split_deg` become two or more, each said when the aircraft reaches
    the one before it.

    A heading word names a DIRECTION, and a direction cannot say which way round to get there.
    Anything that flies the words has to choose, and the only rule available is "the short way" —
    which at a half circle is a coin toss and past it is simply wrong. The reading does know: it
    read the course UNWRAPPED, so the plateau-to-plateau difference carries the real direction
    (+178° and −182° are different numbers there and the same word here). This spends that
    knowledge the only way an absolute-target vocabulary can — on intermediate targets the
    aircraft actually passed through — instead of adding a left/right column to every word.

    The intermediate word is issued when the TURN STARTS (the aircraft must be sent somewhere it
    can reach the short way) and the final one when the aircraft reaches the intermediate heading,
    which is the row the track itself gives."""
    if len(headings) < 2:
        return headings
    out = [headings[0]]
    for previous, instruction in zip(headings, headings[1:]):
        began = previous.settled_s if previous.settled_s is not None else previous.issued_s
        settled = instruction.settled_s if instruction.settled_s is not None else instruction.issued_s
        start_row = int(np.searchsorted(times, began))
        end_row = min(int(np.searchsorted(times, settled)), len(course) - 1)
        turn = float(course[end_row] - course[start_row])       # unwrapped: the way it really went
        parts = int(math.ceil(abs(turn) / vocabulary.turn_split_deg)) if abs(turn) > 0 else 1
        if parts < 2 or end_row <= start_row:
            out.append(instruction)
            continue
        from_deg = float(course[start_row])
        issued = instruction.issued_s
        for part in range(1, parts):
            via = from_deg + turn * part / parts
            word = vocabulary.heading_bin(via)
            if word in (out[-1].word, instruction.word):
                continue                                        # the leg is already this word: no new event
            # the row where the aircraft passes that heading — searchsorted needs the signal to
            # increase, so a left turn is searched on its negation
            span = course[start_row : end_row + 1]
            offset = int(np.searchsorted(span, via) if turn > 0 else np.searchsorted(-span, -via))
            reached = float(times[min(start_row + offset, end_row)])
            if reached <= issued:
                continue
            out.append(Instruction("heading", word, wrap_deg(via), issued, reached, False))
            issued = reached
        out.append(replace(instruction, issued_s=issued) if issued != instruction.issued_s else instruction)
    return out


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
        "speed": plateaus(speed, vocabulary.speed_tolerance_mps, rows),
    }

    def heading_word(value: float) -> tuple[int, float, bool]:
        return vocabulary.heading_bin(value), wrap_deg(value), False

    def speed_word(value: float) -> tuple[int, float, bool]:
        word, clamped = vocabulary.speed_bin(value)
        return word, value, clamped

    instructions: list[Instruction] = []
    absorbed: list[Absorbed] = []
    # The heading and the speed are ABSOLUTE targets, so a plateau — a stretch the aircraft holds
    # — is what an instruction looks like. The vertical word is a RATE and is read differently
    # (`_vertical_instructions`): its segments have to tile the track.
    tolerances = {"heading": vocabulary.course_tolerance_deg, "speed": vocabulary.speed_tolerance_mps}
    min_changes = {kind: (lambda in_force, kind=kind: vocabulary.min_change(kind, in_force))
                   for kind in ("heading", "speed")}
    for kind, signal, to_word in (("heading", course, heading_word), ("speed", speed, speed_word)):
        words, dropped = _manoeuvre_words(kind, times, signal, flats[kind], rows, tolerances[kind], min_changes[kind],
                                          vocabulary.hold_min_s, to_word)
        if kind == "heading":
            words = _split_long_turns(words, times, course, vocabulary)
        instructions.extend(words)
        absorbed.extend(dropped)
    vertical, vertical_absorbed, fit_rms_m, merged = _vertical_instructions(times, height, frame["ground_speed_mps"], vocabulary)
    instructions.extend(vertical)
    absorbed.extend(vertical_absorbed)
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
    events_s, words, duration_clamped = sentence(instructions, vocabulary, ends_s=float(times[-1]))
    return Reading(
        dataset_id=series.dataset_id, flight_id=series.flight_id, instructions=tuple(instructions),
        event_times_s=events_s, words=words, runway=runway_ident, duration_clamped=duration_clamped,
        established_from_start=bool(established[0]), duration_s=float(times[-1] - times[0]),
        absorbed=tuple(sorted(absorbed, key=lambda item: item.start_s)),
        vertical_fit_rms_m=fit_rms_m, vertical_pieces_merged=merged,
    )


def sentence(instructions: Sequence[Instruction], vocabulary: Vocabulary, *, ends_s: float) -> tuple[np.ndarray, np.ndarray, int]:
    """The EVENT SEQUENCE: ``(times, words, clamped)`` (D52, 2026-09-20).

    One row per moment at which something changed, plus a final row for the LANDING at ``ends_s``
    (the record's last time) — never a row that repeats the one before, and no "hold" word.
    ``words`` is ``[E, 6]`` in `INSTRUCTION_KINDS` order — heading, vertical, speed, runway,
    duration, terminal — the duration being the gap to the PREVIOUS event as a word (the first
    event's is 0) and the terminal being `TERMINAL_LANDED` on that last row alone. Because the
    landing is a row, the duration words sum to the flight's own span, which is what lets a
    generated sentence say WHEN it lands.

    This replaced an even grid of one position every τ. The grid did not lose an instruction — two
    of a kind are never closer than the 20 s a level must be held — but it snapped every issue time
    FORWARD by 0–8 s, mean 4 s, and only ever late: 557 m of extra track at 270 kt on average, and
    a bias that accumulates round by round in a closed loop and reads as model error. The event
    time here is the row the labeller read, at the ADS-B grid's own 2 s.
    """
    changes = sorted({item.issued_s for item in instructions})
    if not changes:
        raise ValueError("a sentence needs at least one instruction")
    # THE LANDING IS AN EVENT (2026-09-21). Without it the sentence's last row is the last CHANGE,
    # which on the five-airport cohort sits a median 136 s — about 11 km — before the record ends,
    # and the sentence says "landed" there while carrying nothing at all about the final leg: the
    # duration word is the gap since the PREVIOUS event, so the time from the last change to the
    # landing is in no word. (The altitude word used to leave an event at the threshold crossing —
    # 56 % of its "instructions" were exactly that — so the hole opened when the vertical word
    # replaced it.) The landing row repeats the words in force and carries the terminal word plus
    # the gap since the last change, which is what makes the duration words sum to the flight.
    landed = float(ends_s) > changes[-1] + 1e-9
    moments = [*changes, float(ends_s)] if landed else changes
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
    # The guard is over the CHANGE rows. The landing row is there because the flight ended, not
    # because a word moved, so it repeats the words in force by construction — exempting it is the
    # point, not a loophole; its terminal word is what differs.
    free = [c for c in range(columns) if c not in (duration_column, terminal_column)]
    change_rows = words[: len(changes)]
    if len(changes) > 1 and (np.diff(change_rows, axis=0)[:, free] == 0).all(axis=1).any():
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
    "COURSE_SMOOTHING_S", "COURSE_TOLERANCE_DEG", "HEADING_BIN_DEG",
    "DURATION_BIN_S", "DURATION_MAX_S", "HOLD_MIN_S", "INSTRUCTION_KINDS", "MANDATORY_KINDS", "NO_INTERCEPT", "PLATEAU_MIN_S",
    "READING_RULE", "SMOOTHING_S", "TERMINAL_CONTINUE", "TERMINAL_GO_AROUND", "TERMINAL_LANDED",
    "TERMINAL_WORDS", "NO_TARGET",
    "SPEED_CENTRES_MPS", "SPEED_MIN_CHANGE_FRACTION", "SPEED_TOLERANCE_FRACTION", "SPEED_TOLERANCE_MPS",
    "VERTICAL_FIT_POINTS", "VERTICAL_LEVEL_TOLERANCE_DEG", "VERTICAL_MODES_DEG", "VERTICAL_SEGMENTS",
    "VERTICAL_TOLERANCE_FRACTION", "LEVEL_MODE_DEG",
    "TOKEN_STEP_S", "VOCABULARY_FILE",
    "VOCABULARY_SCHEMA", "ABSORBED_SAME_WORD", "ABSORBED_SHORT_SEGMENT", "ABSORBED_SHORT_TAIL",
    "ABSORBED_SMALL_CHANGE", "Absorbed", "Instruction",
    "Reading", "RunwayVocabulary", "Vocabulary", "course_frame", "departure_row", "flight_runway", "load_vocabulary", "min_rows",
    "plateaus", "read_instructions", "segment_positions_s", "sentence", "smooth", "word_counts", "write_vocabulary",
]
