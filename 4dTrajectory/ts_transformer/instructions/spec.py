"""The vocabulary specification: every grid, class, tolerance and reading parameter, and its sha.

One frozen value. A sentence artefact carries the sha of the spec it was read with, and every
reader refuses an artefact whose sha differs from its own spec (vocabulary §3 of the framework
document) — there is no compatibility path. ``READING_RULE`` names the labelling algorithm:
changing what a field MEANS, or adding one, bumps it, so an old file is refused by name rather
than read with a guessed default.

All quantities are SI (metres, m/s, degrees, seconds). Values taken from ATC text are
converted once here, with the source in the comment.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from geokit import FT_M, KT_MS, NM_M

#: instruction-v5 (`docs/two_tier/design/vocabulary.md` §3, §4): five columns (the runway column
#: holds "go-around", D1, D10), heading words relative to the course of the runway in force (D8), altitude words on a
#: 40-level grid of at most three uniform segments (D22) with "no level-off", as heights above the airport elevation E
#: (D58; instruction-v4 read them as MSL), the go-around read from the track (D18, D19), "unspecified" from the capture
#: row (D4); since instruction-v6 the closed-loop reading's vertical tolerance in the final descent (D66, a field of its own).
READING_RULE = "instruction-v6"
SPEC_SCHEMA = "ts-instruction-spec-v7"

#: FAA JO 7110.65BB 5-9-2 TBL 5-9-1: the largest final-approach interception angle 2 NM or
#: more outside the approach gate — the judge's "lined up" (vocabulary §5.8).
ATC_MAX_INTERCEPT_DEG = 30.0
#: 7110.65BB 5-7-1 b.4: no speed adjustment inside the FAF or 5 NM from the runway.
ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M = 5.0 * NM_M          # 9,260 m
#: 7110.65BB 5-7-1 g NOTE 1: pilots hold an assigned speed within ±10 kt.
ATC_SPEED_COMPLIANCE_MPS = 10.0 * KT_MS                   # 5.144 m/s
#: AIM 4-4-10 d: below 500 ft/min a pilot must tell ATC (the slowest ordinary descent).
ATC_MIN_DESCENT_RATE_MPS = 500.0 * FT_M / 60.0            # 2.54 m/s


@dataclass(frozen=True)
class VocabularySpec:
    """Everything the words mean and how they are read off a track. Every field is required."""

    #: One step of the sentence (the data's own resampling interval).
    step_s: float
    # --- smoothing: centred moving averages over this many seconds, edge-padded
    track_smoothing_s: float
    altitude_smoothing_s: float
    speed_smoothing_s: float
    # --- heading: ground-track targets relative to the course of the runway in force (vocabulary §3.3, D8), read row by row
    #: Each row is labelled with the grid value nearest the track this long later, relative to the course, and the
    #: rows merged into one word while that stays the same grid cell: a word says where the track will be.
    heading_lead_s: float
    heading_step_deg: float
    #: A word's envelope: from each row, the track this far from the word in force `heading_lead_s` earlier, at most
    #: (the grid's half step plus the track's wander).
    heading_tolerance_deg: float
    #: A turn (for the readout's stratum and the turn measurements) is a run of rows turning one way faster than this.
    turn_onset_rate_deg_s: float
    #: The turn rate and the bank of the executor's turns never exceed these (measured on turns of at least
    #: `turn_rate_min_from_deg`: a smaller change of the ground track is mostly wind drift).
    turn_rate_max_deg_s: float
    turn_rate_min_from_deg: float
    turn_bank_max_deg: float
    # --- approach
    #: A threshold is crossed LINED UP when the track is within this of the runway's course (ATC_MAX_INTERCEPT_DEG):
    #: the judge's approach crossing, another runway's crossing and the decision altitude's point (vocabulary §5.8).
    lined_up_deg: float
    #: The capture corridor (only the labeller's capture row reads it, vocabulary §2): its half width is
    #: `corridor_half_width_m` at the threshold and widens by tan(`corridor_widening_deg`) per metre before it (an
    #: angular corridor, as LOC / LPV guidance is); the track stays within `corridor_course_tolerance_deg` of the course.
    corridor_half_width_m: float
    corridor_widening_deg: float
    corridor_course_tolerance_deg: float
    #: The landing (§2.2), the harvest's and the evaluator's condition
    #: (`final_approach.assign.LandingScreen`): the threshold plane crossed within
    #: `landing_cross_limit_m` of the centreline — and within half the spacing to any runway whose
    #: course is within `parallel_course_delta_deg` of this one's — and
    #: within `landing_max_height_m` of the threshold's height (above or below, as the harvest tests).
    landing_cross_limit_m: float
    landing_max_height_m: float
    parallel_course_delta_deg: float
    # --- altitude: heights above the airport elevation E on a grid of uniform segments (vocabulary §3.4, D22, D58)
    #: Segment i holds the levels from the previous segment's top (0 m for the first) plus its step, up to its top, in
    #: steps of ``altitude_segment_steps_m[i]``; the first segment also holds 0 m.
    altitude_segment_steps_m: tuple[float, ...]
    altitude_segment_tops_m: tuple[float, ...]
    #: Largest residual of one straight piece of the altitude-vs-distance fit; a level's band and tube margin is half
    #: its segment's step plus this (`Words.altitude_tolerance_m`).
    altitude_fit_tolerance_m: float
    #: A piece is level when it lasts at least `level_min_s` and every row lies within `level_band_m` of the piece's
    #: own median (a physical test, not the grid's, §4.4).
    level_min_s: float
    level_band_m: float
    # --- descent angle (positive = descending), classes by their edges
    #: K + 1 ascending edges of the K descent classes; the first edge is slightly negative so
    #: a nearly flat stretch inside a descent keeps a descent class.
    descent_angle_edges_deg: tuple[float, ...]
    descent_angle_centres_deg: tuple[float, ...]
    #: The climb class covers climbs up to this angle; its nominal angle is the centre.
    climb_angle_max_deg: float
    climb_angle_centre_deg: float
    # --- speed: ground-speed targets
    #: A raw ground-speed row outside these is bad data (no fixed-wing aircraft flies an
    #: approach slower, none reaches this in a terminal area): the flight is refused.
    ground_speed_floor_mps: float
    ground_speed_ceiling_mps: float
    speed_step_mps: float
    speed_min_mps: float
    speed_max_mps: float
    speed_tolerance_mps: float
    speed_fit_tolerance_mps: float
    #: A speed piece flatter than this, and at least `speed_min_hold_s` long, is a hold.
    speed_flat_accel_mps2: float
    speed_min_hold_s: float
    #: Transition envelope: the largest acceleration magnitude.
    speed_accel_max_mps2: float
    #: From the capture row on the speed is "unspecified" unless a hold of at least this long ends at or after the
    #: capture row and at least `unspecified_distance_m` before the threshold (5-7-1 b.4 / d; vocabulary §4.5, D4).
    unspecified_plateau_s: float
    unspecified_distance_m: float
    # --- go-around (vocabulary §4.6, D18; the rule of R40 `go_around_census`, on the sentence's rows)
    #: A LOW PASS: a run of rows on a candidate's final — at most `go_around_max_cross_m` off its centreline, between
    #: `go_around_along_m` (before the threshold negative, past it positive) and at most `go_around_max_height_m`
    #: above it — with gaps of at most `go_around_max_gap_rows` rows, moving at least `go_around_min_progress_m` along
    #: the landing direction. Its lowest row is the GO-AROUND POINT.
    go_around_max_cross_m: float
    go_around_along_m: tuple[float, float]
    go_around_max_height_m: float
    go_around_max_gap_rows: int
    go_around_min_progress_m: float
    #: A pass is a go-around when a level held `go_around_hold_s` at least `go_around_min_drop_m` above the point comes
    #: before it, and one held as long at least `go_around_min_climb_m` above it comes after it (before the next pass).
    go_around_hold_s: float
    go_around_min_drop_m: float
    go_around_min_climb_m: float
    #: A point past the threshold and at most this high is on the runway (a touch-and-go): not a go-around.
    go_around_on_runway_height_m: float
    # --- the closed-loop reading (vocabulary §4.9, D32; `autopilot.closed_loop`)
    #: A heading correction starts when the flown path is more than `closed_loop_lateral_m` off the observed one, an
    #: angle correction when it is more than the vertical tolerance in force above or below it; each ends under half of
    #: it. The vertical tolerance in force is `closed_loop_final_vertical_m` (H_final) while "no level-off" is in force
    #: (the final descent) and `closed_loop_vertical_m` (H) elsewhere (D66).
    closed_loop_lateral_m: float
    closed_loop_vertical_m: float
    closed_loop_final_vertical_m: float
    reading_rule: str = READING_RULE

    def __post_init__(self) -> None:
        if self.reading_rule != READING_RULE:
            raise ValueError(f"reading rule {self.reading_rule!r} is not this code's {READING_RULE!r}")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{item.name} = {value} is not finite")
        positive = (
            "step_s", "track_smoothing_s", "altitude_smoothing_s", "speed_smoothing_s", "heading_step_deg",
            "heading_tolerance_deg", "turn_onset_rate_deg_s", "turn_rate_max_deg_s",
            "turn_rate_min_from_deg", "turn_bank_max_deg", "lined_up_deg",
            "corridor_half_width_m", "corridor_course_tolerance_deg", "landing_cross_limit_m", "landing_max_height_m",
            "parallel_course_delta_deg", "altitude_fit_tolerance_m", "level_min_s", "level_band_m",
            "climb_angle_max_deg", "climb_angle_centre_deg", "ground_speed_floor_mps", "ground_speed_ceiling_mps",
            "speed_step_mps", "speed_min_mps", "speed_max_mps", "speed_tolerance_mps",
            "speed_fit_tolerance_mps", "speed_flat_accel_mps2", "speed_min_hold_s", "speed_accel_max_mps2",
            "unspecified_plateau_s", "unspecified_distance_m", "go_around_max_cross_m", "go_around_max_height_m",
            "go_around_min_progress_m", "go_around_hold_s", "go_around_min_drop_m", "go_around_min_climb_m",
            "go_around_on_runway_height_m", "closed_loop_lateral_m", "closed_loop_vertical_m",
            "closed_loop_final_vertical_m",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.corridor_widening_deg < 45.0:
            raise ValueError("corridor_widening_deg must lie in [0, 45)")
        if not _divides(360.0, self.heading_step_deg):
            raise ValueError(f"heading_step_deg {self.heading_step_deg} does not divide 360")
        if self.heading_tolerance_deg < self.heading_step_deg / 2:
            raise ValueError("heading_tolerance_deg below half a heading step: the observed track, within half a "
                             "step of its own word, would leave the word's envelope")
        if self.heading_lead_s < 0.0 or not _divides(self.heading_lead_s, self.step_s):
            raise ValueError(f"heading_lead_s {self.heading_lead_s} is not a whole number of steps")
        if not self.turn_bank_max_deg < 90.0:
            raise ValueError("turn_bank_max_deg must be below 90°")
        steps, tops = self.altitude_segment_steps_m, self.altitude_segment_tops_m
        if len(steps) != len(tops) or not steps:
            raise ValueError("the altitude grid needs one step per segment top")
        bottoms = (0.0, *tops[:-1])
        for step, bottom, top in zip(steps, bottoms, tops):
            if not (math.isfinite(step) and step > 0.0 and top > bottom and _divides(top - bottom, step)):
                raise ValueError(f"altitude segment {bottom:g}–{top:g} m is not a positive whole number of {step:g} m steps")
        if not self.lined_up_deg < 90.0:
            raise ValueError("lined_up_deg must be below 90°")
        low, high = self.go_around_along_m
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError("go_around_along_m must be an increasing pair")
        if self.go_around_max_gap_rows < 0:
            raise ValueError("go_around_max_gap_rows must not be negative")
        if not _divides(self.speed_max_mps - self.speed_min_mps, self.speed_step_mps):
            raise ValueError("the speed range must be a whole number of speed steps")
        edges, centres = self.descent_angle_edges_deg, self.descent_angle_centres_deg
        if len(edges) != len(centres) + 1 or len(centres) < 1:
            raise ValueError("descent angle classes need K centres and K + 1 edges")
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("descent angle edges must increase")
        if edges[0] > 0.0:
            raise ValueError("the first descent edge must be at or below zero")
        if any(not lo <= c <= hi for c, lo, hi in zip(centres, edges, edges[1:])):
            raise ValueError("each descent centre must lie inside its class")
        if self.climb_angle_centre_deg > self.climb_angle_max_deg:
            raise ValueError("the climb centre lies beyond the climb range")
        if self.closed_loop_final_vertical_m > self.closed_loop_vertical_m:
            raise ValueError("closed_loop_final_vertical_m above closed_loop_vertical_m: the final descent's tolerance "
                             "is the smaller one (D66)")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in TUPLE_FIELDS:
            data[name] = list(getattr(self, name))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VocabularySpec:
        expected = {item.name for item in fields(cls)}
        missing, extra = expected - set(data), set(data) - expected
        if missing or extra:
            raise ValueError(f"not a {READING_RULE} spec: missing {sorted(missing)}, unexpected {sorted(extra)}")
        values = dict(data)
        for name in TUPLE_FIELDS:
            values[name] = tuple(float(v) for v in data[name])
        return cls(**values)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def rows(self, seconds: float) -> int:
        """Seconds on the step grid, as a whole number of rows (at least one)."""
        return max(1, int(round(seconds / self.step_s)))

    def rows_exact(self, seconds: float) -> int:
        """Seconds that are a whole number of steps (a lead, which may be none), as rows."""
        if not _divides(seconds, self.step_s):
            raise ValueError(f"{seconds} s is not a whole number of {self.step_s} s steps")
        return int(round(seconds / self.step_s))


#: The spec's tuple fields (JSON lists on disk).
TUPLE_FIELDS = ("altitude_segment_steps_m", "altitude_segment_tops_m", "descent_angle_edges_deg",
                "descent_angle_centres_deg", "go_around_along_m")


def _divides(total: float, step: float) -> bool:
    count = total / step
    return abs(count - round(count)) < 1e-9
