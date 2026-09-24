"""The vocabulary specification: every grid, class, tolerance and reading parameter, and its sha.

One frozen value. A sentence artefact carries the sha of the spec it was read with, and every
reader refuses an artefact whose sha differs from its own spec (design §3 of the framework
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

READING_RULE = "instruction-v3"
SPEC_SCHEMA = "ts-instruction-spec-v4"

#: How heading words are read (vocabulary design §10.1, under comparison): ``holds`` — one word per straight
#: hold, larger turns split, an intercept inserted where the heading in force does not reach the final (§3.2);
#: ``per-step`` — every row labelled with the grid heading nearest the track `heading_lead_s` later, a new word
#: whenever that leaves the word in force by more than `heading_band_deg` (half a step: every change of grid cell).
HEADING_READINGS = ("holds", "per-step")

#: FAA JO 7110.65BB 5-9-2 TBL 5-9-1: the largest final-approach interception angle 2 NM or
#: more outside the approach gate.
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
    # --- heading: absolute ground-track targets, flown the shorter way
    #: `HEADING_READINGS`. The hold / split / intercept fields below are read by ``holds`` only, the lead and
    #: the band by ``per-step`` only: §10.1 compares the two, and the reading it rejects goes with its fields.
    heading_reading: str
    heading_lead_s: float
    heading_band_deg: float
    heading_step_deg: float
    #: The hold band about a heading target (covers the grid's half step plus the track's wander).
    heading_tolerance_deg: float
    heading_min_hold_s: float
    #: A heading word's turn from the target in force never exceeds this (from the track at
    #: issue, which may sit anywhere in that target's band: this plus the heading tolerance).
    heading_max_turn_deg: float
    #: Larger turns are cut into equal parts no longer than this ...
    heading_split_part_deg: float
    #: ... and the next part is issued this far before the previous part's target is reached.
    heading_continue_lead_deg: float
    #: A turn begins where the track starts moving toward the next target faster than this.
    turn_onset_rate_deg_s: float
    #: A hold is flown straight: its track's fitted rate stays at or below this.
    heading_hold_max_rate_deg_s: float
    #: A turn is flown at a turn RATE between these (turns are flown at a near-constant rate
    #: whatever the speed; the bank grows with speed), and never beyond `turn_bank_max_deg`. The
    #: lowest rate holds only for a turn of at least `turn_rate_min_from_deg`: a smaller change of
    #: the ground track is mostly the wind drift of a hold, not a banked turn.
    turn_rate_min_deg_s: float
    turn_rate_max_deg_s: float
    turn_rate_min_from_deg: float
    turn_bank_max_deg: float
    #: A turn may start up to this long after its word's row. The row is read off a track smoothed
    #: by two centred windows — the data plane's velocity fit (`flight_scenarios.start_state.
    #: DEFAULT_WINDOW_S`) and `track_smoothing_s` — which see a turn coming half a window early,
    #: so the word's row can lead the flown turn by half their sum. Derived, not fitted.
    turn_start_delay_max_s: float
    # --- approach
    #: The intercept heading the labeller inserts when the track reaches the final from a
    #: heading that does not converge on it (ATC_MAX_INTERCEPT_DEG).
    intercept_angle_deg: float
    #: After capture the corridor's half width is `corridor_half_width_m` at the threshold and
    #: widens by tan(`corridor_widening_deg`) per metre before it (an angular corridor, as LOC /
    #: LPV guidance is); the track stays within `corridor_course_tolerance_deg` of the course.
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
    # --- altitude: geometric MSL targets
    altitude_step_m: float
    altitude_max_m: float
    #: The band about an altitude target, and the tube's margin.
    altitude_tolerance_m: float
    #: Largest residual of one straight piece of the altitude-vs-distance fit.
    altitude_fit_tolerance_m: float
    level_min_s: float
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
    #: After the approach clearance the speed is "unspecified" unless a hold of at least this
    #: long ends at least `unspecified_distance_m` before the threshold (5-7-1 b.4 / d).
    unspecified_plateau_s: float
    unspecified_distance_m: float
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
            "heading_tolerance_deg", "heading_min_hold_s", "heading_max_turn_deg", "heading_split_part_deg",
            "heading_continue_lead_deg", "turn_onset_rate_deg_s", "heading_hold_max_rate_deg_s", "turn_rate_min_deg_s",
            "turn_rate_max_deg_s", "turn_rate_min_from_deg", "turn_bank_max_deg", "turn_start_delay_max_s",
            "intercept_angle_deg",
            "corridor_half_width_m", "corridor_course_tolerance_deg", "landing_cross_limit_m", "landing_max_height_m",
            "parallel_course_delta_deg", "altitude_step_m", "altitude_max_m",
            "altitude_tolerance_m", "altitude_fit_tolerance_m", "level_min_s", "climb_angle_max_deg",
            "climb_angle_centre_deg", "ground_speed_floor_mps", "ground_speed_ceiling_mps", "speed_step_mps", "speed_min_mps", "speed_max_mps", "speed_tolerance_mps",
            "speed_fit_tolerance_mps", "speed_flat_accel_mps2", "speed_min_hold_s", "speed_accel_max_mps2",
            "unspecified_plateau_s", "unspecified_distance_m",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.corridor_widening_deg < 45.0:
            raise ValueError("corridor_widening_deg must lie in [0, 45)")
        if not _divides(360.0, self.heading_step_deg):
            raise ValueError(f"heading_step_deg {self.heading_step_deg} does not divide 360")
        if self.heading_tolerance_deg < self.heading_step_deg / 2:
            raise ValueError("heading_tolerance_deg below half a heading step: a steady track between two "
                             "targets could never be held")
        if self.heading_reading not in HEADING_READINGS:
            raise ValueError(f"heading_reading {self.heading_reading!r} is not one of {HEADING_READINGS}")
        if self.heading_lead_s < 0.0 or not _divides(self.heading_lead_s, self.step_s):
            raise ValueError(f"heading_lead_s {self.heading_lead_s} is not a whole number of steps")
        if self.heading_band_deg < self.heading_step_deg / 2:
            raise ValueError("heading_band_deg below half a heading step: a track between two targets would switch "
                             "words at every row")
        if self.heading_split_part_deg + self.heading_continue_lead_deg > self.heading_max_turn_deg:
            raise ValueError("a split part plus its lead would exceed heading_max_turn_deg")
        if not self.heading_max_turn_deg + self.heading_tolerance_deg < 180.0:
            raise ValueError("heading_max_turn_deg plus the heading tolerance must stay below a half circle")
        if not self.turn_rate_min_deg_s < self.turn_rate_max_deg_s:
            raise ValueError("turn rate range must be increasing")
        if not self.turn_bank_max_deg < 90.0:
            raise ValueError("turn_bank_max_deg must be below 90°")
        if not _divides(self.altitude_max_m, self.altitude_step_m):
            raise ValueError("altitude_max_m must be a whole number of altitude steps")
        if self.altitude_tolerance_m < self.altitude_step_m / 2:
            raise ValueError("altitude_tolerance_m below half an altitude step")
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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["descent_angle_edges_deg"] = list(self.descent_angle_edges_deg)
        data["descent_angle_centres_deg"] = list(self.descent_angle_centres_deg)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VocabularySpec:
        expected = {item.name for item in fields(cls)}
        missing, extra = expected - set(data), set(data) - expected
        if missing or extra:
            raise ValueError(f"not a {READING_RULE} spec: missing {sorted(missing)}, unexpected {sorted(extra)}")
        values = dict(data)
        values["descent_angle_edges_deg"] = tuple(float(v) for v in data["descent_angle_edges_deg"])
        values["descent_angle_centres_deg"] = tuple(float(v) for v in data["descent_angle_centres_deg"])
        return cls(**values)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def rows(self, seconds: float) -> int:
        """Seconds on the step grid, as a whole number of rows (at least one)."""
        return max(1, int(round(seconds / self.step_s)))


def _divides(total: float, step: float) -> bool:
    count = total / step
    return abs(count - round(count)) < 1e-9
