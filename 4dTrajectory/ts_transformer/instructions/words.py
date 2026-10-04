"""The five columns of a sentence step, and each column's class index ↔ physical target (design §3).

Every column has a value "unchanged" (`UNCHANGED`, stored as -1): no new instruction of that
kind at this step. The runway column is a POINTER into the airport's candidate list
(`instructions.airport`), so its classes are per airport and it has no decoder here; its one
other value is "go-around" (`RUNWAY_GO_AROUND`, stored as -2: "abandon this approach", D10), which
does not change the runway in force. The heading column's classes are RELATIVE to the course of the
runway in force (D8): class k is the track ``course + k · step``, so a heading word is turned into
a track only with a course (`Words.heading_track_deg`).
"""

from __future__ import annotations

import math

import numpy as np

from ts_transformer.instructions.spec import VocabularySpec

COLUMNS = ("runway", "heading", "altitude", "angle", "speed")
RUNWAY, HEADING, ALTITUDE, ANGLE, SPEED = range(len(COLUMNS))
UNCHANGED = -1
#: The runway column's "go-around" (design §3.2): every other value of the column is a candidate's index.
RUNWAY_GO_AROUND = -2

ANGLE_LEVEL = 0


class Words:
    """A spec's encoders and decoders. Encoding a value outside a column's range RAISES: the
    labeller turns that into a rejection with its reason, never a clamp."""

    def __init__(self, spec: VocabularySpec) -> None:
        self.spec = spec
        self.n_heading = int(round(360.0 / spec.heading_step_deg))
        levels = [0.0]
        for step, bottom, top in zip(spec.altitude_segment_steps_m, (0.0, *spec.altitude_segment_tops_m[:-1]),
                                     spec.altitude_segment_tops_m):
            levels += [bottom + step * k for k in range(1, int(round((top - bottom) / step)) + 1)]
        gaps = np.diff(levels)
        #: Every level's height (MSL m) and its band's half width: half the larger gap to its neighbours — the largest
        #: rounding error a height said as it can carry — plus the fit residual. Inside a segment that is half the
        #: segment's step + 10 m (§3.4: 40 / 70 / 235 m); a segment's TOP level takes the heights up to half the next
        #: segment's step above it, so its band is that one's (1,260 m: 70 m; 2,700 m: 235 m) — Claude's reading of
        #: §3.4, whose "half its segment's step" would refuse every level held at 1,301–1,320 m or 2,771–2,925 m.
        self.altitude_levels = np.array(levels)
        self.altitude_tolerances = np.maximum(np.concatenate((gaps[:1], gaps)), np.concatenate((gaps, gaps[-1:]))) / 2 \
            + spec.altitude_fit_tolerance_m
        self.n_altitude_levels = len(levels)
        #: The altitude value "no level-off": no target plane.
        self.altitude_no_level_off = self.n_altitude_levels
        self.n_descent = len(spec.descent_angle_centres_deg)
        #: Angle classes: level (0), descent 1..K, climb (K + 1).
        self.angle_climb = self.n_descent + 1
        self.n_speed_levels = int(round((spec.speed_max_mps - spec.speed_min_mps) / spec.speed_step_mps)) + 1
        #: The speed value "unspecified": the pilot's own speed.
        self.speed_unspecified = self.n_speed_levels

    def class_counts(self) -> dict[str, int]:
        """Classes per column, "unchanged" excluded; the runway column is per airport (its candidates and
        "go-around")."""
        return {
            "heading": self.n_heading,
            "altitude": self.n_altitude_levels + 1,
            "angle": self.n_descent + 2,
            "speed": self.n_speed_levels + 1,
        }

    # ---- heading: relative to the course of the runway in force, degrees, clockwise (design §3.3)
    def heading_index(self, relative_deg: float) -> int:
        """The class of a relative heading (track minus the course)."""
        return int(round((relative_deg % 360.0) / self.spec.heading_step_deg)) % self.n_heading

    def heading_relative_deg(self, index: int) -> float:
        """A class's relative heading in (−180°, +180°]: class 0 is the course, class n/2 the opposite direction."""
        self._require(index, self.n_heading, "heading")
        relative = index * self.spec.heading_step_deg
        return relative - 360.0 if relative > 180.0 else relative

    def heading_class(self, track_deg: float, course_deg: float) -> int:
        """The class of a compass track flown with a runway of course ``course_deg`` in force."""
        return self.heading_index(track_deg - course_deg)

    def heading_track_deg(self, index: int, course_deg: float) -> float:
        """The compass track in [0, 360) a class says with a runway of course ``course_deg`` in force."""
        return (course_deg + self.heading_relative_deg(index)) % 360.0

    # ---- altitude: geometric MSL metres, the levels of the spec's segments (design §3.4, D22)
    def altitude_index(self, altitude_m: float) -> int:
        """The level nearest ``altitude_m`` (the lower one at a tie); outside the grid by more than half its outer
        segment's step raises."""
        steps, top = self.spec.altitude_segment_steps_m, self.spec.altitude_segment_tops_m[-1]
        if not -steps[0] / 2 <= altitude_m < top + steps[-1] / 2:
            raise ValueError(f"altitude target {altitude_m:.0f} m outside 0–{top:.0f} m")
        return int(np.argmin(np.abs(self.altitude_levels - altitude_m)))

    def altitude_m(self, index: int) -> float | None:
        """The target plane, or ``None`` for "no level-off"."""
        self._require(index, self.n_altitude_levels + 1, "altitude")
        return None if index == self.altitude_no_level_off else float(self.altitude_levels[index])

    def altitude_tolerance_m(self, index: int) -> float:
        """The half width of an altitude word's band and tube margin (`altitude_tolerances`). "No level-off" has no
        level: it takes the lowest level's, where the final descent flies (Claude's choice, design §3.4 sets ε per level
        only)."""
        self._require(index, self.n_altitude_levels + 1, "altitude")
        return float(self.altitude_tolerances[0 if index == self.altitude_no_level_off else index])

    # ---- descent angle: degrees, descending positive
    def angle_index(self, angle_deg: float) -> int:
        """The class of a straight piece's path angle (not the level class: level is a
        target reached, decided by the altitude reading)."""
        edges = self.spec.descent_angle_edges_deg
        if edges[0] <= angle_deg <= edges[-1]:
            k = int(np.searchsorted(edges, angle_deg, side="right")) - 1
            return 1 + min(k, self.n_descent - 1)
        if -self.spec.climb_angle_max_deg <= angle_deg < edges[0]:
            return self.angle_climb
        raise ValueError(f"path angle {angle_deg:.2f}° outside the classes "
                         f"(climb to {self.spec.climb_angle_max_deg}°, descent to {edges[-1]}°)")

    def angle_bounds(self, index: int) -> tuple[float, float]:
        """``(lowest, steepest)`` descent angle of a class, degrees (a climb is negative)."""
        self._require(index, self.n_descent + 2, "angle")
        if index == ANGLE_LEVEL:
            return 0.0, 0.0
        if index == self.angle_climb:
            return -self.spec.climb_angle_max_deg, self.spec.descent_angle_edges_deg[0]
        edges = self.spec.descent_angle_edges_deg
        return edges[index - 1], edges[index]

    def angle_deg(self, index: int) -> float:
        """The nominal angle a class is flown at."""
        self._require(index, self.n_descent + 2, "angle")
        if index == ANGLE_LEVEL:
            return 0.0
        if index == self.angle_climb:
            return -self.spec.climb_angle_centre_deg
        return self.spec.descent_angle_centres_deg[index - 1]

    def is_descent(self, index: int) -> bool:
        return 1 <= index <= self.n_descent

    # ---- speed: ground speed m/s
    def speed_index(self, speed_mps: float) -> int:
        index = int(round((speed_mps - self.spec.speed_min_mps) / self.spec.speed_step_mps))
        if not 0 <= index < self.n_speed_levels:
            raise ValueError(f"speed target {speed_mps:.1f} m/s outside "
                             f"{self.spec.speed_min_mps:g}–{self.spec.speed_max_mps:g} m/s")
        return index

    def speed_mps(self, index: int) -> float | None:
        """The target, or ``None`` for "unspecified"."""
        self._require(index, self.n_speed_levels + 1, "speed")
        return None if index == self.speed_unspecified else self.spec.speed_min_mps + index * self.spec.speed_step_mps

    @staticmethod
    def _require(index: int, count: int, column: str) -> None:
        if not 0 <= index < count:
            raise ValueError(f"{column} class {index} outside 0..{count - 1}")


def wrap180(angle_deg):
    """Signed angle in [-180, 180); element-wise on arrays."""
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


#: Two tracks a heading word says are one track within this (degrees): the courses of two candidates and a class's
#: relative heading add up the same only to the last digits of a float.
SAME_TRACK_DEG = 1e-9


def same_track(a_deg: float, b_deg: float) -> bool:
    """Whether two compass tracks are one (`SAME_TRACK_DEG`): the one test of "a heading word says nothing new" (the
    2 s sentence, the Δ grid and the closed loop, D46, D48)."""
    return abs(float(wrap180(a_deg - b_deg))) <= SAME_TRACK_DEG


def compass_from_math_rad(psi_rad):
    """Math-ENU heading (0 = East, counter-clockwise) → compass degrees (0 = North, clockwise)."""
    return (90.0 - np.degrees(psi_rad)) % 360.0


def math_rad_from_compass(track_deg):
    """Compass degrees → math-ENU radians (the executor's `psi`): psi = 90° − θ."""
    return np.radians(90.0 - np.asarray(track_deg))


def descent_angle_deg(delta_height_m: float, delta_distance_m: float) -> float:
    """Path angle of a straight piece, descending positive."""
    return math.degrees(math.atan2(-delta_height_m, delta_distance_m))
