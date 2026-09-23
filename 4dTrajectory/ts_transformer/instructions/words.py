"""The six columns of a sentence step, and each column's class index ↔ physical target.

Every column has a value "unchanged" (`UNCHANGED`, stored as -1): no new instruction of that
kind at this step. The runway column is a POINTER into the airport's candidate list
(`instructions.airport`), so its classes are per airport and it has no decoder here.
"""

from __future__ import annotations

import math

import numpy as np

from ts_transformer.instructions.spec import VocabularySpec

COLUMNS = ("runway", "approach", "heading", "altitude", "angle", "speed")
RUNWAY, APPROACH, HEADING, ALTITUDE, ANGLE, SPEED = range(len(COLUMNS))
UNCHANGED = -1

APPROACH_NOT_CLEARED = 0
APPROACH_CLEARED = 1
APPROACH_GO_AROUND = 2
APPROACH_CLASSES = 3

ANGLE_LEVEL = 0


class Words:
    """A spec's encoders and decoders. Encoding a value outside a column's range RAISES: the
    labeller turns that into a rejection with its reason, never a clamp."""

    def __init__(self, spec: VocabularySpec) -> None:
        self.spec = spec
        self.n_heading = int(round(360.0 / spec.heading_step_deg))
        self.n_altitude_levels = int(round(spec.altitude_max_m / spec.altitude_step_m)) + 1
        #: The altitude value "descend to land": no target plane.
        self.altitude_land = self.n_altitude_levels
        self.n_descent = len(spec.descent_angle_centres_deg)
        #: Angle classes: level (0), descent 1..K, climb (K + 1).
        self.angle_climb = self.n_descent + 1
        self.n_speed_levels = int(round((spec.speed_max_mps - spec.speed_min_mps) / spec.speed_step_mps)) + 1
        #: The speed value "unspecified": the pilot's own speed.
        self.speed_unspecified = self.n_speed_levels

    def class_counts(self) -> dict[str, int]:
        """Classes per column, "unchanged" excluded; the runway column is per airport."""
        return {
            "approach": APPROACH_CLASSES,
            "heading": self.n_heading,
            "altitude": self.n_altitude_levels + 1,
            "angle": self.n_descent + 2,
            "speed": self.n_speed_levels + 1,
        }

    # ---- heading: compass degrees, true north, clockwise
    def heading_index(self, track_deg: float) -> int:
        return int(round((track_deg % 360.0) / self.spec.heading_step_deg)) % self.n_heading

    def heading_deg(self, index: int) -> float:
        self._require(index, self.n_heading, "heading")
        return index * self.spec.heading_step_deg

    # ---- altitude: geometric MSL metres
    def altitude_index(self, altitude_m: float) -> int:
        index = int(round(altitude_m / self.spec.altitude_step_m))
        if not 0 <= index < self.n_altitude_levels:
            raise ValueError(f"altitude target {altitude_m:.0f} m outside 0–{self.spec.altitude_max_m:.0f} m")
        return index

    def altitude_m(self, index: int) -> float | None:
        """The target plane, or ``None`` for "descend to land"."""
        self._require(index, self.n_altitude_levels + 1, "altitude")
        return None if index == self.altitude_land else index * self.spec.altitude_step_m

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


def compass_from_math_rad(psi_rad):
    """Math-ENU heading (0 = East, counter-clockwise) → compass degrees (0 = North, clockwise)."""
    return (90.0 - np.degrees(psi_rad)) % 360.0


def math_rad_from_compass(track_deg):
    """Compass degrees → math-ENU radians (the executor's `psi`): psi = 90° − θ."""
    return np.radians(90.0 - np.asarray(track_deg))


def descent_angle_deg(delta_height_m: float, delta_distance_m: float) -> float:
    """Path angle of a straight piece, descending positive."""
    return math.degrees(math.atan2(-delta_height_m, delta_distance_m))
