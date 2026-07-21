"""The runway-aligned frame a final approach is measured in.

Origin = the LANDING threshold (displaced where the runway has one -- see
``trajectory_data_process/acquisition/runways.py``, which is what makes
``runway_thresholds.json`` hold landing thresholds rather than pavement ends).
Axes, right-handed about the landing direction::

    along  +  forward along the landing direction   (NEGATIVE before the threshold)
    cross  +  right of the landing direction
    height +  above threshold elevation

Two sign conventions meet here and are deliberately NOT unified:

  * ``course_deg`` on input is a COMPASS bearing (0 = North, clockwise) because that
    is what ``runway_thresholds.json`` publishes and what CIFP prints on a plate.
  * the project's dynamics model uses math-ENU (0 = East, counter-clockwise), which
    ``approach_constraints.geometry.course_bearing`` returns.

Mixing them reads an aligned aircraft as a 90 deg intercept. This module takes
compass in, converts once in ``RunwayFrame.__post_init__``, and never exposes the
raw angle again -- so a caller cannot pick the wrong one.

The projection is a local flat-earth chart: metres-per-degree from ``geokit``
(``METRES_PER_DEG_LAT`` and ``metres_per_deg_lon`` at the threshold latitude), which
is the same pair every other frame in this project derives from. A final approach
spans single-digit kilometres, so chart curvature is far below the ~1 m the fit
resolves; using the geokit constants (not a hand-rolled 111_320.0) keeps this frame
bit-identical to the optimizer's NE frame and the ts channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon


class TrackPoint(NamedTuple):
    """One observed position. ``alt_m`` shares its datum with the frame elevation."""

    lat: float
    lon: float
    alt_m: float


class Projected(NamedTuple):
    """A :class:`TrackPoint` in the runway frame, metres."""

    along_m: float
    cross_m: float
    height_m: float


@dataclass(frozen=True)
class RunwayFrame:
    """A landing threshold plus the local chart that measures approaches to it.

    ``course_deg`` is the runway's landing direction as a COMPASS bearing. The unit
    vectors and the longitude scale are derived once at construction, so projecting a
    whole track costs two multiplies per point.
    """

    ident: str
    lat: float
    lon: float
    elevation_m: float
    course_deg: float

    _east_hat: float = field(init=False, repr=False)
    _north_hat: float = field(init=False, repr=False)
    _m_per_deg_lon: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        course_rad = math.radians(self.course_deg)
        # Compass -> ENU unit vector along the landing direction.
        object.__setattr__(self, "_east_hat", math.sin(course_rad))
        object.__setattr__(self, "_north_hat", math.cos(course_rad))
        object.__setattr__(self, "_m_per_deg_lon", metres_per_deg_lon(self.lat))

    def project(self, point: TrackPoint) -> Projected:
        """Put one point in the frame."""
        north = (point.lat - self.lat) * METRES_PER_DEG_LAT
        east = (point.lon - self.lon) * self._m_per_deg_lon
        return Projected(
            along_m=east * self._east_hat + north * self._north_hat,
            cross_m=east * self._north_hat - north * self._east_hat,
            height_m=point.alt_m - self.elevation_m,
        )

    def project_all(self, points: Sequence[TrackPoint]) -> list[Projected]:
        """Put a whole track in the frame, in input order."""
        return [self.project(p) for p in points]

    def distance_m(self, point: TrackPoint) -> float:
        """Horizontal distance from the threshold, metres."""
        p = self.project(point)
        return math.hypot(p.along_m, p.cross_m)


def track_course_deg(
    points: Sequence[TrackPoint], anchor_index: int, *, lookback_m: float = 1500.0
) -> float | None:
    """Ground course (compass) over the last ``lookback_m`` into ``points[anchor_index]``.

    Derived from track geometry alone -- no runway, no ADS-B track field -- so it can
    pre-filter runway candidates without assuming the answer. Returns None when the
    track does not extend ``lookback_m`` back from the anchor, which is a normal
    outcome for a short track and is classified by the caller.

    The chart is the same flat-earth pair the frames use, evaluated at the anchor.
    """
    if anchor_index <= 0 or anchor_index >= len(points):
        return None
    anchor = points[anchor_index]
    m_per_deg_lon = metres_per_deg_lon(anchor.lat)
    for i in range(anchor_index - 1, -1, -1):
        north = (anchor.lat - points[i].lat) * METRES_PER_DEG_LAT
        east = (anchor.lon - points[i].lon) * m_per_deg_lon
        if math.hypot(north, east) >= lookback_m:
            return math.degrees(math.atan2(east, north)) % 360.0
    return None


def heading_difference_deg(a: float, b: float) -> float:
    """Smallest absolute angle between two compass bearings, in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)
