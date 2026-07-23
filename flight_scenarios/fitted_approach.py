"""A fitted ADS-B threshold crossing shared by optimization and TS supervision.

The arrival manifest deliberately stops at the last measured sample.  For most harvested
arrivals that sample is still short of the runway, so it is not a physical arrival target.
This module applies :mod:`final_approach` after the modeling seam has converted the flight
to MSL, and exposes two views of the same fit:

* one fitted crossing position for an optimizer target; and
* uniformly timed fitted positions after the last observation for TS supervision.

Only position is inferred.  Callers keep measured terminal kinematics or mask velocity
channels; this module never presents extrapolated velocity as an ADS-B observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from aerodynamic_model.common import GeodeticState
from final_approach import RunwayFrame, SegmentFit, TrackPoint, fit_final_segment
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from .datum import HAE_ALTITUDE_SOURCE, MSL_ALTITUDE_SOURCES
from .start_state import DEFAULT_WINDOW_S


@dataclass(frozen=True)
class TimedFittedPoint:
    """One inferred MSL position on the uniform modeling time grid."""

    time_s: float
    point: TrackPoint
    terminal: bool


@dataclass(frozen=True)
class FittedApproach:
    """The fitted threshold crossing and the timing needed to sample its inferred tail."""

    fit: SegmentFit
    frame: RunwayFrame
    crossing: TrackPoint
    last_observed_along_m: float
    last_observed_time_s: float
    along_rate_mps: float | None
    crossing_time_s: float | None

    def target_state(self, measured_terminal: GeodeticState) -> GeodeticState:
        """Fitted position plus measured terminal ``V/psi/gamma/m``.

        The fit estimates only geometry.  Carrying the last measured LSQ kinematics avoids
        silently treating an extrapolated speed as ground truth or substituting chart Vref,
        which would turn this into the published-runway target mode.
        """
        return replace(
            measured_terminal,
            latitude=self.crossing.lat,
            longitude=self.crossing.lon,
            altitude=self.crossing.alt_m,
        )

    def uniform_tail(self, *, after_time_s: float, dt_s: float) -> list[TimedFittedPoint]:
        """Fitted positions on the regular grid after ``after_time_s``, through crossing.

        The final grid row is clamped to the exact fitted crossing.  Its timestamp is the
        first grid instant at or after the continuous crossing time, so the sequence remains
        uniformly spaced while terminal position error is at most one ``dt_s`` late.
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        if (
            self.crossing_time_s is None
            or self.along_rate_mps is None
            or self.last_observed_along_m >= 0.0
            or self.crossing_time_s <= after_time_s
        ):
            return []

        count = int(math.ceil((self.crossing_time_s - after_time_s) / dt_s))
        rows: list[TimedFittedPoint] = []
        for step in range(1, count + 1):
            time_s = after_time_s + step * dt_s
            terminal = step == count
            along_m = 0.0 if terminal else min(
                0.0,
                self.last_observed_along_m
                + self.along_rate_mps * (time_s - self.last_observed_time_s),
            )
            rows.append(TimedFittedPoint(
                time_s=time_s,
                point=_point_on_fit(self.frame, self.fit, along_m),
                terminal=terminal,
            ))
        return rows


def fit_flight_final_approach(
    flight: dict[str, Any],
    *,
    velocity_window_s: float = DEFAULT_WINDOW_S,
) -> FittedApproach | None:
    """Fit one already-MSL flight to its manifest-published runway, or return ``None``.

    ``runway_target`` supplies the landing-threshold frame.  A missing target or unusable
    final segment is a normal ``None`` for generic/synthetic inputs; an explicit HAE input
    raises because fitting it against the MSL runway would reproduce the historical geoid
    datum bug.
    """
    altitude_source = flight.get("altitude_source")
    if altitude_source == HAE_ALTITUDE_SOURCE:
        raise ValueError(
            "fit_flight_final_approach requires MSL waypoints; call flight_to_msl first"
        )
    if altitude_source not in MSL_ALTITUDE_SOURCES:
        raise ValueError(
            f"fit_flight_final_approach does not recognize altitude_source {altitude_source!r}"
        )
    waypoints = flight.get("waypoints") or []
    target = flight.get("runway_target") or {}
    required = ("lat", "lon", "elevation_msl_m", "course_deg")
    if len(waypoints) < 2 or any(target.get(key) is None for key in required):
        return None

    frame = RunwayFrame(
        ident=str(flight.get("runway") or "?"),
        lat=float(target["lat"]),
        lon=float(target["lon"]),
        elevation_m=float(target["elevation_msl_m"]),
        course_deg=float(target["course_deg"]),
    )
    points = [
        TrackPoint(lat=float(row[2]), lon=float(row[1]), alt_m=float(row[3]))
        for row in waypoints
    ]
    fit = fit_final_segment(points, frame)
    if fit is None or not fit.approaching:
        return None

    projected = frame.project_all(points)
    last_along = float(projected[-1].along_m)
    last_time = float(waypoints[-1][0]) - float(waypoints[0][0])
    along_rate = _terminal_along_rate(waypoints, projected, velocity_window_s)
    crossing_time = (
        last_time - last_along / along_rate
        if last_along < 0.0 and along_rate is not None
        else None
    )
    return FittedApproach(
        fit=fit,
        frame=frame,
        crossing=_point_on_fit(frame, fit, 0.0),
        last_observed_along_m=last_along,
        last_observed_time_s=last_time,
        along_rate_mps=along_rate,
        crossing_time_s=crossing_time,
    )


def _terminal_along_rate(waypoints, projected, window_s: float) -> float | None:
    """Last-window LSQ runway-direction speed in the same chart as the fitted line."""
    if window_s <= 0.0:
        raise ValueError(f"velocity_window_s must be positive, got {window_s}")
    last_t = float(waypoints[-1][0])
    selected = [
        (float(row[0]), float(point.along_m))
        for row, point in zip(waypoints, projected)
        if last_t - float(row[0]) <= window_s
    ]
    if len(selected) < 2:
        selected = [
            (float(row[0]), float(point.along_m))
            for row, point in zip(waypoints[-2:], projected[-2:])
        ]
    mean_t = sum(t for t, _ in selected) / len(selected)
    denominator = sum((t - mean_t) ** 2 for t, _ in selected)
    if denominator <= 0.0:
        return None
    mean_along = sum(x for _, x in selected) / len(selected)
    rate = sum((t - mean_t) * (x - mean_along) for t, x in selected) / denominator
    return rate if math.isfinite(rate) and rate > 1.0 else None


def _point_on_fit(frame: RunwayFrame, fit: SegmentFit, along_m: float) -> TrackPoint:
    """Inverse runway-frame projection for one point on both fitted OLS lines."""
    cross_m = fit.cross.intercept + fit.cross.slope * along_m
    height_m = fit.height.intercept + fit.height.slope * along_m
    course = math.radians(frame.course_deg)
    east_hat, north_hat = math.sin(course), math.cos(course)
    east_m = along_m * east_hat + cross_m * north_hat
    north_m = along_m * north_hat - cross_m * east_hat
    return TrackPoint(
        lat=frame.lat + north_m / METRES_PER_DEG_LAT,
        lon=frame.lon + east_m / metres_per_deg_lon(frame.lat),
        alt_m=frame.elevation_m + height_m,
    )
