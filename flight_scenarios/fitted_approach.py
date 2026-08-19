"""A fitted ADS-B threshold crossing shared by optimization and TS supervision.

The arrival manifest deliberately stops at the last measured sample.  For most harvested
arrivals that sample is still short of the runway, so it is not a physical arrival target.
This module applies :mod:`final_approach` in the flight's declared vertical datum, and
exposes two views of the same fit:

* one fitted crossing state for an optimizer target; and
* uniformly timed fitted positions after the last observation for TS supervision.

The optimizer target uses the spatial fit's tangent plus a constant along-track rate
estimated across the same approach segment.  TS tail velocity channels remain
masked: an inferred crossing state is a modeling boundary, not an ADS-B observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from aerodynamic_model.common import GeodeticState
from final_approach import RunwayFrame, SegmentFit, TrackPoint, fit_final_segment
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from .datum import HAE_ALTITUDE_SOURCE, MSL_ALTITUDE_SOURCES


class UnusableFittedApproach(ValueError):
    """One flight cannot be given a fitted-ADS-B target.

    Its own type so a batch can drop exactly these flights and roster them (see
    ``flight_scenarios.dataset``) without a broad ``except ValueError`` swallowing a real
    contract violation next to it. Measured on the 42,725 rostered arrivals: 35 flights
    (0.08 %) — but before this existed those 35 aborted the whole dataset build for 4 of
    the 5 airports.
    """


@dataclass(frozen=True)
class TimedFittedPoint:
    """One inferred MSL position on the uniform modeling time grid."""

    time_s: float
    point: TrackPoint
    terminal: bool


@dataclass(frozen=True)
class FittedApproach:
    """The fitted threshold crossing and the timing needed to sample its inferred tail."""

    altitude_source: str
    fit: SegmentFit
    frame: RunwayFrame
    crossing: TrackPoint
    last_observed_along_m: float
    last_observed_time_s: float
    along_rate_mps: float | None
    crossing_time_s: float | None

    def target_state(
        self, *, mass_kg: float, hae_minus_msl_m: float = 0.0
    ) -> GeodeticState:
        """Return the fitted position and approach kinematics in the modeling MSL datum.

        The crossing retains the datum in which the fit was performed. Direct callers may
        fit a raw HAE flight, while the standard manifest CLI fits an already-converted MSL
        flight. Applying the runway offset based on that provenance makes both paths correct
        and prevents an idempotent input conversion from being followed by a second, hidden
        conversion here.
        """
        if self.along_rate_mps is None:
            raise UnusableFittedApproach(
                "fitted approach has no usable along-track velocity"
            )
        V, psi, gamma = _kinematics_on_fit(
            self.frame, self.fit, self.along_rate_mps
        )
        altitude_msl = self.crossing.alt_m
        if self.altitude_source == HAE_ALTITUDE_SOURCE:
            altitude_msl -= hae_minus_msl_m
        return GeodeticState(
            latitude=self.crossing.lat,
            longitude=self.crossing.lon,
            altitude=altitude_msl,
            V=V,
            psi=psi,
            gamma=gamma,
            m=mass_kg,
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

        # Avoid manufacturing one extra grid row when an LSQ-derived crossing lies only
        # floating-point epsilon beyond an exact grid instant.
        count = int(math.ceil(
            (self.crossing_time_s - after_time_s) / dt_s - 1e-12
        ))
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


def fit_flight_final_approach(flight: dict[str, Any]) -> FittedApproach | None:
    """Fit one declared-datum flight to its manifest-published runway, or return ``None``.

    Harvested ADS-B is fitted in HAE against the CIFP HAE threshold. Synthetic/local-MSL
    inputs continue to use the MSL threshold.
    """
    altitude_source = flight.get("altitude_source")
    if altitude_source not in MSL_ALTITUDE_SOURCES | {HAE_ALTITUDE_SOURCE}:
        raise ValueError(
            f"fit_flight_final_approach does not recognize altitude_source {altitude_source!r}"
        )
    waypoints = flight.get("waypoints") or []
    target = flight.get("runway_target") or {}
    elevation_key = (
        "elevation_hae_m" if altitude_source == HAE_ALTITUDE_SOURCE else "elevation_msl_m"
    )
    required = ("lat", "lon", elevation_key, "course_deg")
    if len(waypoints) < 2 or any(target.get(key) is None for key in required):
        return None

    frame = RunwayFrame(
        ident=str(flight.get("runway") or "?"),
        lat=float(target["lat"]),
        lon=float(target["lon"]),
        elevation_m=float(target[elevation_key]),
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
    # Fit one constant speed over the same established segment as the spatial lines.
    # Arrival records may continue through rollout, but none of those samples enter here.
    fit_slice = slice(fit.first_sample_index, fit.last_sample_index + 1)
    along_rate = _fit_along_rate(waypoints[fit_slice], projected[fit_slice])
    crossing_time = (
        last_time - last_along / along_rate
        if last_along < 0.0 and along_rate is not None
        else None
    )
    return FittedApproach(
        altitude_source=altitude_source,
        fit=fit,
        frame=frame,
        crossing=_point_on_fit(frame, fit, 0.0),
        last_observed_along_m=last_along,
        last_observed_time_s=last_time,
        along_rate_mps=along_rate,
        crossing_time_s=crossing_time,
    )


def _fit_along_rate(waypoints, projected) -> float | None:
    """Constant LSQ runway-direction speed over the fitted approach segment.

    This constant-rate estimate is the intentionally small seam for a future deceleration
    model: replace the linear ``along(t)`` fit with that model's rate at crossing; the
    spatial tangent and target-state assembly remain unchanged. Consecutive stuck ADS-B
    positions are collapsed before fitting so a long repeated report cannot imply zero
    approach speed.
    """
    selected = []
    previous_position = None
    for row, point in zip(waypoints, projected):
        position = (row[1], row[2])
        if position != previous_position:
            selected.append((float(row[0]), float(point.along_m)))
            previous_position = position
    if len(selected) < 2:
        return None
    mean_t = sum(t for t, _ in selected) / len(selected)
    denominator = sum((t - mean_t) ** 2 for t, _ in selected)
    if denominator <= 0.0:
        return None
    mean_along = sum(x for _, x in selected) / len(selected)
    rate = sum((t - mean_t) * (x - mean_along) for t, x in selected) / denominator
    return rate if math.isfinite(rate) and rate > 1.0 else None


def _kinematics_on_fit(
    frame: RunwayFrame, fit: SegmentFit, along_rate_mps: float
) -> tuple[float, float, float]:
    """``(V, psi, gamma)`` from the fitted 3-D tangent and its along-track rate."""
    course = math.radians(frame.course_deg)
    east_hat, north_hat = math.sin(course), math.cos(course)
    cross_rate = fit.cross.slope * along_rate_mps
    vertical_rate = fit.height.slope * along_rate_mps
    east_rate = along_rate_mps * east_hat + cross_rate * north_hat
    north_rate = along_rate_mps * north_hat - cross_rate * east_hat
    ground_speed = math.hypot(east_rate, north_rate)
    V = math.hypot(ground_speed, vertical_rate)
    psi = math.atan2(north_rate, east_rate)
    gamma = math.atan2(vertical_rate, ground_speed)
    return V, psi, gamma


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
