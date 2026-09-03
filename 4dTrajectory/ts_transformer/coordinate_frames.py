"""Horizontal coordinate-frame implementations: threshold-anchored, and airport-anchored.

The external ``coordinate_frame`` setting is resolved once by :func:`frame_for_state`.
Everything downstream receives a concrete frame object and does not branch on a mode string.

Two anchors exist. ``enu`` / ``runway-aligned`` put the origin at the assigned runway
threshold, so the target IS the origin and the position channels are distance-to-go.
``airport-enu`` puts the origin at the airport reference point instead: one chart per
airport, shared by every runway, in which the target is an ordinary point
(``FlightSeries.target_chart``). Nothing downstream may equate the origin with the
threshold — that is the contract this second anchor exists to test.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from aerodynamic_model.common import GeodeticState
from geokit import METRES_PER_DEG_LAT, WGS84_A, metres_per_deg_lon, wgs84_curvature_radii


@dataclass(frozen=True)
class CoordinateFrame(ABC):
    """Projection utilities shared by concrete horizontal coordinate frames."""

    lat0: float
    lon0: float
    alt0: float

    @classmethod
    def for_state(cls, state: GeodeticState) -> CoordinateFrame:
        return cls(lat0=state.latitude, lon0=state.longitude, alt0=state.altitude)

    @property
    def m_per_deg_lon(self) -> float:
        return metres_per_deg_lon(self.lat0)

    def latlon_from_horizontal(self, first: float, second: float) -> tuple[float, float]:
        """Configured horizontal coordinates in metres -> ``(lat, lon)`` degrees."""
        east, north = self.to_world_horizontal(first, second)
        return (
            self.lat0 + north / METRES_PER_DEG_LAT,
            self.lon0 + east / self.m_per_deg_lon,
        )

    def chart_velocity_factors(self, lat_deg: float, alt_m: float) -> tuple[float, float]:
        """Transport factors from physical ENU velocity to chart derivatives."""
        r_m, r_n = wgs84_curvature_radii(lat_deg)
        cos_lat0 = math.cos(math.radians(self.lat0))
        cos_lat = math.cos(math.radians(lat_deg))
        return (
            WGS84_A * cos_lat0 / ((r_n + alt_m) * cos_lat),
            WGS84_A / (r_m + alt_m),
        )

    @abstractmethod
    def from_world_horizontal(self, east: float, north: float) -> tuple[float, float]:
        """World EN chart components -> this frame's horizontal axes."""

    @abstractmethod
    def to_world_horizontal(self, first: float, second: float) -> tuple[float, float]:
        """This frame's horizontal axes -> world EN chart components."""


@dataclass(frozen=True)
class ENUFrame(CoordinateFrame):
    """Unrotated local east/north/up frame anchored at the runway threshold."""

    def from_world_horizontal(self, east: float, north: float) -> tuple[float, float]:
        return east, north

    def to_world_horizontal(self, first: float, second: float) -> tuple[float, float]:
        return first, second


@dataclass(frozen=True)
class AirportReference:
    """An airport reference point — the anchor of the airport-fixed frame (MSL metres)."""

    code: str
    lat: float
    lon: float
    elevation_msl_m: float


@dataclass(frozen=True)
class AirportENUFrame(ENUFrame):
    """Unrotated east/north/up frame anchored at the AIRPORT reference point.

    Same projection as :class:`ENUFrame`; only the anchor differs, and that difference is
    the experiment: every runway of one airport shares ONE chart, so the physical airspace
    (downwinds, STARs) lands at the same chart coordinates whichever runway is assigned —
    and the target is no longer the origin. Consumers measure distance-to-go, crossing
    planes and approach geometry from ``FlightSeries.target_chart``, never from ``(0, 0)``.
    """

    code: str

    @classmethod
    def for_state(cls, state: GeodeticState) -> AirportENUFrame:
        raise TypeError(
            "AirportENUFrame is anchored at an airport reference point, not at a state; "
            "build it with for_airport"
        )

    @classmethod
    def for_airport(cls, reference: AirportReference) -> AirportENUFrame:
        return cls(
            lat0=reference.lat,
            lon0=reference.lon,
            alt0=reference.elevation_msl_m,
            code=reference.code,
        )


@dataclass(frozen=True)
class RunwayAlignedFrame(CoordinateFrame):
    """Along-runway/cross-runway frame anchored at the runway threshold."""

    heading_rad: float

    @classmethod
    def for_state(cls, state: GeodeticState) -> RunwayAlignedFrame:
        return cls(
            lat0=state.latitude,
            lon0=state.longitude,
            alt0=state.altitude,
            heading_rad=state.psi,
        )

    def from_world_horizontal(self, east: float, north: float) -> tuple[float, float]:
        cosine, sine = math.cos(self.heading_rad), math.sin(self.heading_rad)
        return east * cosine + north * sine, -east * sine + north * cosine

    def to_world_horizontal(self, first: float, second: float) -> tuple[float, float]:
        cosine, sine = math.cos(self.heading_rad), math.sin(self.heading_rad)
        return first * cosine - second * sine, first * sine + second * cosine


COORDINATE_FRAME_ENU = "enu"
COORDINATE_FRAME_RUNWAY_ALIGNED = "runway-aligned"
COORDINATE_FRAME_AIRPORT_ENU = "airport-enu"


def _airport_anchored(
    state: GeodeticState, airport_ref: AirportReference | None
) -> AirportENUFrame:
    del state  # the anchor is the airport, not the target
    if airport_ref is None:
        raise ValueError(
            f"coordinate frame {COORDINATE_FRAME_AIRPORT_ENU!r} requires an airport "
            "reference point (airport_ref)"
        )
    return AirportENUFrame.for_airport(airport_ref)


_FRAME_RESOLVERS = {
    COORDINATE_FRAME_ENU: lambda state, airport_ref: ENUFrame.for_state(state),
    COORDINATE_FRAME_RUNWAY_ALIGNED: (
        lambda state, airport_ref: RunwayAlignedFrame.for_state(state)
    ),
    COORDINATE_FRAME_AIRPORT_ENU: _airport_anchored,
}

# The external setting's vocabulary — config.py imports it, so the two cannot drift.
COORDINATE_FRAMES: tuple[str, ...] = tuple(_FRAME_RESOLVERS)


def frame_for_state(
    state: GeodeticState,
    coordinate_frame: str = COORDINATE_FRAME_ENU,
    *,
    airport_ref: AirportReference | None = None,
) -> CoordinateFrame:
    """Instantiate the concrete frame selected by the external configuration.

    ``state`` is the runway target (the anchor of the threshold frames). ``airport_ref``
    is REQUIRED by ``airport-enu`` and ignored by the threshold-anchored modes.
    """
    try:
        resolve = _FRAME_RESOLVERS[coordinate_frame]
    except KeyError as exc:
        raise ValueError(f"unknown coordinate frame {coordinate_frame!r}") from exc
    return resolve(state, airport_ref)
