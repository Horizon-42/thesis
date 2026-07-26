"""Threshold-centred horizontal coordinate-frame implementations.

The external ``coordinate_frame`` setting is resolved once by :func:`frame_for_state`.
Everything downstream receives a concrete frame object and does not branch on a mode string.
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


_FRAME_CLASSES: dict[str, type[CoordinateFrame]] = {
    "enu": ENUFrame,
    "runway-aligned": RunwayAlignedFrame,
}


def frame_for_state(state: GeodeticState, coordinate_frame: str = "enu") -> CoordinateFrame:
    """Instantiate the concrete frame selected by the external configuration."""
    try:
        frame_class = _FRAME_CLASSES[coordinate_frame]
    except KeyError as exc:
        raise ValueError(f"unknown coordinate frame {coordinate_frame!r}") from exc
    return frame_class.for_state(state)
