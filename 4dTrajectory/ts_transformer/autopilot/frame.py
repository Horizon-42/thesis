"""The dynamics' state read the way the words read a flight (executor design §2.1).

The rollout carries geodetic rows ``(lat, lon, alt, V, ψ, γ, m)``
(`aerodynamic_model.torch_dynamics.STATE_NAMES`), ψ in the math convention (east 0,
counter-clockwise), so a POSITIVE bank turns LEFT. The words speak compass tracks (true north 0,
clockwise), geometric MSL heights, ground speeds, and metres in the airport frame
(`instructions.airport.AirportGeometry.frame`). This module is the one place the two meet.

No wind: the ground track is ψ and the ground speed is ``V cos γ``. The height is the geodetic row's
own altitude — the chart the rollout integrates in carries height above the threshold elevation, not
a tangent plane's up axis, so nothing here corrects for the earth's curvature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from aerodynamic_model.torch_dynamics import STATE_NAMES
from geokit import METRES_PER_DEG_LAT
from ts_transformer.instructions.airport import AirportGeometry

LAT, LON, ALT, SPEED, PSI, GAMMA, MASS = (STATE_NAMES.index(name) for name in
                                          ("lat_deg", "lon_deg", "alt_m", "V", "psi", "gamma", "mass_kg"))


def compass_deg(psi_rad: torch.Tensor) -> torch.Tensor:
    """Math-convention heading → compass degrees in [0, 360) (`instructions.words.compass_from_math_rad`)."""
    return torch.remainder(90.0 - torch.rad2deg(psi_rad), 360.0)


def wrap180(angle_deg: torch.Tensor) -> torch.Tensor:
    """Signed angle in [-180, 180) (`instructions.words.wrap180`)."""
    return torch.remainder(angle_deg + 180.0, 360.0) - 180.0


@dataclass(frozen=True)
class AirportCharts:
    """Each flight's airport frame, one row per flight: the same projection as
    `AirportENUFrame.horizontal_from_latlon`, read off the frame object itself."""

    lat0_deg: torch.Tensor          # [B]
    lon0_deg: torch.Tensor          # [B]
    m_per_deg_lon: torch.Tensor     # [B]

    @classmethod
    def of(cls, geometries: Sequence[AirportGeometry], *, dtype: torch.dtype, device: torch.device) -> AirportCharts:
        def column(values: list[float]) -> torch.Tensor:
            return torch.tensor(values, dtype=dtype, device=device)
        return cls(lat0_deg=column([g.frame.lat0 for g in geometries]),
                   lon0_deg=column([g.frame.lon0 for g in geometries]),
                   m_per_deg_lon=column([g.frame.m_per_deg_lon for g in geometries]))

    def take(self, index: torch.Tensor) -> AirportCharts:
        return AirportCharts(self.lat0_deg[index], self.lon0_deg[index], self.m_per_deg_lon[index])

    def horizontal(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (lon_deg - self.lon0_deg) * self.m_per_deg_lon, (lat_deg - self.lat0_deg) * METRES_PER_DEG_LAT


@dataclass(frozen=True)
class Kinematics:
    """What the laws read, ``[B]`` each: position in the airport frame, geometric MSL height,
    airspeed, compass track, path angle (climbing positive), ground speed, mass."""

    e_m: torch.Tensor
    n_m: torch.Tensor
    height_m: torch.Tensor
    speed_mps: torch.Tensor
    track_deg: torch.Tensor
    gamma_rad: torch.Tensor
    ground_speed_mps: torch.Tensor
    mass_kg: torch.Tensor


def read_state(geodetic: torch.Tensor, charts: AirportCharts) -> Kinematics:
    """``[B,7]`` geodetic rows → :class:`Kinematics`."""
    e, n = charts.horizontal(geodetic[:, LAT], geodetic[:, LON])
    speed, gamma = geodetic[:, SPEED], geodetic[:, GAMMA]
    return Kinematics(e_m=e, n_m=n, height_m=geodetic[:, ALT], speed_mps=speed,
                      track_deg=compass_deg(geodetic[:, PSI]), gamma_rad=gamma,
                      ground_speed_mps=speed * torch.cos(gamma), mass_kg=geodetic[:, MASS])

