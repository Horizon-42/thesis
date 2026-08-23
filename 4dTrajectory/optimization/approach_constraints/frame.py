"""The target-anchored metric frame — convert fixes (lat/lon) into the optimizer's ``(n, e)``.

This is the **reference implementation** of the coordinate transform the ``*Normalized``
collocation schemes use (``collocation.schemes._normalization_cb``). It is
provided complete — getting it wrong silently misaligns the fixes from the aircraft state, so
it is plumbing, not an exercise. Study it; the constraint math (geometry.py, lateral.py,
vertical.py) is what you implement.

The frame is anchored at the optimization **target = the LTP**. With the WGS84 semi-major axis
``R`` and the target latitude ``lat_t`` (8260.58D / design-doc §2):

    n = (lat - lat_t) · R                 # north metres
    e = (lon - lon_t) · R · cos(lat_t)    # east  metres   (cos uses the *target* latitude — one flat frame)

so the **LTP maps to the origin ``(0, 0)``** and a point's ``(n, e)`` is directly comparable to
the optimizer's decision variables. Over a TMA (tens of NM) the equirectangular distortion is
sub-metre. Use the SAME ``R`` and ``lat_t`` as the optimizer (``geokit.WGS84_A``); do not
re-derive Earth radius.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geokit import DEG2RAD, WGS84_A


@dataclass(frozen=True)
class TargetFrame:
    """A flat metric frame centred on the target (LTP).

    ``radius_m`` defaults to the WGS84 semi-major axis, matching the optimizer's
    ``_normalization_cb`` (its ``_EARTH_RADIUS_M = geokit.WGS84_A``).
    """

    lat_t_deg: float
    lon_t_deg: float
    radius_m: float = WGS84_A

    @property
    def _cos_lat_t(self) -> float:
        return float(np.cos(self.lat_t_deg * DEG2RAD))

    def to_ne(self, lat_deg, lon_deg) -> np.ndarray:
        """Convert geodetic ``lat/lon`` (degrees) to ``(n, e)`` metres in this frame.

        Accepts scalars or array-likes; returns shape ``(..., 2)`` (``(2,)`` for scalars).

        Raises for input on the far side of the antimeridian from the target: the raw
        ``lon − lon_t`` difference maps such a fix tens of thousands of km away, and the
        optimizer's ``_normalization_cb`` shares the same non-wrapping assumption — a TMA
        straddling ±180° needs BOTH transforms taught to wrap, not a silent fix here.
        """
        lat = np.asarray(lat_deg, dtype=float)
        lon = np.asarray(lon_deg, dtype=float)
        dlon = lon - self.lon_t_deg
        if np.any(np.abs(dlon) > 180.0):
            raise ValueError(
                "TargetFrame does not span the antimeridian: |lon - lon_t| > 180 deg "
                f"(target lon {self.lon_t_deg})"
            )
        n = (lat - self.lat_t_deg) * DEG2RAD * self.radius_m
        e = dlon * DEG2RAD * self.radius_m * self._cos_lat_t
        return np.stack([n, e], axis=-1)

    def to_latlon(self, ne) -> np.ndarray:
        """Inverse of :meth:`to_ne`: ``(n, e)`` metres back to ``(lat, lon)`` degrees.

        Returns shape ``(..., 2)`` as ``(lat_deg, lon_deg)``. Mainly for tests / debugging.
        """
        ne = np.asarray(ne, dtype=float)
        n = ne[..., 0]
        e = ne[..., 1]
        lat = self.lat_t_deg + n / (DEG2RAD * self.radius_m)
        lon = self.lon_t_deg + e / (DEG2RAD * self.radius_m * self._cos_lat_t)
        return np.stack([lat, lon], axis=-1)
