"""One metric frame per window: the commanded aircraft's target frame.

The constrained solve's decision state is ``(n, e)`` metres from its target (the LTP),
``n = (lat - lat_t)·R``, ``e = (lon - lon_t)·R·cos(lat_t)``. Recorded positions, runway thresholds
and the judge's positions are put in the same frame, so a separation row is a function of the
decision variables alone. The formula is the solver's own (``collocation.schemes._normalization_cb``).
It is a local equirectangular frame: an east distance is off by about ``tan(lat)·Δlat`` (Δlat from
the target, radians) — 0.46 % at 36° N and 40 km (design §4.4); the row margin covers it.
"""

from __future__ import annotations

import math

import casadi as ca
import numpy as np

from aerodynamic_model.common import GeodeticState
from collocation.schemes import _normalization_cb


class TargetFrame:
    """``(lat, lon)`` degrees -> ``(n, e)`` metres in the frame anchored at ``target``."""

    def __init__(self, target: GeodeticState) -> None:
        c, b = _normalization_cb(ca.DM([math.radians(target.latitude), math.radians(target.longitude), 0, 0, 0, 0]))
        self._c = np.array(c).reshape(-1)[:2]
        self._b = np.array(b).reshape(-1)[:2]

    def to_ne(self, lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray]:
        return ((np.radians(lat_deg) - self._b[0]) / self._c[0],
                (np.radians(lon_deg) - self._b[1]) / self._c[1])

    def east_scale_error(self, lat_deg) -> float:
        """The largest relative east-distance error of the frame at the given latitudes."""
        lat_t = self._b[0]
        return float(np.max(np.abs(np.tan(lat_t) * (np.radians(lat_deg) - lat_t))))
