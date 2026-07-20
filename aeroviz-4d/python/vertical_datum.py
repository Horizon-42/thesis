"""MSL -> ellipsoidal (HAE) at the modeling -> viewer seam.

Modeling-plane records (``*_states.json`` / predictions) carry MSL altitudes: observed
tracks are converted HAE->MSL on ingest, and everything downstream (runway thresholds,
CIFP, the evaluation gates) is MSL. Cesium, however, defines CZML ``cartographicDegrees``
altitude as metres above the WGS84 ELLIPSOID -- so record altitudes must gain the EGM96
geoid undulation back on the way out, or every modeled path renders ~|N| (33.5 m at KRDU)
above its own observed reference and the terrain.

Deliberate MIRROR of the vertical-datum handling in ``flight_scenarios/datum.py``
(frontend tooling must not import the modeling tree -- precedent: ``flight_identity.py``).
Both modules probe the same vector -- KRDU, EGM96 N = -33.53 m; change them together.
"""

from __future__ import annotations

import functools
import os
from typing import Sequence

# EPSG:4326+5773 = WGS84 2D + EGM96 height (MSL); EPSG:4979 = WGS84 3D (ellipsoidal).
_MSL_CRS = "EPSG:4326+5773"
_HAE_CRS = "EPSG:4979"


@functools.cache
def _geoid_transformer():
    """The EGM96 MSL->HAE transformer, built once (a raising build retries next call).

    ``pyproj`` needs the EGM96 grid (``us_nga_egm96_15.tif``). Without it PROJ silently
    falls back to a "ballpark" no-op returning the input unchanged, so the transform is
    verified against a known undulation before it is ever used on real data.
    """
    try:
        import pyproj
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "the MSL->HAE conversion needs pyproj (EGM96 geoid grid); install pyproj, "
            "or the comparison CZML would render ~30 m off the observed reference"
        ) from exc
    # An explicit PROJ_NETWORK from the operator wins (the probe below still fails loudly
    # if the grid then isn't reachable); otherwise enable network so PROJ can fetch and
    # cache the grid.
    if "PROJ_NETWORK" not in os.environ:
        pyproj.network.set_network_enabled(True)
    transformer = Transformer.from_crs(_MSL_CRS, _HAE_CRS, always_xy=True)
    # KRDU: MSL 0 must come back as HAE = N = -33.53 m; a no-op transform returns 0.0.
    # Written as not-(within-tolerance) so a NaN probe also raises.
    _, _, probe = transformer.transform(-78.7794, 35.8792, 0.0)
    if not (abs(probe + 33.53) <= 1.0):
        raise RuntimeError(
            "PROJ returned a ballpark (no-op) vertical transform: the EGM96 grid "
            "'us_nga_egm96_15.tif' is missing and PROJ network access is unavailable. "
            f"Expected a KRDU geoid undulation near -33.53 m, got {probe:.2f} m. "
            "Fetch the grid (PROJ_NETWORK=ON, or projsync) before building comparison CZML."
        )
    return transformer


def msl_to_hae(lons: Sequence[float], lats: Sequence[float], alts: Sequence[float]) -> list[float]:
    """MSL altitudes -> ellipsoidal (HAE), one per point: h_HAE = H_MSL + N(lat, lon)."""
    _, _, hae = _geoid_transformer().transform(list(lons), list(lats), list(alts))
    return list(hae)
