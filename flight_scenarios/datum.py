"""Vertical datum at the data -> modeling seam: observed ADS-B altitude is NOT MSL.

OpenSky's ``geoaltitude`` -- the only altitude the harvest keeps (``altitude_source:
"opensky_history_geoaltitude_m"``) -- is GNSS geometric altitude, i.e. height above the
WGS84 **ellipsoid** (HAE). Everything the modeling plane measures it against is **mean sea
level** (orthometric): runway threshold elevations, CIFP procedure altitudes, and the
8260.58D gates in ``evaluation/thresholds.py``. The two differ by the geoid undulation N::

    h_HAE = H_MSL + N          =>          H_MSL = h_HAE - N

N is roughly -25 to -33 m over the continental US, so an uncorrected observed track sits
about 30 m BELOW its own approach. Measured on 996 KRDU arrivals whose fitted glidepath is
a textbook 3.08 deg: the extrapolated threshold crossing came out 29.2 m low and the
vertical gate passed 0.5 % of real, completed airline landings. KRDU's lowest observed
sample confirms the mechanism independently -- 99.1 m, against a field elevation of
132.59 m and N = -33.53 m, i.e. 99.06 m predicted.

WHY HERE AND NOT IN THE HARVEST
-------------------------------
The harvest feeds two consumers with opposite requirements:

    *_landings.json (HAE, as the sensor reported)
       |
       +-- CZML -> Cesium      wants HAE  (``aeroviz-4d/src/types/czml.d.ts``: "altitude in
       |                       METERS above WGS84 ellipsoid") -- correct as recorded
       +-- flight_scenarios    wants MSL  (thresholds / CIFP / gates are MSL)

Converting at the harvest would fix the modeling plane and break the visualisation by the
same ~33 m. So the harvest stays a faithful record of what the sensor said, and the datum
choice is made once, here, on the way into the modeling plane.

The conversion is keyed on the flight's declared ``altitude_source``, so a file can be
loaded twice without being converted twice, and an unrecognised source fails loudly rather
than being silently assumed to be one datum or the other.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Iterable, Sequence

# Source tag written by ``trajectory_data_process/processing/czml_export.py``.
HAE_ALTITUDE_SOURCE = "opensky_history_geoaltitude_m"
# What this module rewrites it to, so the conversion is visible and non-repeatable.
MSL_ALTITUDE_SOURCE = "opensky_history_geoaltitude_m_to_msl_egm96"
# Sources that are ALREADY MSL and must not be converted:
#   "synthetic" -- ``ts_transformer/synthetic.py`` builds waypoints as
#                  ``threshold["elevation_m"] + height``, and threshold elevations are MSL.
MSL_ALTITUDE_SOURCES = frozenset({MSL_ALTITUDE_SOURCE, "synthetic"})

# EPSG:4979 = WGS84 3D (ellipsoidal height); EPSG:4326+5773 = WGS84 2D + EGM96 height.
_HAE_CRS = "EPSG:4979"
_MSL_CRS = "EPSG:4326+5773"

@functools.cache
def _geoid_transformer():
    """The EGM96 transformer, built once (a raising build is not cached, so it retries).

    ``pyproj`` needs the EGM96 grid (``us_nga_egm96_15.tif``). Without it PROJ silently
    falls back to a "ballpark" no-op that returns the input unchanged -- which would look
    exactly like a correctly-applied zero correction -- so the transform is verified against
    a known undulation before it is ever used on real data.
    """
    try:
        import pyproj
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "the HAE->MSL conversion needs pyproj (EGM96 geoid grid); install pyproj into "
            "the thesis env, or the observed altitudes will be ~30 m off their datum"
        ) from exc
    # An explicit PROJ_NETWORK from the operator wins (the probe below still fails loudly
    # if the grid then isn't reachable); otherwise enable network so PROJ can fetch and
    # cache the grid.
    if "PROJ_NETWORK" not in os.environ:
        pyproj.network.set_network_enabled(True)
    transformer = Transformer.from_crs(_HAE_CRS, _MSL_CRS, always_xy=True)
    # KRDU: EGM96 N = -33.53 m. A no-op transform returns 0.0 here. Written as
    # not-(within-tolerance) so a NaN probe also raises -- every comparison with NaN is
    # False, and `> 1.0` would have cached a transformer that NaNs every altitude.
    _, _, probe = transformer.transform(-78.7794, 35.8792, 0.0)
    if not (abs(probe - 33.53) <= 1.0):
        raise RuntimeError(
            "PROJ returned a ballpark (no-op) vertical transform: the EGM96 grid "
            "'us_nga_egm96_15.tif' is missing and PROJ network access is unavailable. "
            f"Expected a KRDU geoid undulation near -33.53 m, got {-probe:.2f} m. "
            "Fetch the grid (PROJ_NETWORK=ON, or projsync) before building scenarios."
        )
    return transformer


def geoid_undulation_m(lats: Sequence[float], lons: Sequence[float]) -> list[float]:
    """EGM96 geoid undulation N = h_HAE - H_MSL, metres, one per point."""
    _, _, msl_of_zero = _geoid_transformer().transform(list(lons), list(lats), [0.0] * len(lats))
    # Transforming HAE 0 gives -N, so N is its negation.
    return [-z for z in msl_of_zero]


def waypoints_to_msl(waypoints: Iterable[Sequence[float]]) -> list[list[float]]:
    """``[t, lon, lat, alt_HAE]`` -> ``[t, lon, lat, alt_MSL]``, altitudes only.

    The altitudes are transformed directly: EGM96's N depends only on (lat, lon), so the
    transform of ``alt_HAE`` IS ``alt_HAE - N``, with no intermediate undulation list and
    no sign flip to get wrong (``geoid_undulation_m`` stays as the diagnostic API).
    """
    rows = [list(w) for w in waypoints]
    if not rows:
        return rows
    _, _, msl = _geoid_transformer().transform(
        [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]
    )
    for row, alt in zip(rows, msl):
        row[3] = alt
    return rows


def flight_to_msl(flight: dict[str, Any]) -> dict[str, Any]:
    """One CZML-input flight with its track converted HAE -> MSL.

    Already-converted flights pass through untouched; an unknown ``altitude_source``
    raises, because guessing a datum is how a 30 m error survives review.
    """
    source = flight.get("altitude_source")
    if source in MSL_ALTITUDE_SOURCES:
        return flight
    if source != HAE_ALTITUDE_SOURCE:
        raise ValueError(
            f"flight {flight.get('id')!r} declares altitude_source {source!r}; the modeling "
            f"plane only knows how to convert {HAE_ALTITUDE_SOURCE!r} (ellipsoidal) to MSL"
        )
    converted = dict(flight)
    converted["waypoints"] = waypoints_to_msl(flight.get("waypoints") or [])
    converted["altitude_source"] = MSL_ALTITUDE_SOURCE
    return converted


def flights_to_msl(flights: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every flight's track converted HAE -> MSL."""
    return [flight_to_msl(f) for f in flights]
