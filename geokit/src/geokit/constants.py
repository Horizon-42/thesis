"""Single source of truth for Earth / unit constants used across the project.

Why this module exists: the same constants (Earth radius, metres-per-degree, nm/ft/kt
factors) were re-derived in ~40 places, in several mutually-inconsistent forms. They now
live here once. The frontend (TypeScript, which cannot import Python) mirrors these via a
generated ``geoConstants.json`` — see ``geokit/scripts/export_constants_json.py``.

Radius policy
-------------
- **Spherical helpers** (haversine / great-circle / small-angle distance) default to the
  WGS84 semi-major axis ``a`` via :data:`SPHERE_RADIUS_M`. The IUGG mean radius
  :data:`EARTH_RADIUS_MEAN_M` is kept as the **switchable** alternative: set
  ``SPHERE_RADIUS_M = EARTH_RADIUS_MEAN_M`` here to flip the default everywhere, or pass
  ``radius_m=...`` to an individual helper.
- **Ellipsoidal helpers** (ECEF, curvature radii ``R_M``/``R_N``, geodetic dynamics) use
  the full WGS84 ellipsoid (``WGS84_A`` + ``WGS84_E2``) and are unaffected by the switch.
"""

from __future__ import annotations

import math

# ── WGS84 ellipsoid ──────────────────────────────────────────────────────────
WGS84_A = 6_378_137.0                       # semi-major axis (equatorial), metres
WGS84_F = 1.0 / 298.257223563               # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)        # first eccentricity squared
WGS84_E_PRIME2 = WGS84_E2 / (1.0 - WGS84_E2)
WGS84_B = WGS84_A * (1.0 - WGS84_F)         # semi-minor axis (polar), metres

# ── Spherical-Earth radius for great-circle / haversine helpers ──────────────
EARTH_RADIUS_MEAN_M = 6_371_008.8           # IUGG mean radius R1 = (2a + b)/3
SPHERE_RADIUS_M = WGS84_A                    # default; set to EARTH_RADIUS_MEAN_M to switch

# ── Flat-Earth metres-per-degree (cheap local-tangent helpers) ───────────────
METRES_PER_DEG_LAT = 111_320.0              # mean degree of latitude, metres

# ── Unit conversions ─────────────────────────────────────────────────────────
NM_M = 1852.0                               # nautical mile -> metre (exact, by definition)
FT_M = 0.3048                               # international foot -> metre (exact)
KT_MS = NM_M / 3600.0                        # knot -> metre/second (= 0.5144444...)
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
