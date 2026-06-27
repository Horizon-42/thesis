"""Estimate an aircraft's initial physics state from an observed track.

The point-mass model state is ``(lat, lon, alt, V, psi, gamma, m)``:

- ``lat, lon, alt`` — read straight off the first track sample.
- ``m`` — the aircraft mass (passed in; the track says nothing about mass).
- ``V, psi, gamma`` — the kinematic components, which the track does **not** store
  directly. They are *estimated* by finite-differencing two samples a short window apart
  at the start of the track:

      V     = speed along the flight path         = |displacement| / dt
      psi   = heading (0 = North, CW, radians)    = great-circle bearing of the step
      gamma = flight-path angle (+ = climb, rad)  = atan2(vertical rise, horizontal run)

This is the one piece of real physics in the package. It is left as a guided TODO — see
the ``TODO ①`` block below and ``flight_scenarios/README.md``.
"""

from __future__ import annotations

import math

from geokit import bearing_rad, haversine_m

from aerodynamic_model.common import GeodeticState

# How far apart (seconds) the two samples used for the finite difference should be. A few
# seconds smooths 1 Hz ADS-B jitter without losing the start geometry.
DEFAULT_WINDOW_S = 5.0

# A CZML-input waypoint is [offset_sec, lon_deg, lat_deg, alt_m].
Waypoint = list  # [float, float, float, float]


def initial_state_from_track(
    waypoints: list[Waypoint],
    *,
    mass_kg: float,
    window_s: float = DEFAULT_WINDOW_S,
) -> GeodeticState:
    """Estimate the :class:`GeodeticState` at the **start** of ``waypoints``.

    ``waypoints`` are ``[offset_sec, lon_deg, lat_deg, alt_m]`` rows, ascending in time
    (the CZML-input format). The anchor is the first sample; the second sample is the
    first one at least ``window_s`` later (or the last sample, for very short tracks).
    """
    if len(waypoints) < 2:
        raise ValueError("need at least two waypoints to estimate V / psi / gamma")

    t0, lon0, lat0, alt0 = waypoints[0]
    # Second point: the first sample at least `window_s` after the anchor.
    p1 = next((wp for wp in waypoints[1:] if wp[0] - t0 >= window_s), waypoints[-1])
    t1, lon1, lat1, alt1 = p1

    dt = t1 - t0
    if dt <= 0:
        raise ValueError("waypoint time offsets must be strictly increasing")

    # Geometry of the step (provided; these are the inputs to the estimate):
    horizontal_m = haversine_m(lat0, lon0, lat1, lon1)  # ground distance over the window
    vertical_m = alt1 - alt0                              # altitude change over the window

    # ── TODO ① — estimate V, psi, gamma from (horizontal_m, vertical_m, dt) ──────────
    # You have everything you need above. The three formulas:
    #
    #     V     = math.sqrt(horizontal_m**2 + vertical_m**2) / dt     # m/s, along-path speed
    #     psi   = bearing_rad(lat0, lon0, lat1, lon1)                 # rad, 0 = N, CW (geokit)
    #     gamma = math.atan2(vertical_m, horizontal_m)                # rad, + = climbing
    #
    # Assign those three, then return the state below (delete this raise):
    raise NotImplementedError(
        "Implement the V / psi / gamma estimate in initial_state_from_track "
        "(flight_scenarios/start_state.py). The three formulas are in the TODO ① comment; "
        "see flight_scenarios/README.md for the derivation."
    )

    # return GeodeticState(
    #     latitude=lat0,
    #     longitude=lon0,
    #     altitude=alt0,
    #     V=V,
    #     psi=psi,
    #     gamma=gamma,
    #     m=mass_kg,
    # )
