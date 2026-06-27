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

    V = math.sqrt(horizontal_m**2 + vertical_m**2) / dt
    psi = bearing_rad(lat0, lon0, lat1, lon1)
    gamma = math.atan2(vertical_m, horizontal_m)

    return GeodeticState(
        latitude=lat0,
        longitude=lon0,
        altitude=alt0,
        V=V,
        psi=psi,
        gamma=gamma,
        m=mass_kg,
    )


def final_state_from_track(
    waypoints: list[Waypoint],
    *,
    mass_kg: float,
    window_s: float = DEFAULT_WINDOW_S,
) -> GeodeticState:
    """Estimate the :class:`GeodeticState` at the **end** of ``waypoints`` (the target).

    The mirror of :func:`initial_state_from_track`, anchored at the *last* sample: the
    velocity is the step *into* the final point (from a sample about ``window_s`` earlier),
    and the position is the final sample. Same V/psi/gamma kinematics, evaluated at the end.
    """
    if len(waypoints) < 2:
        raise ValueError("need at least two waypoints to estimate V / psi / gamma")

    tN, lonN, latN, altN = waypoints[-1]
    # Earlier point: the last sample at least `window_s` before the end.
    p0 = next((wp for wp in reversed(waypoints[:-1]) if tN - wp[0] >= window_s), waypoints[0])
    t0, lon0, lat0, alt0 = p0

    dt = tN - t0
    if dt <= 0:
        raise ValueError("waypoint time offsets must be strictly increasing")

    horizontal_m = haversine_m(lat0, lon0, latN, lonN)
    vertical_m = altN - alt0

    # Same three formulas as initial_state_from_track, evaluated at the track's end:
    V = math.sqrt(horizontal_m**2 + vertical_m**2) / dt
    psi = bearing_rad(lat0, lon0, latN, lonN)
    gamma = math.atan2(vertical_m, horizontal_m)

    return GeodeticState(
        latitude=latN,
        longitude=lonN,
        altitude=altN,
        V=V,
        psi=psi,
        gamma=gamma,
        m=mass_kg,
    )
