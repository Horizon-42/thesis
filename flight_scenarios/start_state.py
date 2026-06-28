"""Estimate an aircraft's physics state at the start / end of an observed track.

The point-mass model state is ``(lat, lon, alt, V, psi, gamma, m)``:

- ``lat, lon, alt`` — read straight off the boundary track sample.
- ``m`` — the aircraft mass (passed in; the track says nothing about mass).
- ``V, psi, gamma`` — the kinematic components, which the track does **not** store
  directly. They are *estimated* from the motion over a short window:

      V     = speed along the flight path         = |velocity|
      psi   = heading (0 = North, CW, radians)    = atan2(east_velocity, north_velocity)
      gamma = flight-path angle (+ = climb, rad)  = atan2(vertical_velocity, ground_speed)

The velocity is a **least-squares fit** of the ENU position against time over the window,
not a 2-point finite difference. Low-altitude ADS-B is jittery — duplicate / "stuck"
position reports near the ground make a 2-point estimate wildly under/over-read the speed —
and a line fit over ~15 s of samples (after dropping stuck reports) is robust to that.
"""

from __future__ import annotations

import math

from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from aerodynamic_model.common import GeodeticState

# Seconds of track used to estimate the velocity. A line fit over this window rides out 1 Hz
# ADS-B jitter; ~15 s spans a stabilized-approach segment without smearing across maneuvers.
DEFAULT_WINDOW_S = 15.0

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
    (the CZML-input format). The position is the first sample; the velocity is fit over the
    first ``window_s`` seconds.
    """
    if len(waypoints) < 2:
        raise ValueError("need at least two waypoints to estimate V / psi / gamma")

    _t0, lon0, lat0, alt0 = waypoints[0]
    V, psi, gamma = _velocity_lsq(_window_from_start(waypoints, window_s))
    return GeodeticState(
        latitude=lat0, longitude=lon0, altitude=alt0, V=V, psi=psi, gamma=gamma, m=mass_kg
    )


def final_state_from_track(
    waypoints: list[Waypoint],
    *,
    mass_kg: float,
    window_s: float = DEFAULT_WINDOW_S,
) -> GeodeticState:
    """Estimate the :class:`GeodeticState` at the **end** of ``waypoints`` (the target).

    The mirror of :func:`initial_state_from_track`: the position is the last sample, and the
    velocity is fit over the last ``window_s`` seconds (the motion *into* the final point).
    """
    if len(waypoints) < 2:
        raise ValueError("need at least two waypoints to estimate V / psi / gamma")

    _tN, lonN, latN, altN = waypoints[-1]
    V, psi, gamma = _velocity_lsq(_window_from_end(waypoints, window_s))
    return GeodeticState(
        latitude=latN, longitude=lonN, altitude=altN, V=V, psi=psi, gamma=gamma, m=mass_kg
    )


# ── Velocity estimation ───────────────────────────────────────────────────────

def _window_from_start(waypoints: list[Waypoint], window_s: float) -> list[Waypoint]:
    """Samples within ``window_s`` of the first one (at least the first two)."""
    t0 = waypoints[0][0]
    window = [wp for wp in waypoints if wp[0] - t0 <= window_s]
    return window if len(window) >= 2 else waypoints[:2]


def _window_from_end(waypoints: list[Waypoint], window_s: float) -> list[Waypoint]:
    """Samples within ``window_s`` of the last one (at least the last two)."""
    tN = waypoints[-1][0]
    window = [wp for wp in waypoints if tN - wp[0] <= window_s]
    return window if len(window) >= 2 else waypoints[-2:]


def _velocity_lsq(window: list[Waypoint]) -> tuple[float, float, float]:
    """Least-squares velocity ``(V, psi, gamma)`` from a window of ``[t, lon, lat, alt]``.

    Projects each sample into a local ENU frame (metres) anchored at the window's first
    sample, fits east / north / up against time (the slopes are the velocity components),
    and reads off speed, heading, and flight-path angle. Consecutive samples with an
    identical horizontal position (stuck ADS-B reports) are dropped first — they would
    otherwise bias the ground speed toward zero.
    """
    samples = [window[0]]
    for wp in window[1:]:
        if (wp[1], wp[2]) != (samples[-1][1], samples[-1][2]):  # new horizontal position
            samples.append(wp)
    if len(samples) < 2:
        samples = window  # every report was at one point -> keep raw (velocity ~ 0)

    t0, lon0, lat0, alt0 = samples[0]
    m_per_deg_lon = metres_per_deg_lon(lat0)
    ts = [s[0] - t0 for s in samples]
    east = [(s[1] - lon0) * m_per_deg_lon for s in samples]
    north = [(s[2] - lat0) * METRES_PER_DEG_LAT for s in samples]
    up = [s[3] - alt0 for s in samples]

    ve = _slope(ts, east)
    vn = _slope(ts, north)
    vu = _slope(ts, up)

    ground_speed = math.hypot(ve, vn)
    V = math.sqrt(ve * ve + vn * vn + vu * vu)
    psi = math.atan2(ve, vn) if ground_speed > 0.0 else 0.0
    gamma = math.atan2(vu, ground_speed) if ground_speed > 0.0 else 0.0
    return V, psi, gamma


def _slope(ts: list[float], xs: list[float]) -> float:
    """Least-squares slope ``d(x)/d(t)`` (0 if the times don't vary)."""
    n = len(ts)
    mean_t = sum(ts) / n
    denom = sum((t - mean_t) ** 2 for t in ts)
    if denom == 0.0:
        return 0.0
    mean_x = sum(xs) / n
    return sum((t - mean_t) * (x - mean_x) for t, x in zip(ts, xs)) / denom
