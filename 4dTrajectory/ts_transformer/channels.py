"""What the network actually sees: a local chart anchored at the runway threshold.

The evaluation contract's state is ``(lat, lon, alt, V, psi, gamma, m)`` — degrees, metres,
and two angles in radians. That is a poor thing to regress directly:

- ``lat``/``lon`` are ~1e2 degrees of offset carrying ~1e-5 degree signal; float32 attention
  on raw degrees wastes its dynamic range on the airport's absolute position.
- ``psi`` wraps at +/-pi. A model regressing it will average 179deg and -179deg to 0deg —
  pointing the aircraft backwards — right where the turn onto final happens.
- ``m`` is not observable from ADS-B at all. It is carried, never predicted.

So the channels are the **chart coordinates and their exact time derivatives**, in a local
chart whose origin is the runway threshold:

    e, n, u           horizontal-axis 1 / horizontal-axis 2 / up, metres from the threshold
    edot, ndot, udot  d(e)/dt, d(n)/dt, d(u)/dt — metres/second IN THE CHART

The stable channel names represent east/north in :class:`coordinate_frames.ENUFrame` and
along-runway/cross-runway in :class:`coordinate_frames.RunwayAlignedFrame`. The concrete
frame owns that distinction; the channel encoder and model remain frame-agnostic.

``psi`` and ``gamma`` then fall out of the velocity channels on the way back
(:func:`states_from_channels`) with the right convention *by construction* —
after undoing the transport factors, ``psi = atan2(V_north, V_east)`` is exactly the
modeling layer's math-ENU heading, so there is no place left to accidentally substitute a
compass bearing. The wrap problem disappears with it: ``(edot, ndot)`` is continuous
across the branch cut that ``psi`` is not.

**Projection.** ``east = (lon - lon0) * metres_per_deg_lon(lat0)``,
``north = (lat - lat0) * METRES_PER_DEG_LAT`` — the same form + anchor as the
``approach_constraints`` NE frame (``ltp_ne = (0, 0)``, which scales degrees by
``WGS84_A·DEG2RAD`` — the definition geokit's ``METRES_PER_DEG_LAT`` aligns to, so the two
frames share one constant; ``metres_per_deg_lon`` is that times ``cos(lat)``).
The flat form itself deviates from a true tangent-plane ENU by up to ~40 m at the ring edge
(the ``e·n·tanφ/R`` cross term — zero on the pure east/north axes), and ``u = Δalt``
ignores the ~49 m curvature drop there deliberately: the channel is height above the
THRESHOLD, not tangent-plane z. None of that reaches the metrics: predictions, references
and the loss all live in the SAME projection, so the systematic distortion cancels; what
survives on a measured displacement is the local scale distortion (~0.2% at the ring edge,
→ 0 at the threshold, where the gates judge). Matching the existing frame is what matters —
a different projection here would show up as apparent model error against records built by
the other packages.

**Transport-consistent velocity channels.** A state's ``(V, psi, gamma)`` is a PHYSICAL
ENU velocity — the modeling layer's contract (``V_east = V·cosγ·cosψ`` feeding
``lat_dot = V_north/(R_M+h)`` in the geodetic RHS). The chart coordinates above are scaled
*angles* (``n = Δlat_rad·WGS84_A``, ``e = Δlon_rad·WGS84_A·cos lat0`` — exact, because
``METRES_PER_DEG_LAT = WGS84_A·π/180``), so the chart derivative of a physical velocity
picks up the full-transport Jacobian the optimizer's geodetic RHS encodes:

    ndot = V_north · WGS84_A / (R_M + h)
    edot = V_east  · WGS84_A·cos(lat0) / ((R_N + h)·cos(lat))
    udot = V·sin(gamma)                                        (u = Δalt is already exact)

with ``R_M``/``R_N`` the exact WGS84 curvature radii (``geokit.wgs84_curvature_radii``).
The factors are 1 ± ~0.3% over a TMA (``a/R_M`` ≈ 1.0033 at 36°, the cos ratio ≤ 0.3% at
the 25 km ring) — small, but before this change the velocity channels stored the physical
components RAW, so integrating them did not reproduce the position channels (the 2026-07-20
finding A7: an O(0.2–0.5%) mismatch, a few metres per minute of integration). Now the six
channels describe ONE curve consistently: for any state sequence whose velocities are the
true derivatives of its positions, ``∫ edot dt`` IS ``e``. The rename (``ve/vn/vu`` →
``edot/ndot/udot``) is deliberate: the channel tuple is serialised into every checkpoint
and ``train.load_checkpoint`` refuses a mismatch, so pre-change checkpoints fail loudly
instead of silently mis-scaling every velocity.

Fixed at the same time, one seam up: ``state_samples_from_track``'s least-squares fit now
projects through the true tangent scales (``R_M+h``, ``(R_N+h)·cos lat``), so the
``(V, psi, gamma)`` it hands this module really IS physical velocity — it used to carry the
flat-chart scales, an O(0.3%) estimator bias that would otherwise have survived here as
north integration drift (measured: fixing only the chart moved the median whole-track north
drift from 2.7 to 8.6 m/min; fixing both restores consistency, leaving only LSQ smoothing).

**Ordering trap.** CZML-input waypoints are ``[t, lon, lat, alt]`` — *lon before lat* — while
every state dict downstream is ``lat`` before ``lon``. This module is the only place the two
orders meet; nothing else should be indexing a waypoint by position.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from geokit import METRES_PER_DEG_LAT

from aerodynamic_model.common import GeodeticState
from coordinate_frames import CoordinateFrame

# The channel contract. Order AND names are load-bearing — the tuple indexes every tensor,
# the normalizer's per-channel statistics, and the checkpoint; load_checkpoint refuses a
# mismatch. The velocity names encode their semantics (chart derivatives, NOT raw physical
# ENU components) precisely so a semantics change cannot reuse a name and load silently.
CHANNELS: tuple[str, ...] = ("e", "n", "u", "edot", "ndot", "udot")

# Column indices, so nothing downstream hard-codes a number.
IDX = {name: i for i, name in enumerate(CHANNELS)}
POSITION_IDX = (IDX["e"], IDX["n"], IDX["u"])
VELOCITY_IDX = (IDX["edot"], IDX["ndot"], IDX["udot"])


def channels_from_states(
    samples: Sequence[tuple[float, GeodeticState]], frame: CoordinateFrame
) -> tuple[np.ndarray, np.ndarray]:
    """``[(t, GeodeticState), ...]`` -> ``(times[N], channels[N, C])``.

    Feed this the output of ``flight_scenarios.state_samples_from_track`` rather than
    differencing raw waypoints yourself: that function's centred least-squares velocity fit
    (with stuck-ADS-B-report rejection) is what *defines* the reference ``V/psi/gamma``, and
    re-deriving velocity a second way here would show up as model error when predictions are
    compared against reference records built from it.
    """
    if not samples:
        raise ValueError("need at least one state sample to build channels")

    m_per_deg_lon = frame.m_per_deg_lon
    times = np.empty(len(samples), dtype=np.float64)
    out = np.empty((len(samples), len(CHANNELS)), dtype=np.float64)
    for i, (t, s) in enumerate(samples):
        ground_speed = s.V * math.cos(s.gamma)
        f_e, f_n = frame.chart_velocity_factors(s.latitude, s.altitude)
        times[i] = t
        first, second = frame.from_world_horizontal(
            (s.longitude - frame.lon0) * m_per_deg_lon,
            (s.latitude - frame.lat0) * METRES_PER_DEG_LAT,
        )
        first_dot, second_dot = frame.from_world_horizontal(
            ground_speed * math.cos(s.psi) * f_e,
            ground_speed * math.sin(s.psi) * f_n,
        )
        out[i] = (
            first,
            second,
            s.altitude - frame.alt0,                         # u
            first_dot,
            second_dot,
            s.V * math.sin(s.gamma),                         # udot
        )
    return times, out


def states_from_channels(
    times: np.ndarray, values: np.ndarray, frame: CoordinateFrame, *, mass_kg: float
) -> list[tuple[float, GeodeticState]]:
    """``(times[N], channels[N, C])`` -> ``[(t, GeodeticState), ...]``.

    The exact inverse of :func:`channels_from_states` — the transport factors are undone
    at the position the channels themselves encode, so the round trip is the identity —
    and deliberately the same shape ``flight_scenarios.state_samples_from_track`` returns,
    which is what ``4dTrajectory/optimization/evaluation_export.reference_evaluation_record``
    consumes. Producing that shape here means the evaluation record is emitted by the SAME
    function the optimizer's reference records go through, instead of this package
    hand-rolling a second copy of the JSON contract.

    ``m`` is carried from ``mass_kg``, not predicted: ADS-B never observed it, so the model
    was never given it and cannot return it. ``V`` is the TOTAL physical speed along the
    flight path (including the vertical component), matching the evaluation contract — not
    ground speed, and not a chart quantity.
    """
    if len(times) != len(values):
        raise ValueError(f"times ({len(times)}) and values ({len(values)}) must align")

    states: list[tuple[float, GeodeticState]] = []
    for t, row in zip(times, values):
        e, n, u, edot, ndot, udot = (float(v) for v in row)
        lat, lon = frame.latlon_from_horizontal(e, n)
        altitude = frame.alt0 + u
        f_e, f_n = frame.chart_velocity_factors(lat, altitude)
        world_edot, world_ndot = frame.to_world_horizontal(edot, ndot)
        v_east = world_edot / f_e
        v_north = world_ndot / f_n
        ground_speed = math.hypot(v_east, v_north)
        states.append((float(t), GeodeticState(
            latitude=lat,
            longitude=lon,
            altitude=altitude,
            V=math.sqrt(v_east * v_east + v_north * v_north + udot * udot),
            # math-ENU: 0 = East, CCW toward North. atan2(V_north, V_east), NOT the compass
            # atan2(V_east, V_north) — see the module docstring.
            psi=math.atan2(v_north, v_east) if ground_speed > 0.0 else 0.0,
            gamma=math.atan2(udot, ground_speed) if ground_speed > 0.0 else 0.0,
            m=float(mass_kg),
        )))
    return states


def resample_uniform(times: np.ndarray, values: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate channels onto a uniform ``dt_s`` grid starting at ``times[0]``.

    Transformers assume a regular grid; ADS-B does not provide one (nominally 1 Hz, in
    practice ragged, with dropouts). The grid ends at the last sample at or before
    ``times[-1]``, so this never extrapolates past the observed track.

    Interpolating in CHANNEL space rather than state space is the point: interpolating
    ``psi`` would average across the +/-pi branch cut, and interpolating ``lat``/``lon``
    then re-fitting velocity would double-smooth it. Position and velocity components are
    all linear quantities, so a straight interpolation is well defined.
    """
    if len(times) < 2:
        raise ValueError("need at least two samples to resample")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")

    span = float(times[-1] - times[0])
    n_steps = int(math.floor(span / dt_s)) + 1
    if n_steps < 2:
        raise ValueError(
            f"track spans {span:.1f}s, too short for a {dt_s}s grid (would give {n_steps} step(s))"
        )
    grid = times[0] + np.arange(n_steps, dtype=np.float64) * dt_s
    out = np.empty((n_steps, values.shape[1]), dtype=np.float64)
    for c in range(values.shape[1]):
        # One-dimensional linear interpolation
        out[:, c] = np.interp(grid, times, values[:, c])
    return grid, out


def horizontal_distance_m(values: np.ndarray) -> np.ndarray:
    """Horizontal distance from the runway-threshold frame origin, in metres."""
    return np.hypot(values[:, IDX["e"]], values[:, IDX["n"]])
