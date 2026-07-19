"""What the network actually sees: a local ENU frame anchored at the runway threshold.

The evaluation contract's state is ``(lat, lon, alt, V, psi, gamma, m)`` — degrees, metres,
and two angles in radians. That is a poor thing to regress directly:

- ``lat``/``lon`` are ~1e2 degrees of offset carrying ~1e-5 degree signal; float32 attention
  on raw degrees wastes its dynamic range on the airport's absolute position.
- ``psi`` wraps at +/-pi. A model regressing it will average 179deg and -179deg to 0deg —
  pointing the aircraft backwards — right where the turn onto final happens.
- ``m`` is not observable from ADS-B at all. It is carried, never predicted.

So the channels are **position and velocity in metres and metres/second**, in a local
ENU tangent plane whose origin is the runway threshold:

    e, n, u     east / north / up, metres from the threshold
    ve, vn, vu  velocity components, metres/second

``psi`` and ``gamma`` then fall out of the velocity components on the way back
(:func:`states_from_channels`) with the right convention *by construction* —
``psi = atan2(vn, ve)`` is exactly the modeling layer's math-ENU heading, so there is no
place left to accidentally substitute a compass bearing. The wrap problem disappears with
it: ``(ve, vn)`` is continuous across the branch cut that ``psi`` is not.

**Projection.** ``east = (lon - lon0) * metres_per_deg_lon(lat0)``,
``north = (lat - lat0) * METRES_PER_DEG_LAT`` — the same local flat projection
``flight_scenarios/start_state.py`` fits velocities in, and the same frame
``approach_constraints`` anchors at ``ltp_ne = (0, 0)``. Over a 25 km entry ring the flat
approximation costs well under a metre, and matching the existing frame matters more than
the last decimetre: a different projection here would show up as apparent model error when
compared against records built by the other packages.

**Ordering trap.** CZML-input waypoints are ``[t, lon, lat, alt]`` — *lon before lat* — while
every state dict downstream is ``lat`` before ``lon``. This module is the only place the two
orders meet; nothing else should be indexing a waypoint by position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from aerodynamic_model.common import GeodeticState

# The channel contract. Order is load-bearing — it indexes every tensor, the normalizer's
# per-channel statistics, and the checkpoint. Changing it invalidates trained checkpoints.
CHANNELS: tuple[str, ...] = ("e", "n", "u", "ve", "vn", "vu")

# Column indices, so nothing downstream hard-codes a number.
IDX = {name: i for i, name in enumerate(CHANNELS)}
POSITION_IDX = (IDX["e"], IDX["n"], IDX["u"])


@dataclass(frozen=True)
class Frame:
    """A local ENU tangent plane anchored at a runway threshold.

    ``alt0`` is the threshold's altitude (MSL metres), so ``u`` is height above the
    threshold rather than above the ellipsoid — which keeps the vertical channel centred
    near zero at the one point the evaluation gates care about.
    """

    lat0: float
    lon0: float
    alt0: float

    @property
    def m_per_deg_lon(self) -> float:
        return metres_per_deg_lon(self.lat0)

    def latlon_from_en(self, e: float, n: float) -> tuple[float, float]:
        """``(east, north)`` metres -> ``(lat, lon)`` degrees: the inverse projection.

        The single home of the inverse formula — :func:`states_from_channels` and
        ``synthetic.py``'s waypoint generator both go through it, so a projection change
        cannot land on one side only and read as model error on the other.
        """
        return self.lat0 + n / METRES_PER_DEG_LAT, self.lon0 + e / self.m_per_deg_lon


def frame_for_state(state: GeodeticState) -> Frame:
    """The ENU frame anchored at a target state — normally the runway threshold.

    Pair with ``flight_scenarios.threshold_target_state``, which builds that target from
    the published threshold (position at threshold-crossing height, runway heading, Vref).
    """
    return Frame(lat0=state.latitude, lon0=state.longitude, alt0=state.altitude)


def channels_from_states(
    samples: Sequence[tuple[float, GeodeticState]], frame: Frame
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
        times[i] = t
        out[i] = (
            (s.longitude - frame.lon0) * m_per_deg_lon,   # e
            (s.latitude - frame.lat0) * METRES_PER_DEG_LAT,  # n
            s.altitude - frame.alt0,                       # u
            ground_speed * math.cos(s.psi),                # ve
            ground_speed * math.sin(s.psi),                # vn
            s.V * math.sin(s.gamma),                       # vu
        )
    return times, out


def states_from_channels(
    times: np.ndarray, values: np.ndarray, frame: Frame, *, mass_kg: float
) -> list[tuple[float, GeodeticState]]:
    """``(times[N], channels[N, C])`` -> ``[(t, GeodeticState), ...]``.

    The exact inverse of :func:`channels_from_states`, and deliberately the same shape
    ``flight_scenarios.state_samples_from_track`` returns — which is what
    ``4dTrajectory/optimization/evaluation_export.reference_evaluation_record`` consumes.
    Producing that shape here means the evaluation record is emitted by the SAME function
    the optimizer's reference records go through, instead of this package hand-rolling a
    second copy of the JSON contract.

    ``m`` is carried from ``mass_kg``, not predicted: ADS-B never observed it, so the model
    was never given it and cannot return it. ``V`` is the TOTAL speed along the flight path
    (including the vertical component), matching the evaluation contract — not ground speed.
    """
    if len(times) != len(values):
        raise ValueError(f"times ({len(times)}) and values ({len(values)}) must align")

    states: list[tuple[float, GeodeticState]] = []
    for t, row in zip(times, values):
        e, n, u, ve, vn, vu = (float(v) for v in row)
        ground_speed = math.hypot(ve, vn)
        lat, lon = frame.latlon_from_en(e, n)
        states.append((float(t), GeodeticState(
            latitude=lat,
            longitude=lon,
            altitude=frame.alt0 + u,
            V=math.sqrt(ve * ve + vn * vn + vu * vu),
            # math-ENU: 0 = East, CCW toward North. atan2(vn, ve), NOT the compass
            # atan2(ve, vn) — see the module docstring.
            psi=math.atan2(vn, ve) if ground_speed > 0.0 else 0.0,
            gamma=math.atan2(vu, ground_speed) if ground_speed > 0.0 else 0.0,
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
        out[:, c] = np.interp(grid, times, values[:, c])
    return grid, out


def horizontal_distance_m(values: np.ndarray) -> np.ndarray:
    """Horizontal distance from the frame origin (the threshold) for each row, metres.

    Used to decide where a predicted approach reaches the runway — see
    ``forecast.truncate_at_threshold``.
    """
    return np.hypot(values[:, IDX["e"]], values[:, IDX["n"]])
