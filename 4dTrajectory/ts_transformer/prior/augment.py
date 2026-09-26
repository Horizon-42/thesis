"""Augmented starts (post-training design §4): a train-day flight's start moved to a situation like it that is not in
the data — one rigid change applied alike to the observed rows the prior reads and to the executor's state it flies from.

- **rotated about the airport** by δ (compass degrees, clockwise): each position keeps its distance to the airport
  reference point (the arrival slice's entry ring is centred there) and every track turns with it, so the flight still
  heads where it headed relative to the airport;
- **raised** by Δh (all rows and the state);
- **sped up** by 1 + κ: the airspeed scales with the path angle kept, and the observed rows stretch by the same factor
  about the first predicted step's row — positions and heights alike, so their 2 s spacing still matches the speed, the
  climb the prior reads from them (height differences between rows) scales like the state's, and row 0 moves along its
  path by up to κ × its distance flown (~100 m at 5 %).

The flight's airport, day and times stay (its landing context and the airport's landing direction are the source
flight's), and so do its type, mass and dynamics. Drawn uniformly within `LIMITS` (user 2026-09-25). Torch-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.prior.scene import N_LOOK


@dataclass(frozen=True)
class Augmentation:
    rotation_deg: float        # compass, clockwise
    altitude_m: float
    speed_scale: float         # 1 + κ


@dataclass(frozen=True)
class Limits:
    """The half-widths of the uniform draws (design §4.2, §8)."""

    rotation_deg: float = 15.0
    altitude_m: float = 150.0
    speed_fraction: float = 0.05


LIMITS = Limits()
#: An augmented start's time limit from the first predicted step: its source flight's observed remaining time × this
#: (design §4.5; a real start's is the executor spec's ``timeout_factor``, 1.5).
TIMEOUT_FACTOR = 2.0


def draw(rng: np.random.Generator, limits: Limits = LIMITS) -> Augmentation:
    return Augmentation(rotation_deg=float(rng.uniform(-limits.rotation_deg, limits.rotation_deg)),
                        altitude_m=float(rng.uniform(-limits.altitude_m, limits.altitude_m)),
                        speed_scale=1.0 + float(rng.uniform(-limits.speed_fraction, limits.speed_fraction)))


def rotate(e: np.ndarray, n: np.ndarray, rotation_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Positions about the airport frame's origin turned ``rotation_deg`` clockwise (compass): a bearing β becomes β + δ."""
    turn = math.radians(rotation_deg)
    c, s = math.cos(turn), math.sin(turn)
    e, n = np.asarray(e, dtype=np.float64), np.asarray(n, dtype=np.float64)
    return e * c + n * s, n * c - e * s


def augment_signals(signals: FlightSignals, augmentation: Augmentation) -> FlightSignals:
    """The flight's rows moved (module docstring): rotated, raised, stretched about the first predicted step's row
    (positions and heights), tracks turned, speeds scaled; the times, the day and the identity kept."""
    e, n = rotate(signals.e_m, signals.n_m, augmentation.rotation_deg)
    k = augmentation.speed_scale
    e = e[N_LOOK] + k * (e - e[N_LOOK])
    n = n[N_LOOK] + k * (n - n[N_LOOK])
    h = np.asarray(signals.altitude_m, dtype=np.float64)
    h = h[N_LOOK] + augmentation.altitude_m + k * (h - h[N_LOOK])
    return replace(signals, e_m=e, n_m=n, altitude_m=h,
                   track_deg=np.asarray(signals.track_deg) + augmentation.rotation_deg,
                   ground_speed_mps=np.asarray(signals.ground_speed_mps) * k,
                   vertical_rate_mps=np.asarray(signals.vertical_rate_mps) * k)


def augment_state(lat_deg: float, lon_deg: float, altitude_m: float, speed_mps: float, psi_rad: float,
                  geometry: AirportGeometry, augmentation: Augmentation) -> tuple[float, float, float, float, float]:
    """The executor's state at the first predicted step moved alike: ``(lat, lon, altitude, airspeed, ψ)`` — ψ in the
    dynamics' math convention (east 0, counter-clockwise), so a clockwise turn of δ takes δ off it. The row is the
    stretch's centre, so the stretch leaves its position."""
    e, n = geometry.frame.horizontal_from_latlon(lat_deg, lon_deg)
    e, n = rotate(np.float64(e), np.float64(n), augmentation.rotation_deg)
    lat, lon = geometry.frame.latlon_from_horizontal(float(e), float(n))
    return (float(lat), float(lon), altitude_m + augmentation.altitude_m, speed_mps * augmentation.speed_scale,
            psi_rad - math.radians(augmentation.rotation_deg))
