"""Point-mass kinematics shared by the evaluator and the modeling tree (stdlib only).

The dynamics model (``aerodynamic_model/casadi_simulator.py``) rotates the velocity
vector with two equations,

    gamma_dot = g (n cos mu - cos gamma) / V
    psi_dot   = g n sin mu / (V cos gamma)

so the load factor a flown trajectory IMPLIES is the inversion of those two: the lift
component along the flight-path normal is ``cos gamma + V gamma_dot / g`` and the
component across it is ``V cos gamma psi_dot / g``; their resultant, in units of the
weight, is ``n``. ``evaluation.arrival`` uses it over a window of ADS-B samples to
anchor the observed baseline's speed gate on the load factor the flight actually flew.
(``ts_transformer/flyability.py`` still carries its own per-sample copy of the same
two lines -- it also needs the bank angle; folding it onto this function is a listed
follow-up, deferred while a campaign runs from that tree.)
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import Sequence

from aircraft.aero_params import GRAVITY_M_S2


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Ordinary-least-squares slope of ``ys`` against ``xs``.

    Raises on fewer than two samples or on a degenerate abscissa; the callers hold
    strictly increasing timestamps, so neither can happen from a validated record.
    (``flight_scenarios.start_state._slope`` is the same operation returning 0.0 on a
    degenerate abscissa -- a second copy listed for folding onto this one.)
    """
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("linear_slope needs at least two paired samples")
    x_mean = fmean(xs)
    y_mean = fmean(ys)
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("linear_slope needs at least two distinct x values")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sxx


def load_factor_from_rates(
    speed_ms: float,
    gamma_rad: float,
    gamma_rate_rad_s: float,
    psi_rate_rad_s: float,
) -> float:
    """The load factor ``n`` the point-mass rotational equations imply.

    Signs cancel in the resultant, so a left or right turn and a pull-up or push-over
    read the same magnitude; a straight, steady path returns ``cos gamma`` (0.9986 on
    a 3 deg glidepath). The rotating-frame transport terms the geodetic model adds
    (``V cos gamma / R``, ~1e-5 rad/s) are 7e-5 in ``n`` and are not applied here.
    """
    normal = math.cos(gamma_rad) + speed_ms * gamma_rate_rad_s / GRAVITY_M_S2
    lateral = speed_ms * math.cos(gamma_rad) * psi_rate_rad_s / GRAVITY_M_S2
    return math.hypot(normal, lateral)
