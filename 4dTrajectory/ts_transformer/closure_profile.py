"""Speed and height profiles for the closure decoder (scene design §五 P1.b).

The closure geometry gives WHERE the aircraft flies; a profile along that path gives
WHEN and how high. The profiles here are parametrised by a few knots over the path's
progress fraction ``f = s / L`` so that a decoder head can regress them and the P1.b
oracle can measure what each parametrisation can reach on the truth path itself
(geometry error zero — the timing floor of the design).

* Speed: piecewise-linear SLOWNESS ``w(f) = 1 / v(f)`` (s/m) at ``K + 1`` knots, because
  time along the path is then LINEAR in the knots, ``t(s) = L · ∫₀^{s/L} w(f) df``, and a
  fit to the truth times is a bounded linear least-squares problem (``fit_slowness_knots``;
  bounds keep the speed inside ``[SPEED_MIN_MPS, SPEED_MAX_MPS]``). The duration is the
  integral's last value, so a decoder that predicts the knots predicts the duration with
  them; ``scale_to_duration`` rescales any time profile onto a given duration (the "naive
  shape + duration head" reading).
* Height: piecewise-linear in ``f`` at ``K + 1`` knots, fitted by linear least squares
  with the LAST knot pinned to the threshold height (``fit_height_knots``: an unpinned fit
  trades the landing for the interior and ends up to 33 m below the threshold on a
  quarter of the flights); the geometry's own profile (linear to the glidepath at the
  join, then the glidepath) is the alternative the oracle compares it with.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from geometric_metrics import cumulative_arc_m

SPEED_MIN_MPS = 40.0
SPEED_MAX_MPS = 160.0


def progress(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """``(f, L)``: each row's fraction of the horizontal arc length, and the length."""
    s = cumulative_arc_m(xy)
    if s[-1] <= 0.0:
        raise ValueError("a path of zero horizontal length has no progress fraction")
    return s / s[-1], float(s[-1])


def hat_basis(f: np.ndarray, knots: int) -> np.ndarray:
    """``[N, knots + 1]`` piecewise-linear hat functions on ``knots`` equal intervals of [0, 1]."""
    centres = np.linspace(0.0, 1.0, knots + 1)
    width = 1.0 / knots
    return np.clip(1.0 - np.abs(f[:, None] - centres[None, :]) / width, 0.0, None)


def integrated_hat_basis(f: np.ndarray, knots: int) -> np.ndarray:
    """``[N, knots + 1]`` with ``B[i, k] = ∫₀^{f_i} hat_k``, so ``B @ w`` is the integral
    of the piecewise-linear function with knot values ``w`` — time when ``w`` is slowness
    per unit of progress (i.e. ``L · slowness``)."""
    centres = np.linspace(0.0, 1.0, knots + 1)
    width = 1.0 / knots
    x = f[:, None]
    # Rising half: hat = (t − lo) / width on [centre − width, centre]; its integral up to
    # x, less the part below f = 0 (the first hat's rising half lies outside [0, 1]).
    a = np.clip(x - (centres - width), 0.0, width)
    a0 = np.clip(-(centres - width), 0.0, width)
    rising = (a ** 2 - a0 ** 2) / (2.0 * width)
    # Falling half: hat = (hi − t) / width on [centre, centre + width], capped at f = 1
    # (the last hat's falling half lies outside [0, 1]).
    b = np.clip(x - centres, 0.0, np.minimum(width, 1.0 - centres))
    falling = b - b ** 2 / (2.0 * width)
    return rising + falling


def times_from_slowness(f: np.ndarray, length_m: float, slowness_knots: np.ndarray) -> np.ndarray:
    """Time at every progress fraction from slowness knots (s/m)."""
    return length_m * (integrated_hat_basis(f, len(slowness_knots) - 1) @ slowness_knots)


def speed_from_slowness(f: np.ndarray, slowness_knots: np.ndarray) -> np.ndarray:
    return 1.0 / (hat_basis(f, len(slowness_knots) - 1) @ slowness_knots)


def fit_slowness_knots(f: np.ndarray, length_m: float, t: np.ndarray, knots: int) -> np.ndarray:
    """Slowness knots whose integral best matches the truth times (bounded linear least
    squares; ``t[0] == 0`` at ``f[0] == 0``)."""
    basis = length_m * integrated_hat_basis(f, knots)
    bounds = (1.0 / SPEED_MAX_MPS, 1.0 / SPEED_MIN_MPS)
    return lsq_linear(basis, t, bounds=bounds, method="bvls").x


def scale_to_duration(t: np.ndarray, duration_s: float) -> np.ndarray:
    """The same time SHAPE stretched onto ``duration_s`` (a duration head over a fixed profile)."""
    if t[-1] <= 0.0:
        raise ValueError("a time profile of zero duration has no shape to stretch")
    return t * (duration_s / t[-1])


def fit_height_knots(f: np.ndarray, u: np.ndarray, knots: int, *, terminal_m: float = 0.0) -> np.ndarray:
    """Height knots (m) whose piecewise-linear profile best matches ``u`` in least squares,
    the last knot pinned to ``terminal_m`` (the threshold: 0 in the threshold chart)."""
    basis = hat_basis(f, knots)
    free = np.linalg.lstsq(basis[:, :-1], u - basis[:, -1] * terminal_m, rcond=None)[0]
    return np.append(free, terminal_m)


def height_from_knots(f: np.ndarray, height_knots: np.ndarray) -> np.ndarray:
    return hat_basis(f, len(height_knots) - 1) @ height_knots
