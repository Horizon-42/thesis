"""Greedy piecewise-straight fit with a bounded residual — the altitude and speed readings share it.

From each piece's first row, the piece is extended as far as the least-squares line through its
rows keeps every residual within the tolerance (galloping, then bisection, on the end row).
Pieces tile the rows without overlap; a piece has at least two rows unless it is the last row
left over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Piece:
    start: int          # first row
    stop: int           # one past the last row
    slope: float        # dy/dx of the least-squares line
    intercept: float    # y at x = 0 of that line

    @property
    def rows(self) -> int:
        return self.stop - self.start


def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) == 1:
        return 0.0, float(y[0])
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    denominator = float(dx @ dx)
    slope = 0.0 if denominator == 0.0 else float(dx @ (y - ym)) / denominator
    return slope, float(ym - slope * xm)


def _fits(x: np.ndarray, y: np.ndarray, start: int, stop: int, tolerance: float) -> bool:
    slope, intercept = _line(x[start:stop], y[start:stop])
    return bool(np.max(np.abs(y[start:stop] - (slope * x[start:stop] + intercept))) <= tolerance)


def fit_pieces(x: np.ndarray, y: np.ndarray, tolerance: float) -> list[Piece]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must be non-empty and aligned")
    if np.any(np.diff(x) < 0):
        raise ValueError("x must not decrease")
    pieces: list[Piece] = []
    start, n = 0, len(x)
    while start < n:
        if n - start <= 2:
            stop = n
        else:
            good, step = start + 2, 1
            while good + step <= n and _fits(x, y, start, good + step, tolerance):
                good += step
                step *= 2
            bad = min(good + step, n + 1)
            while bad - good > 1:                 # good fits, bad does not (or is past the end)
                middle = (good + bad) // 2
                if _fits(x, y, start, middle, tolerance):
                    good = middle
                else:
                    bad = middle
            stop = good
            if n - stop == 1:                     # never leave a single row behind
                stop = n
        slope, intercept = _line(x[start:stop], y[start:stop])
        pieces.append(Piece(start, stop, slope, intercept))
        start = stop
    return pieces


def moving_average(values: np.ndarray, window_rows: int) -> np.ndarray:
    """Centred moving average over ``window_rows`` (made odd), edges padded with the end values."""
    values = np.asarray(values, dtype=np.float64)
    window = max(1, int(window_rows) | 1)
    if window == 1 or len(values) == 0:
        return values.copy()
    half = window // 2
    padded = np.concatenate((np.full(half, values[0]), values, np.full(half, values[-1])))
    kernel = np.full(window, 1.0 / window)
    return np.convolve(padded, kernel, mode="valid")
