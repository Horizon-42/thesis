"""Small aggregate helpers shared by the metric modules (stdlib only)."""

from __future__ import annotations

import math
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q`` (0..1) percentile with linear interpolation between order statistics."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def magnitude_spread(values: Sequence[float]) -> dict[str, float] | None:
    """Spread of non-negative magnitudes; ``None`` for an empty sequence.

    Returns ``{"mean", "p95", "max"}`` — plain summary statistics of ``values``,
    all in the input's unit. This dict shape is what report fields documented as
    "magnitude spread" carry.
    """
    if not values:
        return None
    return {"mean": mean(values), "p95": percentile(values, 0.95), "max": max(values)}


def signed_spread(values: Sequence[float]) -> dict[str, float] | None:
    """Spread of signed values; ``None`` for an empty sequence.

    Returns ``{"mean_signed", "mean_abs", "p95_abs", "max_abs"}`` —
    ``mean_signed`` keeps the sign (cancellation shows a bias), the ``*_abs``
    entries are statistics of ``|value|``. All in the input's unit; this dict
    shape is what report fields documented as "signed spread" carry.
    """
    if not values:
        return None
    magnitudes = [abs(v) for v in values]
    return {
        "mean_signed": mean(values),
        "mean_abs": mean(magnitudes),
        "p95_abs": percentile(magnitudes, 0.95),
        "max_abs": max(magnitudes),
    }
