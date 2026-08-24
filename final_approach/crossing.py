"""Shared threshold-plane crossing interpolation: one fraction, one channel blend.

A sampled trajectory crosses the threshold plane between two samples, and every
resolver in this project answers it with the SAME two-point operation: a fraction
along the bracketing segment from the signed along-track coordinates, then a
linear blend of each channel at that fraction (angles blended on the shortest
arc). The harvest's direct threshold bracket, the evaluator's computed-record
crossing, and the ts dataset's supervision truncation each used to hand-roll it —
three copies of one formula, each one edit away from disagreeing.

Pure stdlib on purpose (this package's charter): callers bring their own
projection and their own channel mappings.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

# ── The crossing-span marker: a record says WHERE its threshold crossing lives ──
#
# Serialized under record ``source["crossing_span"]`` by observed-record producers
# and validated here — the ONE schema both the producer side (flight_scenarios)
# and the reading side (evaluation) import, same seam pattern as
# ``event_contract.validate_event``. Computed records never carry one: their
# crossing is derived by the evaluator from the raw states, so the artifact under
# test cannot author the quantity it is graded on.
CROSSING_SPAN_KEY = "crossing_span"
MEASURED_BRACKET_KIND = "measured_bracket"
FITTED_TAIL_KIND = "fitted_tail"


def validate_crossing_span(marker: Mapping[str, Any], state_count: int) -> str:
    """Validate one serialized marker against its record's state count; return kind.

    ``measured_bracket``: the instrument-selected direct bracket —
    ``left_index``/``left_index + 1`` must both be real states and ``fraction``
    must be a genuine interpolation, in (0, 1].
    ``fitted_tail``: states from ``start_index`` on are inferred; exactly one
    appended crossing row is the current contract, so ``start_index`` names the
    last state and at least one measured row must precede it.
    Extra keys (``v_source``, ``time_source``, …) are producer audit metadata and
    ride through untouched.
    """
    kind = marker.get("kind")
    if kind == MEASURED_BRACKET_KIND:
        left = marker.get("left_index")
        fraction = marker.get("fraction")
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or not 0 <= left < state_count - 1
        ):
            raise ValueError(
                f"crossing_span.left_index must name a bracketing state pair "
                f"inside {state_count} states, got {left!r}"
            )
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) <= 1.0
        ):
            raise ValueError(
                f"crossing_span.fraction must lie in (0, 1], got {fraction!r}"
            )
        return kind
    if kind == FITTED_TAIL_KIND:
        start = marker.get("start_index")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or not 1 <= start < state_count
        ):
            raise ValueError(
                f"crossing_span.start_index must leave at least one measured state "
                f"and name an existing one, got {start!r} of {state_count}"
            )
        if start != state_count - 1:
            raise ValueError(
                "crossing_span currently appends exactly one inferred crossing row; "
                f"start_index {start} of {state_count} states leaves "
                f"{state_count - start}"
            )
        return kind
    raise ValueError(f"crossing_span has unsupported kind {kind!r}")


def bracket_fraction(before_along_m: float, after_along_m: float) -> float:
    """Fraction of the segment at which the along-track coordinate reaches zero.

    Requires a genuine bracket — ``before <= 0 <= after`` with positive span —
    because a fraction computed from a non-bracketing segment is an
    extrapolation wearing interpolation's name.
    """
    span = after_along_m - before_along_m
    if not (before_along_m <= 0.0 <= after_along_m) or span <= 0.0:
        raise ValueError(
            f"segment [{before_along_m:.3f}, {after_along_m:.3f}] m does not "
            "bracket the threshold plane"
        )
    return -before_along_m / span


def interpolate_channels(
    before: Mapping[str, float],
    after: Mapping[str, float],
    fraction: float,
    *,
    keys: Sequence[str],
    angular_keys: Sequence[str] = (),
) -> dict[str, float]:
    """Blend the named channels linearly at ``fraction``; angles take the short arc.

    ``angular_keys`` must be a subset of ``keys``: an angle blended as a plain
    number reads a due-west heading reached on the other branch as a 2π swing.
    """
    missing_angular = set(angular_keys) - set(keys)
    if missing_angular:
        raise ValueError(f"angular keys {sorted(missing_angular)} are not in keys")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction {fraction} is outside [0, 1]")
    blended = {
        key: before[key] + (after[key] - before[key]) * fraction for key in keys
    }
    for key in angular_keys:
        blended[key] = before[key] + (
            math.remainder(after[key] - before[key], math.tau) * fraction
        )
    return blended
