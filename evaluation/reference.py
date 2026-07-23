"""Compare a trajectory record against its observed reference (ADS-B track).

The reference is itself a record file (the observed track expressed in the same
contract: derived-kinematics states, EMPTY controls) pointed at by the record's
``reference_file``, resolved relative to the record's own directory. Two
comparisons:

  * **flight-time delta** — ``optimized − observed`` (negative = the optimizer
    is faster than the flight as flown, over the same start→target journey; the
    scenario's initial state is derived from the track start, so the durations
    are directly comparable).
  * **path deviation** — both paths resampled at ``N_RESAMPLE`` fractions of
    their OWN horizontal arc length, then compared point-by-point (horizontal
    great-circle distance + signed altitude difference). This is a path-SHAPE
    comparison: the two trajectories fly different speed profiles by design
    (the optimizer minimizes time), so matching by time would conflate timing
    with geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geokit import haversine_m

from evaluation.records import TrajectoryRecord, load_record
from evaluation.stats import magnitude_spread, signed_spread

N_RESAMPLE = 101


@dataclass(frozen=True)
class ReferenceComparison:
    """One record vs its observed reference (see the module docstring).

    The two dict fields summarize the per-fraction offsets between the
    arc-length-matched paths, in metres, using the ``evaluation.stats`` dict
    shapes (keys documented there, once).
    """

    reference_file: str
    reference_flight_time_s: float
    flight_time_delta_s: float                # optimized − observed, s (negative = faster)
    path_lateral_m: dict[str, float]          # stats.magnitude_spread of |horizontal offset|
    path_vertical_m: dict[str, float]         # stats.signed_spread of alt diff (+ = optimized higher)


def load_reference(record: TrajectoryRecord) -> TrajectoryRecord:
    """Load the record's reference, resolving the pointer against its directory."""
    if record.reference_file is None:
        raise ValueError(f"record {record.path or record.source} has no reference_file")
    base = record.path.parent if record.path is not None else Path(".")
    reference = load_record(base / record.reference_file)
    identity_keys = ("id", "icao24", "landing_time_utc", "runway")
    mismatches = [
        key
        for key in identity_keys
        if record.source.get(key) != reference.source.get(key)
    ]
    if mismatches:
        ours = {key: record.source.get(key) for key in identity_keys}
        theirs = {key: reference.source.get(key) for key in identity_keys}
        raise ValueError(
            f"record/reference flight identity mismatch in {mismatches}: "
            f"record={ours}, reference={theirs}"
        )
    return reference


def horizontal_arc_length_m(states: list[dict[str, float]]) -> float:
    """Total along-track horizontal length (metres) of a record's state samples.

    Zero for empty, single-sample, or stationary paths — the degenerate cases
    that cannot be arc-length matched (``resample_by_arc_length`` raises on
    them); callers use this to SKIP such paths instead of crashing on them.
    """
    return _cumulative_arc_m([(s["lat"], s["lon"]) for s in states])[-1]


def _cumulative_arc_m(points: list[tuple[float, float]]) -> list[float]:
    """Per-point along-track "mileage markers": ``out[i]`` = horizontal metres from
    ``points[0]`` to ``points[i]`` (``[0.0]`` for empty/single-point input)."""
    cumulative = [0.0]
    for (lat_a, lon_a), (lat_b, lon_b) in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + haversine_m(lat_a, lon_a, lat_b, lon_b))
    return cumulative


def compare_to_reference(
    record: TrajectoryRecord,
    reference: TrajectoryRecord,
    *,
    n: int = N_RESAMPLE,
) -> ReferenceComparison:
    """Compare a SOLVED record's path + duration against its observed reference."""
    ours = resample_by_arc_length(record.states, n)
    theirs = resample_by_arc_length(reference.states, n)
    lateral = [haversine_m(a[0], a[1], b[0], b[1]) for a, b in zip(ours, theirs)]
    vertical = [a[2] - b[2] for a, b in zip(ours, theirs)]
    return ReferenceComparison(
        reference_file=record.reference_file,
        reference_flight_time_s=float(reference.final_time_s),
        flight_time_delta_s=float(record.final_time_s) - float(reference.final_time_s),
        path_lateral_m=magnitude_spread(lateral),
        path_vertical_m=signed_spread(vertical),
    )


def resample_by_arc_length(
    states: list[dict[str, float]],
    n: int = N_RESAMPLE,
) -> list[tuple[float, float, float]]:
    """``n`` points ``(lat, lon, alt)`` evenly spaced along the path's horizontal arc.

    ``states`` are record state samples (the ``records.py`` contract) — only
    their ``lat``/``lon`` (degrees) and ``alt`` (metres) keys are read. Each
    output tuple is ``(lat_deg, lon_deg, alt_m)`` in that order.

    The ``n`` outputs are NOT the input points: they are NEW points found by walking
    equal horizontal distances (0, total/(n-1), 2·total/(n-1), …, total) along the
    track and, at each target distance, linearly interpolating ``(lat, lon, alt)``
    between the two original points that bracket it. Only the first and last outputs
    coincide with original points; every other output is an interpolated intermediate.

    Mechanics: ``cumulative[i]`` is the along-track distance from the start to input
    point ``i`` (consecutive ``haversine_m`` hops — a per-input "mileage marker", same
    length as ``states``, NOT the result). For each target distance we advance to the
    bracketing segment and interpolate at its ``fraction`` (0 = segment start, 1 = end).
    Arc length is HORIZONTAL only (altitude excluded from the distance), so the spacing
    is uniform in ground distance; ``alt`` is still interpolated at each resampled point.

    Raises on a path with zero horizontal extent (nothing to parametrize by)
    and on ``n < 2`` (the endpoints are always included).
    """
    if n < 2:
        raise ValueError("n must be >= 2 (the endpoints are always included)")
    points = [(s["lat"], s["lon"], s["alt"]) for s in states]
    cumulative = _cumulative_arc_m([(p[0], p[1]) for p in points])
    total = cumulative[-1]
    if total <= 0.0:
        raise ValueError("cannot resample a path with zero horizontal length")

    resampled: list[tuple[float, float, float]] = []
    segment = 0
    for k in range(n):
        target_s = total * k / (n - 1)
        while segment < len(points) - 2 and cumulative[segment + 1] < target_s:
            segment += 1
        span = cumulative[segment + 1] - cumulative[segment]
        fraction = (target_s - cumulative[segment]) / span if span > 0.0 else 0.0
        a, b = points[segment], points[segment + 1]
        resampled.append(tuple(a[i] + (b[i] - a[i]) * fraction for i in range(3)))
    return resampled
