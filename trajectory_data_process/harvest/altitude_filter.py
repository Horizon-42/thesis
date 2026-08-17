"""Read-time repair of ADS-B altitude outliers. A view, never a rewrite.

WHY THIS IS A LAYER AND NOT A SCRIPT
------------------------------------
``tracks/`` is the sensor reconstruction and stays byte-identical to what was broadcast:
``arrivals/manifest.json`` pins every source track by SHA-256, ``reclassify`` re-derives
assignment from those exact samples, and ``source_integrity`` counts rows against them.
Editing a track file in place invalidates all three at once -- the arrival roster stops
loading, and the manifest's own provenance becomes a claim about bytes that no longer
exist. So the repair happens where a stored track is READ INTO A DERIVED VIEW, and only
there: the observed CZML, the model-ready arrival slices, and the evaluation records.

WHAT AN OUTLIER LOOKS LIKE IN THIS DATA
--------------------------------------
A single state vector reports an altitude that is nowhere near the ones on either side of
it -- measured extremes in the current harvest are 20 147 m between neighbours at 724 m,
and 35 189 m between neighbours at 556 m. Rendered, that is the needle-like vertical peak
this filter exists to remove; fitted, it drags a least-squares final-approach line with it.

The detector is therefore a robust-baseline test, not a jump test. Jump tests attribute
one bad sample to three (the outlier fails, and so do both of its neighbours, whose chord
runs through it): measured on this harvest a chord test reported 363 runs of exactly three
where the truth was 363 isolated samples.

THE POLICY, AND WHY THESE NUMBERS
---------------------------------
A sample is an outlier when BOTH hold:

  1. ``|alt - median(centred window)| > min_deviation_m``.  Over all 20 851 436 assigned
     samples the residual against this baseline is < 25 m for 20 847 051 of them, 3 625
     fall in [25, 50) -- the 25 ft (7.62 m) and 100 ft (30.5 m) reporting lattices plus
     real vertical motion -- and only 189 exceed 50 m.  100 m sits at 2x the largest
     residual genuine flight produces and 13x the finer quantum.
  2. ``|deviation| > max_vertical_rate_m_s * min(adjacent time gap)``.  An excursion that
     is gone by the next sample had to be flown to and back; at 25 m/s (4 900 fpm) over a
     0.5 s gap that is 12.5 m, so this bound is slack for a needle and decisive across a
     coverage gap.  It is what separates the two cases, and it is not cosmetic: it spares
     10 real descents that stepped 107-160 m across 9-14 s gaps in reception, every one
     of which a bare deviation threshold repairs into a lie.

Together they flag 561 samples in 451 of 44 622 assigned tracks (0.0027 % of samples),
479 of them isolated, the longest run six.

REPAIR REPLACES THE ALTITUDE, IT NEVER DROPS THE SAMPLE
-------------------------------------------------------
Sample INDICES are load-bearing across the harvest: ``landing_sample_index``,
``first_sample_index``/``last_sample_index`` in the arrival roster, the threshold event's
``source_sample_range``, and the ``reported_ground_speeds_m_s`` parallel array all index
this array. Deleting a row silently renumbers every one of them. Only ``samples[i][3]``
changes here -- count, times and horizontal positions are exactly what the receiver
reported -- so every index contract survives untouched.

The replacement is a linear interpolation in TIME between the nearest retained samples on
either side. An outlier at the very start or end of a track has retained samples on one
side only; there it holds the nearest retained altitude rather than extrapolating a rate
off data that is itself the thing under suspicion. Both cases are labelled in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FILTER_SCHEMA_VERSION = "altitude-outlier-filter-v1"

INTERPOLATED = "interpolated"
HELD = "held"


@dataclass(frozen=True)
class AltitudePolicy:
    """The one place the outlier criterion is defined. See the module docstring."""

    half_window: int = 2
    min_deviation_m: float = 100.0
    max_vertical_rate_m_s: float = 25.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "half_window": self.half_window,
            "min_deviation_m": self.min_deviation_m,
            "max_vertical_rate_m_s": self.max_vertical_rate_m_s,
        }


DEFAULT_POLICY = AltitudePolicy()


@dataclass(frozen=True)
class AltitudeOutlier:
    """One repaired sample, with everything needed to audit the decision."""

    index: int
    time_offset_s: float
    observed_alt_m: float
    replacement_alt_m: float
    correction_m: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "time_offset_s": self.time_offset_s,
            "observed_alt_m": self.observed_alt_m,
            "replacement_alt_m": self.replacement_alt_m,
            "correction_m": round(self.correction_m, 1),
            "method": self.method,
        }


@dataclass(frozen=True)
class FilteredSamples:
    """The derived sample array plus the audit trail that produced it."""

    samples: list[list[float]]
    outliers: tuple[AltitudeOutlier, ...]
    policy: AltitudePolicy

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": FILTER_SCHEMA_VERSION,
            "policy": self.policy.to_dict(),
            "outlier_count": len(self.outliers),
            "outliers": [outlier.to_dict() for outlier in self.outliers],
        }


def detect_altitude_outliers(
    samples: list[list[float]], *, policy: AltitudePolicy = DEFAULT_POLICY
) -> tuple[int, ...]:
    """Indices whose altitude the sensor cannot have measured. Pure, no repair."""
    count = len(samples)
    flagged: list[int] = []
    for index in range(count):
        low = max(0, index - policy.half_window)
        high = min(count, index + policy.half_window + 1)
        deviation = samples[index][3] - _median(
            [samples[j][3] for j in range(low, high)]
        )
        if abs(deviation) <= policy.min_deviation_m:
            continue
        if abs(deviation) <= policy.max_vertical_rate_m_s * _tightest_gap_s(samples, index):
            continue
        flagged.append(index)
    return tuple(flagged)


def filter_altitude_outliers(
    samples: list[list[float]], *, policy: AltitudePolicy = DEFAULT_POLICY
) -> FilteredSamples:
    """Repair every detected outlier's altitude; keep the array's length and timing.

    Nothing is edited in place. The result is a new array whose repaired rows are new
    lists and whose untouched rows are the caller's own -- a full copy of every sample in
    a 21-million-row harvest to rewrite 561 of them would be the expensive half of this
    module.
    """
    flagged = detect_altitude_outliers(samples, policy=policy)
    filtered = list(samples)
    if not flagged:
        return FilteredSamples(samples=filtered, outliers=(), policy=policy)

    rejected = set(flagged)
    outliers: list[AltitudeOutlier] = []
    for index in flagged:
        left = _nearest_retained(rejected, index, len(samples), step=-1)
        right = _nearest_retained(rejected, index, len(samples), step=1)
        replacement, method = _replacement(samples, index, left, right)
        repaired = list(samples[index])
        repaired[3] = replacement
        filtered[index] = repaired
        outliers.append(
            AltitudeOutlier(
                index=index,
                time_offset_s=float(samples[index][0]),
                observed_alt_m=float(samples[index][3]),
                replacement_alt_m=replacement,
                correction_m=replacement - float(samples[index][3]),
                method=method,
            )
        )
    return FilteredSamples(samples=filtered, outliers=tuple(outliers), policy=policy)


def filtered_track(
    track: dict[str, Any], *, policy: AltitudePolicy = DEFAULT_POLICY
) -> dict[str, Any]:
    """One stored track as a derived view: same record, repaired altitudes.

    The input dict is never mutated -- callers that also hash or re-serialise the source
    record keep the bytes they read.
    """
    filtered = filter_altitude_outliers(track["samples"], policy=policy)
    view = dict(track)
    view["samples"] = filtered.samples
    view["altitude_filter"] = filtered.provenance()
    return view


def outlier_count(track: dict[str, Any]) -> int:
    """How many samples the filter replaced in a view produced by :func:`filtered_track`."""
    return int(track["altitude_filter"]["outlier_count"])


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _tightest_gap_s(samples: list[list[float]], index: int) -> float:
    """The shorter of the two adjacent sample intervals -- the time the excursion had."""
    gaps = []
    if index > 0:
        gaps.append(float(samples[index][0]) - float(samples[index - 1][0]))
    if index + 1 < len(samples):
        gaps.append(float(samples[index + 1][0]) - float(samples[index][0]))
    tightest = min(gaps)
    if tightest <= 0.0:
        raise ValueError(
            f"sample times must increase; sample {index} sits {tightest:g} s from its "
            "neighbour"
        )
    return tightest


def _nearest_retained(
    rejected: set[int], index: int, count: int, *, step: int
) -> int | None:
    candidate = index + step
    while 0 <= candidate < count:
        if candidate not in rejected:
            return candidate
        candidate += step
    return None


def _replacement(
    samples: list[list[float]], index: int, left: int | None, right: int | None
) -> tuple[float, str]:
    if left is not None and right is not None:
        t_left, t_right = float(samples[left][0]), float(samples[right][0])
        fraction = (float(samples[index][0]) - t_left) / (t_right - t_left)
        alt_left, alt_right = float(samples[left][3]), float(samples[right][3])
        return round(alt_left + (alt_right - alt_left) * fraction, 1), INTERPOLATED
    edge = left if left is not None else right
    if edge is None:
        raise ValueError(
            f"sample {index} has no retained neighbour to repair from; every sample in "
            "the track is an altitude outlier"
        )
    return round(float(samples[edge][3]), 1), HELD
