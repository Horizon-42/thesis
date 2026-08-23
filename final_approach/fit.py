"""Least-squares fit of a flown final-approach segment, extrapolated to the threshold.

WHY A STRAIGHT LINE IS THE RIGHT MODEL (not a convenience)
----------------------------------------------------------
A stabilised final approach IS geometrically straight in both planes: the aircraft
tracks the extended centreline and descends on a constant-angle glidepath. So the
functional form is given by the physics, and ordinary least squares on it is the
correct estimator -- not an approximation of a curve.

This is also why a kernel Gaussian process would be the WRONG tool here. A GP with an
RBF/Matern kernel is an interpolator: extrapolated beyond its data it reverts to the
prior mean with exploding variance, which would drag the crossing toward zero. You
reach for a GP when the functional form is unknown. Here it is known, and the fitted
glidepath coming out 3.02-3.11 deg across five airports is the evidence.

WHY THE FIT IS MANDATORY FOR ASSIGNMENT AND UNOBSERVED CROSSINGS
----------------------------------------------------------------
OpenSky's ``geoaltitude`` is quantised to 25 ft = 7.62 m (verified: all 482 distinct
altitudes in the KRDU set lie on that lattice). A fit supplies stable geometry for
runway assignment and is required when reception ends before the threshold. A valid
pair that physically brackets the threshold is handled by the harvest event producer
instead and does not run this fitter.

WHAT THIS MODULE DOES NOT DO
----------------------------
No thresholds, no ``established`` flag, no verdict. ``SegmentFit`` reports what
the data says; the harvest may use it for assignment or an unobserved crossing,
and evaluation later applies policy to the serialized event. ``assign_runway``
applies only a relative comparison. See the package docstring for why that
separation is load-bearing.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from final_approach.frame import Projected, RunwayFrame, TrackPoint

# Along-track window of the ESTABLISHED segment, metres (negative = before threshold).
#
# Outer edge -5000 m: on a 3 deg path that is ~900 ft above threshold elevation, just
# below the commonly used 1000 ft stabilised-approach reference. Reaching further out
# measurably biases the extrapolation HIGH (KRDU
# median crossing +5.43 m from -8000 m vs +3.66 m from -5000 m) because aircraft are
# still intercepting the glidepath from above out there.
#
# Inner edge -300 m: ~107 ft above threshold elevation, comfortably above flare
# initiation (~50 ft), so the fit never ingests the round-out. It also sits inside
# where ADS-B coverage typically stops (median last sample -325 m at KRDU), which is
# the truncation this fit exists to bridge.
#
# Shrinking the window further does not help: below ~3 km the baseline is too short
# to pin the threshold intercept reliably.
DEFAULT_WINDOW_M = (-5000.0, -300.0)

DEFAULT_MIN_SAMPLES = 8
DEFAULT_MIN_SPAN_M = 500.0

# How far the along-track coordinate may retreat between consecutive samples before the
# backward walk decides the aircraft was not on ONE inbound run (see _final_inbound_run).
# One 1 Hz sample advances ~77 m on approach, so 100 m absorbs GPS jitter and the slight
# along-track stall of a late centreline intercept without tolerating a real reversal.
#
# Public because the harvest walks FORWARD from this fit's last sample to find the
# closest observed support on the same pass, and that walk must accept exactly the
# jitter this one does -- two 100.0 literals would be one edit away from disagreeing.
INBOUND_TOLERANCE_M = 100.0

# OpenSky geometric altitude is encoded on a 25 ft lattice.  A robust residual
# scale may collapse to zero on a clean quantised descent, so half-distribution
# standard deviation of one lattice step is the minimum meaningful scale.
_ALTITUDE_QUANTUM_M = 25.0 * 0.3048
_ALTITUDE_RESIDUAL_SCALE_FLOOR_M = _ALTITUDE_QUANTUM_M / math.sqrt(12.0)
_MAD_SCALE_FACTOR = 1.482602218505602
# A fixed 3.03-scale gross-outlier cut. It is deliberately generous: ordinary
# quantisation scatter remains, while isolated kilometre-scale corruptions do not.
_ROBUST_RESIDUAL_Z_MAX = math.sqrt(9.21034037197618)
_MAX_ROBUST_SEED_SAMPLES = 64

# Empirical lateral residual floor. The closest-to-threshold 500 m seed must be much
# less curved than this, and earlier samples may join the straight final only while
# their robust residual remains within three such scales. The clean five-airport control
# maxes out at 3.62 m; 10.5 m deliberately leaves generous room for ADS-B position
# noise without allowing a base-to-final turn hundreds of metres wide to define the
# threshold intercept.
_LATERAL_RESIDUAL_FLOOR_M = 10.5


@dataclass(frozen=True)
class LineFit:
    """An OLS line ``y = intercept + slope * x`` evaluated at the threshold."""

    intercept: float
    slope: float
    rms_residual_m: float
    max_abs_residual_m: float


@dataclass(frozen=True)
class SegmentFit:
    """The established final-approach segment of ONE track against ONE runway.

    Facts only -- see the module docstring. ``cross_*`` is signed right-positive;
    ``median_abs_cross_m`` is the robust statistic ``assign_runway`` compares between
    runways (a misassigned parallel scores ~the runway separation, a correct one
    scores metres).
    """

    runway: str
    n_samples: int
    span_m: float
    window_m: tuple[float, float]
    first_sample_index: int
    last_sample_index: int

    cross: LineFit
    height: LineFit

    median_abs_cross_m: float
    nearest_sample_along_m: float
    along_progress_m: float
    rejected_sample_indices: tuple[int, ...]

    @property
    def cross_at_threshold_m(self) -> float:
        """Signed lateral offset at the threshold, right-positive."""
        return self.cross.intercept

    @property
    def height_at_threshold_m(self) -> float:
        """Height above threshold elevation at the threshold crossing."""
        return self.height.intercept

    @property
    def glidepath_deg(self) -> float:
        """Fitted descent angle. Positive = descending toward the threshold."""
        return math.degrees(math.atan(-self.height.slope))

    @property
    def approaching(self) -> bool:
        """True when the track moved TOWARD this threshold across the window.

        The one test that separates the two ends of the same runway. Both ends share
        one extended centreline, so cross-track offset -- the statistic that separates
        PARALLEL runways -- cannot tell them apart at all: a track flying 05L scores
        near-zero against 23R too. Direction of travel is what differs, and in this
        frame that is simply the sign of the along-track progression.
        """
        return self.along_progress_m > 0.0


def fit_line(xs: Sequence[float], ys: Sequence[float]) -> LineFit:
    """Fit one OLS line and retain only diagnostics serialized by the producer.

    Public because the harvest's threshold-event producer extrapolates the reported
    ground speed to the threshold with the SAME estimator the position components
    use — two OLS implementations would be one edit away from disagreeing. The
    ``*_m`` diagnostic names read as metres here; a caller fitting another quantity
    relabels them when it serializes.
    """
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    s_xx = sum((x - x_bar) ** 2 for x in xs)
    s_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    slope = s_xy / s_xx
    intercept = y_bar - slope * x_bar

    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in residuals)

    return LineFit(
        intercept=intercept,
        slope=slope,
        rms_residual_m=math.sqrt(sse / n),
        max_abs_residual_m=max(abs(r) for r in residuals),
    )


def _final_inbound_run(
    projected: Sequence[Projected], window_m: tuple[float, float]
) -> list[tuple[int, Projected]]:
    """The LAST contiguous stretch of one inbound approach inside ``window_m``.

    Selecting purely by along-track range is not enough, because a real arrival can
    occupy the same along-track band more than once: a downwind leg abeam the runway,
    vectoring, a go-around, or -- measured on the shipped KSJC data -- a track exported
    against the wrong runway end so that it contains the whole approach AND the landing
    roll. Those tracks are not monotonic in along-track (one had along ranging over
    -23.5 km to +18.7 km yet ending at +2.6 km), so a range filter mixed samples from a
    downwind leg into the fit and produced a median cross-track of 8.7 km -- a number
    with no physical meaning that would then have decided a runway assignment.

    So the run is walked BACKWARD from the last sample at or inside the window's inner
    edge, and stops as soon as the aircraft was closer to the threshold earlier (a
    reversal, i.e. a different pass) or had not yet reached the window's outer edge.
    What survives is one continuous inbound stretch, which is what "the established
    final approach" means.
    """
    inner = next(
        (i for i in range(len(projected) - 1, -1, -1) if projected[i].along_m <= window_m[1]),
        None,
    )
    if inner is None:
        return []
    run = [(inner, projected[inner])]
    # Compare every earlier point with the most negative along-track coordinate
    # already accepted in the suffix.  Comparing only adjacent points renews the
    # 100 m allowance on every sample: a steady 50 m/s reversal can then accumulate
    # for kilometres without ever tripping the old guard.
    suffix_min_along_m = projected[inner].along_m
    for i in range(inner - 1, -1, -1):
        if projected[i].along_m < window_m[0]:
            break
        if projected[i].along_m > suffix_min_along_m + INBOUND_TOLERANCE_M:
            break
        run.append((i, projected[i]))
        suffix_min_along_m = min(suffix_min_along_m, projected[i].along_m)
    run.reverse()
    return run


def _robust_line_residuals(
    xs: Sequence[float], ys: Sequence[float]
) -> tuple[float, float, list[float]]:
    """Return a bounded Theil-Sen seed and its centred robust residual scale."""
    seed_indices = _robust_seed_indices(len(xs))
    intercept, slope = _median_line(
        [xs[index] for index in seed_indices],
        [ys[index] for index in seed_indices],
    )
    residuals = [
        value - (intercept + slope * along)
        for along, value in zip(xs, ys)
    ]
    center = statistics.median(residuals)
    return intercept + center, slope, [value - center for value in residuals]


def _straight_final_suffix(
    indexed: list[tuple[int, Projected]],
    *,
    min_samples: int,
    min_span_m: float,
) -> list[tuple[int, Projected]]:
    """Return the final straight suffix after centreline capture.

    Along-track monotonicity identifies one physical pass, but an aircraft can keep
    moving inbound while turning from base onto final.  Seed the line from the
    closest-to-threshold suffix that is itself long enough to fit, require that seed
    to be laterally coherent, then extend it backward only while the same line still
    explains the source positions.  This is model-validity selection, not a runway
    or evaluation gate: a straight approach with a constant lateral offset survives.
    """
    if len(indexed) < min_samples:
        return []
    anchor = indexed[-1][1]
    seed_start = next(
        (
            index
            for index in range(len(indexed) - min_samples, -1, -1)
            if anchor.along_m - indexed[index][1].along_m >= min_span_m
        ),
        None,
    )
    if seed_start is None:
        return []

    seed = [point for _, point in indexed[seed_start:]]
    seed_xs = [point.along_m for point in seed]
    seed_crosses = [point.cross_m for point in seed]
    intercept, slope, residuals = _robust_line_residuals(
        seed_xs,
        seed_crosses,
    )
    seed_scale = _MAD_SCALE_FACTOR * statistics.median(
        abs(value) for value in residuals
    )
    if seed_scale > _LATERAL_RESIDUAL_FLOOR_M:
        return []

    residual_limit = _ROBUST_RESIDUAL_Z_MAX * max(
        seed_scale,
        _LATERAL_RESIDUAL_FLOOR_M,
    )
    start = seed_start
    for index in range(seed_start - 1, -1, -1):
        point = indexed[index][1]
        residual = point.cross_m - (intercept + slope * point.along_m)
        if abs(residual) > residual_limit:
            break
        start = index
    return indexed[start:]


def _median_line(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Return a deterministic Theil-Sen seed for gross-outlier rejection."""
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs) - 1)
        for j in range(i + 1, len(xs))
        if abs(xs[j] - xs[i]) > 1e-9
    ]
    if not slopes:
        raise ValueError("robust line needs distinct along-track coordinates")
    slope = statistics.median(slopes)
    intercept = statistics.median(
        y - slope * x for x, y in zip(xs, ys)
    )
    return intercept, slope


def _robust_seed_indices(sample_count: int) -> list[int]:
    """Return an evenly spaced, endpoint-preserving bounded seed roster.

    Exact Theil-Sen is quadratic.  Some real 5 km windows contain hundreds of
    aggregated receiver rows, so applying it to every row makes reclassification
    scale with the square of source density.  The seed only proposes a line; the
    residual gate still checks every source row before the final OLS fit.
    """
    if sample_count <= _MAX_ROBUST_SEED_SAMPLES:
        return list(range(sample_count))
    last = sample_count - 1
    return [
        round(index * last / (_MAX_ROBUST_SEED_SAMPLES - 1))
        for index in range(_MAX_ROBUST_SEED_SAMPLES)
    ]


def _line_inliers(
    xs: Sequence[float], values: Sequence[float], *, scale_floor_m: float
) -> set[int]:
    """Indices of the samples consistent with one line, via a bounded robust seed."""
    _intercept, _slope, residuals = _robust_line_residuals(xs, values)
    scale_m = max(
        _MAD_SCALE_FACTOR * statistics.median(abs(value) for value in residuals),
        scale_floor_m,
    )
    return {
        index
        for index, residual in enumerate(residuals)
        if abs(residual) <= _ROBUST_RESIDUAL_Z_MAX * scale_m
    }


def fit_final_segment(
    points: Sequence[TrackPoint],
    frame: RunwayFrame,
    *,
    window_m: tuple[float, float] = DEFAULT_WINDOW_M,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_span_m: float = DEFAULT_MIN_SPAN_M,
) -> SegmentFit | None:
    """Fit ``points``' established segment against ``frame``, or None if it cannot be.

    None means "this track does not carry a fittable final segment for this runway" --
    too few samples in the window, or too short a baseline to pin a slope. That is a
    NORMAL outcome (a track truncated early by ADS-B coverage, or one still on base
    leg), not an error. Producer-side callers classify and count it rather than drop
    it, since it is a statement about coverage, not about how the approach was flown.

    ``points`` MUST be in time order -- ``along_progress_m`` reads its sign as the
    direction of travel, which is the only thing separating the two ends of one
    runway. Order is not verified here; the harvest sorts by timestamp upstream.
    """
    # The two tuning knobs are checked once, here, because this is the package's
    # public entry point: below this line they are invariants, not arguments.
    if min_samples < 3:
        raise ValueError("min_samples must be >= 3 (the line fit needs n - 2 degrees of freedom)")
    if min_span_m <= 0.0:
        raise ValueError("min_span_m must be > 0 (a zero-span segment cannot pin a slope)")
    projected = frame.project_all(points)
    indexed = _straight_final_suffix(
        _final_inbound_run(projected, window_m),
        min_samples=min_samples,
        min_span_m=min_span_m,
    )
    # The suffix, when it exists, already satisfies both minima by construction.
    if not indexed:
        return None
    source_indices = [index for index, _ in indexed]
    source_projected = [point for _, point in indexed]
    source_alongs = [p.along_m for p in source_projected]

    # Both components are fitted over the SAME samples, so the cross-track and
    # vertical answers always describe one segment. The catastrophic failures in the
    # measured cohort were isolated geometric-altitude values (983 m and 10,828 m
    # inside otherwise normal descents), which the vertical MAD scale rejects without
    # any reference to TCH or to a verdict limit.
    kept = sorted(
        _line_inliers(
            source_alongs,
            [point.height_m for point in source_projected],
            scale_floor_m=_ALTITUDE_RESIDUAL_SCALE_FLOOR_M,
        )
        & _line_inliers(
            source_alongs,
            [point.cross_m for point in source_projected],
            scale_floor_m=_LATERAL_RESIDUAL_FLOOR_M,
        )
    )
    if len(kept) < min_samples:
        return None
    kept_set = set(kept)
    rejected = [
        index for index in range(len(source_projected)) if index not in kept_set
    ]
    projected = [source_projected[index] for index in kept]
    sample_indices = [source_indices[index] for index in kept]
    alongs = [point.along_m for point in projected]
    span = max(alongs) - min(alongs)
    if span < min_span_m:
        return None

    crosses = [p.cross_m for p in projected]
    return SegmentFit(
        runway=frame.ident,
        n_samples=len(projected),
        span_m=span,
        window_m=window_m,
        first_sample_index=sample_indices[0],
        last_sample_index=sample_indices[-1],
        cross=fit_line(alongs, crosses),
        height=fit_line(alongs, [p.height_m for p in projected]),
        median_abs_cross_m=statistics.median(abs(c) for c in crosses),
        nearest_sample_along_m=max(alongs),
        # Time-ordered, so the SIGN is direction of travel (see SegmentFit.approaching).
        # ``points`` is required to be in time order; projection preserves input order.
        along_progress_m=alongs[-1] - alongs[0],
        rejected_sample_indices=tuple(source_indices[index] for index in rejected),
    )
