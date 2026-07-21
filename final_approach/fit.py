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

WHY THE FIT IS MANDATORY, NOT COSMETIC
--------------------------------------
OpenSky's ``geoaltitude`` is quantised to 25 ft = 7.62 m (verified: all 482 distinct
altitudes in the KRDU set lie on that lattice). The vertical evaluation window is
9.15 m wide, so ONE sample carries +/-3.81 m of rounding -- it cannot resolve the
window even in principle. Averaging a few hundred lattice-quantised samples is what
drives the crossing's standard error below 2 m. The fit recovers sub-quantum
precision; the last raw sample never could.

WHY THE UNCERTAINTY IS AUTOCORRELATION-CORRECTED
------------------------------------------------
Textbook OLS variance assumes ``n`` independent samples. At 1 Hz on a smooth descent
they are not: the aircraft crosses one 7.62 m quantisation step every ~2 samples
(measured 3.81 m of descent per sample), so consecutive residuals share the same
rounding error -- 54.8% of consecutive KRDU samples report an IDENTICAL raw altitude.
Measured lag-1 residual autocorrelation is rho ~ 0.43, i.e. n_eff ~ 0.40 n.

Both variance terms must be deflated -- ``Sxx`` is a sum over the same correlated
samples as ``1/n``. Correcting only the first understates sigma (it gives a 1.15x
inflation where the honest figure is 1.58x). On a 9.15 m gate that difference changes
verdicts, so the naive number is not offered as an option.

WHAT THIS MODULE DOES NOT DO
----------------------------
No thresholds, no ``established`` flag, no verdict. ``SegmentFit`` reports what the
data says; ``evaluation/arrival.py`` applies the regulation gates and
``assign_runway`` applies a relative comparison. See the package docstring for why
that separation is load-bearing.
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
# below the 1000 ft stabilisation gate every airline SOP requires an approach to be
# stable by. Reaching further out measurably biases the extrapolation HIGH (KRDU
# median crossing +5.43 m from -8000 m vs +3.66 m from -5000 m) because aircraft are
# still intercepting the glidepath from above out there.
#
# Inner edge -300 m: ~107 ft above threshold elevation, comfortably above flare
# initiation (~50 ft), so the fit never ingests the round-out. It also sits inside
# where ADS-B coverage typically stops (median last sample -325 m at KRDU), which is
# the truncation this fit exists to bridge.
#
# Shrinking the window further does not help: below ~3 km the baseline is too short
# to pin the slope and sigma grows sharply (2.50 m at a 2 km window vs 1.68 m at 5 km).
DEFAULT_WINDOW_M = (-5000.0, -300.0)

DEFAULT_MIN_SAMPLES = 8
DEFAULT_MIN_SPAN_M = 500.0

# How far the along-track coordinate may retreat between consecutive samples before the
# backward walk decides the aircraft was not on ONE inbound run (see _final_inbound_run).
# One 1 Hz sample advances ~77 m on approach, so 100 m absorbs GPS jitter and the slight
# along-track stall of a late centreline intercept without tolerating a real reversal.
_INBOUND_TOLERANCE_M = 100.0

# Negative residual autocorrelation would SHRINK the variance. Clamping at zero keeps
# the correction one-sided (it can only ever widen the interval), so a noisy rho
# estimate cannot manufacture precision.
_RHO_CLAMP = (0.0, 0.95)
_MIN_EFFECTIVE_SAMPLES = 3.0


@dataclass(frozen=True)
class LineFit:
    """One ordinary-least-squares line ``y = intercept + slope * x``, fitted over a
    window of x and reported AT x = 0 (the threshold).

    ``sigma_at_zero`` is the standard error of the fitted MEAN response at x = 0 --
    a confidence interval on where the aircraft's true smooth path crossed, which is
    the question being asked. It is deliberately not a prediction interval: that would
    add the per-sample scatter ``s``, i.e. the 25 ft quantisation noise of a
    hypothetical extra measurement, which is not part of where the aircraft was.
    """

    intercept: float
    slope: float
    rms_residual_m: float
    max_abs_residual_m: float
    rho: float
    n_effective: float
    sigma_at_zero: float


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

    cross: LineFit
    height: LineFit

    median_abs_cross_m: float
    nearest_sample_along_m: float
    along_progress_m: float

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
    def extrapolation_m(self) -> float:
        """How far past the last usable sample the crossing was extrapolated."""
        return abs(self.nearest_sample_along_m)

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


def _fit_line(xs: Sequence[float], ys: Sequence[float]) -> LineFit:
    """OLS with an autocorrelation-corrected standard error at x = 0.

    Requires at least 3 points (residual variance needs n - 2 degrees of freedom)
    and non-identical xs; ``fit_final_segment`` validates both at its boundary.
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
    s = math.sqrt(sse / (n - 2))

    # Lag-1 residual autocorrelation -> effective sample size. Both variance terms are
    # deflated: Sxx is a sum over the same correlated samples as the 1/n term.
    rho = sum(residuals[i] * residuals[i + 1] for i in range(n - 1)) / sse if sse > 0 else 0.0
    rho = min(max(rho, _RHO_CLAMP[0]), _RHO_CLAMP[1])
    n_eff = max(n * (1.0 - rho) / (1.0 + rho), _MIN_EFFECTIVE_SAMPLES)
    s_xx_eff = s_xx * (n_eff / n)
    sigma = s * math.sqrt(1.0 / n_eff + (x_bar * x_bar) / s_xx_eff)

    return LineFit(
        intercept=intercept,
        slope=slope,
        rms_residual_m=math.sqrt(sse / n),
        max_abs_residual_m=max(abs(r) for r in residuals),
        rho=rho,
        n_effective=n_eff,
        sigma_at_zero=sigma,
    )


def _final_inbound_run(
    projected: Sequence[Projected], window_m: tuple[float, float]
) -> list[Projected]:
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
    run = [projected[inner]]
    for i in range(inner - 1, -1, -1):
        if projected[i].along_m < window_m[0]:
            break
        if projected[i].along_m > projected[i + 1].along_m + _INBOUND_TOLERANCE_M:
            break
        run.append(projected[i])
    run.reverse()
    return run


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
    leg), not an error, and callers classify it: the harvest as ``unassignable``,
    evaluation as ``not_established``. Both must count it rather than drop it, since
    it is a statement about coverage, not about how the approach was flown.

    Projection is done once; both lines are fitted over the same sample set, so the
    cross-track and vertical answers always describe the same segment.

    ``points`` MUST be in time order -- ``along_progress_m`` reads its sign as the
    direction of travel, which is the only thing separating the two ends of one
    runway. Order is not verified here; the harvest sorts by timestamp upstream.
    """
    if min_samples < 3:
        raise ValueError("min_samples must be >= 3 (the line fit needs n - 2 degrees of freedom)")
    if min_span_m <= 0.0:
        raise ValueError("min_span_m must be > 0 (a zero-span segment cannot pin a slope)")
    projected = _final_inbound_run(frame.project_all(points), window_m)
    if len(projected) < min_samples:
        return None
    alongs = [p.along_m for p in projected]
    span = max(alongs) - min(alongs)
    if span < min_span_m:
        return None

    crosses = [p.cross_m for p in projected]
    return SegmentFit(
        runway=frame.ident,
        n_samples=len(projected),
        span_m=span,
        window_m=window_m,
        cross=_fit_line(alongs, crosses),
        height=_fit_line(alongs, [p.height_m for p in projected]),
        median_abs_cross_m=statistics.median(abs(c) for c in crosses),
        nearest_sample_along_m=max(alongs),
        # Time-ordered, so the SIGN is direction of travel (see SegmentFit.approaching).
        # ``points`` is required to be in time order; projection preserves input order.
        along_progress_m=alongs[-1] - alongs[0],
    )
