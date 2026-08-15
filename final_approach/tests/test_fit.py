"""The segment fit recovers one straight pass and rejects gross corruption."""

from __future__ import annotations

import math

import pytest

from final_approach import Projected, RunwayFrame, TrackPoint, fit_final_segment
from final_approach.fit import _robust_seed_indices
from final_approach.tests.factories import (
    ALTITUDE_QUANTUM_M,
    FRAME,
    synthetic_approach,
)


def test_robust_seed_is_deterministically_bounded_for_dense_real_tracks():
    indices = _robust_seed_indices(1_753)

    assert len(indices) == 64
    assert indices[0] == 0
    assert indices[-1] == 1_752
    assert indices == sorted(set(indices))


def test_recovers_glidepath_and_crossing_from_a_truncated_track():
    """The whole point: the track stops 325 m short, the crossing is still recovered."""
    fit = fit_final_segment(synthetic_approach(glidepath_deg=3.0, tch_m=17.5), FRAME)
    assert fit is not None
    assert fit.glidepath_deg == pytest.approx(3.0, abs=0.01)
    assert fit.height_at_threshold_m == pytest.approx(17.5, abs=0.2)
    assert fit.nearest_sample_along_m == pytest.approx(-325.0, abs=80.0)


def test_recovers_a_lateral_offset():
    fit = fit_final_segment(synthetic_approach(cross_m=230.0), FRAME)
    assert fit is not None
    assert fit.cross_at_threshold_m == pytest.approx(230.0, abs=1.0)
    assert fit.median_abs_cross_m == pytest.approx(230.0, abs=1.0)


def test_cross_track_sign_is_preserved():
    left = fit_final_segment(synthetic_approach(cross_m=-150.0), FRAME)
    assert left is not None and left.cross_at_threshold_m < 0
    assert left.median_abs_cross_m > 0  # the robust score is unsigned


def test_quantised_altitude_still_recovers_the_crossing():
    """25 ft quantisation is 83% of the 9.15 m vertical gate; one sample cannot resolve
    it, but the fit averages the lattice away."""
    fit = fit_final_segment(synthetic_approach(tch_m=17.5, quantise=True), FRAME)
    assert fit is not None
    assert fit.height_at_threshold_m == pytest.approx(17.5, abs=1.5)
    assert fit.height.max_abs_residual_m <= ALTITUDE_QUANTUM_M


def test_perfect_line_has_zero_residual():
    fit = fit_final_segment(synthetic_approach(quantise=False), FRAME)
    assert fit is not None
    assert fit.height.rms_residual_m == pytest.approx(0.0, abs=1e-6)


def test_an_inbound_track_fits_and_is_marked_approaching():
    fit = fit_final_segment(synthetic_approach(), FRAME)
    assert fit is not None and fit.approaching
    assert fit.along_progress_m > 0


def test_an_outbound_track_yields_no_fit_at_all():
    """Direction is enforced by the inbound-run walk, before any line is fitted -- so a
    departure down the same centreline cannot produce a 'perfect' backwards approach."""
    assert fit_final_segment(list(reversed(synthetic_approach())), FRAME) is None


def test_an_earlier_pass_through_the_window_is_excluded():
    """A go-around: the same along-track band is occupied twice, 4 km off centreline the
    first time. Only the final inbound run may reach the fit."""
    missed = synthetic_approach(cross_m=4000.0, end_along_m=-1000.0)
    final = synthetic_approach(cross_m=0.0)
    fit = fit_final_segment([*missed, *final], FRAME)
    assert fit is not None
    assert fit.median_abs_cross_m < 5.0  # the 4 km pass contributed nothing
    assert fit.cross_at_threshold_m == pytest.approx(0.0, abs=1.0)


def test_small_step_reversal_cannot_accumulate_into_the_final_inbound_run():
    """The reversal tolerance is cumulative, not a renewable per-sample allowance.

    The first leg moves steadily away from the threshold in 50 m steps.  Every
    individual step is below the 100 m jitter allowance, but together it is a
    different leg and must not be admitted into the final inbound fit.
    """
    outbound = [
        FRAME.unproject(Projected(float(along_m), 4_000.0, 300.0))
        for along_m in range(-400, -2_001, -50)
    ]
    final = synthetic_approach(
        start_along_m=-2_000.0,
        end_along_m=-300.0,
        step_m=100.0,
    )

    fit = fit_final_segment([*outbound, *final], FRAME)

    assert fit is not None
    assert fit.approaching
    assert fit.median_abs_cross_m < 5.0
    assert fit.height_at_threshold_m == pytest.approx(17.5, abs=0.5)


def test_base_to_final_turn_is_excluded_before_fitting_the_aligned_suffix():
    """Monotonic along-track progress alone does not establish a straight final.

    The aircraft turns onto the centreline inside the 3 km event window.  Treating
    that entire curve as one line extrapolates a fictitious crossing hundreds of
    metres off the runway even though the final 700 m are exactly aligned.
    """
    slope = math.tan(math.radians(3.0))
    points = []
    for along_m in range(-3_000, -299, 100):
        if along_m >= -1_000:
            cross_m = 0.0
        else:
            turn_fraction = (-along_m - 1_000.0) / 2_000.0
            cross_m = 16_000.0 * turn_fraction * turn_fraction
        points.append(
            FRAME.unproject(
                Projected(
                    float(along_m),
                    cross_m,
                    17.5 - slope * along_m,
                )
            )
        )

    fit = fit_final_segment(points, FRAME, window_m=(-3_000.0, -300.0))

    assert fit is not None
    assert fit.first_sample_index >= 20
    assert fit.cross_at_threshold_m == pytest.approx(0.0, abs=1.0)
    assert fit.height_at_threshold_m == pytest.approx(17.5, abs=0.5)


def test_no_fit_when_the_nearest_minimum_span_is_still_turning():
    """Without 500 m of coherent straight final, extrapolation is unavailable."""
    slope = math.tan(math.radians(3.0))
    points = [
        FRAME.unproject(
            Projected(
                float(along_m),
                16_000.0 * ((-along_m - 300.0) / 2_700.0) ** 2,
                17.5 - slope * along_m,
            )
        )
        for along_m in range(-3_000, -299, 100)
    ]

    assert fit_final_segment(
        points,
        FRAME,
        window_m=(-3_000.0, -300.0),
    ) is None


def test_one_extreme_altitude_sample_is_rejected_before_the_final_fit():
    points = synthetic_approach()
    corrupted = list(points)
    index = next(
        i
        for i, point in enumerate(corrupted)
        if -1_000.0 < FRAME.project(point).along_m < -900.0
    )
    source = corrupted[index]
    corrupted[index] = TrackPoint(
        source.lat,
        source.lon,
        source.alt_m + 10_000.0,
    )

    fit = fit_final_segment(corrupted, FRAME)

    assert fit is not None
    assert fit.height_at_threshold_m == pytest.approx(17.5, abs=1.0)
    assert index in fit.rejected_sample_indices


def test_none_when_the_window_holds_too_few_samples():
    """A track truncated before the window fills is a coverage fact, not an error."""
    stub = synthetic_approach(start_along_m=-600.0, end_along_m=-325.0, step_m=77.0)
    assert fit_final_segment(stub, FRAME) is None


def test_none_when_the_baseline_is_too_short_to_pin_a_slope():
    dense = synthetic_approach(start_along_m=-700.0, end_along_m=-400.0, step_m=10.0)
    assert fit_final_segment(dense, FRAME, min_span_m=500.0) is None


def test_degenerate_fit_parameters_are_rejected_at_the_boundary():
    """min_samples < 3 starves the n-2 dof; min_span_m <= 0 lets s_xx hit zero."""
    with pytest.raises(ValueError, match="min_samples"):
        fit_final_segment(synthetic_approach(), FRAME, min_samples=2)
    with pytest.raises(ValueError, match="min_span_m"):
        fit_final_segment(synthetic_approach(), FRAME, min_span_m=0.0)


def test_window_bounds_are_respected():
    fit = fit_final_segment(
        synthetic_approach(start_along_m=-9000.0),
        FRAME,
        window_m=(-5000.0, -300.0),
    )
    assert fit is not None
    assert fit.span_m <= 5000.0 - 300.0
    assert fit.nearest_sample_along_m <= -300.0
