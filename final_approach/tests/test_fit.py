"""The segment fit: does it recover a known approach, and is its sigma honest?"""

from __future__ import annotations

import math

import pytest
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from final_approach import RunwayFrame, TrackPoint, fit_final_segment

FRAME = RunwayFrame(ident="05L", lat=35.8745, lon=-78.802, elevation_m=111.86, course_deg=45.0)

QUANTUM_M = 25.0 * 0.3048  # OpenSky geoaltitude is reported on a 25 ft lattice.


def synthetic_approach(
    *,
    glidepath_deg: float = 3.0,
    tch_m: float = 17.5,
    cross_m: float = 0.0,
    start_along_m: float = -12000.0,
    end_along_m: float = -325.0,
    step_m: float = 77.0,
    quantise: bool = False,
    frame: RunwayFrame = FRAME,
) -> list[TrackPoint]:
    """A textbook straight approach, sampled from far to near (time order).

    Defaults mirror the measured KRDU case: ~77 m of ground track per 1 Hz sample and
    a truncation 325 m short of the threshold. The start is well outside the fit
    window so the track also carries a realistic descent for the landing screen.
    """
    slope = math.tan(math.radians(glidepath_deg))
    m_per_deg_lon = metres_per_deg_lon(frame.lat)
    east_hat, north_hat = math.sin(math.radians(frame.course_deg)), math.cos(math.radians(frame.course_deg))
    points = []
    along = start_along_m
    while along <= end_along_m:
        height = tch_m + (-along) * slope
        if quantise:
            height = round((height + frame.elevation_m) / QUANTUM_M) * QUANTUM_M - frame.elevation_m
        east = along * east_hat + cross_m * north_hat
        north = along * north_hat - cross_m * east_hat
        points.append(
            TrackPoint(
                lat=frame.lat + north / METRES_PER_DEG_LAT,
                lon=frame.lon + east / m_per_deg_lon,
                alt_m=frame.elevation_m + height,
            )
        )
        along += step_m
    return points


def test_recovers_glidepath_and_crossing_from_a_truncated_track():
    """The whole point: the track stops 325 m short, the crossing is still recovered."""
    fit = fit_final_segment(synthetic_approach(glidepath_deg=3.0, tch_m=17.5), FRAME)
    assert fit is not None
    assert fit.glidepath_deg == pytest.approx(3.0, abs=0.01)
    assert fit.height_at_threshold_m == pytest.approx(17.5, abs=0.2)
    assert fit.extrapolation_m == pytest.approx(325.0, abs=80.0)


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
    assert fit.height.max_abs_residual_m <= QUANTUM_M  # scatter bounded by the lattice
    assert fit.height.sigma_at_zero < QUANTUM_M / 2.0  # yet the MEAN is far better pinned


def test_autocorrelation_correction_only_ever_widens_sigma():
    """rho is clamped at zero, so a noisy estimate cannot manufacture precision."""
    fit = fit_final_segment(synthetic_approach(quantise=True), FRAME)
    assert fit is not None
    assert 0.0 <= fit.height.rho <= 0.95
    assert fit.height.n_effective <= fit.n_samples


def test_perfect_line_has_zero_residual_and_zero_sigma():
    fit = fit_final_segment(synthetic_approach(quantise=False), FRAME)
    assert fit is not None
    assert fit.height.rms_residual_m == pytest.approx(0.0, abs=1e-6)
    assert fit.height.sigma_at_zero == pytest.approx(0.0, abs=1e-6)


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


def test_none_when_the_window_holds_too_few_samples():
    """A track truncated before the window fills is a coverage fact, not an error."""
    stub = synthetic_approach(start_along_m=-600.0, end_along_m=-325.0, step_m=77.0)
    assert fit_final_segment(stub, FRAME) is None


def test_none_when_the_baseline_is_too_short_to_pin_a_slope():
    dense = synthetic_approach(start_along_m=-700.0, end_along_m=-400.0, step_m=10.0)
    assert fit_final_segment(dense, FRAME, min_span_m=500.0) is None


def test_window_bounds_are_respected():
    fit = fit_final_segment(synthetic_approach(start_along_m=-9000.0), FRAME, window_m=(-5000.0, -300.0))
    assert fit is not None
    assert fit.span_m <= 5000.0 - 300.0
    assert fit.nearest_sample_along_m <= -300.0


def test_a_shorter_baseline_yields_a_larger_sigma():
    """Why the window is not shrunk further: extrapolation needs a long lever arm."""
    long_fit = fit_final_segment(synthetic_approach(quantise=True), FRAME, window_m=(-5000.0, -300.0))
    short_fit = fit_final_segment(synthetic_approach(quantise=True), FRAME, window_m=(-2000.0, -300.0))
    assert long_fit is not None and short_fit is not None
    assert short_fit.height.sigma_at_zero > long_fit.height.sigma_at_zero
