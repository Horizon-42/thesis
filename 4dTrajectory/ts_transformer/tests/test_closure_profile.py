"""The closure decoder's speed / height profiles: the knot parametrisations are linear in
their knots, integrate exactly, and recover a profile they generated."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (TS_DIR, REPO_ROOT, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import closure_profile as cp  # noqa: E402


def test_hat_basis_partitions_unity_and_its_integral_is_exact():
    f = np.linspace(0.0, 1.0, 1001)
    for knots in (1, 4, 8):
        basis = cp.hat_basis(f, knots)
        assert basis.shape == (len(f), knots + 1)
        assert np.allclose(basis.sum(axis=1), 1.0)
        # ∫₀^x of each hat, checked against a fine numerical integral.
        integral = cp.integrated_hat_basis(f, knots)
        numeric = np.concatenate([[0.0], np.cumsum(0.5 * (basis[1:] + basis[:-1]) * np.diff(f)[:, None], axis=0).sum(axis=1)])
        assert np.allclose(integral.sum(axis=1), numeric, atol=1e-6)
        assert np.allclose(integral[-1], np.where((np.arange(knots + 1) == 0) | (np.arange(knots + 1) == knots), 0.5 / knots, 1.0 / knots))


def test_constant_slowness_gives_constant_speed_and_linear_time():
    xy = np.stack([np.linspace(0.0, 10_000.0, 51), np.zeros(51)], 1)
    f, length = cp.progress(xy)
    knots = np.full(5, 1.0 / 80.0)                        # 80 m/s everywhere
    t = cp.times_from_slowness(f, length, knots)
    assert t[0] == 0.0 and t[-1] == pytest.approx(10_000.0 / 80.0)
    assert np.allclose(np.diff(t), 200.0 / 80.0)
    assert np.allclose(cp.speed_from_slowness(f, knots), 80.0)


def test_slowness_fit_recovers_a_profile_and_respects_the_speed_bounds():
    xy = np.stack([np.linspace(0.0, 20_000.0, 401), np.zeros(401)], 1)
    f, length = cp.progress(xy)
    truth_knots = 1.0 / np.array([120.0, 110.0, 95.0, 80.0, 70.0])
    t = cp.times_from_slowness(f, length, truth_knots)
    fitted = cp.fit_slowness_knots(f, length, t, knots=4)
    assert np.allclose(fitted, truth_knots, rtol=1e-6)
    assert np.allclose(cp.times_from_slowness(f, length, fitted), t, atol=1e-6)
    # A profile faster than the bound is clipped to it: the fit cannot claim 300 m/s.
    fast = cp.fit_slowness_knots(f, length, t * 0.2, knots=4)
    assert np.all(1.0 / fast <= cp.SPEED_MAX_MPS + 1e-9)


def test_duration_scaling_keeps_the_shape_and_refuses_a_zero_duration():
    t = np.array([0.0, 10.0, 25.0, 50.0])
    scaled = cp.scale_to_duration(t, 100.0)
    assert scaled[-1] == 100.0 and np.allclose(scaled, t * 2.0)
    with pytest.raises(ValueError, match="zero duration"):
        cp.scale_to_duration(np.zeros(3), 100.0)
    with pytest.raises(ValueError, match="zero horizontal length"):
        cp.progress(np.zeros((3, 2)))


def test_height_fit_recovers_a_piecewise_linear_profile():
    f = np.linspace(0.0, 1.0, 301)
    truth_knots = np.array([900.0, 700.0, 450.0, 200.0, 0.0])
    u = cp.height_from_knots(f, truth_knots)
    fitted = cp.fit_height_knots(f, u, knots=4)
    assert np.allclose(fitted, truth_knots, atol=1e-6)
    assert np.allclose(cp.height_from_knots(f, fitted), u)
