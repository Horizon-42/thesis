"""Backend-agnosticism: the SAME functions must work for NumPy (tests) and CasADi (optimizer).

``test_mathx_*`` pass now (the dispatch layer is provided). The equivalence tests xfail until the
corresponding TODO is implemented — once it is, they prove the function returns a CasADi
expression that evaluates to the same number as the NumPy call (the "写完即可用" guarantee).
"""

import math

import numpy as np
import pytest

ca = pytest.importorskip("casadi")

from constraints import geometry as geo  # noqa: E402
from constraints import mathx, vertical   # noqa: E402
from constraints.segments import StepDown  # noqa: E402

A = np.array([0.0, 0.0])
B = np.array([10.0, 0.0])


def test_mathx_dispatch_numpy():
    assert np.isclose(mathx.fabs(-3.0), 3.0)
    assert np.isclose(mathx.atan2(1.0, 0.0), np.pi / 2)
    assert np.allclose(mathx.fmax(np.array([1.0, 5.0]), 3.0), [3.0, 5.0])
    assert np.allclose(mathx.if_else(np.array([True, False]), 1.0, 0.0), [1.0, 0.0])


def test_mathx_dispatch_casadi():
    x = ca.SX.sym("x")
    expr = mathx.fabs(x) + mathx.sqrt(mathx.fmax(x, 1.0))
    assert isinstance(expr, ca.SX)
    f = ca.Function("f", [x], [expr])
    assert np.isclose(float(f(4.0)), 4.0 + 2.0)


@pytest.mark.xfail(reason="needs TODO ① ② implemented", strict=False)
def test_along_and_cross_track_casadi_match_numpy():
    n_sym, e_sym = ca.SX.sym("n"), ca.SX.sym("e")
    s_sym = geo.along_track(n_sym, e_sym, A, B)
    x_sym = geo.cross_track(n_sym, e_sym, A, B)
    assert isinstance(s_sym, ca.SX) and isinstance(x_sym, ca.SX)
    f = ca.Function("f", [n_sym, e_sym], [s_sym, x_sym])
    s_val, x_val = f(3.0, 7.0)
    assert np.isclose(float(s_val), geo.along_track(3.0, 7.0, A, B))
    assert np.isclose(float(x_val), geo.cross_track(3.0, 7.0, A, B))


@pytest.mark.xfail(reason="needs TODO ⑨ implemented", strict=False)
def test_moc_floor_casadi_matches_numpy():
    steps = [StepDown(2000.0, 1524.0), StepDown(5000.0, 914.0)]
    s = ca.SX.sym("s")
    floor = vertical.moc_floor(s, steps)
    assert isinstance(floor, ca.SX)
    f = ca.Function("f", [s], [floor])
    for query in (1000.0, 3000.0, 6000.0):
        assert np.isclose(float(f(query)), float(vertical.moc_floor(query, steps)))
