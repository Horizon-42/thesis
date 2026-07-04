"""Backend-agnosticism: the SAME functions must work for NumPy (tests) and CasADi (optimizer).

The equivalence tests prove each function returns a CasADi expression that evaluates to the same
number as the NumPy call (the "写完即可用" guarantee).
"""

import math

import numpy as np
import pytest

ca = pytest.importorskip("casadi")

from approach_constraints import geometry as geo  # noqa: E402
from approach_constraints import mathx, vertical   # noqa: E402
from approach_constraints.segments import StepDown  # noqa: E402

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


def test_along_and_cross_track_casadi_match_numpy():
    n_sym, e_sym = ca.SX.sym("n"), ca.SX.sym("e")
    s_sym = geo.along_track(n_sym, e_sym, A, B)
    x_sym = geo.cross_track(n_sym, e_sym, A, B)
    assert isinstance(s_sym, ca.SX) and isinstance(x_sym, ca.SX)
    f = ca.Function("f", [n_sym, e_sym], [s_sym, x_sym])
    s_val, x_val = f(3.0, 7.0)
    assert np.isclose(float(s_val), geo.along_track(3.0, 7.0, A, B))
    assert np.isclose(float(x_val), geo.cross_track(3.0, 7.0, A, B))


def test_moc_floor_casadi_matches_numpy():
    steps = [StepDown(2000.0, 1524.0), StepDown(5000.0, 914.0)]
    s = ca.SX.sym("s")
    floor = vertical.moc_floor(s, steps)
    assert isinstance(floor, ca.SX)
    f = ca.Function("f", [s], [floor])
    for query in (1000.0, 3000.0, 6000.0):
        assert np.isclose(float(f(query)), float(vertical.moc_floor(query, steps)))


def test_gated_final_segment_casadi_matches_numpy():
    # The flexible-join vertical gate (if_else on d vs d_faf) and the join primitives must lower
    # identically to both backends — they run inside the CasADi NLP.
    from approach_constraints import examples, lateral
    from approach_constraints.segments import segment_violations_from_components

    final = examples.build_example_segments()[-1]       # gated: d_faf_m + prefaf_floor_m set
    n_sym, e_sym, h_sym = ca.SX.sym("n"), ca.SX.sym("e"), ca.SX.sym("h")
    viol = segment_violations_from_components(final, n_sym, e_sym, h_sym, ca.SX(0.0))
    keys = sorted(k for k in viol if "glidepath" in k or "prefaf" in k)
    f = ca.Function("g", [n_sym, e_sym, h_sym], [viol[k] for k in keys])
    for node in ((final.lpv.d_faf_m + 4000.0, 0.0, 720.0),     # upstream of the FAF
                 (final.lpv.d_faf_m - 2000.0, 0.0, 500.0)):    # downstream
        got = [float(v) for v in f(*node)]
        want_all = segment_violations_from_components(
            final, np.array([node[0]]), np.array([node[1]]), np.array([node[2]]), np.zeros(1))
        want = [float(np.ravel(want_all[k])[0]) for k in keys]
        assert np.allclose(got, want)

    # join primitives: SX in, same numbers out
    tol = 926.0
    fix = np.array([1000.0, -500.0])
    expr = lateral.fix_passage_violation(n_sym, e_sym, fix, tol)
    g = ca.Function("g", [n_sym, e_sym], [expr])
    assert np.isclose(float(g(1000.0, 436.0)),
                      float(lateral.fix_passage_violation(1000.0, 436.0, fix, tol)))
    win = lateral.fac_join_window_violation(n_sym, e_sym, final.lpv, 9260.0, 2500.0)
    w = ca.Function("w", [n_sym, e_sym], list(win))
    got = [float(v) for v in w(10000.0, 0.0)]
    want = [float(v) for v in lateral.fac_join_window_violation(10000.0, 0.0, final.lpv, 9260.0, 2500.0)]
    assert np.allclose(got, want)
