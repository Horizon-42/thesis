"""End-to-end: builder + segments + examples (feasible/infeasible scoring, unit-aware report)."""

import math
from dataclasses import replace

import numpy as np

from approach_constraints import examples
from approach_constraints.builder import ConstraintSet
from approach_constraints.segments import segment_violations_from_components


def test_constraint_set_structure_is_wired():
    cset = examples.build_example_constraint_set()
    assert len(cset.segments) == 4
    nodes = examples.feasible_segment_nodes()
    assert [n.shape for n in nodes] == [(6, 7)] * 4


def test_feasible_trajectory_is_feasible():
    cset = examples.build_example_constraint_set()
    report = cset.evaluate(examples.feasible_segment_nodes())
    assert report.is_feasible(tol_m=1.0), report.summary()


def test_infeasible_trajectory_flags_final_corridor():
    cset = examples.build_example_constraint_set()
    report = cset.evaluate(examples.infeasible_segment_nodes())
    assert not report.is_feasible(tol_m=1.0)
    worst_key = max(report.violations, key=lambda k: float(np.ravel(report.violations[k]).max()))
    assert "final_lpv" in worst_key and "lateral" in worst_key


def test_vertical_gate_publishes_semantics_around_the_faf():
    # README §4b: with d_faf_m set, the glidepath window binds only from the FAF toward the
    # runway; UPSTREAM of the FAF (an early FAC join) the published pre-FAF floor applies.
    final = examples.build_example_segments()[-1]        # d_faf_m = PFAF, prefaf_floor = 700 m
    lpv = final.lpv
    d_faf = lpv.d_faf_m
    # Two nodes ON the course: one 4 km UPSTREAM of the FAF at 720 m — above the 700 m floor but
    # ~50 m BELOW the extended glidepath window (gp ≈ 830, low bound ≈ 770 there) — and one
    # downstream at 100 m below the glidepath (below_m = 60).
    n = np.array([d_faf + 4000.0, d_faf - 2000.0])
    e = np.array([0.0, 0.0])
    h = np.array([
        720.0,                                            # upstream: floor-legal, window-illegal
        examples._glidepath_alt(d_faf - 2000.0) - 100.0,  # downstream: 100 m low
    ])
    viol = segment_violations_from_components(final, n, e, h, np.zeros(2))
    low = np.ravel(viol[[k for k in viol if k.endswith("glidepath_low")][0]])
    floor = np.ravel(viol[[k for k in viol if k.endswith("prefaf_floor")][0]])
    assert low[0] <= 0.0          # upstream node: the window does NOT bind (gated off)
    assert low[1] > 0.0           # downstream node: 100 m low -> violated as before
    assert floor[0] <= 0.0        # upstream node: above the published floor
    assert floor[1] <= 0.0        # downstream node: floor row inactive there
    # a violated pre-FAF floor is caught
    viol_low = segment_violations_from_components(
        final, np.array([d_faf + 4000.0]), np.array([0.0]), np.array([600.0]), np.zeros(1))
    floor_key = [k for k in viol_low if k.endswith("prefaf_floor")][0]
    assert float(np.ravel(viol_low[floor_key])[0]) > 0.0  # 600 m < the 700 m floor

    # gate OFF (d_faf_m None): pre-existing behavior — the window binds everywhere, no floor row
    ungated = replace(final, lpv=replace(lpv, d_faf_m=None, prefaf_floor_m=None))
    viol_off = segment_violations_from_components(ungated, n, e, h, np.zeros(2))
    assert not any(k.endswith("prefaf_floor") for k in viol_off)
    low_off = np.ravel(viol_off[[k for k in viol_off if k.endswith("glidepath_low")][0]])
    assert low_off[0] > 0.0       # upstream node IS below the extended window when ungated


def test_descent_violation_is_not_masked_by_the_metre_tolerance():
    # REGRESSION (mixed units): the descent rows are RADIANS. A 10 deg-too-steep descent is a
    # violation of ~0.18 rad — far under the 1.0 METRE tolerance — and used to read as feasible.
    nodes = examples.feasible_segment_nodes()
    nodes[0][:, 5] = math.radians(-15.0)      # feeder cap is 4.7 deg -> ~10 deg too steep
    report = examples.build_example_constraint_set().evaluate(nodes)
    assert not report.is_feasible()
    assert report.max_angular_violation() > math.radians(5.0)
    assert report.max_violation() <= 1.0      # the metre rows are still fine
    # the summary reports the descent row in degrees and flags it
    line = next(l for l in report.summary().splitlines() if "feeder" in l and "descent" in l)
    assert "deg" in line and "VIOLATED" in line
