"""End-to-end: builder + segments + examples (feasible/infeasible scoring, unit-aware report)."""

import math

import numpy as np

from approach_constraints import examples
from approach_constraints.builder import ConstraintSet


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
