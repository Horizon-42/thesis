"""End-to-end (builder + segments + examples). Plumbing passes; the feasibility checks xfail
until the geometry/lateral/vertical TODOs are implemented."""

import numpy as np
import pytest

from constraints import examples
from constraints.builder import ConstraintSet, split_contiguous


def test_constraint_set_structure_is_wired():
    # plumbing only (no TODO math) — passes now.
    cset = examples.build_example_constraint_set()
    assert len(cset.segments) == 4
    nodes = examples.feasible_segment_nodes()
    assert [n.shape for n in nodes] == [(6, 7)] * 4


def test_split_contiguous_roundtrip():
    traj = np.arange(24 * 7).reshape(24, 7).astype(float)
    parts = split_contiguous(traj, [6, 6, 6, 6])
    assert sum(p.shape[0] for p in parts) == 24
    assert np.array_equal(np.concatenate(parts), traj)


@pytest.mark.xfail(reason="needs TODO ①–⑨ implemented", strict=False)
def test_feasible_trajectory_is_feasible():
    cset = examples.build_example_constraint_set()
    report = cset.evaluate(examples.feasible_segment_nodes())
    assert report.is_feasible(tol_m=1.0), report.summary()


@pytest.mark.xfail(reason="needs TODO ①–⑨ implemented", strict=False)
def test_infeasible_trajectory_flags_final_corridor():
    cset = examples.build_example_constraint_set()
    report = cset.evaluate(examples.infeasible_segment_nodes())
    assert not report.is_feasible(tol_m=1.0)
    worst_key = max(report.violations, key=lambda k: float(np.ravel(report.violations[k]).max()))
    assert "final_lpv" in worst_key and "lateral" in worst_key
