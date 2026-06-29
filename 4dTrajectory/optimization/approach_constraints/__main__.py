"""CLI demo: build the synthetic LPV approach, evaluate a feasible and an infeasible trajectory.

    python -m approach_constraints          # from the 4dTrajectory/optimization directory
    # or, from the repo root:
    PYTHONPATH=4dTrajectory/optimization python -m approach_constraints

Before you implement the TODOs it prints which one to do first; afterwards it prints a violation
report for both trajectories (feasible → all ≤ 0; infeasible → the final-leg corridor flags).
"""

from __future__ import annotations

from . import examples
from .builder import ConstraintSet


def main() -> None:
    cset: ConstraintSet = examples.build_example_constraint_set()
    print("Synthetic straight-in LPV approach:")
    for seg in cset.segments:
        print(f"  {seg.kind.value:<13s} {seg.start_ident:>5s} -> {seg.end_ident:<5s}")
    print()

    try:
        feasible = cset.evaluate(examples.feasible_segment_nodes())
        infeasible = cset.evaluate(examples.infeasible_segment_nodes())
    except NotImplementedError as exc:
        print("Not implemented yet — finish the TODOs in geometry/lateral/vertical.")
        print(f"  -> {exc}")
        print("\nSee constraints/README.md for the checklist (TODO ①…⑩).")
        return

    print("FEASIBLE trajectory (expect max ≤ 0):")
    print(feasible.summary())
    print("\nINFEASIBLE trajectory (final leg pushed off the corridor):")
    print(infeasible.summary())


if __name__ == "__main__":
    main()
