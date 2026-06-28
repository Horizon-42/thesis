"""constraints — approach-procedure constraints for the trajectory optimizer (teaching scaffold).

Turns an LPV approach (segments + FAS geometry) into a set of inequality constraints
``g(z) ≤ 0`` over the optimizer's NORMALIZED state nodes ``z = (n, e, h, V, psi, gamma, m)``.

Architecture (what is provided vs. what YOU implement):

    frame.py     TargetFrame: fixes (lat/lon) → (n, e)            [provided — matches optimizer]
    state.py     decision-state column layout                      [provided]
    geometry.py  along/cross-track, bearing, intercept angle       [TODO ①②③④]
    lateral.py   RNP box corridor + LPV angular corridor           [TODO ⑤⑥⑦]
    vertical.py  glidepath window, step-down floor, descent cap     [TODO ⑧⑨, ⑩ bonus]
    segments.py  SegmentSpec + per-segment assembly                [provided — composes the above]
    builder.py   ConstraintSet + ConstraintReport                  [provided]
    examples.py  a synthetic straight-in LPV approach               [provided]

Convention: every ``*_violation`` returns ``g`` with ``g ≤ 0  ⇔  satisfied``.

See ``constraints/README.md`` for the math, the TODO checklist, and the optimizer integration.
Design source: ``4dTrajectory/docs/optimization_constraint_design.md`` +
``4dTrajectory/docs/lpv_final_segment.en.html``.
"""

from __future__ import annotations

from .builder import ConstraintReport, ConstraintSet, split_contiguous
from .frame import TargetFrame
from .segments import LpvFinalSpec, SegmentKind, SegmentSpec, StepDown, segment_violations

__all__ = [
    "TargetFrame",
    "SegmentKind",
    "SegmentSpec",
    "LpvFinalSpec",
    "StepDown",
    "segment_violations",
    "ConstraintSet",
    "ConstraintReport",
    "split_contiguous",
]
