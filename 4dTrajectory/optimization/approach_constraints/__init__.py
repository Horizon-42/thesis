"""approach_constraints — approach-procedure constraints for the trajectory optimizer.

Turns an LPV approach (segments + FAS geometry) into a set of inequality constraints
``g(z) ≤ 0`` over the optimizer's NORMALIZED state nodes ``z = (n, e, h, V, psi, gamma, m)``.

Module map:

    frame.py     TargetFrame: fixes (lat/lon) → (n, e)              [matches the optimizer]
    mathx.py     NumPy/CasADi op dispatch (fabs, atan2, if_else, …)
    state.py     decision-state column layout
    geometry.py  along/cross-track, course bearing, intercept angle
    lateral.py   RNP box corridor + LPV angular corridor
    vertical.py  glidepath window, step-down floor, descent cap
    segments.py  SegmentSpec + per-segment assembly                  [composes the primitives]
    builder.py   ConstraintSet + ConstraintReport                    [NumPy evaluation path]
    examples.py  a synthetic straight-in LPV approach

Conventions: every ``*_violation`` returns ``g`` with ``g ≤ 0  ⇔  satisfied``; headings/courses
use the DYNAMICS MODEL's convention (0 = East, CCW toward North — see geometry.course_bearing).
The optimizer consumer is ``collocation.optimizer`` (one phase per segment, symbolic node
columns through ``segment_violations_from_components``).

See ``approach_constraints/README.md`` for the math and the optimizer integration.
Design source: ``4dTrajectory/docs/optimization_constraint_design.md`` +
``4dTrajectory/docs/lpv_final_segment.en.html``.
"""

from __future__ import annotations

from .builder import (
    DEFAULT_TOL_M,
    DEFAULT_TOL_RAD,
    ConstraintReport,
    ConstraintSet,
)
from .frame import TargetFrame
from .lateral import DEFAULT_K_MARGIN
from .segments import (
    DEFAULT_GLIDEPATH_ABOVE_M,
    DEFAULT_GLIDEPATH_BELOW_M,
    STANDARD_INTERCEPT_MAX_DEG,
    LpvFinalSpec,
    SegmentKind,
    SegmentSpec,
    StepDown,
    segment_violations,
    segment_violations_from_components,
)

__all__ = [
    "TargetFrame",
    "SegmentKind",
    "SegmentSpec",
    "LpvFinalSpec",
    "StepDown",
    "segment_violations",
    "segment_violations_from_components",
    "ConstraintSet",
    "ConstraintReport",
    "DEFAULT_GLIDEPATH_BELOW_M",
    "DEFAULT_GLIDEPATH_ABOVE_M",
    "DEFAULT_K_MARGIN",
    "DEFAULT_TOL_M",
    "DEFAULT_TOL_RAD",
    "STANDARD_INTERCEPT_MAX_DEG",
]
