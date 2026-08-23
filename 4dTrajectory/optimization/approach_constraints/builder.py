"""Assemble a whole approach into one constraint set, and evaluate a trajectory against it.

A :class:`ConstraintSet` holds the ordered segments; ``evaluate`` runs each segment's
:func:`~approach_constraints.segments.segment_violations` over that segment's state nodes and
flattens everything into one violation vector with the package convention ``g ≤ 0 ⇔ satisfied``.

**This is the NumPy evaluation path** (the demo and trajectory scoring). The optimizer does NOT
use it: ``collocation.optimizer`` models one PHASE per segment and feeds each phase's symbolic
node columns straight into :func:`~approach_constraints.segments.segment_violations_from_components`
— the same primitives, so the two paths cannot drift.

**Units.** Violation rows are metres everywhere EXCEPT the descent-gradient rows
(``*.descent``), which are RADIANS (``gamma`` vs the cap). The report keeps the two families
apart: one shared max/tolerance across mixed units would let a violated descent cap (up to
~57°) hide under a metre tolerance.

**Multiple IAFs.** Per design-doc §5, do NOT encode "nearest of several IAFs" with a ``min``
(non-convex, non-smooth). Build one :class:`ConstraintSet` per candidate IAF and solve each as a
separate problem, then keep the best objective. (This module models a single chosen route.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .segments import SegmentSpec, segment_violations

# Violation-name suffix marking the radian-valued rows (see module docstring).
_ANGULAR_SUFFIX = ".descent"
# Default feasibility tolerances, one per unit family.
DEFAULT_TOL_M = 1.0
DEFAULT_TOL_RAD = math.radians(0.1)


@dataclass
class ConstraintReport:
    """The result of evaluating a trajectory: named violation arrays (``≤ 0`` = satisfied)."""

    violations: dict[str, np.ndarray]

    def vector(self) -> np.ndarray:
        """All violations flattened into one 1-D vector (the NLP ``g(x)``; mixed units)."""
        if not self.violations:
            return np.zeros(0)
        return np.concatenate([np.ravel(v) for v in self.violations.values()])

    def _worst(self, angular: bool) -> float:
        vals = [
            np.ravel(v)
            for name, v in self.violations.items()
            if name.endswith(_ANGULAR_SUFFIX) == angular
        ]
        if not vals:
            return 0.0
        vec = np.concatenate(vals)
        return float(vec.max()) if vec.size else 0.0

    def max_violation(self) -> float:
        """Worst metre-row violation (lateral / glidepath / floor); ``≤ 0`` = satisfied."""
        return self._worst(angular=False)

    def max_angular_violation(self) -> float:
        """Worst radian-row violation (the descent-gradient caps); ``≤ 0`` = satisfied."""
        return self._worst(angular=True)

    def is_feasible(self, tol_m: float = DEFAULT_TOL_M, tol_rad: float = DEFAULT_TOL_RAD) -> bool:
        """True if every metre row holds within ``tol_m`` AND every radian row within ``tol_rad``."""
        return self.max_violation() <= tol_m and self.max_angular_violation() <= tol_rad

    def summary(self, tol_m: float = DEFAULT_TOL_M, tol_rad: float = DEFAULT_TOL_RAD) -> str:
        """Per-row report. Pass the SAME tolerances the caller judges feasibility with —
        the headline ``feasible=`` and the per-row flags use them, so a custom-tolerance
        caller no longer gets a printout that contradicts its own verdict."""
        lines = [
            f"max violation = {self.max_violation():+.2f} m / "
            f"{math.degrees(self.max_angular_violation()):+.3f} deg  "
            f"(feasible={self.is_feasible(tol_m, tol_rad)})"
        ]
        for name, v in self.violations.items():
            v = np.ravel(v)
            worst = float(v.max()) if v.size else 0.0
            if name.endswith(_ANGULAR_SUFFIX):
                flag = "  <-- VIOLATED" if worst > tol_rad else ""
                lines.append(f"  {name:<40s} worst={math.degrees(worst):+9.3f} deg{flag}")
            else:
                flag = "  <-- VIOLATED" if worst > tol_m else ""
                lines.append(f"  {name:<40s} worst={worst:+9.2f} m{flag}")
        return "\n".join(lines)


class ConstraintSet:
    """An ordered list of approach segments to evaluate a trajectory against (NumPy path)."""

    def __init__(self, segments: list[SegmentSpec]):
        if not segments:
            raise ValueError("a ConstraintSet needs at least one segment")
        self.segments = segments

    def evaluate(self, segment_nodes: list[np.ndarray]) -> ConstraintReport:
        """Evaluate, given one ``(k_i, 7)`` state-node array per segment (aligned with order)."""
        if len(segment_nodes) != len(self.segments):
            raise ValueError(
                f"expected {len(self.segments)} node-groups, got {len(segment_nodes)}"
            )
        violations: dict[str, np.ndarray] = {}
        for index, (seg, nodes) in enumerate(zip(self.segments, segment_nodes)):
            for name, value in segment_violations(seg, np.asarray(nodes, dtype=float)).items():
                # Violation names embed the segment's idents, which default to "" — two
                # same-kind default-ident legs (or a route through the same fix pair
                # twice) collide, and a dict update silently DROPPED the earlier leg's
                # rows, reading a violating trajectory as feasible. Disambiguate by
                # segment position — as a PREFIX, so the ".descent" suffix keeps
                # classifying the radian-unit rows.
                violations[name if name not in violations else f"seg{index}:{name}"] = value
        return ConstraintReport(violations)
