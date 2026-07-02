"""Assemble a whole approach into one constraint set, and evaluate a trajectory against it.

Provided wiring. A :class:`ConstraintSet` holds the ordered segments; ``evaluate`` runs each
segment's :func:`~constraints.segments.segment_violations` over that segment's state nodes and
flattens everything into one violation vector with the package convention ``g ≤ 0 ⇔ satisfied``.

**Node → segment mapping.** Which collocation nodes belong to which leg is decided by the
optimizer (by along-track position; design-doc §7). Here ``evaluate`` takes the nodes
*already grouped per segment* — one ``(k_i, 7)`` array per segment — so the package stays a pure
constraint library and does not guess membership. :func:`split_contiguous` is a convenience for
the common case where the trajectory array is ordered start→runway.

**Multiple IAFs.** Per design-doc §5, do NOT encode "nearest of several IAFs" with a ``min``
(non-convex, non-smooth). Build one :class:`ConstraintSet` per candidate IAF and solve each as a
separate problem, then keep the best objective. (This module models a single chosen route.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segments import SegmentSpec, segment_violations


@dataclass
class ConstraintReport:
    """The result of evaluating a trajectory: named violation arrays (``≤ 0`` = satisfied)."""

    violations: dict[str, np.ndarray]

    def vector(self) -> np.ndarray:
        """All violations flattened into one 1-D vector (the NLP ``g(x)``)."""
        if not self.violations:
            return np.zeros(0)
        return np.concatenate([np.ravel(v) for v in self.violations.values()])

    def max_violation(self) -> float:
        """Worst (largest) violation; ``≤ 0`` means the whole trajectory is feasible."""
        vec = self.vector()
        return float(vec.max()) if vec.size else 0.0

    def is_feasible(self, tol_m: float = 1.0) -> bool:
        """True if every constraint holds to within ``tol_m`` (metres)."""
        return self.max_violation() <= tol_m

    def summary(self) -> str:
        lines = [f"max violation = {self.max_violation():+.2f} m  "
                 f"(feasible={self.is_feasible()})"]
        for name, v in self.violations.items():
            v = np.ravel(v)
            worst = float(v.max()) if v.size else 0.0
            flag = "  <-- VIOLATED" if worst > 1.0 else ""
            lines.append(f"  {name:<40s} worst={worst:+9.2f} m{flag}")
        return "\n".join(lines)


class ConstraintSet:
    """An ordered list of approach segments to evaluate a trajectory against.

    NumPy evaluation path — used by the demo/tests to score a concrete trajectory. The optimizer
    does NOT use this; it builds the symbolic rows directly (inline in
    ``collocation.optimizer``), using the same `segment_violations_*`
    primitives, fed CasADi vectors). Keep the two in sync if you change the violation set.
    """

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
        for seg, nodes in zip(self.segments, segment_nodes):
            violations.update(segment_violations(seg, np.asarray(nodes, dtype=float)))
        return ConstraintReport(violations)


def split_contiguous(traj: np.ndarray, counts: list[int]) -> list[np.ndarray]:
    """Split an ordered ``(K, 7)`` trajectory into per-segment groups of the given sizes."""
    traj = np.asarray(traj, dtype=float)
    if sum(counts) != traj.shape[0]:
        raise ValueError(f"counts sum to {sum(counts)} but trajectory has {traj.shape[0]} nodes")
    out, i = [], 0
    for c in counts:
        out.append(traj[i:i + c])
        i += c
    return out


def partition_node_indices(num_nodes: int, spans: list[tuple[float, float]]) -> list[list[int]]:
    """Assign each of ``num_nodes`` ordered state nodes to a segment — a FIXED partition.

    ``spans[j] = (along_start_m, along_end_m)`` is segment ``j``'s along-track interval on the
    procedure (cumulative distance, ascending and contiguous, starting at 0). Node ``i`` is placed
    at the nominal along-track ``(i + 0.5)/num_nodes · total`` and assigned to the segment whose
    interval contains it. This is **constant** (it does not depend on the decision variables) —
    required, because node→segment membership can't be decided from the (variable) node positions
    inside the NLP. It is an approximation (nodes are time-spaced, not distance-spaced); see
    design-doc §7. Returns one index list per segment (some may be empty).
    """
    groups: list[list[int]] = [[] for _ in spans]
    if not spans:
        return groups
    total = spans[-1][1]
    if total <= 0.0:                      # degenerate -> everything in the first segment
        groups[0] = list(range(num_nodes))
        return groups
    edges = [s[0] for s in spans] + [total]
    last = len(spans) - 1
    for i in range(num_nodes):
        along = (i + 0.5) / num_nodes * total
        j = last
        for jj in range(len(spans)):
            if along < edges[jj + 1]:
                j = jj
                break
        groups[j].append(i)
    return groups
