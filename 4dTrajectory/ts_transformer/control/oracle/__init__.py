"""Future-aware teachers. Train-only by construction: everything here reads a real future.

``shooting`` + ``curriculum`` are the direct-shooting oracle; the remaining modules are the
imitation-pretraining path that was previously the separate ``oracle_teacher`` package —
two halves of one idea that had no reason to be two top-level entries.
"""

from control.oracle.targets import OracleTeacherTarget, build_inverse_dynamics_target

__all__ = ("OracleTeacherTarget", "build_inverse_dynamics_target")
