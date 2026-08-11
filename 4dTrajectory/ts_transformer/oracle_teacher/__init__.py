"""Train-only future-aware control teachers for isolated experiments."""

from oracle_teacher.targets import OracleTeacherTarget, build_inverse_dynamics_target

__all__ = ("OracleTeacherTarget", "build_inverse_dynamics_target")
