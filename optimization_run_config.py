"""Pure optimizer-run provenance shared by the optimizer CLI and its root runner."""

from __future__ import annotations

from typing import Any

FITTING_SCHEMES = {
    "hs": "hermiteSimpsonNormalizedFullTransport",
    "trapezoidal": "trapezoidalNormalizedFullTransport",
    "rk4": "rk4NormalizedFullTransport",
}
DEFAULT_MAX_DURATION_S = 2000.0
DEFAULT_ROLLOUT_DT_S = 0.5


def build_optimization_config(
    *,
    constrained_iaf: bool,
    fitting: str,
    n_segments: int,
    n_seg_per_phase: int,
    state_substeps: int | None,
    max_duration_s: float,
    rollout_dt_s: float,
    iaf_selection: str = "shortest",
) -> dict[str, Any]:
    """Return the exact active solver/rollout recipe persisted in ``summary.json``."""
    if fitting not in FITTING_SCHEMES:
        raise ValueError(
            f"unknown fitting {fitting!r}; choose from {sorted(FITTING_SCHEMES)}"
        )
    config: dict[str, Any] = {
        "mode": "constrained_iaf" if constrained_iaf else "unconstrained",
        "fitting": fitting,
        "transcription_scheme": FITTING_SCHEMES[fitting],
        "control_mesh": (
            {"segments_per_phase": n_seg_per_phase}
            if constrained_iaf
            else {"segments": n_segments}
        ),
        "state_substeps": state_substeps,
        "max_duration_s": max_duration_s,
        "rollout_dt_s": rollout_dt_s,
    }
    if constrained_iaf:
        config["iaf_selection"] = iaf_selection
    return config
