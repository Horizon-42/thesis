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
# MUST match collocation.components.DEFAULT_MAX_ITERATIONS. This module is deliberately
# import-light (the pipeline runner shells out and never imports casadi), so it cannot
# import it; scenario_optimization.py DOES import the real one and is the single source.
DEFAULT_MAX_ITERATIONS = 3000


def build_optimization_config(
    *,
    constrained_iaf: bool,
    fitting: str,
    n_segments: int,
    n_seg_per_phase: int,
    state_substeps: int | None,
    max_duration_s: float,
    rollout_dt_s: float,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
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
        # In the persisted recipe because it is verdict-relevant, not just a speed knob:
        # a lower cap turns slow-but-converging solves into failures, so a batch run at a
        # different cap is a different experiment and `--skip-optimize` must not reuse it.
        "max_iterations": max_iterations,
    }
    if constrained_iaf:
        config["iaf_selection"] = iaf_selection
    return config
