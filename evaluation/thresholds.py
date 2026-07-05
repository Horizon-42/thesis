"""Regulation-derived gates on the final-state deviation.

Both gates are threshold-referenced (the targets in this project anchor on the
CIFP landing threshold) and cite FAA Order 8260.3F / 8260.58D — the documents in
``docs/regulation/``. Override any value per run (constructor / CLI flags).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviationThresholds:
    """Pass/fail limits for the final state vs the target state.

    lateral_max_m — FAA Order 8260.58D §3-1-3(3), Formula 3-1-1: the LPV lateral
        guidance course SEMIWIDTH at the threshold (LTP) is
        ``max(350 ft, tan(1.5°) · d_GARP)``. The 350 ft floor ≈ 106.75 m (the
        order rounds course widths to 0.25 m increments) is the TIGHTEST
        full-scale deflection any LPV final can have at the threshold — a final
        state farther off laterally is outside the lateral guidance sector of
        every LPV approach.

    vertical_below_max_m / vertical_above_max_m — FAA Order 8260.58D §1-3-2
        (Threshold Crossing Height): the default TCH is designed for a 30 ft
        wheel crossing height (WCH); the minimum WCH is 20 ft and the maximum is
        50 ft. Relative to a target placed AT the published TCH point, the
        acceptable window is therefore −10 ft (3.05 m low) to +20 ft (6.10 m
        high). This assumes the target altitude is threshold elevation + TCH,
        which holds for this project's CIFP-anchored optimizer targets.
    """

    lateral_max_m: float = 106.75
    vertical_below_max_m: float = 3.05
    vertical_above_max_m: float = 6.10
