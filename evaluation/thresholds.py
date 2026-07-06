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

    lateral_max_m — FAA Order 8260.58D §3-1-5.c(3) "Course width at threshold"
        (pages 3-7/3-8), Formula 3-1-1: the course width at the threshold (LTP)
        is ``greater of 350 ft or tan(1.5°) · d_GARP`` — a ONE-SIDED value (the
        order's term is "course width", but Figure 3-1-7 annotates it ±350 ft
        about the centerline, i.e. what this package calls the semiwidth /
        full-scale deflection). 350 ft = 106.68 m, and the order's own rule
        ("convert the result to meters and round to the nearest 0.25-meter
        increment") makes it 106.75 m — the value never appears verbatim in the
        PDF. It is the TIGHTEST deflection any LPV final can have at the
        threshold — a final state farther off laterally is outside the lateral
        guidance sector of every LPV approach.

    vertical_below_max_m / vertical_above_max_m — FAA Order 8260.58D
        §1-3-1.f(2)(b) "Threshold crossing height" (page 1-27), item 1: "The
        default TCH provides a 30-foot WCH. The minimum WCH is 20 feet and the
        maximum WCH is 50 feet." Relative to a target placed AT the published
        TCH point (WCH 30), the acceptable window is therefore −10 ft (3.05 m
        low) to +20 ft (6.10 m high) — derived offsets, not verbatim numbers.
        This assumes the target altitude is threshold elevation + TCH, which
        holds for this project's CIFP-anchored optimizer targets.
    """

    lateral_max_m: float = 106.75
    vertical_below_max_m: float = 3.05
    vertical_above_max_m: float = 6.10
