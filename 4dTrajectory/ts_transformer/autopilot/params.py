"""The executor's own parameters (executor design §10), and the constraints the design derives them under.

Every parameter here is the executor's or its judge's; what the vocabulary already fixes (tolerances, the lined-up
angle, the classes' angles, the speed band, the turn rates and the bank limit, the lead) is read from the
`VocabularySpec`, never restated. Every value here is the roll rate the procedure standards cite or one of the
design's fixed choices (the runner's constants); the decision-altitude check's bounds are no parameter (vocabulary §5.8,
D38: one definition with the evaluation module) — the executor takes nothing from data (the user's rule, 2026-09-24); this container only checks that a set of values is
one the laws can fly:

- every rate, factor, time constant and period finite and positive (first: the other checks divide by them);
- the sentence's row interval (the data's step, or a coarser Δ, vocabulary §4.8) a whole number of cycles (the executor
  hears a row's words on the cycle that starts it);
- ``τ_γ ≥ 2 Δt`` (the vocabulary's bank limit is checked against the grader's where it is flown,
  `inverse.attitude`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ts_transformer.instructions.spec import VocabularySpec


@dataclass(frozen=True)
class ExecutorParams:
    cycle_s: float                 # Δt, the control period
    bank_rate_deg_s: float         # p, how fast the bank moves
    path_time_constant_s: float    # τ_γ, the path-angle inner loop
    path_rate_factor: float        # γ̇_max as a multiple of the tube's lower bound (§5.1)
    timeout_factor: float          # the flight's own remaining time × this (§8.3)

    def check(self, spec: VocabularySpec, row_interval_s: float) -> None:
        positive = (self.cycle_s, self.bank_rate_deg_s, self.path_time_constant_s, self.path_rate_factor,
                    self.timeout_factor)
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("every rate, factor, time constant and period must be finite and positive")
        for name, seconds in (("step", spec.step_s), ("row interval", row_interval_s)):
            if abs(seconds / self.cycle_s - round(seconds / self.cycle_s)) > 1e-9:
                raise ValueError(f"the sentence's {name} {seconds:g} s is not a whole number of {self.cycle_s:g} s cycles")
        if self.path_time_constant_s < 2.0 * self.cycle_s:
            raise ValueError(f"τ_γ {self.path_time_constant_s:g} s is under 2 Δt")
