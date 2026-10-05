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
  `inverse.attitude`);
- the start rule one of `START_RULES` (D77).

THE START RULE (vocabulary D77): how the executor's start state takes its airspeed, track and path angle from the
observed 2 s rows at or before the row it starts at (`flights.start_kinematics`), and how the stored observed rows of a
closed-loop sentence take their track, ground speed and vertical rate. A rule is a least-squares line through the rows of
the seconds before the row: ``displacement-2s`` (the row and the one before), ``trailing-fit-8s``, ``trailing-fit-15s``.
``centred-fit-15s`` is the data plane's own velocity (a fit over 15 s centred on each sample, which reads up to 7.5 s
after the row): measured beside the others for the user's choice only (A33), and never written into a spec by
`experiments/executor_spec.py` (`FORMAL_START_RULES`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ts_transformer.instructions.spec import VocabularySpec

#: Each start rule's window, the seconds before the row its line is fitted through (None: the data plane's centred fit).
START_RULES: dict[str, float | None] = {"displacement-2s": 2.0, "trailing-fit-8s": 8.0, "trailing-fit-15s": 15.0,
                                         "centred-fit-15s": None}
#: The rules an executor spec may hold (`experiments/executor_spec.py`); the centred fit is A33's comparison only.
FORMAL_START_RULES = ("displacement-2s", "trailing-fit-8s", "trailing-fit-15s")


@dataclass(frozen=True)
class ExecutorParams:
    cycle_s: float                 # Δt, the control period
    bank_rate_deg_s: float         # p, how fast the bank moves
    path_time_constant_s: float    # τ_γ, the path-angle inner loop
    path_rate_factor: float        # γ̇_max as a multiple of the tube's lower bound (§5.1)
    timeout_factor: float          # the flight's own remaining time × this (§8.3)
    start_rule: str                # how the start state reads the observed rows before it (D77, `START_RULES`)

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
        if self.start_rule not in START_RULES:
            raise ValueError(f"start rule {self.start_rule!r} is none of {tuple(START_RULES)}")
