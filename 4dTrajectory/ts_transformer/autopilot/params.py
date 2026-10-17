"""The executor's own parameters (executor design §9), and the constraints the design derives them under.

Every parameter here is the executor's or its judge's; what the vocabulary already fixes (the classes' angles, the leads,
the tolerances) is read from the `VocabularySpec`, never restated. Every value here is the roll rate the user decided
(D2), an assumed actuator time constant (§9.2: not identified from data, the acceptance runs over the literature's range)
or one of the design's fixed choices (the runner's constants); the decision-altitude check's bounds are no parameter
(vocabulary §5.8, D38: one definition with the evaluation module) — the executor takes nothing from data (the user's rule,
2026-09-24); this container only checks that a set of values is one the laws can fly:

- every rate, factor, time constant and period finite and positive (first: the other checks divide by them);
- the sentence's row interval (the data's step, or a coarser Δ, vocabulary §4.8) a whole number of cycles (the executor
  hears a row's words on the cycle that starts it);
- the plant's integrator stable at every time constant (`plant.executor_dynamics`: the lag model's own check, C16);
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
    cycle_s: float                    # Δt, the control period
    bank_rate_deg_s: float            # p, the largest rate the actual bank is rolled at (§4.4, D2)
    thrust_time_constant_s: float     # τ_x, the thrust actuator's first-order lag (§2.2; assumed, §9.2)
    bank_time_constant_s: float       # τ_φ
    load_time_constant_s: float       # τ_n
    timeout_factor: float             # the flight's own remaining time × this (§8.3)
    start_rule: str                   # how the start state reads the observed rows before it (D77, `START_RULES`)

    @property
    def time_constants_s(self) -> tuple[float, float, float]:
        """``(τ_x, τ_φ, τ_n)``, in the order of the commands (thrust fraction, bank, load factor)."""
        return self.thrust_time_constant_s, self.bank_time_constant_s, self.load_time_constant_s

    def check(self, spec: VocabularySpec, row_interval_s: float) -> None:
        from ts_transformer.autopilot.plant import executor_dynamics

        positive = (self.cycle_s, self.bank_rate_deg_s, *self.time_constants_s, self.timeout_factor)
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("every rate, factor, time constant and period must be finite and positive")
        for name, seconds in (("step", spec.step_s), ("row interval", row_interval_s)):
            if abs(seconds / self.cycle_s - round(seconds / self.cycle_s)) > 1e-9:
                raise ValueError(f"the sentence's {name} {seconds:g} s is not a whole number of {self.cycle_s:g} s cycles")
        executor_dynamics(self)          # the lag model refuses a time constant its integrator cannot hold (C16)
        if self.start_rule not in START_RULES:
            raise ValueError(f"start rule {self.start_rule!r} is none of {tuple(START_RULES)}")
