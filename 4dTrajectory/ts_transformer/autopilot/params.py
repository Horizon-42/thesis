"""The executor's own parameters (executor design §10), and the constraints the design derives them under.

Every parameter here is the executor's or its judge's; what the vocabulary already fixes (tolerances, the lined-up
angle, the classes' angles, the speed band, the turn rates and the bank limit, the lead) is read from the
`VocabularySpec`, never restated. Every value here comes from the vocabulary (method A, `derive`), is the roll rate the
procedure standards cite, is one of the design's fixed choices (the runner's constants), or is one of the decision-
altitude check's two tolerances, which the user gives (design §5.8, O2: REQUIRED on the spec runner's command line) —
the executor takes nothing from data (the user's rule, 2026-09-24); this container only checks that a set of values is
one the laws can fly:

- every rate, factor, time constant, period and tolerance finite and positive (first: the other checks divide by them);
- the sentence's row interval (the data's step, or a coarser Δ, design §4.8) a whole number of cycles (the executor
  hears a row's words on the cycle that starts it);
- ``τ_ψ ≥ 2 Δt`` (§4.1 constraint 1: each cycle removes at most half the heading error; the heading words' own law
  floors its time left at the same 2 Δt, `lateral.word_rate`);
- ``τ_γ ≥ 2 Δt`` (the vocabulary's bank limit is checked against the grader's where it is flown,
  `inverse.attitude`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ts_transformer.autopilot.sentence import CLOCKS
from ts_transformer.instructions.spec import VocabularySpec


@dataclass(frozen=True)
class ExecutorParams:
    cycle_s: float                 # Δt, the control period
    heading_time_constant_s: float  # τ_ψ, the roll-out of the executor's own turns
    bank_rate_deg_s: float         # p, how fast the bank moves
    path_time_constant_s: float    # τ_γ, the path-angle inner loop
    path_rate_factor: float        # γ̇_max as a multiple of the tube's lower bound (§5.1)
    timeout_factor: float          # the flight's own remaining time × this (§8.3)
    word_clock: str                # which clock a replay says a truth sentence's words on (`sentence.CLOCKS`, §11)
    #: The decision-altitude check (design §5.8, O2): at the DA point the lateral offset within this share of the FAS
    #: cone's half-width at that distance (1 = the cone itself), and the height within this of the published glidepath.
    decision_cone_share: float
    decision_glidepath_tolerance_m: float

    def check(self, spec: VocabularySpec, row_interval_s: float) -> None:
        positive = (self.cycle_s, self.heading_time_constant_s, self.bank_rate_deg_s, self.path_time_constant_s,
                    self.path_rate_factor, self.timeout_factor, self.decision_cone_share,
                    self.decision_glidepath_tolerance_m)
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("every rate, factor, time constant, period and tolerance must be finite and positive")
        for name, seconds in (("step", spec.step_s), ("row interval", row_interval_s)):
            if abs(seconds / self.cycle_s - round(seconds / self.cycle_s)) > 1e-9:
                raise ValueError(f"the sentence's {name} {seconds:g} s is not a whole number of {self.cycle_s:g} s cycles")
        if self.heading_time_constant_s < 2.0 * self.cycle_s:
            raise ValueError(f"τ_ψ {self.heading_time_constant_s:g} s is under 2 Δt ({2 * self.cycle_s:g} s)")
        if self.path_time_constant_s < 2.0 * self.cycle_s:
            raise ValueError(f"τ_γ {self.path_time_constant_s:g} s is under 2 Δt")
        if self.word_clock not in CLOCKS:
            raise ValueError(f"word_clock {self.word_clock!r} is none of {CLOCKS}")
