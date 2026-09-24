"""The executor's own parameters (executor design §10), and the constraints the design derives them under.

Every parameter here is the executor's; what the vocabulary already fixes (tolerances, the intercept
angle, the corridor, the classes' angles, the speed band, the turn rates and the bank limit, the lead) is read from
the `VocabularySpec`, never restated. Where each value comes from is the design's §9–§10 and the executor spec's
(E7) — the vocabulary, the executor's own constraints (method A), or, for the values still measured (the speed
changes' pace and the landing aim, the vocabulary-only plan's second stage), the data; this container only checks
that a set of values is one the laws can fly:

- the sentence's step a whole number of cycles (the judge reads the flown track at the sentence's rows);
- ``τ_ψ ≥ 2 Δt`` (§4.1 constraint 1: each cycle removes at most half the heading error; the heading words' own law
  floors its time left at the same 2 Δt, `lateral.word_rate`);
- ``τ_γ ≥ 2 Δt``; the vocabulary's bank limit inside the grader's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ts_transformer.autopilot.inverse import BANK_MAX_RAD
from ts_transformer.autopilot.sentence import CLOCKS
from ts_transformer.instructions.spec import VocabularySpec


@dataclass(frozen=True)
class ExecutorParams:
    cycle_s: float                 # Δt, the control period
    heading_time_constant_s: float  # τ_ψ, the roll-out of the executor's own turns
    bank_rate_deg_s: float         # p, how fast the bank moves
    path_time_constant_s: float    # τ_γ, the path-angle inner loop
    path_rate_factor: float        # γ̇_max as a multiple of the tube's lower bound (§5.1)
    decel_mps2: float              # a_dec
    accel_mps2: float              # a_acc
    unspecified_decel_mps2: float  # a_unspec
    land_aim_height_m: float       # where "descend to land" aims: this far above the pointed threshold
    land_window_low_m: float       # ... and the heights it may cross at instead, to stay inside the word's tube
    land_window_high_m: float
    timeout_factor: float          # the flight's own remaining time × this (§8.3)
    word_clock: str                # which clock a replay says a truth sentence's words on (`sentence.CLOCKS`, §11)

    def check(self, spec: VocabularySpec) -> None:
        if abs(spec.step_s / self.cycle_s - round(spec.step_s / self.cycle_s)) > 1e-9:
            raise ValueError(f"the sentence's step {spec.step_s:g} s is not a whole number of {self.cycle_s:g} s cycles")
        if self.heading_time_constant_s < 2.0 * self.cycle_s:
            raise ValueError(f"τ_ψ {self.heading_time_constant_s:g} s is under 2 Δt ({2 * self.cycle_s:g} s)")
        if self.path_time_constant_s < 2.0 * self.cycle_s:
            raise ValueError(f"τ_γ {self.path_time_constant_s:g} s is under 2 Δt")
        if not 0.0 < spec.turn_bank_max_deg <= math.degrees(BANK_MAX_RAD):
            raise ValueError(f"the vocabulary's bank limit {spec.turn_bank_max_deg:g}° is outside the grader's "
                             f"(0, {math.degrees(BANK_MAX_RAD):g}°]")
        if not 0.0 <= self.land_window_low_m <= self.land_aim_height_m <= self.land_window_high_m <= spec.landing_max_height_m:
            raise ValueError(f"the landing aim {self.land_aim_height_m:g} m and its window {self.land_window_low_m:g}–"
                             f"{self.land_window_high_m:g} m are not inside the landing condition's "
                             f"0–{spec.landing_max_height_m:g} m, the aim inside its window")
        if self.word_clock not in CLOCKS:
            raise ValueError(f"word_clock {self.word_clock!r} is none of {CLOCKS}")
        positive = (self.cycle_s, self.bank_rate_deg_s, self.path_rate_factor,
                    self.decel_mps2, self.accel_mps2, self.unspecified_decel_mps2, self.timeout_factor)
        if min(positive) <= 0.0:
            raise ValueError("every rate, factor and period must be positive")
