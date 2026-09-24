"""The executor's own parameters (executor design §10), and the constraints the design derives them under.

Every parameter here is the executor's; what the vocabulary already fixes (tolerances, the intercept
angle, the corridor, the classes' angles, the speed band) is read from the `VocabularySpec`, never
restated. Where each value comes from — data, method A, method B — is the design's §9–§10 and the
executor spec's (E7); this container only checks that a set of values is one the laws can fly:

- the sentence's step a whole number of cycles (the judge reads the flown track at the sentence's rows);
- ``τ_ψ ≥ 2 Δt`` (§4.1 constraint 1: each cycle removes at most half the heading error);
- ``r_turn · τ_ψ ≤ heading_continue_lead_deg`` (§4.1 constraint 2: a split turn's next word arrives
  before the roll-out begins, so the aircraft does not level between the parts);
- ``τ_γ ≥ 2 Δt``; ``φ_cap`` inside the vocabulary's bank ceiling and the grader's;
- every delay ``≥ 0`` (a word acts once said) and ``d_ψ`` inside the late start the turn envelope allows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ts_transformer.autopilot.inverse import BANK_MAX_RAD
from ts_transformer.autopilot.sentence import Delays
from ts_transformer.instructions.spec import VocabularySpec


@dataclass(frozen=True)
class ExecutorParams:
    cycle_s: float                 # Δt, the control period
    turn_rate_deg_s: float         # r_turn, the steady turn rate
    bank_cap_deg: float            # φ_cap
    heading_time_constant_s: float  # τ_ψ, the roll-out
    bank_rate_deg_s: float         # p, how fast the bank moves
    path_time_constant_s: float    # τ_γ, the path-angle inner loop
    path_rate_factor: float        # γ̇_max as a multiple of the tube's lower bound (§5.1)
    decel_mps2: float              # a_dec
    accel_mps2: float              # a_acc
    unspecified_decel_mps2: float  # a_unspec
    land_aim_height_m: float       # where "descend to land" aims: this far above the pointed threshold
    delays: Delays                 # d_ψ, d_h, d_v
    timeout_factor: float          # the flight's own remaining time × this (§8.3)

    def check(self, spec: VocabularySpec, *, early_words: bool = False) -> None:
        """``early_words``: allow a negative delay — a word acting before it is said. Only the sensitivity
        check's probes pass it (they measure what the labeller's late reading costs); a spec never does."""
        if abs(spec.step_s / self.cycle_s - round(spec.step_s / self.cycle_s)) > 1e-9:
            raise ValueError(f"the sentence's step {spec.step_s:g} s is not a whole number of {self.cycle_s:g} s cycles")
        if self.heading_time_constant_s < 2.0 * self.cycle_s:
            raise ValueError(f"τ_ψ {self.heading_time_constant_s:g} s is under 2 Δt ({2 * self.cycle_s:g} s)")
        if self.turn_rate_deg_s * self.heading_time_constant_s > spec.heading_continue_lead_deg:
            raise ValueError(f"r_turn · τ_ψ = {self.turn_rate_deg_s * self.heading_time_constant_s:.1f}° exceeds the "
                             f"{spec.heading_continue_lead_deg:g}° a split turn's next word leads by")
        if self.path_time_constant_s < 2.0 * self.cycle_s:
            raise ValueError(f"τ_γ {self.path_time_constant_s:g} s is under 2 Δt")
        if not 0.0 < self.bank_cap_deg <= min(spec.turn_bank_max_deg, math.degrees(BANK_MAX_RAD)):
            raise ValueError(f"φ_cap {self.bank_cap_deg:g}° outside (0, {spec.turn_bank_max_deg:g}°]")
        if not early_words and min(self.delays.heading_s, self.delays.vertical_s, self.delays.speed_s) < 0.0:
            raise ValueError(f"{self.delays}: a word cannot take effect before it is said")
        if self.delays.heading_s > spec.turn_start_delay_max_s:
            raise ValueError(f"d_ψ {self.delays.heading_s:g} s exceeds the {spec.turn_start_delay_max_s:g} s a turn "
                             f"may start late (vocabulary §2.3)")
        if not 0.0 <= self.land_aim_height_m <= spec.landing_max_height_m:
            raise ValueError(f"the landing aim {self.land_aim_height_m:g} m is outside the landing condition's "
                             f"0–{spec.landing_max_height_m:g} m")
        positive = (self.cycle_s, self.turn_rate_deg_s, self.bank_rate_deg_s, self.path_rate_factor,
                    self.decel_mps2, self.accel_mps2, self.unspecified_decel_mps2, self.timeout_factor)
        if min(positive) <= 0.0:
            raise ValueError("every rate, factor and period must be positive")
