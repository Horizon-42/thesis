"""Method A (executor design §9): the executor's own parameters, from the vocabulary alone — the user's rule of
2026-09-24: the executor takes no information beyond the vocabulary (a parameter mined from data means it will not
generalise).

- ``heading_time_constant_s`` (τ_ψ): the ease-out of the executor's OWN turns (its intercept, a go-around, the capture,
  the line) — the heading words have their own law, which arrives on each word a lead after it is heard
  (`lateral.word_rate`). The lead L, the time a word gives to arrive, so the executor's own turns settle on the
  scale the words do (at least 2Δt, `params.ExecutorParams.check`).

The roll rate p is the procedure standards' 5°/s, a fixed choice of the spec runner (`experiments/executor_spec.py`), not
derived here; the vocabulary sets only its floor, ``stopping_roll_rate_deg_s``: the slowest p at which the executor's own
turns (``e / τ_ψ``, `lateral.rate_for_error`) never outrun the rate a turn can still be stopped at before the bank limit
binds. The flown checks that set p before are archived: `archive/executor_vocabulary_only_2026_09/`.
"""

from __future__ import annotations

import math

from ts_transformer.instructions.spec import VocabularySpec


def heading_time_constant_s(spec: VocabularySpec, cycle_s: float) -> float:
    """τ_ψ, the executor's own turns' ease-out: the lead (module docstring); at least 2Δt."""
    if spec.heading_lead_s < 2.0 * cycle_s:
        raise ValueError(f"τ_ψ {spec.heading_lead_s:g} s would be under 2 Δt: the lead is too short")
    return spec.heading_lead_s


def stopping_roll_rate_deg_s(spec: VocabularySpec, heading_time_constant_s: float) -> float:
    """The slowest roll rate p, deg/s, at which the executor's own turns need no stopping limit: ``e / τ_ψ`` exceeds the
    stopping rate ``sqrt(2 g p e / V)`` only where it asks for a bank of ``tan φ > 2 p τ_ψ``, so the vocabulary's bank
    limit binds first when ``2 p τ_ψ ≥ tan φ_max`` (32° and 4 s: 4.48°/s)."""
    return math.degrees(math.tan(math.radians(spec.turn_bank_max_deg)) / (2.0 * heading_time_constant_s))
