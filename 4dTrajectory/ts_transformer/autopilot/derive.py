"""Method A (executor design §9): the executor's own parameters, from the vocabulary alone — the user's rule of
2026-09-24: the executor takes no information beyond the vocabulary (a parameter mined from data means it will not
generalise).

- ``heading_time_constant_s`` (τ_ψ): the ease-out of the executor's OWN turns (its intercept, a go-around, the capture,
  the line) — the heading words have their own law, which arrives on each word a lead after it is heard
  (`lateral.word_rate`). The lead L, the time a word gives to arrive, so the executor's own turns settle on the
  scale the words do (at least 2Δt, `params.ExecutorParams.check`);
- ``roll_rate_deg_s`` (p): the vocabulary's bank limit over the lead — the executor can bank as steeply as a word
  admits within the time a word gives to arrive (32° / 4 s = 8°/s; the user, 2026-09-24).

The flown checks that set p before (the executor's own largest turn, turns said word by word on synthetic flights)
are archived: `archive/executor_vocabulary_only_2026_09/`.
"""

from __future__ import annotations

from ts_transformer.instructions.spec import VocabularySpec


def heading_time_constant_s(spec: VocabularySpec, cycle_s: float) -> float:
    """τ_ψ, the executor's own turns' ease-out: the lead (module docstring); at least 2Δt."""
    if spec.heading_lead_s < 2.0 * cycle_s:
        raise ValueError(f"τ_ψ {spec.heading_lead_s:g} s would be under 2 Δt: the lead is too short")
    return spec.heading_lead_s


def roll_rate_deg_s(spec: VocabularySpec) -> float:
    """p, deg/s: the vocabulary's bank limit reached within one lead (module docstring)."""
    return spec.turn_bank_max_deg / spec.heading_lead_s
