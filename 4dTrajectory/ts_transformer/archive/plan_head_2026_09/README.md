# plan_head_2026_09 — the plan-and-guidance HEAD (design v5, 2026-09-09 … 09-12)

Archived 2026-09-18 when the manoeuvre-token plan
(`docs/2026-09-18_manoeuvre_token_plan.zh.md`, §5.1) replaced it. Design, steps and every
measurement (§12.1–§12.11: the oracle ceilings, rolled windows, the order hold, the K = 4
mixture, the assigned time, the pooled five-airport head): `docs/2026-09-09_plan_and_guidance_design.md`
(superseded). Off the import path on purpose: nothing live imports it.

**What stayed live**: the guidance layer — `outputs/plan/guidance/` and `outputs/plan/skeleton.py`
moved to **`outputs/guidance/`** — is the manoeuvre-token plan's second executor (plan §2.6) and
the baseline its gate X reads. Two constants the guidance took from here now live with it
(`route.ON_COURSE_FIX_M`, `timing.SPEED_MAX_MPS`).

| file | was |
|---|---|
| `model.py`, `strategy.py`, `labels.py`, `extractors.py`, `forecast.py`, `rolled.py`, `plan__init__.py` (the package docstring; renamed so the archive stays unimportable) | `outputs/plan/` — the head (operating parameters + next instruction, the mixture), its labels and extractors, the rolled-window table, the `plan` strategy |
| `plan_oracle.py`, `plan_oracle_pair.py`, `plan_rolled_windows.py`, `plan_fan_readout.py`, `plan_next_readout.py`, `plan_extractors.py` | `experiments/` — the oracle / lockstep runner and its readouts |
| `runway_intent/` | `experiments/runway_intent_r2*.py`, `r3*.py`, `r31*.py`, `r32_diagnosis.py` — the runway-intent R2 (mixture over runway hypotheses flown by the plan head) and R3 (scheduler) stages, which read the head; R0 / R1 / R1.1 (the runway head itself, what R1.2 builds on) stayed live |
| `docs_scripts/phase0_intent_diagnostics.py` | `docs/` — the Phase 0 intent diagnostics (read the closure geometry too) |

Their tests are archived beside them, unmodified (`tests/`); they do not run. The campaign tree
`4dTrajectory/outputs/KRDU/experiments/plan_guidance_20260910/` stays in place: its lockstep JSON
(`step3d_lockstep_l1`) is the rule-guidance baseline the new gates read, and its published
categories stay. The `PREDICTION_PLAN` name survives only in `config.PREDICTION_OUTPUTS_RETIRED`;
a stored plan config is refused at load.
