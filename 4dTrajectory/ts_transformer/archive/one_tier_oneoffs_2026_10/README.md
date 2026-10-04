# one_tier_oneoffs_2026_10 — finished one-tier diagnostics and the runway-intent R0 / R1.1 runners

Archived 2026-10-04 on the user's decision ("A 类可以搬") after a check of the test suite: each module here had no entry in
`docs/reference/runners.md`, no live importer, and no mention in the two-tier design documents or `intents.json`. They
were still listed by `run_ts.py --list` only because `experiments/__main__.py` discovers every module in `experiments/`.
Off the import path on purpose (`tests/test_architecture.py`): nothing live imports it.

| file | was |
|---|---|
| `experiments/kinematic_ablation.py` | state path: kinematic-loss weight chosen by held-out accuracy band, then smoothness |
| `experiments/overfit_diagnostic.py` | state path: can the model memorise a small fixed outer-train subset (2026-07-27, `docs/history/2026-07_state_path/2026-07-27_small_sample_overfit_diagnostic.zh.md`) |
| `experiments/clock_attribution.py` | control path: a checkpoint's error split by the learned duration, geometry and clock (validation only) |
| `experiments/control_capacity_ceiling.py` | control path: per-flight control reconstruction ceiling, the trained network only initialising (development flights only) |
| `experiments/runway_intent_r0.py`, `runway_intent_r0_readout.py` | runway-intent R0: how well causal context alone names the landing runway; the five airports' R0a / R0b in one table set |
| `experiments/runway_intent_r11.py`, `runway_intent_r11_readout.py` | runway-intent R1.1: a candidate-symmetric listwise head against R1's per-runway head |

Plans and results: `docs/history/2026-08_control_path/control_parameter_prediction.zh.md` (§5, §7) for the first four,
`docs/history/2026-09_runway_intent/2026-09-13_runway_intent_plan.zh.md` (§7, §14–§15) for the runway-intent runners.
Their tests are archived beside them, unmodified (`tests/`); they do not run.

What stayed live: `experiments/runway_intent_r1.py` and `runway_intent_r1_readout.py` (`run_naming.py` imports r1; R1.1's
archived runner imports both), `experiments/runway_hypotheses.py` (R0b's runner; with R0's readout archived nothing live
reads it), and the three design figures (`approach_clock_figure`, `approach_legs_figure`, `scene_sample_figure`: they draw
the SVGs committed under `docs/two_tier/figures/`). The published outputs these runners made stay where they are:
`4dTrajectory/outputs/POOLED/experiments/runway_intent_r0_20260913/`, `runway_intent_r11_20260913/`,
`runway_intent_r11b_20260913/`.
