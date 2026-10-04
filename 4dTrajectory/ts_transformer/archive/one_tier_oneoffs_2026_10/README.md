# one_tier_oneoffs_2026_10 — finished one-tier diagnostics, the runway-intent R0 / R1.1 runners and three readouts

Archived 2026-10-04 on the user's decisions ("A 类可以搬"; then "B 类的也搬进去"; `eta_error_readout`, `latent_probe` and
`latent_fan_readout` of that second list moved, see the end) after a check of the test suite. The first group (kinematic ablation
to runway-intent R1.1) had no entry in `docs/reference/runners.md`, no live importer, and no mention in the two-tier design
documents or `intents.json`; the three readouts had a `runners.md` entry each and no live importer. They were still listed by
`run_ts.py --list` only because `experiments/__main__.py` discovers every module in `experiments/`.
Off the import path on purpose (`tests/test_architecture.py`): nothing live imports it.

| file | was |
|---|---|
| `experiments/kinematic_ablation.py` | state path: kinematic-loss weight chosen by held-out accuracy band, then smoothness |
| `experiments/overfit_diagnostic.py` | state path: can the model memorise a small fixed outer-train subset (2026-07-27, `docs/history/2026-07_state_path/2026-07-27_small_sample_overfit_diagnostic.zh.md`) |
| `experiments/clock_attribution.py` | control path: a checkpoint's error split by the learned duration, geometry and clock (validation only) |
| `experiments/control_capacity_ceiling.py` | control path: per-flight control reconstruction ceiling, the trained network only initialising (development flights only) |
| `experiments/runway_intent_r0.py`, `runway_intent_r0_readout.py` | runway-intent R0: how well causal context alone names the landing runway; the five airports' R0a / R0b in one table set |
| `experiments/runway_intent_r11.py`, `runway_intent_r11_readout.py` | runway-intent R1.1: a candidate-symmetric listwise head against R1's per-runway head |
| `experiments/eta_error_readout.py` | R4 (B0): the ETA error per stratum out of existing `summary.json` files, and q50's error beside the rollout's |
| `experiments/latent_probe.py` | R5 (L2.f): a latent checkpoint's prior / posterior densities, displacement in prior sigmas, per-dimension KL |
| `experiments/latent_fan_readout.py` | R6 (4(a)): the latent sample fan read by the quantile fan's gate protocol |

Plans and results: `docs/history/2026-08_control_path/control_parameter_prediction.zh.md` (§5, §7) for the first four,
`docs/history/2026-09_runway_intent/2026-09-13_runway_intent_plan.zh.md` (§7, §14–§15) for the runway-intent runners.
Their tests are archived beside them, unmodified (`tests/`); they do not run. One exception: the two tests of
`eta_error_readout` sat inside `test_two_head_duration.py` beside tests of live code, so `tests/test_two_head_duration_eta_readout.py`
is an EXCERPT — the file's header and imports, the helpers those two tests call, and the two tests, as they stood at
`dev-two-tier-v4` `4efbf5d3` plus the test-fixture move of the same day — and they left the live file (its other 20 tests are unchanged).
The runners' `docs/reference/runners.md` entries (R4, R5, R6) are in `docs/reference/entries.md`; the live file keeps a heading line each.

What stayed live: `experiments/runway_intent_r1.py` and `runway_intent_r1_readout.py` (`run_naming.py` imports r1; R1.1's
archived runner imports both), `experiments/runway_hypotheses.py` (R0b's runner; with R0's readout archived nothing live
reads it), and the three design figures (`approach_clock_figure`, `approach_legs_figure`, `scene_sample_figure`: they draw
the SVGs committed under `docs/two_tier/figures/`). The published outputs these runners made stay where they are:
`4dTrajectory/outputs/POOLED/experiments/runway_intent_r0_20260913/`, `runway_intent_r11_20260913/`,
`runway_intent_r11b_20260913/`.

Asked for and NOT moved, with the reason: `experiments/eta_calibration.py` (the only producer of the conformal table that the
live `predict --cta-from-quantiles` reads, and `cli/predict.py` tells the user to run it; its stub is also used by a live test
of `test_two_head_duration`) and `experiments/chain_sensitivity.py` (its tests are the only ones pinning `inference/receding.py`,
which the live no-token lockstep imports; `test_publish_ts_experiment_trajectories` imports its `RECORDS_BLOCK`).
