# one_tier_runners_2026_10 — one-tier runners that nothing imports any more

Archived 2026-10-13 on the user's decision (the runner list of that day: "一层 ts 线中无人导入的"; "archive的同时，把测试也带过去
而且archive代码一律不做测试"). The one-tier prediction line (state / control output, CV, ablations, readouts) is no longer
the thesis line; these 20 runners had no live importer and are not used by the two-tier model. The one-tier runners that
something live still uses stay in `experiments/` (`pipeline`, `frame_ablation`, `anytime_curve`, `eta_calibration`,
`chain_sensitivity` and others). Off the import path on purpose (`tests/test_architecture.py`): nothing live imports it.

| file | was |
|---|---|
| `experiments/cv.py`, `plot_results.py` | the pooled CV stage with the project defaults; its plots and CSV tables |
| `experiments/coordinate_ablation.py` | ENU against runway-aligned frame, paired CV, one-shot test release |
| `experiments/history_ablation.py` | lookback lengths compared at one common anchor floor |
| `experiments/predictability_report.py` | validation-only error, dispersion and coverage report |
| `experiments/latent_readout.py` | multimodal readout of a latent control arm (C8) |
| `experiments/lead_time_error.py` | displacement error against lead time from the anchor, per stratum |
| `experiments/quantile_fan_readout.py` | B3: does the quantile fan contain the flight (R3) |
| `experiments/runway_hypotheses.py`, `runway_intent_r1.py`, `runway_intent_r1_readout.py` | runway intent: one forecast per candidate runway; R1's learned runway head and its readout |
| `experiments/score_control_arms.py`, `compare_control_arms_paired.py`, `compare_control_arms_stratified.py` | the bank-wiggle metrics per arm; paired and stratified arm comparisons |
| `experiments/compare_frame_arms.py`, `compare_constraint_arms.py` | the airport-frame and final-approach constraint campaign readouts |
| `experiments/specific_force_teacher_distribution.py` | where the truth lies in each longitudinal coordinate (specific-force design §2.4) |
| `experiments/straight_in_residual_readout.py`, `wind_residual_readout.py` | where the straight-in residual lives; does the METAR headwind explain it |
| `experiments/trace_architecture.py` | the traced architecture SVGs of the tutorials (`docs/tutorials/arch_*.svg`) |

Their tests are archived beside them, unmodified (`tests/`); they do not run. Five EXCERPTS, as at
`two-tier2-eval-landing` `7621d5b3` (header, imports, the helpers those tests call, the tests; the live file keeps the
rest): `tests/test_anchor_policies_history_ablation.py` (1 test), `test_cta_from_quantiles_fan_readout.py` (2),
`test_latent_control_latent_readout.py` (1; the live file keeps that test without its readout part), `test_ts_pipeline_cv.py` (8: the CV runner, the plotter, the coordinate
ablation) and `test_ts_predictability_report.py` (7; the live file keeps 2).
The runners.md entry R3 is in `docs/reference/entries.md`; the live file keeps its heading line. The package README's
passages on `cv`, `plot_results`, `history_ablation`, `coordinate_ablation` and `predictability_report` are in
`docs/cut_sections.md`. Their outputs on disk stay where they are.

What stays live with no live caller now (not deleted; the user decides): `data/runway_features.py` (R1's features; its
other users were archived on 2026-09 and 2026-10-04) and `inference/arm_readout.py` (the arm table of
`compare_frame_arms` and the two residual readouts); each keeps its own live test.
