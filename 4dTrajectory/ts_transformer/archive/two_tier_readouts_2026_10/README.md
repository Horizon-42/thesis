# two_tier_readouts_2026_10 — finished two-tier readouts: measurements whose choice is made

Archived 2026-10-13 on the user's decision (the runner list of that day: "已定论的测量" and `multi_val_evaluation`;
"archive的同时，把测试也带过去 而且archive代码一律不做测试"). Each runner measured one thing for one choice of the
user, and the choice is recorded in the design documents; or, for `multi_val_evaluation`, it was a second grading path
beside the window loop's own landing verdict (the training line now reads landing from one judge only,
`autopilot/judge.landing_verdict` → `evaluation.evaluate_record`). Off the import path on purpose
(`tests/test_architecture.py`): nothing live imports it.

| file | was | the choice it served |
|---|---|---|
| `experiments/executor_turns.py` | A13: how much of the offset after a turn is the executor's heading law, how much the words' (R51) | D47 |
| `experiments/final_descent_tolerance.py` | A24: the closed loop's vertical tolerance H_final in the final descent (R52) | D66; H_final 7.5 m (2026-10-12) |
| `experiments/start_rules.py` | A33: the train closed loop read once for each start rule (R55) | D77 |
| `experiments/lead_ablation.py` | V12: the words' leads L, L_γ, L_v, one column at a time (R71) | V12 / D9: a lead of 4 s |
| `experiments/multi_windows.py` | MC1: the census of stage D's windows for the span L, its kinds and c_min (no runners.md entry) | D146 |
| `experiments/pass_gradients.py` | the gradient of each part of the loss at the start of a round's pass, stages C and D (R70) | stage D log §22 |
| `experiments/multi_val_evaluation.py` | a stage D round's val windows flown again, every commanded track graded by the evaluation package (R74) | the r7 val read of 2026-10-12 |

Their tests are archived beside them, unmodified (`tests/`); they do not run (`pyproject.toml` `norecursedirs`).
`test_final_descent_tolerance.py` includes the check that the test fixture's DA lies on the test final, and
`test_multi_val_evaluation.py` the test of `multi_train.selection_readout(window_ends=)`: both went with their files.
One EXCERPT: `tests/test_multi_windows_runner.py` — the header, imports and the one runner test of
`test_multi_windows.py`, as at `two-tier2-eval-landing` `7621d5b3`; the live file keeps its 10 tests of `multi/`.
The runners.md entries (R51, R52, R55, R70, R71, R74) are in `docs/reference/entries.md`; the live file keeps a heading
line each. Their outputs on disk stay where they are.

What stays live with no live caller now (not deleted; the user decides): `multi_train.selection_readout`'s
`window_ends` argument (only `multi_val_evaluation` passed it).
