# manoeuvre_lockstep_2026_10 — two-tier plan v3's no-token closed loop, its gates and its queue

Archived 2026-10-13 on the user's decision (the runner list of that day: "旧 v3 无词表线"; "archive的同时，把测试也带过去 而且
archive代码一律不做测试"). Plan v3 (2026-09-18, `docs/history/2026-09_two_tier_v3/`) was the manoeuvre line: an
executor network flown in lockstep with no second layer, its (lookback, segment) grid, the gates read off it. Its
intent-code layer was archived on 2026-09-20 (`archive/manoeuvre_codes_2026_09/`); the no-token part stayed live as
"stage A". The two-tier model that exists (instruction words, the rule executor `autopilot/`, the prior) replaced it
and imports none of it. Off the import path on purpose (`tests/test_architecture.py`): nothing live imports it.

| file | was |
|---|---|
| `manoeuvre/__init__.py`, `lockstep.py` | the lockstep: the executor predicts again every round on its own flown rows, protocol `none` (schema `ts-manoeuvre-lockstep-v4`) |
| `manoeuvre/gates.py` | plan v3 §3.3's gates: the grid cell (A1) and a reading against its protocol-none baseline (A3, B, B1) |
| `manoeuvre/failure_modes.py` | how a non-crossing closed-loop flight failed, six modes in the course frame (A2) |
| `experiments/manoeuvre_lockstep.py` | flies the lockstep over the executor's val cohort and writes the readout |
| `experiments/executor_grid_gate.py`, `executor_relative_gate.py`, `executor_failure_modes.py` | the three readings above, off lockstep artefacts |
| `experiments/two_tier_grid_queue.py` | trains the (L, Δ) grid one cell at a time and reads each cell |
| `experiments/plan_cohort.py` | the development cohort a random-anchor run trains on, one per grid cell (`--arms`) |

Their tests are archived beside them, unmodified (`tests/`); they do not run. Three EXCERPTS, as at
`two-tier2-eval-landing` `7621d5b3` (header, imports, the helpers those tests call, the tests): `tests/test_two_tier_v3_grid.py`
(8 tests; the live file keeps the test of `dataset.series_from_row`), `tests/test_architecture_manoeuvre.py` (the 2
tests of the `manoeuvre/` package's import rules) and `tests/test_publish_ts_experiment_trajectories_lockstep.py` (the
publisher's variant blocks against the runners'; the live file keeps that test with its two asserts on live blocks). `test_executor_failure_modes.py` includes the test of
`inference/export.copy_record_subset`. The live `test_architecture.py` lost `NOT_TWO_TIER_RUNNERS` (these gates) and
`EVALUATION_NAMES` (`multi_val_evaluation`'s, archived in `two_tier_readouts_2026_10`).
The runners.md entries R8 and R9 are in `docs/reference/entries.md`; the live R8 keeps `frame_ablation --only`, which
stays. The publisher keeps its `manoeuvre_lockstep` / `failure_modes` record blocks, so the published categories stay
readable; the campaign trees under `4dTrajectory/outputs/KRDU/experiments/` stay where they are.

Deleted from the live code on 2026-10-13 (the user's order): `inference/export.copy_record_subset` with
`metrics_from_row`, its only user. `executor_failure_modes` here does not import any more; the deleted code is at
two-tier2 `b8426ff5`.
