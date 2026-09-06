# Archived: the 2026-08 oracle-teacher campaign

A **completed** campaign, kept in the repository as the record behind published numbers and
taken off the import path. Nothing here runs against the current package.

## What it was

Two halves of one idea — teach a control head what a known future implies:

- **Direct shooting** (`control_oracle/shooting.py`, `curriculum.py`): fit one flight's
  piecewise-constant schedule against its own observed track, with a segment-count
  curriculum and restarts. Driven by `runners/run_ts_control_oracle.py`.
- **Imitation pretraining** (`cohort.py`, `targets.py`, `optimization.py`, `evaluation.py`,
  `imitation.py`, `pretraining.py`): solve a cohort's schedules once, cache them to an
  `.npz`, then warm-start the network on them before ordinary training
  (`CachedSchedulePretrainer`, reached from `__main__ train --control-teacher-schedules`).
  Driven by `runners/run_ts_oracle_teacher_{audit,optimize}.py` and
  `runners/run_ts_simple_teacher_paired_cv.py`; `runners/plot_teacher_training.py` plots the
  `model_pretraining.history` block those runs wrote.

It was superseded by the **in-training imitation term** of the `simple-v3` recipe
(`train.control_imitation_mse` + `control_imitation_loss_weight`), which needs no cached
teacher and no separate optimisation stage. `imitation.py::control_imitation_loss` here is a
DIFFERENT formula from the live one; only one "imitation target" stays alive, and it is the
live one.

## The documents that cite it

- `docs/2026-08-02_oracle_teacher_experiment.zh.md` — the campaign itself.
- `docs/2026-08-16_control_simple_v1_development.zh.md` — `simple-v1`'s development, whose
  frozen teacher schedule (`SIMPLE_V1_TEACHER_SCHEDULE_SHA256`) is pinned in
  `control_oracle/pretraining.py`.
- `docs/control_parameter_prediction.zh.md` — the module-by-module call graph as it stood.

## Taken from

Commit `882d048` (`dev-t2`), package audit T2, 2026-09-07.

## Why it no longer runs

- Its checkpoints were already refused by `load_checkpoint` (the 2026-08-18 control-unit
  change: newtons → fraction of installed thrust).
- The `arc-length-geometry` objective and the dual terminal clock its formal recipe used were
  deleted in T1-11 / T1-13, and the `transport-chart-velocity` dynamics backend its runners
  pinned was retired with this archive.
- `control_oracle/optimization.py` still imports `control.oracle.basis`; that module is the
  one member with a live consumer and moved OUT to `control/basis_fit.py` (it is a fit, not a
  teacher — `run_ts_control_basis_oracle.py` is the live successor of the width study).
- Five names this campaign was the sole consumer of were deleted from the live package by the
  same commit and are VENDORED into `control_oracle/shooting.py` under an "vendored 2026-09-07"
  banner, so the unit is self-contained: `PHYSICAL_CRITERIA_DISTANCE_SCALE_M`,
  `PHYSICAL_CRITERIA_SMOOTH_MAX_TEMPERATURE`, `smooth_maximum` and `physical_criteria_loss`
  (originals: `git show 882d048:4dTrajectory/ts_transformer/physical_criteria.py`) and
  `refine_piecewise_constant_schedule` (original: `git show
  882d048:4dTrajectory/ts_transformer/control/dynamics/inverse.py`). Everything else it
  imports — `config`, `dataset`, `channels`, `control.envelope`, `control.loss.fixed_dt`,
  `physical_criteria.fixed_dt_position_ade_m` — is still live, and drifts without notice.
- `tests/` here are the campaign's own tests, renamed off pytest's `test_*.py` pattern so the
  suite does not collect them.
