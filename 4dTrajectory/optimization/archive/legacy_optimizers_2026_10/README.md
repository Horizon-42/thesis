# Legacy optimizers (archived 2026-10-05)

These six optimizers are archived without change, with their tests:

- `transcription_optimizor.py`, `least_squares_transcription_optimizor.py`, `warm_start_transcription_optimizor.py`,
  `variable_time_warm_start_transcription_optimizor.py` (scipy transcription with alpha controls),
- `single_shooting_optimizor.py` (scipy single shooting),
- `casadi_optimizer.py` (casadi multiple shooting, `casadiIpopt`).

Why: the batch and the backend use only `collocation.CollocationOptimizer` (direct collocation, free and procedure-
constrained). The backend menu showed these six beside it. They added tests and maintenance and no current result
uses them (`4dTrajectory/docs/multi_aircraft_optimization/code_review.md` F13; the user accepted the finding on
2026-10-05).

`tests/` holds their tests from `4dTrajectory/optimization/tests/`; `tests_4dTrajectory/` holds the second copy of
the single-shooting test from `4dTrajectory/tests/`. The repository's pytest configuration does not collect
`archive/` directories, and nothing imports these files. Do not edit them; to use one again, move it back with its
tests.
