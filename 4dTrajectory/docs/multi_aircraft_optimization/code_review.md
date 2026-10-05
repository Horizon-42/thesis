# Optimizer code review: refactor and speed findings

Style: ASD-STE100 (Simplified Technical English). Units: SI.

## 0. Status

| Item | Value |
|---|---|
| Scope | `4dTrajectory/optimization/` (collocation, approach_constraints, scenario_optimization, the older optimizers, the study scripts) and its two callers: `aeroviz_backend/optimization_backend.py` and `run_scenario_optimization.py` |
| Base commit | `dev-two-tier` `a804633e` |
| Code changes in this branch | None. This document only records findings |
| Findings | 13 (F1 to F13). 11 are verified on the code. 2 are judgement (F12 speed, F13 study scripts) |
| Findings that the multi-aircraft design needs first | F4, F5, F6, F7 (see `design.md` §8, step T0) |
| Decision state | Open. The user decides which findings to fix. Each finding that the user accepts gets one row in `docs/code-health-followups.md` |

A finding is **verified** when this review read the code line and the line shows the defect. A finding is
**judgement** when it is an opinion about cost or structure that no measurement supports yet.

## 1. Summary

The core is in good condition. `collocation/optimizer.py` has one optimizer for the free and the
procedure-constrained solves. Each constraint family is one function with one contract
(`(expr, lb, ub)`, `optimizer.py:89-94`). `approach_constraints` is the one source of the corridor,
glidepath and course math.

The problems are at the edges:

1. `scenario_optimization.py` is 1,634 lines and holds five different jobs (F6).
2. The batch imports the HTTP backend package, against the stated dependency direction (F4).
3. The backend and the batch build the constrained solve in two places (F5, the O4 seam class).
4. The optimizer has no point where a caller can add constraint rows (F7). Multi-aircraft
   separation needs this point.
5. Some code does not obey the project conventions: dead parameters (F2), fallbacks with
   `.get(...) or DEFAULT` (F3), stale docstrings (F1).

## 2. Findings

Priority: **P1** = fix before the multi-aircraft work. **P2** = fix soon, low risk. **P3** = user decision or low value.

### F1 · Stale docstrings (verified, P2)

- `optimization/scenario_optimization.py:1-27` says the module is a "teaching scaffold" and that the
  optimizer call (TODO ①) and the rollout (TODO ②) are not done. Both are done (`optimize_scenario`,
  `rollout_controls`).
- `optimization/scenario_optimization.py:1036` labels the IAF section "(NEW)".
- `optimization/approach_constraints/README.md:17` and `optimization/approach_constraints/state.py:5` say
  "We optimize with the `trapezoidalNormalizedFullTransport` scheme". The default is Hermite-Simpson
  (`hermiteSimpsonNormalizedFullTransport`, `collocation/optimizer.py` `_DEFAULT_CONSTRAINED_SCHEME`; K3 in
  `4dTrajectory/docs/optimizer_reference.md`). The frame statement stays true for all normalized schemes.

Fix: write the current state in the three docstrings. Risk: none.

### F2 · Dead parameters (verified, P2)

- `dt` (`DEFAULT_DT = 1.0`, `scenario_optimization.py:104`, comment "API parity") goes through
  `optimize_scenario` (`:230`), `optimize_scenarios` (`:737`), the two IAF functions (`:1310`, `:1361`),
  `_solve_iaf` and the CLI flag `--dt`. No solve reads it.
- `n_segments` goes into `_solve_iaf`. Its docstring (`:1237`) says that it is "accepted for CLI/API
  parity but unused".
- `optimization_config` (`optimization_run_config.py`) does not record `dt`. Thus the records do not
  depend on it, and removal changes no artefact.

A parameter that does nothing tells the reader that it does something. This is the same problem as the
project rule "a bound that can never bind is worse than no bound". Fix: remove `dt` from the batch API
and the CLI. Remove `n_segments` from the constrained path. Keep the backend `dt` only where an older
optimizer reads it.

### F3 · Fallbacks against the coding conventions (verified, P1 for the first item, P2 for the others)

The root `CLAUDE.md` forbids `.get(key, default)` fallbacks and silent approximations. These lines do one
of the two:

| Line (`scenario_optimization.py`) | Code | Effect |
|---|---|---|
| `:1209` | `apt = airport or scenario.source.get("arr_airport") or DEFAULT_AIRPORT` | A scenario with no `arr_airport` reads the KRDU procedures. The 150 m threshold check (`_require_procedure_threshold_agrees`) then fails with the message "built against different runway data". The message names the wrong cause |
| `:1159` | `(w.geometry_alt_ft or 0.0)` | A waypoint with no altitude reads as 0 m in the IAF ranking proxy `_path_length_m`. This can change the IAF that `"shortest"` selects. Nothing tells the user |
| `:428`, `:1419` | `scenario.source.get("id") or f"scenario{index}"` | A flight without `id` gets an invented name in the log |
| `:965` | `float(src.get("window_s") or DEFAULT_WINDOW_S)` | A scenario without `window_s` silently uses 15 s for its reference record |
| `:545` | `record.get("reason") or "unsolved (resumed)"` | A failed record without a reason gets an invented reason |
| `:1418` | `params.pop("selection", "shortest")` | The batch always sends `selection`. The default is a second copy of the CLI default |
| CLI `--airport` (`:1562`) | "airport ICAO fallback" | The flag exists only to feed the fallback of `:1209` |

Fix: make `arr_airport`, `id` and `window_s` required at the scenario boundary (`FlightScenario.from_dict`
already refuses other missing fields). Refuse a waypoint without altitude in `_path_length_m`, or state the
approximation as an option with a notice. Remove the `--airport` flag.

### F4 · The batch imports the HTTP backend (verified, P1)

- `scenario_optimization.py:1116` imports `aeroviz_backend.procedure_constraint.ProcedureConstraint`.
- `scenario_optimization.py:1241` imports `aeroviz_backend.procedure_segments.build_constraint_segments`.
- `aeroviz_backend/procedure_segments.py:3-8` states the direction: "backend → {constraints, optimizer};
  optimizer → constraints; constraints → nothing". The batch breaks this direction. The imports are inside
  the functions, so the cycle does not show at import time.

The procedure model (`ProcedureConstraint`), the bridge to `SegmentSpec` (`build_constraint_segments`) and
the IAF path assembly (`_iaf_full_paths`, `_concat_to_runway`, `_recompute_distances`,
`_identifier_match`, `scenario_optimization.py:1071-1135`) are not HTTP code. They are procedure geometry.

Fix: move them to one module outside the backend. Two candidates:
`approach_constraints/procedure.py` (next to the geometry it feeds), or `flight_scenarios/procedure_final.py`
(it already resolves the procedure file path). The backend and the batch then import the same module. The
multi-aircraft package needs this module too (the "established" test reads the final approach geometry).

### F5 · Two constructions of the constrained solve (verified, P1)

- The backend builds the segments and the `CollocationOptimizer` itself
  (`aeroviz_backend/optimization_backend.py:197-213`).
- The batch builds them again in `_solve_iaf` (`scenario_optimization.py:1222-1261`).
- `4dTrajectory/docs/optimizer_reference.md` O4 records three bugs that came from a change on one side only.
  `_run_batch` removed the duplication inside the batch. The duplication between the batch and the
  backend remains.

Fix: one public function in `optimization/`, for example
`solve_constrained(initial, target, segments, aircraft, options) -> ConstrainedSolve`. The backend and the
batch call it. The multi-aircraft rows (design §5) then reach both callers through one change.

### F6 · `scenario_optimization.py` holds five jobs (verified, P1 for the batch driver, P2 for the rest)

The file is 1,634 lines. It holds:

1. One solve and its record (`optimize_scenario`, `StateSample`, `ScenarioOptimization`).
2. The replay (`rollout_controls`, `_GroundCheckedSimulator`, `rollout_guard_altitude_m`).
3. The batch driver (`_run_batch`, `_resumable_record`, `_clear_stale_records`, the worker functions).
4. The reference records (`write_reference_records`, the observed-track store and its sweep).
5. The IAF procedure assembly and selection (F4 and `_IAF_SELECTORS`).

Fix: split it into modules with the same functions: `solve.py`, `replay.py`, `batch.py`, `references.py`,
`iaf.py`. Keep `scenario_optimization.py` as the CLI. Move the tests with their functions. Do not change
behaviour. The multi-aircraft batch (design §7) needs item 3 as a reusable driver that accepts a different
worker and a different record writer.

### F7 · No extension point for caller rows (verified, P1)

`CollocationOptimizer._build` (`collocation/optimizer.py:390-613`) makes every row inside one method.
The dispatcher (`:524-545`) calls only the procedure families. A caller cannot:

- add a constraint row,
- read the symbolic time of a node (the time is `start + Σ T_p` plus a fraction of the current phase; the
  code computes it only as numbers, after the solve, in `dense_node_times`, `:173`),
- add a term to the objective.

The multi-aircraft separation rows need all three (design §5.3). Fix: add one optional argument
`extra_rows(nodes, node_times, phase_of_node) -> list[(expr, lb, ub)]` to `optimize_free_time` and
`optimize_trajectory`. Use the same row contract that the families use (`optimizer.py:89-94`). Return the
symbolic node times from `_build` in `layout`.

Gate for this change: `4dTrajectory/docs/optimizer_reference.md` O2 says that IPOPT is sensitive to the
order of symbol creation. Thus with `extra_rows=None`, the change must not create a symbol before the
present symbols. Check this with a behaviour test: solve a fixed set of scenarios before and after the
change; the final times and the states must be bit-identical.

### F8 · Two different interactive IPOPT caps (verified, P2)

- `aeroviz_backend/optimization_backend.py:50`: `DEFAULT_MAX_ITERATIONS = 1000`. The HTTP request default
  is 1,000 iterations.
- `scenario_optimization.py:114-115` says: "`components.DEFAULT_MAX_ITERATIONS` (3000) is the
  interactive/backend default". K4 in `optimizer_reference.md` says the same.

The code and the two documents do not agree. Fix: the user selects one value. Then make the backend import
it from `collocation.components`, or write in K4 that the backend has its own cap and why.

### F9 · Private names in the public API (verified, P3)

`collocation/__init__.py` exports `_DEFECT_SCHEMES` and `_SOLVER_BACKENDS` in `__all__`. The leading
underscore says "private". Fix: export public names (`DEFECT_SCHEMES`, `SOLVER_BACKENDS`), or stop the
export.

### F10 · Two control and state envelopes (verified, P3)

- `collocation/components.py` `make_control_bounds` and `optimization/casadi_optimizer.py`
  `make_control_bounds` are two copies. A comment says they must be identical. Both hard-code the bank
  limit ±45°.
- The state bounds differ: γ in [−6°, 15°] in `components.make_state_bounds`, γ in [−10°, 30°] in
  `casadi_optimizer.make_state_bounds`.
- `CollocationOptimizer._aircraft_meta` (`collocation/optimizer.py:738-752`) hard-codes the load factor
  range 0.5 to 2.0 for every aircraft type.

Fix: one envelope module, imported by both optimizers. Write the source of each limit (aircraft data or a
stated modelling choice).

### F11 · Resume compares four identity fields (verified, P3)

`_resumable_record` (`scenario_optimization.py:527-531`) compares `id`, `runway`, `icao24` and
`landing_time_utc`. `write_reference_records` uses `flight_key` since 2026-09-25 (O10 note). The two
select the same flights. Fix: compare `flight_key` (`flight_scenarios.identity.flight_key`), the one
definition of flight identity.

### F12 · The NLP is built again for each solve (judgement, P2, measure first)

- `_build` writes the complete symbolic graph for each solve. Each Hermite-Simpson sub-interval calls the
  right-hand side three times inline (`collocation/schemes.py:62-83`). A 30-node phase thus has about 90
  copies of the dynamics graph.
- `last_solve_timings` (`optimizer.py:357`, `:375`) records the solve time. It does not record the build
  time. Thus nobody knows the build fraction.
- The multi-aircraft loop (design §5.4) solves one window several times. The build cost then repeats.

Proposal:

1. Add the build time to `last_solve_timings`. Measure it on a fixed sample of 50 scenarios per mode.
2. If the build is a large fraction, make one cached casadi `Function` for one sub-interval defect (like
   `_geodetic_rhs`, `schemes.py:48`) and call it at each node. This makes the graph smaller.
3. Gate: the same behaviour test as F7. A change in the graph can change the IPOPT path (O2). Accept the
   change only when the solve rate and the final times do not change on the fixed sample, or when the user
   accepts the measured change.

### F13 · Older optimizers and study scripts (judgement, P3, user decision)

- The batch uses three schemes only (`optimization_run_config.FITTING_SCHEMES`). `_DEFECT_SCHEMES` has 16
  (`collocation/schemes.py:287`). The backend menu shows 20 collocation names
  (17 single-phase and 3 multiphase, `optimization_backend.py:63-97`) and six older optimizers: `transcription_optimizor`,
  `warm_start_transcription_optimizor`, `variable_time_warm_start_transcription_optimizor`,
  `least_squares_transcription_optimizor`, `single_shooting_optimizor`, `casadi_optimizer`.
- Five study scripts have no importer: `fixed_enu_frame_error.py`, `geodetic_vs_reanchored_error.py`,
  `transport_term_comparison.py`, `solver_backend_benchmark.py`, `collocation_scheme_comparison.py`. Their
  data files (`*_data.json`) are next to the code.
- K3 says that the trapezoidal and rk4 fittings are kept "for comparison studies only".

Each of these items adds tests and maintenance. Proposal (the user decides): remove the older optimizers
from the backend menu, and move them with their tests into `archive/` unchanged (the project's archive
rule). Move the study scripts and their data into `optimization/studies/`.

## 3. What this review did not find

- No defect in the collocation math, the normalization or the constraint families. Their tests and
  docstrings agree with the code that this review read.
- No thread-safety defect: the casadi caches (`schemes.py:37-59`) document that callers serialize.

## 4. Recommended order

1. F7 and F4, then F5 (P1, needed by the multi-aircraft design, step T0).
2. F6 item 3 (the batch driver), then the rest of F6.
3. F3 (`:1209` first), F1, F2, F8, F11 (small and low risk).
4. F12 (measure first).
5. F9, F10, F13 (the user decides).
