# 4dTrajectory — optimizer, constraints, batch tooling

`optimization/` holds the casadi/IPOPT direct-collocation optimizer, approach constraints and
batch runners. `ts_transformer/` (learned prediction) has its **own** `CLAUDE.md` — read that
one for anything torch-side.

## Gotchas (recurring, verified)

- **casadi symbolic construction is NOT thread-safe** (global SXElem pools; concurrent NLP builds
  → heap corruption → C++ `abort()`/SIGABRT that Python can't catch). Casadi-heavy endpoints run
  in an isolated worker subprocess (`aeroviz_backend/isolated_backend.py`); all in-process casadi
  entry points serialize on `casadi_lock.CASADI_LOCK`. Toggle isolation off with
  `AEROVIZ_ISOLATE_SOLVER=0` (to get a native traceback).
- **IPOPT is sensitive to CasADi symbol-creation order** — `make_dynamics()` must run before NLP
  decision symbols are created.
- **Heading convention**: the dynamics model ψ is math-ENU (0 = East, CCW toward North;
  `V_east = V·cosγ·cosψ`). `approach_constraints.geometry.course_bearing` returns this convention
  (`atan2(Δn, Δe)`). Mixing in compass convention reads an aligned aircraft as a 90° intercept.
- **"Batch edition" seam class**: the batch callers in `scenario_optimization.py` duplicate wiring
  the backend HTTP path also has — several bugs (stale `_solve_iaf` unpack, trapezoidal left
  behind after the HS flip, missing `n_seg_per_phase`) came from updating one path and missing
  the other. When changing optimizer wiring, update BOTH and their seam tests. The worst
  intra-file instance is gone: `optimize_scenarios` and `optimize_scenarios_constrained_iaf`
  are thin fronts over ONE `_run_batch` driver (2026-08-23) — batch mechanics changes go there,
  once.
- **The dense plan export carries the solver's OWN node times, never an even spread.**
  Multiphase node spacing is `(T_p/n_seg)/m_sub_p` with `m_sub` auto-selected per phase, so
  spreading dense nodes evenly over `[0, T]` time-warps every constrained plan (the orange
  "Optimizer plan" CZML track animated wrong until 2026-08-23). `CollocationOptimizer`
  exposes `last_dense_state_times_s` (pure helper `dense_node_times`); any new exporter of
  `last_dense_states_geo` must consume it. States files written before the fix have wrong
  `optimizer_states[].t` (positions correct; eval/rollout records unaffected).
- **The constrained solve does NOT move its target, and must not.** `_iaf_setup` used to
  snap it onto the procedure document's last waypoint; that waypoint and the arrival
  manifest's `runway_target` are two renderings of one CIFP threshold and round differently
  (0.05-0.22 m over the 25 runways in service; KRDU 32 = 2.98 m, KSMF 35R = 39.45 m), which
  put every `runway_cons` record outside `evaluation`'s 1 cm target check and killed the
  whole sweep. `_require_procedure_threshold_agrees` now validates the procedure AGAINST the
  scenario target at `_FRAME_ANCHOR_TOLERANCE_M` (150 m) instead — the displaced-threshold
  case the snap existed to catch (KSJC 12L, 390 m against the NASR config) still fails loudly.
  The optimizer, the evaluator and the arrival manifest all read `harvest.airports.Runway`.
- **`--max-iterations` is the batch's biggest cost lever.** Measured serially on KRDU: a
  scenario that ends `Maximum_Iterations_Exceeded` costs ~56 s (the full 3000-iteration
  budget) against ~4.3 s for one that solves — **6.7 % of the flights, ~48 % of an
  unconstrained batch's CPU**. Plumbed from both runners through both solve paths, and
  recorded in `summary.json`'s `optimization_config`: a lower cap turns slow successes into
  failures, so it is a different experiment and `--skip-optimize` refuses to reuse across it.
  The default stays 3000 — lowering it is a research decision, not a performance one.
- **`--resume` exists because `summary.json` is written only at the end.** A 70k-solve batch
  runs for tens of hours; before, a crash discarded every finished record with it. Resume
  reads back complete record pairs whose identity matches a CURRENT scenario and solves the
  rest. `_clear_stale_records` still sweeps orphans — resume narrows which files survive, it
  never turns the sweep off — and summary counts come from the roster, not from what the
  process happened to write (a roster that ends up incomplete raises).
- **Resume verifies the solver CONFIG, not just identity** (2026-08-23): every eval record
  (solved and failed) is stamped with the batch's `optimization_config`, and
  `_resumable_record` rejects a mismatch or a missing stamp — a resume across a changed
  `--max-iterations`/`--fitting`/`--rollout-dt` re-solves instead of laundering the old
  records into a summary that claims the new config (which `--skip-optimize` would then
  trust). Pre-stamp records are therefore never resumable. Summary rows quote the EVAL
  record's `final_time_s` (the replay's last sample, shorter than the plan for
  guard-truncated replays), so fresh and resumed rows agree.
- **Reference records quote a SHARED observed track.** The two prepared target datasets
  reference the same flights and their reference records differ only in `target_state`, so
  the states live once in `shared_references/observed_tracks/` and each record points at
  them through the contract's `states_ref` (134 KB/flight -> 64 KB). Cache contract
  `optimization-references-v3-shared-tracks` hashes the track alongside the record; the store
  is swept against the UNION of all sibling reference dirs, never one dataset's roster (that
  would delete the other's tracks).
- **Stale-artifact hygiene (write side)**: `_clear_stale_records` deletes top-level
  `*_states.json`/`*_eval.json` at optimization-batch start; `write_reference_records` clears
  reference records when rebuilding and its v3 manifest anchors every record by source identity +
  SHA-256 (record AND shared observed track). Comparison publication never pre-deletes the live batch: immutable
  generation-suffixed CZML/report files are written first, `comparison_index.json` is the atomic
  commit point, and only then are unreferenced generations pruned. `--skip-optimize` validates
  the complete summary/eval/states/reference roster, reference hashes, identities, and available
  prepared-input signatures before reuse. Record-filename suffixes + `REFERENCES_DIR` + the
  `summary.json` row shape (`summary_row`) + the reference cache contract
  (`REFERENCE_CACHE_SCHEMA`, `file_sha256`, `observed_track_path`) are single-sourced in
  `optimization/evaluation_export.py` — the pipeline runner imports them (its restated
  mirror is gone).
- Stale docs (historically inaccurate, kept): `4dTrajectory/docs/direct_collocation_hermite_simpson.zh.md`
  §5 and `geodetic_dynamics_transport.zh.html` describe the old HS-planner + RK4-polish pipeline.

## Key defaults & constants (current)

- **Mesh**: `collocation/optimizer.py` `DEFAULT_N_SEGMENTS = 8`, `DEFAULT_N_SEG_PER_PHASE = 3`
  (single source; backend + batch import them; `run_scenario_optimization.py` mirrors 8/3 with a
  "MUST match" comment). Frontend/backend unconstrained `n_segments` default = 10 (a different
  knob). Multiphase mesh = n_seg_per_phase × legs (`n_segments` doesn't apply).
- **State substeps**: auto per phase ≈ 3 s state step, cap 16 (`_TARGET_STATE_STEP_S` /
  `_MAX_STATE_SUBSTEPS`); explicit `--state-substeps` / frontend "State substeps" (0 = auto,
  clamp 0–64) overrides. Do NOT lower below auto (M=4 → 14.5 km rollout error); on unconstrained
  solves M=32 improves accuracy/optimum; constrained solves don't need big M (per-node inequality
  rows make big M explode solve time).
- **Fitting**: constrained + unconstrained default = Hermite-Simpson
  (`hermiteSimpsonNormalizedFullTransport`; frontend
  `DEFAULT_TRAJECTORY_OPTIMIZER = casadiMultiphaseNormalizedFullTransport`). `FITTING_SCHEMES`:
  `hs` / `trapezoidal` / `rk4` via `--fitting-type`. Trapezoidal is dynamically unfaithful on
  aggressive min-time floor-riding solves (5–15 km rollout drift vs HS metres); rk4 is
  basin-fragile there (needs M=64) — both kept for comparison studies only.
- **IPOPT**: `components.DEFAULT_MAX_ITERATIONS = 3000` (`ipopt.max_iter` set explicitly on BOTH
  IPOPT constructions — verbose and quiet; request `maxIterations` reaches both backend
  branches). The third construction in that function is the `sqpmethod` backend, which
  **hardcodes `max_iter: 100` and ignores `max_iterations` entirely**. Linear solver:
  `AEROVIZ_IPOPT_LINSOL` (default `mumps`) + `AEROVIZ_IPOPT_HSLLIB` — HSL hook dormant (free MA27
  measured 3–27× slower than MUMPS on these small NLPs; kept for a future MA57 attempt). Batch
  speed levers, in order of measured effect: `--max-iterations` (see above), then `--jobs`
  (the pipeline driver defaults to cores−4; the library auto is half the cores).
- **Altitude floor**: `altitude_floor_m(target) = target − 5 m` (a real operational floor min-time
  solves ride; was −300 m which they dove to). Transition-phase floor = min(start alt, first
  leg's published entry floor `_first_leg_entry_floor_m`) − margin. Rollout guard:
  `ROLLOUT_GUARD_MARGIN_M = 5.0`, `rollout_guard_altitude_m(target) = altitude_floor_m(target) − 5`
  (zero margin truncated faithful floor-riding replays on cm integration noise). `min_altitude_m`
  is REQUIRED on `rollout_controls`/`simulate_controls` (no silent sea-level default).
- **ψ corridor**: constrained solves bound the heading variable to the route's heading hull ± 90°
  (`_route_psi_profile`, `_PSI_CORRIDOR_SLACK_RAD`) — this killed the whole looping/crawling
  local-optimum family. Terminal ψ pinned on the route-unwrapped branch (the first element of
  `_route_psi_profile`'s return); per-phase heading guesses = own leg course.
- **Join/passage constraints** (`approach_constraints` + `collocation/optimizer.py`): the ONE
  forced fix passage is the pre-FAF fix, within its leg's k·RNP disc; FAC join = on-course
  (`fac_cross_track = 0`) with along-course distance in
  `[d_FAF + L_final/5, d_FAF + max_offset]` (max_offset auto = half the leg into the FAF; join
  guess = window middle; `max_join_offset_m=0` → the single point 1/5 before the FAF);
  branch-aware intercept box `|ψ_join − course_branch| ≤ 30°`; two-tier FAC alignment ±30°
  join→FAF, ±10° (`_FAC_ALIGN_TIGHT_DEG`) FAF→threshold; vertical glidepath window binds only
  `d ≤ d_faf_m`, upstream the published FAF minimum (`prefaf_floor_m`) applies. Tiny
  duration-split regularizer `1e-4·Σ(Tp/T_max)²` on free-time multi-phase solves. Constraint
  families are explicit functions dispatched from `_build`: `_terminal_pin_rows` /
  `_fac_join_rows` / `_prefaf_fix_rows` / `_leg_path_rows` / `_fac_alignment_rows`.
- **Transition phase**: prepended (unconstrained) when the start is farther than
  `_first_fix_join_tolerance_m` from the first fix (= first leg's k·RNP when it has one, 2 km
  fallback for LPV-first). Frame-anchor contract validated loudly (`segments[-1].end_ne` +
  `lpv.ltp_ne` at origin ±150 m). `DEFAULT_K_MARGIN` + `STANDARD_INTERCEPT_MAX_DEG`
  single-sourced in `approach_constraints`.
- **Playback drift guard**: `playbackDriftM` on every optimize response; stderr WARNING above
  `PLAYBACK_DRIFT_WARN_M = 50`.

## Open items

- **KRDU RW32 is systematically hard, and it is NOT the old truncation artifact.** The full
  2026-07-20 batch re-run (post-truncation/floor/HS/identity fixes, all 15 airport×category cells
  fresh) kept the concentration: KRDU runway_cons RW32 = 79 offTarget + 59 failed vs 60 clean
  solves (198 flights; every other runway ≤ 9 offTarget), and KRDU **asdb RW32 fails 197/198**
  (IPOPT infeasible). Runway/procedure-specific — check against the per-leg-RNP item below
  (H05LZ is RNP-AR) before touching solver knobs. KSTL runway_cons has a milder cluster (12R
  53/200, 30R 41/168, 30L 30/200, 24 21/80; the "all IAF(s) infeasible" rows repeatedly name
  `PAULY`).
- Per-leg RNP is not extracted from CIFP — RNP-AR procedures (H05LZ) get the default RNP 1.0 disc
  (~926 m at k=0.5) instead of ~278 m (RNP 0.3).
- CIFP leg speed restrictions not extracted (no speed-bearing data source in the dataset yet; the
  canonical `speedMaxKt` field is ready).
- HSL linear-solver hook dormant (free MA27 measured slower than MUMPS); revisit with an MA57
  academic license.
- **Pre-existing numpy failure in `collocation/tests/test_optimizer.py::test_fixed_time_objective_weights_control_effort_at_one`
  is BACK (2026-07-21).** `float(np.array(grad(x0))[0])` raises
  `TypeError: only 0-dimensional arrays can be converted to Python scalars` under numpy 2.x. It
  went green on 2026-07-20 and failed again on 07-21 with no optimizer change in between, so it
  tracks the numpy version, not the code. Verified unrelated to any working-tree change by
  re-running with the tree stashed. Modeling suite is otherwise 588 pass.
