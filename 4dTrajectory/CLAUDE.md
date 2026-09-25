# 4dTrajectory — optimizer, constraints, batch tooling

`optimization/` holds the casadi/IPOPT direct-collocation optimizer, approach constraints and
batch runners. `ts_transformer/` (learned prediction) has its **own** `CLAUDE.md` — read that
one for anything torch-side.

**This file is an index**: each line ends in the ID of its full text in
`docs/optimizer_reference.md` (`O` gotchas, `K` defaults; moved there verbatim 2026-09-16). A new
fact gets a new ID there and one line here. Open items (KRDU RW32, per-leg RNP, CIFP speed
restrictions, HSL): repo `docs/open-items.md`, "Optimizer".

## Gotchas (recurring, verified)

- **casadi symbolic construction is NOT thread-safe** (heap corruption → a SIGABRT Python cannot
  catch): casadi-heavy endpoints run in an isolated worker subprocess
  (`aeroviz_backend/isolated_backend.py`), in-process entry points serialize on
  `casadi_lock.CASADI_LOCK`; `AEROVIZ_ISOLATE_SOLVER=0` for a native traceback (O1).
- **IPOPT is sensitive to CasADi symbol-creation order** — `make_dynamics()` must run before the
  NLP decision symbols are created (O2).
- **ψ is math-ENU** (0 = East, CCW toward North; `approach_constraints.geometry.course_bearing` =
  `atan2(Δn, Δe)`); mixing in compass convention reads an aligned aircraft as a 90° intercept (O3).
- **"Batch edition" seam class**: the batch callers in `scenario_optimization.py` duplicate wiring
  the backend HTTP path has — change BOTH and their seam tests; batch mechanics go in the ONE
  `_run_batch` driver (O4).
- The dense plan export carries the solver's OWN node times (`last_dense_state_times_s`), never
  an even spread; states files from before 2026-08-23 have wrong `optimizer_states[].t` (O5).
- **The constrained solve does NOT move its target, and must not** —
  `_require_procedure_threshold_agrees` validates the procedure against the scenario target at
  `_FRAME_ANCHOR_TOLERANCE_M` (150 m) instead (O6).
- **`--max-iterations` is the batch's biggest cost lever** (6.7 % of flights hit the cap ≈ 48 % of
  an unconstrained batch's CPU); a lower cap is a different experiment (`--skip-optimize` refuses
  to reuse across it) — the default stays 3000 (O7).
- `--resume` reads back complete record pairs whose identity matches a CURRENT scenario (O8) AND
  whose stamped `optimization_config` matches the batch — pre-stamp records are never
  resumable (O9).
- Reference records point at ONE shared observed track through `states_ref`
  (`optimization-references-v3-shared-tracks`); the store is swept against the UNION of sibling
  reference dirs, never one dataset's roster; a scenario finds its observed flight by `flight_key` (O10).
- Write-side hygiene: stale records are swept at batch start; publication writes immutable
  generation-suffixed files, commits via `comparison_index.json`, then prunes; filename suffixes,
  `REFERENCES_DIR`, `summary_row` and the reference cache contract are single-sourced in
  `optimization/evaluation_export.py`; a summary row's identity fields are required (refused when
  null), `callsign` optional (absent, never null) (O11).
- `docs/direct_collocation_hermite_simpson.zh.md` §5 and `geodetic_dynamics_transport.zh.html`
  describe the OLD HS-planner + RK4-polish pipeline (kept, historically inaccurate) (O12).

## Key defaults & constants (current)

- Mesh: `collocation/optimizer.py` `DEFAULT_N_SEGMENTS = 8`, `DEFAULT_N_SEG_PER_PHASE = 3`
  (single source; `run_scenario_optimization.py` mirrors them); the frontend/backend
  unconstrained `n_segments` default 10 is a different knob (K1).
- State substeps: auto ≈ 3 s state step per phase, cap 16; do NOT lower below auto (M=4 → 14.5 km
  rollout error); constrained solves do not need big M (K2).
- Fitting: Hermite-Simpson (`hermiteSimpsonNormalizedFullTransport`) for constrained and
  unconstrained; `trapezoidal` / `rk4` are kept for comparison studies only (K3).
- IPOPT: `components.DEFAULT_MAX_ITERATIONS = 3000` on both IPOPT constructions; the `sqpmethod`
  backend hardcodes `max_iter: 100` and ignores it; linear solver `AEROVIZ_IPOPT_LINSOL`
  (default `mumps`; the HSL hook is dormant) (K4).
- Altitude floor `altitude_floor_m(target) = target − 5 m`; rollout guard = floor − 5 m;
  `min_altitude_m` is REQUIRED on `rollout_controls` / `simulate_controls` (K5).
- ψ corridor: constrained solves bound heading to the route's heading hull ± 90° — this killed the
  looping/crawling local-optimum family (K6).
- Join/passage: the ONE forced fix passage is the pre-FAF fix within its k·RNP disc; the FAC join
  is on-course within `[d_FAF + L_final/5, d_FAF + max_offset]`; alignment ±30° join→FAF, ±10°
  FAF→threshold; the glidepath window binds only inside the FAF (K7).
- Transition phase: prepended when the start is farther than `_first_fix_join_tolerance_m` from
  the first fix; frame-anchor contract validated at ±150 m (K8).
- Playback drift guard: `playbackDriftM` on every optimize response; WARNING above
  `PLAYBACK_DRIFT_WARN_M = 50` (K9).
- Target speed: `runway`-mode pins the terminal V to the airframe's PUBLISHED approach speed at the
  solve mass (`approach.reference_speed_ms(m)`, the speed gate's own law; since 2026-09-24, was 145 kt
  for every 5.7–150 t type); velocity floor `1.10 · V_s1g` (its V_ref cap could not bind); the
  interactive optimizer's default floor is the published lower edge at the solve mass; `--resume`
  re-solves records flown to another target (K10).
