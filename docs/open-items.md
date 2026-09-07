# Open items

Cross-subsystem status: what is done, what is blocked, and what is measured but not fixed.
Moved out of the root `CLAUDE.md` so it is read when planning work rather than injected into
every session. The root file keeps only the hazards that must fire unprompted.

Convention: add items as they arise, delete them when resolved. Findings noticed *outside* the
change you are making go in `docs/code-health-followups.md` instead.

---


- **The v5 re-roster is DONE (verified on disk 2026-09-03):** all five
  `outputs/harvest/<ICAO>/arrivals/manifest.json` are `harvest-arrivals-v5-takeoff-excluded`
  (KRDU 14,435 · KSJC 11,082 · KSTL 8,767 · KSMF 4,219 · KMSY 4,147) and every one has its
  `lateral_pass_eligibility.json`. Do NOT run `--evaluate-only` again without need — it
  deletes that roster, and since 2026-09-07 it rewrites it as
  `harvest-arrivals-v6-published-vertical-path` with KRDU 32 + KSMF 35R included (+1,876
  arrivals; readers accept both versions, the ts splits do not survive the change). **ts checkpoints trained before 2026-08-24 still predate the v5
  cohort**; the 2026-09-03 airport-frame arms are the first state checkpoints on it. →
  `trajectory_data_process/CLAUDE.md`, `docs/2026-08-21_ksjc_route_mix_and_ade.md`
- **Per-airport ADE/FDE must be quoted with its route mix, never bare.** KSJC's apparent 1.7×
  advantage is composition: reweighted to the pooled stratum mix it goes 483 → 1526 m, best of
  five to worst. `summary.json` now carries the covariates per row and an
  `accuracy.difficulty` block; published tables predate them and need re-deriving. →
  `4dTrajectory/ts_transformer/CLAUDE.md`
- **Observed airframe coverage (2026-09-08): 10,541 of 42,746 observed rows (24.7 %) carried no type; 9,056 of them are recoverable by code, 1,485 need a better identity source.** Classified by re-resolving every untyped icao24: 9,056 rows / 3,006 airframes resolve to an ICAO type that OpenAP does not model (BCS3 1,582, E55P 719, CRJ7 674, BCS1 482, P28A 409, CL35 359, PC12 336 … 167 types) and were dropped only because the observed writer demanded dynamics for a mass the gate never reads — fixed in `resolve_airframe` / `harvest/observed.py` (type written whenever the identity resolves; `mass_source` on the record) and applied with `--observed-only`; their speed rows then grade wherever `aircraft/reference_speeds.json` has the type WITH a published minimum mass (the pack is being extended to those 167 types). The other 1,485 rows / 632 airframes have no identity: 1,164 "FAA model has no unambiguous ICAO typecode" (US-registered, the FAA→ICAO crosswalk is ambiguous for the model), 283 unmatched in both FAA and OpenSky (67 Mexican `0D`, 54 `4C`, …), 38 OpenSky codes absent from the Doc 8643 snapshot (H900 19, CL61 6, G450 5, F2EX 4, F2LX 2, G650 1, AS29 1 — non-standard codes that an alias table would map to H25B/CL60/GLF4/F2TH/GLF6). Follow-ups #24/#25.
- **The five canonical observed reports are v9 as of 2026-09-07 08:14, and the ts data provenance moved with them.** `run_all_evaluations.py --kind observed --html` rewrote `outputs/harvest/<ICAO>/approach/evaluation_report.{json,html}` and the published `comparison/observed/` copies (verdict counts identical to `evaluation_reports/observed_v9_2026-09-07/`), and the five `lateral_pass_eligibility.json` rosters were refreshed against them — eligible sets IDENTICAL (KRDU 14,378 / KSJC 11,076 / KSTL 8,761 / KSMF 4,216 / KMSY 4,140), only the recorded `evaluation_report_sha256` / roster digests changed. Consequence: `data_provenance` of any NEW ts run differs from every checkpoint trained before, so `--skip-train`, `evaluate-fit` and `freeze-test` refuse to replay an OLD checkpoint against the current data (the equality guard). Arm-vs-arm comparisons inside a campaign are unaffected. The v6 reports, HTML, published copies and rosters are backed up in `outputs/evaluation_reports/observed_v6_backup_2026-09-07/<ICAO>/` — copying them back restores the old hashes.
  **FIXED IN CODE 2026-09-08 (`dev-provenance`, not yet merged): the ts data identity is the eligible SET, never the roster's bytes** (`ts-arrival-data-v4-eligible-set`). A regenerated observed report no longer refuses anything — a v4 run compares `eligible_set_sha256`, and a checkpoint carrying the retired v3 fingerprint carries the set itself (`source_records` IS the eligible set), so it is compared to today's rosters per airport. Verified on the real KRDU data: `L1b_hr16_tv1_full` (recorded roster `e2b0fa52…`, a generation no longer on disk) refuses at HEAD and predicts 1,404 val flights after; sweeping all 200 on-disk checkpoints, 65 → 66 accepted, zero newly refused. Restoring old roster bytes is no longer a prerequisite for replaying a checkpoint. → `4dTrajectory/ts_transformer/CLAUDE.md`
  - **Operator note: an IN-FLIGHT `cross_validation/cv_candidate_progress.json` written before that change is refused on resume** — the CV run contract names `eligible_sets` where it named the roster files, and `_load_candidate_progress` raises rather than silently re-running. Delete that one file to restart the search from candidate 0; finished `cv_results.json` are read, not refused.
- **TODO (owner-scheduled, between GPU campaigns): regenerate the 15 optimizer reports under the v9 published-V_ref gate.** **The optimizer solves ARE on disk (2026-09-07 correction).**
  `4dTrajectory/outputs/<ICAO>/{runway,fitted_adsb,runway_cons}`: 15 batches, 70,267 v6-evaluated
  records, solved from the 2026-08-23 scenarios. Their v6 reports read speed-indeterminate on every
  row (they predate `source.landing_aero`, which the v9 gate no longer reads anyway); every record
  carries `source.dynamics_typecode`, so `run_all_evaluations.py` regrades them as-is (~70 GB of
  record reads). The `backfill_landing_aero.py` tool was deleted with v9. **`flight_scenarios/outputs` is NOT empty**: all ten `*_scenarios.json` +
  `.selection.json`, 92 MB, built 2026-08-23; the arrival manifests were rewritten 2026-08-24,
  whether the scenarios can be reused as-is is UNVERIFIED — let the runner's prepared-input
  signature check decide.
  The arrival manifests were re-harvested 2026-08-15…17 for all five airports (the old
  "KSJC and KSTL need a re-harvest" item is closed) but need the v5 re-roster above first; after it,
  `prepare_scenario_inputs.py --skip-observed` is safe and skips rebuilding the observed
  CZML/report tail.
  **Scale**: **42,650** rostered arrivals (re-measured 2026-09-06; this said 42,725, which was
  the 2026-08-19 roster generation). At the default `--max-per-runway 2000`, read off the
  `.selection.json` files on disk, the batch is **23,429 flights / 70,287 solves** for `runway`
  — `fitted_adsb` selects the same flights and then drops the 20 `UnusableFittedApproach` ones
  (23,409 / 70,227) — estimated ~30 h at `--jobs 24` and **12.3 GiB** of
  artifacts (`--rollout-dt 1.0` → 8.1 GiB). Free space is the binding constraint and moves
  with the ts_transformer experiments, so check it right before launching.
  The runner refuses to start if the estimate does not fit. Order: prepare → optimize
  (`--resume` is cheap to restart) → the CZML/report tails run automatically per cell.
- Optimizer: KRDU RW32 systematically hard (not a truncation artifact); per-leg RNP not extracted
  from CIFP; CIFP leg speed restrictions not extracted; HSL linear-solver hook dormant;
  pre-existing numpy 2.x failure in `test_optimizer.py`. → `4dTrajectory/CLAUDE.md`
- Optimizer quality, measured 2026-08-19 and NOT fixed: on 120 random KRDU `runway` flights,
  **15 of 120 (12.5 %) fail only because the replay stops 1–10 m short of the threshold
  plane** (`event_status: not_reached` → lateral/vertical indeterminate → fail). Recovering
  just those would move the pass rate 60 % → 72.5 %, so any quoted gate rate should say
  whether it counts them. A further 18 solved flights end genuinely far short (median 610 m).
- Optimizer determinism: `_limit_solver_threads()` only runs when `jobs > 1`, so BLAS threading
  differs between `--jobs 1` and `--jobs N` and a borderline scenario can solve in one and hit
  `Maximum_Iterations_Exceeded` in the other (observed once in a 120-flight sample). The
  batch driver's docstring now states this caveat instead of claiming worker-count-independent
  output; the threading difference itself remains open.
- ts_transformer: **the threshold anchor is target conditioning, measured 2026-09-03** —
  an airport-anchored chart (`coordinate_frame="airport-enu"`) makes the deterministic model
  average across each parallel-runway pair at KRDU (endpoints nearer the sibling runway
  1.5 % → 12–15 %, minority runway pulled ~600 m), feeding the target's coordinates as input
  channels (`target_conditioning="channels"`) does not undo it, and the "route stability"
  gain for vectored flights did not survive a second seed. Keep `enu`. →
  `4dTrajectory/ts_transformer/docs/2026-09-03_airport_frame_ablation_results.md`
- ts_transformer: **the final-approach corridor as a bounded output works; as a penalty it does
  not (2026-09-05).** `state_position_reference="corridor-bounded"` improves pooled FDE on all
  four runs (KRDU/KSJC × two seeds) without triggering the pre-registered veto (a vectored
  regression on both seeds) and is the candidate default; the
  runway-scale hinge penalty diverges under dual ascent and costs accuracy at parity; the
  row-by-row on-final projection recovers most of B's KRDU FDE gain post hoc but not its
  violation rate, and the FAF-gated projection wrecks vectored flights. →
  `4dTrajectory/ts_transformer/docs/2026-09-05_final_constraint_results.zh.md`
- ts_transformer: **control-output constraint = a predict-time safety layer, measured
  2026-09-06 on KRDU + KSJC.** The v2 barrier command hook (lag-aware, lead-position margins,
  load-coordinated; `predict --command-hook barrier --hook-saturation soft`) applied to the
  `simple-v3` baseline without retraining improves pooled FDE 12 % / 7 % with no flight worse
  by 1 km; no arm that trained THROUGH a hook (six, two hooks, two airports) beat its
  predict-time counterpart; the nominal-law hook needs its thrust held to the unhooked rollout's speed and is the
  vertical complement. → `4dTrajectory/ts_transformer/docs/2026-09-06_control_hooks_results.zh.md`
- ts_transformer: KRDU run DONE (three generations, quote current artifacts only); gate-pass
  conclusion needs re-deriving after the datum fix; only KRDU trained; flyability measured but
  not fixed; single-aircraft + deterministic by scope. **All control-output checkpoints are
  stale as of 2026-08-18** — the control contract changed units (newtons → fraction of installed
  thrust) and `TSConfig` gained required fields, so `load_checkpoint` refuses them; `state`
  checkpoints are unaffected. The lagged flight model (`simple-v1-lag`) has no published
  train→predict→evaluate result yet; its τ_bank CV sweep is the open experiment.
  → `4dTrajectory/ts_transformer/CLAUDE.md`
- Viewer: local terrain vs aircraft CZML disagree by ~33 m; Observe 3-colour comparison overlay
  not yet fed to the approach view (+ ungated `useCzmlLoader` clock write); approach-view
  interior-gap `break` latent. → `aeroviz-4d/CLAUDE.md`
