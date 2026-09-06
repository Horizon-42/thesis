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
  deletes that roster. **ts checkpoints trained before 2026-08-24 still predate the v5
  cohort**; the 2026-09-03 airport-frame arms are the first state checkpoints on it. →
  `trajectory_data_process/CLAUDE.md`, `docs/2026-08-21_ksjc_route_mix_and_ade.md`
- **Per-airport ADE/FDE must be quoted with its route mix, never bare.** KSJC's apparent 1.7×
  advantage is composition: reweighted to the pooled stratum mix it goes 483 → 1526 m, best of
  five to worst. `summary.json` now carries the covariates per row and an
  `accuracy.difficulty` block; published tables predate them and need re-deriving. →
  `4dTrajectory/ts_transformer/CLAUDE.md`
- **The optimizer batch has NOT been run since the harvest grew; no SOLVES are on disk.**
  `4dTrajectory/outputs/<ICAO>` holds only ts artifacts, so `--skip-optimize` has nothing to find
  and the solve run is from scratch. **`flight_scenarios/outputs` is NOT empty** (this item said
  it was until 2026-09-06): all ten `*_scenarios.json` + their `.selection.json` are there, 92 MB,
  built 2026-08-23. Whether they can be reused as-is is UNVERIFIED — the arrival manifests were
  rewritten 2026-08-24, one day later, though their per-airport `available` counts still match the
  selections exactly. Let the runner's prepared-input signature check decide; do not assume either
  way.
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
