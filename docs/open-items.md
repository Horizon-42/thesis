# Open items

Cross-subsystem status: what is done, what is blocked, and what is measured but not fixed.
Moved out of the root `CLAUDE.md` so it is read when planning work rather than injected into
every session. The root file keeps only the hazards that must fire unprompted.

Convention: add items as they arise, delete them when resolved. Findings noticed *outside* the
change you are making go in `docs/code-health-followups.md` instead.

---

- **机型识别与最小重量已补（2026-09-23，用户要求“数据源必须靠谱，宁缺”）。** 用户当天决定并已执行：
  (1) 五个机场的观测报告已重评并重发前端（三门通过率 82.6% → 85.8%，横向/垂直判定无一变化，横向名单的合格集合
  不变）；(2) `cohorts_v7_20260923` 已按新机型库重建（每单元 train −4、val −3，全部是 openap-direct 资格变化的航班）；
  (3) 10 个机型只有厂家旧网页的互联网档案馆存档（E55P、CL30、C750、C680、GLF5、G280、LJ60、BE9L、C510、E50P）——
  **先记录，不进参考表**，数字与出处在 `docs/reference_speeds/README.md`“2026-09-23: type certificate data sheets”，
  以后需要时再决定。
  分析与数字：`evaluation/docs/2026-09-23_observed_baseline_pass_rate.zh.md`。
- **跟踪项：Falcon 7X/8X 新判得了速度之后 54 条里 22 条偏快**（窗口上限约 124 kt，中位数 123 kt）。上限锚在 FAA
  飞机特性数据库的进近速度 104 / 106 kt；需要另一份公开来源核对这个数是否偏低，再决定是否改表。
- **跟踪项：教练机在速度门偏慢（用户 2026-09-23：“先按教练机来，加入后期跟踪项，看后面需不需要改”）。**
  KRDU 的 P28A 2,084 条里 168 条（8%）比速度窗口下限慢，中位数约 3 kt；占 KRDU 速度门 fail 的 15%。
  现在把它解释为训练飞行的实际飞法，门不改。以后需要回答：通航/教练机是否该用单独的速度窗口（有没有权威的
  训练进近速度来源），还是维持现状并在结论里注明。分析：`evaluation/docs/2026-09-23_observed_baseline_pass_rate.zh.md`。
- **到达清单已重建并合并新数据**（用户 2026-09-23 决定；2026-09-20 的“不重建”就此作废）。live 根目录是
  v5 数据加 8/22–9/22 新下载，v7 名单 72,574 条、eligible 72,247 条，KRDU 32 与 KSMF 35R 已加回（跑道覆盖：
  KRDU 5 条，只缺没有 RNAV 程序的 14）。旧 v5 名单冻结为 `outputs/harvest-v5-20260823`，旧 checkpoint 按指纹回放。
  状态与数字：`trajectory_data_process/docs/11-2026-09-23-merge-new-data.zh.md`。
  已完成：`new_data_9_22` 核对后删除；前端 observed 已更新；stage A 网格 cohort 已按新数据重建到
  `4dTrajectory/outputs/KRDU/experiments/cohorts_v7_20260923/two_tier_v3_grid/`（同日晚按新机型库再重建一次，见上条）。前端 `training/` 下已归档词表的导出
  （box_v3、prior_*）在 `check-publication` 里报错（读取规则/schema 已变），与合并无关，待词表重写后处理。

- **`clean_pipeline_data.py` DELETES the 70,267 optimizer records the next item says to
  regenerate from** (verified 2026-09-20). Root `CLAUDE.md` carries both halves of the
  contradiction: the Build & Dev block tells you to run the cleaner to clear "allow-listed
  regenerable outputs", and the Open Items block says
  `4dTrajectory/outputs/<ICAO>/{runway,fitted_adsb,runway_cons}` hold 15 batches / **70,267
  records** whose v6 reports must be **regenerated from them**, a re-solve costing ~30 h at
  `--jobs 24` and 12.3 GiB that does not fit on this disk. `clean_pipeline_data.py --airport
  KRDU --dry-run` prints `optimizer + standalone predictions (allow-listed)  77862 files
  2.2 GB` and would take them; ~7.0 GB over the five airports. **Do not run the cleaner
  unattended; exclude those directories by hand.** The durable fix is either to teach the
  cleaner to protect them while the regeneration is pending, or to do the regeneration and then
  release them — the user's call, because it decides whether those records are still needed.

- **Two-tier — PAUSED 2026-09-18 01:00 by the user; the handover is
  `4dTrajectory/ts_transformer/docs/2026-09-17_two_tier_plan_v2.zh.md` §13** (state, how to resume, the
  decision list). Results: gate L1 FAIL (all contracts, with and without the token-using recipe), gate L2
  PASS on the segments axis at seq_len 61 / 91 / 121, gate E2E FAIL on every contract; the usable formal
  recipe is pa + masking 0 + smoothness off (`L1c_pa_*`: E2E ADE ~1270 / fully flyable 0.99 / straight-in
  established 0.97–0.99), whose two misses are both vectored (ADE ~3050–3130 > 2745, established 0.04–0.09).
  A longer L2 lookback improves the open-loop vectored 60–120 s waypoints by a third but not the closed loop
  (§12.3 hypothesis: the longer window reads more of the tracker's own flown rows). Open: the B121 E2E died in
  the arc-length geometry on a <2-point truth segment (§12.4, unfixed); no records were written for L1b/L1c/
  L2c/L2d (publishing needs a re-run); the full test suite has not run since 40b19b9; 13d5367 (review fixes)
  is unreviewed. Nothing is running; the runs worktree must be moved to the latest commit before any launch.


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
- **KSMF 35R's configured threshold is 39.4 m off, and every one of its 259 observed crossings fails lateral because of it (found 2026-09-08 when `--observed-only` first included the LNAV/VNAV runways).** The 24 LPV thresholds come from the CIFP Path Point LTP and agree with the CIFP runway (PG) records within 0.4 m; the two non-LPV thresholds (KRDU 32, KSMF 35R) still come from OurAirports, whose KSMF row gives BOTH ends of 17L/35R the same longitude (−121.580002) — the 35R end is 39.4 m east of the CIFP PG threshold (38.683514, −121.580456), exactly where the aircraft cross (their fitted cross-track is −40 m on every row). KRDU 32 is 3.0 m off (its crossings centre at −1.2 m; harmless). Fix = read the non-LPV threshold coordinates from the CIFP PG runway record (same cycle, same authority as the LTP, cross-checked at ≤ 0.5 m on the 24 LPV runways), never the JSON. **Consequence: the 35R physical frame changes, so KSMF's stored threshold events go stale (`require_matching_threshold_frame`) and KSMF needs `--reclassify-existing` + an arrivals rebuild (+259 arrivals → its ts split changes) before its observed pipeline runs again — owner decision; until then the published KSMF report carries 259 false lateral fails on 35R.** KRDU 32 would move 3 m (a KRDU reclassify adds its 1,604 RW32 arrivals to the roster — the v6 growth already noted above). **The fix is WRITTEN and parked on branch `dev-cifp-runway-thresholds` (`c4ae552`, 2026-09-07; its worktree was removed 2026-09-18, the branch kept)**: `cifp.read_runway_records` (landing threshold lat/lon incl. displacement, threshold elevation, ellipsoidal height, width), decode-pinned against the 24 LPV Path Point LTPs within 0.4 m, and `airports._build_runway` reading it for every non-LPV runway. It merges cleanly except `docs/CHANGELOG.md` and `trajectory_data_process/CLAUDE.md` (the latter now an index — its note becomes one line plus a TD section). Merging it makes the three non-LPV physical frames change, so it is gated on an owner-scheduled `--reclassify-existing` + arrivals rebuild window, not on the code.
- **Observed airframe coverage (2026-09-08): 10,541 of 42,746 observed rows (24.7 %) carried no type; 9,056 of them are recoverable by code, 1,485 need a better identity source. DONE for the code half — after the fix, the 172-type table and the rebuild (which added KRDU 32 / KSMF 35R, 44,609 rows) the speed-indeterminate share is 12.5 %: 1,657 rows unresolved identities, ~3,900 rows bizjet types with a FAA speed but no published minimum mass (E55P, CL30, E545, C680, C750, GLF5, G280, H25B, C25A, GLF4, C560 …; `docs/reference_speeds/README.md` lists each failed source).** Classified by re-resolving every untyped icao24: 9,056 rows / 3,006 airframes resolve to an ICAO type that OpenAP does not model (BCS3 1,582, E55P 719, CRJ7 674, BCS1 482, P28A 409, CL35 359, PC12 336 … 167 types) and were dropped only because the observed writer demanded dynamics for a mass the gate never reads — fixed in `resolve_airframe` / `harvest/observed.py` (type written whenever the identity resolves; `mass_source` on the record) and applied with `--observed-only`; their speed rows then grade wherever `aircraft/reference_speeds.json` has the type WITH a published minimum mass (the pack is being extended to those 167 types). The other 1,485 rows / 632 airframes have no identity: 1,164 "FAA model has no unambiguous ICAO typecode" (US-registered, the FAA→ICAO crosswalk is ambiguous for the model), 283 unmatched in both FAA and OpenSky (67 Mexican `0D`, 54 `4C`, …), 38 OpenSky codes absent from the Doc 8643 snapshot (H900 19, CL61 6, G450 5, F2EX 4, F2LX 2, G650 1, AS29 1 — non-standard codes that an alias table would map to H25B/CL60/GLF4/F2TH/GLF6). Follow-ups #24/#25.
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
  **2026-09-23: regrading them now judges them in the NEW runway frame.** `evaluation/arrival.py`
  builds the grading frame from today's `load_airport` course (the CIFP centreline, TD21), while
  these records were solved toward the old whole-degree OurAirports course (up to 0.45° off,
  KMSY; ~0.3° at KSTL). The threshold crossing is unaffected; path-shape and cross-track readouts
  along the final move by up to ~40 m at 5 km. The same holds for re-evaluating any ts prediction
  made before 2026-09-23. State it with any regraded number.
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
- Optimizer (moved verbatim from `4dTrajectory/CLAUDE.md` "Open items", 2026-09-16):
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
- Viewer (moved verbatim from `aeroviz-4d/CLAUDE.md` "Open items", 2026-09-16):
  - **Local terrain and aircraft CZML disagree by ~33 m in the viewer.** `local-terrain` heightmaps
    come from USGS TNM DSM (NAVD88 ≈ MSL) and the metadata records
    `vertical: "Source GeoTIFF elevation values, used directly as metres"` — no datum handling —
    while Cesium expects ellipsoidal heights and the aircraft CZML correctly supplies them. Found
    while chasing the datum bug; not investigated further.
  - Approach view: the Observe 3-colour comparison overlay is a separate datasource not yet fed to
    the view (Observe-with-comparison plots neither source); the pre-existing `useCzmlLoader` clock
    write is still ungated for the Observe+comparison two-writer case.
  - Approach-view interior-gap `break` is latent (current CZMLs are single-interval); the 07-07
    approach-view changes were verified via tests/tsc/build but not re-checked in-browser.
