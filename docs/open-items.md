# Open items

Cross-subsystem status: what is done, what is blocked, and what is measured but not fixed.
Moved out of the root `CLAUDE.md` so it is read when planning work rather than injected into
every session. The root file keeps only the hazards that must fire unprompted.

Convention: add items as they arise, delete them when resolved. Findings noticed *outside* the
change you are making go in `docs/code-health-followups.md` instead.

---

- **观测航迹里有不可能的位置点（ADS-B 跳点 / 跨缺口插值），未修（2026-10-05，用户：先记为未决项）。**
  两个相邻 2 s 行之间航迹转向超过 120°（每秒 60° 以上，真实飞机做不到）的航班：v12_20261005 的 train 44,703 架里
  53 架，select 6,913 架里 21 架（val 未数）。例：KSJC CPJ007（2 s 内向后跳 1,719 m，约 860 m/s）、KSTL DAL1400
  （以约 7 m/s 倒退数行、步长相同，像跨缺口的线性插值）、KSMF SWA1414（2 s 内由南向北反向）。
  - 已有的一道关：标注器已按 "impossible ground speed" 拒绝一部分（v12 的 train + select 共 166 架）；上面的 74 架是
    在全部信号里数的，没有分开已被拒的和有句子的。
  - 影响：这些点在信号、标注和闭环读数的观测路径里。A37 的拐点规则（D83）在这 3 架的近乎掉头处曾把 e_y 判成相反
    符号；已改为转角超过 170° 按掉头处理（`autopilot/closed_loop.py` `REVERSAL_TURN_DEG`），A34 的产物不用重建。
  - **A34 的 26 km 误差和动力学失败也来自这类点（2026-10-05 查）**：KMSY N12EC（train 8 s）第一个预测步前 2 s 内位置跳 959 m，
    起点规则 displacement-2s 只看这一步，起点地速 478 m/s，飞离 26 km 超时（A30 的居中 15 s 拟合平滑掉了，落地）；
    KSMF SWA3140、KSTL DAL1400（select）路径里有近乎掉头的点，匹配点卡在那里不再前进，修正拉不回来，漂到 24–26 km
    超时（A30 也超时，只漂 2–3 km）。A34 每格 4–14 条动力学失败（A30 0–1）共 53 条，起点地速都只有 7–58 m/s（进近
    正常约 70–130 m/s），第一个预测步前多有 150–790 m 的 2 s 大步（19 条 > 400 m）：像位置停更几秒后补跳，2 s 的窗正落在
    停更上（Claude 的读法，未逐条核）。输出：`.claude/worktrees/two-tier-v4-a37/smoke_v4/data/a37/big_lateral.txt`。
  - **已定（用户 2026-10-05，D111）**：落地了的坏航迹航班也打“坏数据”标记，预训练不用——读取时按规则标记（一步 2 s 的位移超过
    周围 10 步中位数的 3 倍或不到 1/3、或相邻两行转向超过 120°），不改产物、不重建；A 阶段 A40 给函数，B 阶段 B11 在
    `landed` 选择里去掉它们；自由生成和后训练的起点不变。Δ = 4 s 有闭环句子的航班中标记 train 476（落地 448）、select 172
    （落地 154）。下面这条仍未定的是数据本身要不要修。
  - 未定（用户）：修在哪里——harvest 的读取时修复（像高度离群值那样，影响所有使用者）、只在指令信号构建处，或标注器
    按名拒绝这类航班；哪种都要重建 A34 的产物。“不可能”的界线（转角、隐含速度）也要定。
  - 数据：`.claude/worktrees/two-tier-v4-a37/smoke_v4/data/a37/vertex.json`（3 架的拐点）；数法见阶段 A 实现日志 A37 行。

- **KAUS 测试机场：数据齐了，代码还没接（2026-09-27）。** 数据在 `trajectory_data_process/outputs/harvest-heldout/KAUS`
  （TD27），不在 live 根，默认的机场列表看不到它——这是故意的。要用它做测试时：
  - `prior/mva.py`：`FACILITY` 加 `"KAUS": "AUS"`，`CHARTS_DATE` 改为 `2026-09-27`（五个机场的图逐字节不变）；
    `test_prior_mva.py` 钉着五个机场。改了会进先验运行记录的元数据，所以等用户点头、别在运行中改。
  - 读测试机场的 runner 要能指定 heldout 根（`repo_layout.arrival_manifest_path(airport, root)` 已支持参数）。
  - 北向跑道已补（用户 09-27）：冬季 2025-12-24 → 2026-01-29 下载后合并进 heldout 根，每跑道 18L 20,204 / 18R 3,989 /
    36L 3,516 / 36R 1,265——只有 36R 不到 2000（冬季北向运行时到达主要落 36L）。合并用的源目录
    `outputs/harvest-heldout-winter/`（1.3 GB）还留着，删不删由用户定。

- **观测的速度通道含有未来 7.5 s 的信息，不能当因果模型的输入（2026-09-24 核实代码；影响大小未量）。**
  `flight_scenarios.state_samples_from_track` 对每个 ADS-B 点在前后各 7.5 s（`DEFAULT_WINDOW_S = 15`）的窗口里做
  最小二乘拟合，得到速度、航迹、升降角；ts 数据集（`data/dataset.py`，默认 `reference_velocity_source = track-fit`）和
  句子产物的信号都用它。位置不受影响。合成例子：100 m/s、第 0 s 起 2°/s 右转，居中拟合在 −2 s 已偏 0.89°、0 s 偏
  2.30°，只用过去 15 s 的拟合在 0 s 为 0。
  - 事后用（evaluation、参考记录、前端、标注器读完整航迹）没有问题。
  - 有问题的是模型输入：两层框架的先验（第 t 步输入这一行的航迹 / 地速 / 升降率，航向词正好标在转弯开始那一行）；
    旧 ts 预测模型的历史窗口（最后几行的速度通道看到起点后最多 7.5 s）。2026-08-02 的参考速度消融按预测精度选了
    `track-fit`（因果平滑差分 FDE 1202.9 → 1367.1 m），这个差距里有多少来自未来信息没有分开过。
  - 待定（用户）：先验在下一代产物里改用因果速度（在 `prior/data.py` 里由位置算，不动数据平面）；旧 ts 结果要不要重量。

- **缺气动参数的机型一律用 A320（2026-09-23 分析完成）。** train 50,693 条里 13,164 条（26.0 %）用 A320 飞，其中 4,599 条的机型进近速度落在 A320 模型的失速分支里或更低（活塞机全部）。报告与 199 个机型的替代表：`docs/aircraft_performance/2026-09-23_missing_performance_substitution.zh.md`。用户的决定（09-23 / 09-24）：
  - **进近参考速度改用各机型公布值、按质量换算（方案 A）—— 已完成**（分支 `docs-aero-substitution`，FS5 / K10）。09-24 前准备的 `flight_scenarios/outputs/*_threshold_scenarios.json` 加载时被拒，重新求解前要重跑 `prepare_scenario_inputs.py`（先问用户）。
  - **第 1–3 项做成一张索引表 —— 已完成**（`aircraft/performance_index.json`，FS6）：own 31 / substitute 53 / exclude 125（含 10 个没有航班的同义机型）；
    flight_scenarios 与优化器场景默认不再用 A320。
  - **ts 去掉 A320 回退，按需丢弃 —— 已完成**（用户 09-24：“B，去掉 all 的 A320 回退”“ts 用2 按需丢弃”；C31）：
    `all-flights`（状态输出、指令标注器，保留没有动力学的航班）/ `modelled`（控制输出）/ `openap-direct`，默认按需。
    **合并后要用户决定何时重建**（都没删）：28 个旧 `all` 状态预测 run（2026-09-04 起的 final_constraint 等）的配置不再
    能加载；`4dTrajectory/outputs/POOLED/instruction_language/{v1_20260923,v2_20260924}` 的信号是 v1，新代码拒读，
    标注链要从 `instruction_signals` 重跑（数值不变，只有 `typecode` 变：v1 里没有动力学的航班写的是 A320）；前端
    `training/` 的 v5 样本（另一个会话 09-24 在 `a66c3276` 从 `v2_20260924` 导出的）要在信号重建后重新导出成 v6。
    另一个会话正在做标注器，合并时间要和它协调。215 个 `openap-direct` run 不受影响。
  - **待办**：写信问 EASA（environment@easa.europa.eu），公开的 ANP v2.3 表的系数能否在论文里引用（唯一有襟翼档进近速度规律的公开来源）。先不做。
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

- **Two-tier model: status lives in `4dTrajectory/ts_transformer/docs/two_tier/two_tier_stage_notes.zh.md`**
  (every stage, the current artefacts, the user's decisions, the next step). The 2026-09-18 pause of plan v2 is over:
  v2 and v3 were superseded by the controller-word design (`docs/two_tier/two_tier_framework.zh.md`).

- **ts one-tier paths — the decisions still open** (moved 2026-09-26 from the retired ts `OPEN_ITEMS.md`, now
  `4dTrajectory/ts_transformer/docs/history/OPEN_ITEMS_2026-09-18.md`, which keeps the full one-tier history):
  - **N7′ adoption is the user's call** (`control_thrust_parameterization=specific-force+path-angle`, defaults row D27:
    passes every pre-registered gate on both seeds; one regression, vectored FDE p50 +625 / +704 m, unexplained).
  - **C-4, the duration floor under `uniform`.** `control_duration_uniform_floor` is read by the
    `factorized` head only, but its default is 0.8 and every recipe pins 0.0, so 88 stored
    `uniform` runs carry a non-default inert value and a refusal would stop them loading. It
    stays a field on `DurationSpec` after the §4.3 split (2026-09-10): a `Factorized(floor)`
    variant would refuse the `0.0` every recipe pins under `uniform`, i.e. every recipe. A custom
    `uniform` run can still wear `duration-floor=`; retiring the pin is a recipe-version change.
  - **C-7, the test-release ledger is bound to the DIRECTORY.** `test_release.json` sits beside
    `checkpoint.pt`; a copy of the checkpoint elsewhere can be frozen and released again. The fix
    is a registry keyed by the checkpoint digest outside the run directory — where it lives and
    how tests isolate it is the decision. No ledger exists on disk today, so it costs nothing
    whenever it is made.
  - **C-9, `cv_results.json` misnames the selection metric as a loss.** `mean_/std_val_macro_loss`
    at the candidate level hold the SELECTION value (ADE, metres); the fold rows' `best_val_macro_loss`
    is the loss. Renaming means a schema bump, after which the two stored files (2026-08-16
    `POOLED/ts_patchtst_normalized_time`, 2026-08-18 `KSJC/experiments/cv_tau_bank_20260818`)
    are no longer reusable by `--skip-cv`. The writer says so in a comment; nothing renamed.

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
- **KSMF 35R's threshold was 39.4 m off — FIXED 2026-09-23 (`803605e0`, TD21).** The non-LPV thresholds (KRDU 32, KSMF 35R) came from OurAirports, whose KSMF row gives both ends of 17L/35R the same longitude, so every 35R crossing failed lateral. Every non-LPV threshold now comes from the CIFP runway (PG) record and every course from the line through both ends' records; the v7 roster was rebuilt on that code (KSMF 35R 643/643 lateral pass). The 2026-09-07 fix parked on branch `dev-cifp-runway-thresholds` (`c4ae552`) was superseded by it and the branch deleted 2026-09-26. → `trajectory_data_process/CLAUDE.md`, `trajectory_data_process/docs/11-2026-09-23-merge-new-data.zh.md`
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
  `4dTrajectory/ts_transformer/docs/history/2026-09_frames/2026-09-03_airport_frame_ablation_results.md`
- ts_transformer: **the final-approach corridor as a bounded output works; as a penalty it does
  not (2026-09-05).** `state_position_reference="corridor-bounded"` improves pooled FDE on all
  four runs (KRDU/KSJC × two seeds) without triggering the pre-registered veto (a vectored
  regression on both seeds) and is the candidate default; the
  runway-scale hinge penalty diverges under dual ascent and costs accuracy at parity; the
  row-by-row on-final projection recovers most of B's KRDU FDE gain post hoc but not its
  violation rate, and the FAF-gated projection wrecks vectored flights. →
  `4dTrajectory/ts_transformer/docs/history/2026-09_constraints/2026-09-05_final_constraint_results.zh.md`
- ts_transformer: **control-output constraint = a predict-time safety layer, measured
  2026-09-06 on KRDU + KSJC.** The v2 barrier command hook (lag-aware, lead-position margins,
  load-coordinated; `predict --command-hook barrier --hook-saturation soft`) applied to the
  `simple-v3` baseline without retraining improves pooled FDE 12 % / 7 % with no flight worse
  by 1 km; no arm that trained THROUGH a hook (six, two hooks, two airports) beat its
  predict-time counterpart. (The nominal-law hook measured there as the vertical complement is archived since T2,
  2026-09-07: not a live option.) → `4dTrajectory/ts_transformer/docs/history/2026-09_constraints/2026-09-06_control_hooks_results.zh.md`
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
  - Approach view: the Evaluation 3-colour comparison overlay is a separate datasource not yet fed to
    the view (Evaluation-with-comparison plots neither source); the pre-existing `useCzmlLoader` clock
    write is still ungated for the Evaluation+comparison two-writer case.
  - Approach-view interior-gap `break` is latent (current CZMLs are single-interval); the 07-07
    approach-view changes were verified via tests/tsc/build but not re-checked in-browser.
