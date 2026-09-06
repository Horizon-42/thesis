# evaluation 审核修复计划（2026-09-07）

分支 `dev-evaluation-review-fixes`（worktree `.claude/worktrees/lnav-vnav`，基线
`3029b09` on `dev-leg-ctrl`）。审核对象：`evaluation/` 全包 + 阈值点速度门的物理假设。
每个里程碑：代码 → 定向测试 → opus 子代理评审 → 提交。合并条件见 §6。

## 1. 状态表

| # | 里程碑 | 状态 | 提交 | 备注 |
|---|---|---|---|---|
| M1 | LNAV/VNAV 分支接通：进近航段竖直角 + TCH 进 `Runway`，生产端不再拒绝 KRDU 32 / KSMF 35R，`baro_vnav_approved` 由程序事实推出 | 完成，已评审 | `92c74ed` | KRDU 14 无程序，仍排除（13 条）；26 条跑道 frame fingerprint 与 23 条 LPV context fingerprint 钉死不变 |
| M2 | 速度门：真实载荷因子 n、报告自相矛盾、HTML 说明、过时注释与死路径、schema v7（四处镜像） | 完成，已评审 | `bb643d5` | 政策 B，见 §3.1；前端 v6 归为 prior（"graded at 1 g"），v5 仍 legacy |
| M3 | 结构：roster 缺字段抛错、visualize 单路径、边界校验集中、followups #7/#8/#10、守卫列表不一致、`arrival` 收拾、`__init__` 清理 | 完成，已评审 | `bb643d5` | `contexts_for_input` 取代两条路径；`load_records` 仅为 ts 测试保留 |
| M4 | 测试与文档：常量 import、正向断言、重复/错位测试、覆盖缺口；文档归档与版本头；`landing_aero` 回填脚本；根 CLAUDE.md Open Items 更新 | 完成，已评审 | `bb643d5` | 回填脚本只写不跑，见 §5；`validate_event` 测试移至 `final_approach/tests/test_event_contract.py` |
| — | 两轮 opus 评审（M1；M2–M4）的发现已全部处理 | 完成 | | 阻塞项：RNAV 前缀误匹配环形程序、arrivals schema 未 bump、观测记录 `hae_minus_msl_m` 硬要求打断 ts 参考记录、观测 roster 无 `arr_airport`；其余见 §7 |
| — | 合并到 `dev-leg-ctrl` | 分支已吸收 `dev-leg-ctrl`（`1fa4875`，仅 CHANGELOG 冲突，两侧条目都保留），合并树上全部套件复跑通过后 fast-forward | | campaign 在合并前已自行结束（见 §6.6） |

测试状态（2026-09-07，worktree）：Python 定向套件 619 passed（evaluation、harvest、final_approach、
flight_scenarios、aircraft、aeroviz_backend、aeroviz-4d/python、ts `test_lateral_eligibility`、
optimization `test_backfill_landing_aero`）；前端 vitest 524 passed，`tsc --noEmit` 干净。
未跑：ts_transformer 全套（torch，与正在跑的 GPU campaign 争内存）。

## 2. 审核测得的事实（改动依据，数字来自 2026-09-06/07 盘上数据）

- 优化批次：`4dTrajectory/outputs/<ICAO>/{runway,fitted_adsb,runway_cons}` 共 15 个批次
  70,267 条 v6 记录，speed 全部 indeterminate、verdict pass = 0，原因是 08-23 的 scenarios
  早于 08-24 的 `source.landing_aero`。根 CLAUDE.md Open Items 仍写"no SOLVES on disk"，已过时。
- 穿越点载荷因子（优化记录 `controls[-1].load_factor`，按 `dynamics_typecode` 重建失速事实）：

  | 批次 | 已判定 | n 中位 | n p95 | n 最大 | A 方案翻转 | B 方案翻转 | 通过但 V < V_s1g·√n |
  |---|---|---|---|---|---|---|---|
  | KMSY/runway | 3,439 | 1.010 | 1.015 | 1.231 | 227 | 1 | 0 |
  | KMSY/fitted_adsb | 3,308 | 1.010 | 1.015 | 1.128 | 158 | 20 | 0 |
  | KMSY/runway_cons | 2,853 | 0.997 | 1.005 | 1.160 | 4 | 0 | 0 |
  | KSJC/runway | 3,383 | 1.011 | 1.015 | 1.372 | 192 | 0 | 0 |
  | KSTL/runway_cons | 4,485 | 0.996 | 1.040 | 1.142 | 73 | 6 | 0 |

  A = 上下边都按 √n 缩放；B = 只缩放下边且 n 钳制 ≥ 1。A 的翻转来自上边缘堆积：A320 族
  Cl_max 3.0 后 64.5 t 的窗口约 126.7–146.7 kt，类默认目标 145 kt 离上边 1.7 kt。
- 无 LPV 跑道：全机队 assigned 44,622 条落地，1,876 条（4.2 %）因"publishes no LPV TCH"
  被 arrivals 排除：KRDU 32 1,604、KRDU 14 13、KSMF 35R 259。
- 同一份 CIFP 里的竖直参考（PF 进近程序最后一段，列 85–89 过点高度 ft、103–106 竖直角
  0.01°；续记录列 41+ 的最低标准行）：KRDU RNAV (GPS) RWY 32：LNAV/VNAV，−3.50°，
  470 − 425 = 45 ft；KSMF RNAV (GPS) Y RWY 35R：LNAV/VNAV，−3.00°，86 − 22 = 64 ft；
  KRDU 14 无任何程序。跑道记录（PG）TCH 与 Path Point TCH 在 23 条 LPV 跑道上 20 条差
  ≤ 0.4 ft，KSTL 06/24 差约 4 ft（按程序发布的量，不用跑道记录值）。
- 三条无 LPV 跑道的 `threshold_frame_fingerprint` 只哈希位置、标高、航向、周期与
  `position_source`/`vertical_source`，不含 TCH/下滑角（`airports.threshold_frame_snapshot`
  的设计），所以给它们补 TCH 不会让已存事件变"stale"。

## 3. 决策

### 3.1 速度门（M2）
- 下边 = 1.23 · V_s1g · √max(n, 1)，上边保持 1g 的 V_ref + 20 kt（法规 V_REF 按定义锚在
  1g 的 V_SR；上边是能量/冲出判据，不随 n 上移；n < 1 的推杆不放宽下边，拉平仍需 n ≥ 1）。
- n 的来源：记录带 `controls` 时取 `controls[-1].load_factor`（`rollout.py` 每个样本携带产生
  该样本的那段控制，`terminal_state` 与 `interpolated_threshold` 两分支同一段）；`controls == []`
  的记录（观测、状态输出的 ts 预测）声明 n = 1，行上写来源 `assumed_1g`，不从 ADS-B 反演
  （25 ft 量化的 γ̇ 只有噪声）。
- `aircraft.aero_params.stall_speed_ms` 增加 `load_factor=1.0` 关键字；优化器地板调用不变。
- 行上新增 `crossing_load_factor`、`crossing_load_factor_source`；criterion id 改为
  `vref_1p23_vs_n_to_vref_1g_plus_20kt`（观测加 `_ground_speed_proxy`）。
- 报告 schema v6 → v7，四处镜像一起动：`evaluation.metrics.REPORT_SCHEMA_VERSION`、ts 接缝
  import、前端 `EVALUATION_REPORT_SCHEMA_VERSION`（v6 进入可读旧版本集合）、fixtures 走常量。
  ts 接缝改为接受 `READABLE_REPORT_SCHEMA_VERSIONS = (v6, v7)`，盘上 v6 报告与正在跑的实验都不受影响。
- `METHODOLOGY["observed_crossing_ground_speed"]` 改为与 `terminal_speed.subjects` 一致；
  `visualize._TEMPLATE` 的说明与 V 列改为按 subject 显示被判定的那个速度。

### 3.2 LNAV/VNAV（M1）
- `harvest/cifp.py` 新增 `read_approach_verticals`：每条跑道取 RNAV 进近（程序 id 首字母
  R/H）最后一段（fix 为 `RWxx`、section/subsection `PG`）的竖直角与过点高度，再减该跑道的
  MSL 标高得 TCH；续记录的最低标准行含 `LNAV/VNAV` 记为 `baro_vnav_minima`。解码用
  Path Point 交叉钉死：LPV 跑道上航段 TCH 与 Path Point TCH 差 ≤ 1 ft、角度相等的比例 ≥ 75 %。
- `Runway` 新增 `tch_source`（`faa_cifp_path_point` | `faa_cifp_approach_leg` | None，不进
  fingerprint）与 `baro_vnav_minima: bool`；非 LPV 跑道有航段时填 TCH/下滑角。
- `evaluation.context.assessment_for_runway` 去掉 `baro_vnav_approved` 参数，由
  `runway.baro_vnav_minima` 推出；非 LPV 的 `procedure_source` = `runway.tch_source`
  或 `no_published_vertical_guidance`。
- 生产端条件不变（TCH/下滑角为 None 才排除），理由文案改为"publishes no vertically guided
  approach: no LPV path point and no LNAV/VNAV final leg"。KRDU 14 继续排除。
- 不重跑 harvest。重跑后 arrivals 会多出 KRDU +1,617、KSMF +259 条，ts 数据集与分割随之改变，
  必须等当前 campaign 结束并重建 `lateral_pass_eligibility.json` 后再做。
- 前端"未评判 vs indeterminate"灰色问题（`UNJUDGED_RUNWAY_VERDICT_GAP.md`）不在本次范围；接通后
  它缩小到 KRDU 14 的 13 条。

### 3.3 结构（M3）
- `roster_context_keys`：roster 行缺 `arr_airport`/`runway` 抛错，不再回退到全量物化路径；
  `summary.json` 缺失返回 None 走 `record_files` 的报错（followup #7）。单文件输入加载一次取机场。
- `visualize`：只保留流式 `build_payload(input_path, ...)`；`load_records` 保留为
  `list(iter_records())`（ts 测试在用）；`contexts_from_args` 删除。
- `record_from_dict` 集中校验 `source.arr_airport`、`source.runway`、`landing_aero` 形状、
  观测记录的 `hae_minus_msl_m`；`t` 严格递增（followup #8）；空批次 `subject: "empty"`（#10）。
- `evaluate_batch` 参考比较守卫改用 `measured_states`；`compare_to_reference` 接收已算好的
  span；`observed_availability` 在第一条非 observed 记录处抛错。
- `arrival.py`：`_STATE_KEYS = ("t", *STATE_KEYS)`；import 顺序；`_authoritative_frame` 帮助函数；
  `dataclasses.replace`。`thresholds.Verdict` 作为 `ComponentResult` 的别名保留一个定义。
  `_row` 的 criterion 由 `evaluate_record` 决定一次并带在 `TrajectoryEvaluation` 上。

## 4. 里程碑改动清单（文件）

- M1：`trajectory_data_process/harvest/cifp.py`、`airports.py`、`arrivals.py`、`observed.py`；
  `evaluation/context.py`；测试 `harvest/tests/test_cifp.py`、`evaluation/tests/test_terminal_standard.py`；
  文档 `trajectory_data_process/CLAUDE.md`、`evaluation/CLAUDE.md`、`evaluation/README.md`、
  `FINAL_APPROACH_VERDICT_STANDARD.md`、`UNJUDGED_RUNWAY_VERDICT_GAP.md`。
- M2：`aircraft/aero_params.py`；`evaluation/speed_gate.py`、`arrival.py`、`metrics.py`、`visualize.py`；
  `4dTrajectory/ts_transformer/lateral_eligibility.py`；`aeroviz-4d/src/data/evaluationReport.ts` + 测试；
  `evaluation/tests/test_speed_gate.py`、`factories.py`；`THRESHOLD_SPEED_GATE.md`。
- M3：`evaluation/records.py`、`cli.py`、`__main__.py`、`visualize.py`、`reference.py`、`metrics.py`、
  `arrival.py`、`thresholds.py`、`__init__.py` + 测试。
- M4：`evaluation/tests/*`、`final_approach/tests/test_event_contract.py`（新）、
  `4dTrajectory/optimization/backfill_landing_aero.py`（新）、`docs/CHANGELOG.md`、（注：该工具已于同日 v9 删除——发布 V_ref 门只需记录里的 `dynamics_typecode`，见 `2026-09-07_observed_speed_gate_plan.zh.md` §6）
  `docs/code-health-followups.md`、根 `CLAUDE.md` Open Items、`evaluation/docs/archive/`。

## 5. 留给使用者的操作（本分支不执行）

- `python 4dTrajectory/optimization/backfill_landing_aero.py --root 4dTrajectory/outputs --dry-run`：
  按 `dynamics_typecode` 走 `aircraft_for_code → aero_params_for_aircraft` 为 70,267 条记录补
  `source.landing_aero`，然后按批次重跑 `python -m evaluation`。约 70 GB 读 I/O，campaign 结束后再跑。
- 重跑 harvest（`--reclassify-existing` 或 `--evaluate-only`）以收回 1,863 条无 LPV 跑道航迹。
  注意根 CLAUDE.md 的警告：`--evaluate-only` 会重写 v5 arrivals roster。

## 7. 评审后追加的决定

- arrivals manifest 写出版本改为 `harvest-arrivals-v6-published-vertical-path`，读取端接受 v5 与 v6
  （形状不变；盘上五个 v5 roster 与正在跑的实验都不受影响）。
- 进近航段解码只读 `R` + 数字的程序（排除环形的 `RNV-A`），最后航段的跑道 fix 只允许出现一次、
  高度必须是精确值；Y/Z 两个程序不一致时记为 `conflict`，仅在非 LPV 跑道需要它时抛错。
  Path Point 钉定改为与 Path Point 解码同样的 75 % 置信规则、容差 2 ft；无 LPV 程序的机场不钉定。
- `Runway.tch_source` 默认 None，谁设 TCH 谁命名来源；`baro_vnav_minima` 而无航段路径直接抛错，
  `AssessmentContext` 也不再接受"approved 但无 TCH"，`limits()` 的那个死分支删除。
- 观测记录的 `hae_minus_msl_m` 只在带 `observed_threshold_event` 时要求（优化器/ts 的参考记录不带）。
- roster 行缺 `arr_airport` 时取 `summary["airport"]`（harvest 观测 roster 的写法）。
- 观测 criterion id 改为 `vref_1p23_vs1g_to_vref_1g_plus_20kt_ground_speed_proxy`（观测窗口就是 1 g 的）。
- 空窗口（n 大到下边越过上边）按 fail 判定并在行上显示两边，不抛错（与评审建议不同，理由见速度门文档 §9）。
- `crossing_load_factor` 批量聚合含 `min`、`below_1g`、`assumed_1g`。
- 回填脚本按生产端的紧凑序列化写回（默认分隔符会让每条记录大 8 %），并带 `--aircraft-provider`。

## 6. 实验隔离与合并条件

正在跑：`run_ts_frame_ablation.py --campaign 4dTrajectory/outputs/KRDU/experiments/l2_beta_ladder_20260907`
（PID 823458 → train 823486，3 个 arm，每 arm 约 1 h，从主树 `3029b09` 运行，每步一个新子进程，
manifest 逐 arm 记录 commit）。合并进 `dev-leg-ctrl` 会改变后续 arm 实际运行的代码，所以：

1. 本分支不改 `4dTrajectory/ts_transformer/` 除 `lateral_eligibility.py` 的版本集合外的任何文件；
2. 23 条 LPV 跑道的 `threshold_frame_fingerprint` 与 `AssessmentContext` 内容用测试钉死不变；
3. ts 接缝接受 v6 与 v7；
4. 合并前再看一次 campaign 进度：若某个 arm 正处于 train 与 eval 之间，等它完成该 arm。
5. 已处理的一个隐患：worktree 目录 `.claude/worktrees/lnav-vnav` 在主树里显示为 untracked，
   而 formal run 的 `experiment_index.begin_run` 遇到脏树会拒绝启动，下一个 arm 会因此中止。
   已在主树 `.git/info/exclude` 加入 `.claude/worktrees/`（本机本地设置），主树 `git status` 恢复干净。
   正在跑的 arm 1 的 manifest 记录 `commit 3029b09, dirty False`，未受影响。
6. 观察到但非本分支所为：arm 2（`L2b_beta0p01`）的 train 于 17:43:49 正常写出 manifest
   （`status: running`，脏树检查已通过），随后 17:44–17:45 之间整个 wrapper 与 train 进程消失，
   日志无 traceback，内存充裕（31 GB 总量、14 GB 空闲），同一时刻另一个交互会话在主树上活动。
   最可能是该会话主动停止了 campaign；本分支没有触碰主树。合并时主树无实验在跑。
