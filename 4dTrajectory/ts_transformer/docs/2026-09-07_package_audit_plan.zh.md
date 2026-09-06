# ts_transformer 包审计与优化方案（2026-09-07）

三个 opus 审计并行完成（A：config/train/dataset 核心；B：`control/` + closure + oracle；C：测试 /
`docs/*.py` / 运行器 / 文档），每条结论都经 grep / 运行验证，不是读出来的。本文把三份报告去重、
按"价值 ÷ 工作量"排序，给出可以逐个提交的工作包。基线：`dev-l2 @ 765303d`，套件 667 项通过。

## 〇、状态表（压缩 context 后从这里继续）

| 工作包 | 状态 | 内容 | 工作量 |
|---|---|---|---|
| T0 零风险清理 | **完成并通过 review**（`55af0b7`、`57e68fc`、`7350638`；review 的 10 项修复见 CHANGELOG 2026-09-07 T0 条目）；L2 启动前合入 | 8 项（§二） | S×8 |
| T1 删除已废除设计 | **9、12 完成**（`0702669` 跟踪器、`0898291` regularization）；10、11、13、14 未开始 | 跟踪器、horizon curriculum、arc-length 目标族、regularization | M×3 + S |
| T2 归档 2026-08 教师机器 | 未开始 | `control/oracle/` 八模块 + 三个运行器 + nominal 律 hook | L |
| T3 结构重排 | 未开始 | `objective.py` / `validation.py` 拆出 train、config 校验拆分、dataset 拆分、CLI 拆分、后端合并 | M×5 |
| T4 测试与运行器 | 未开始 | 13 条红测试搬家修复、`tests/support.py`、`runner_support.py`、拆 6.5k 行测试文件、20→14 运行器 | M×4 |
| T5 `docs/*.py` 迁移 | 未开始 | 13 个脚本的迁移表（枢纽三个先） | M |
| T6 文档卫生 | **完成**（本提交） | OPEN_ITEMS 重写头部、CLAUDE.md 降级 closure + 两根新轴、README 加 banner、7 处死链、`30d`、两份取代 banner、`notes_7_20` 并入 | S |

**时序约束**：所有改动在 `dev-l2`（worktree）上做；**只在 campaign 之间合入 `dev-leg-ctrl`**——正式
campaign 的 predict/eval 步会 import 主树的当前代码，中途合入会污染它的来源。T0 在 L2 campaign
启动前合入（都是 S、零行为变化，且 L2 检查点应带干净的契约）；T1–T5 在 L2 跑的时候做、跑完合入。

## 一、审计的总判断

1. **包里有整整一代已废除的设计仍然是活代码**：closure 跟踪器（设计文档 §四 已批准废除、带未修
   BLOCKER，`CLAUDE.md` 却仍称它为交付形态）、horizon curriculum（没有任何 recipe 或臂设过）、
   `arc-length-geometry` 目标族（16 个 config 字段，2026-08-02 之后无人使用，包自己的状态文档从不
   提它）、2026-08 的教师机器（`control/oracle/` 1,410 行 + 三个运行器，被 `simple-v3` 的训练内
   模仿项取代）。`control/` 5,382 行中 **11 % 今天可删，41 % 归档后可删**。
2. **有三处"单一来源"被打破且会改数字**：`STRAIGHT_TORTUOSITY` 四处定义，`docs/score_control_arms.py`
   用 **1.02**——`CLAUDE.md` 让读者据以判断 bank skill 的那组数，算在与所有 ADE 表**不同的分层**上；
   `compare_control_arms_stratified.py` 的 vectored 掩码缺 `& ~established`。`simple-v1/v2/v3` 有
   8 个字段钉在**可变的模块默认值**上，改默认值会静默重定义已发表的配对比较。
3. **测试套件的退出码不携带信息**：`run_ts_pipeline.py` 的 877 行测试放在 `trajectory_data_process/tests/`，
   **13 条今天是红的**，而 `CLAUDE.md` 给的命令永远不跑它们。
4. **结构**：`train.py` 3,246 行、`config.__post_init__` 742 行 123 个 raise、`__main__.main` 714 行、
   `test_ts_transformer.py` 6,507 行。`batching ↔ train` 靠函数内延迟 import 打破循环——原因是目标函数
   住在训练循环里，而 `batch_contract.py` 存在的理由正是让它不必如此。
5. **文档落后一代**：`OPEN_ITEMS.md`、`CLAUDE.md`、`README.md` 都把已废除的工作当现状，`latent` / `cta`
   零提及；`README` 仍说 control 路径"没有发表过精度结果"、"只训过 KRDU"。

## 二、T0 — 零风险清理（每项一次提交，全部 S）

| # | 改什么 | 证据 | 来源 |
|---|---|---|---|
| 1 | **删 `control/duration.py`**，`models.CONTROL_OUTPUT_MODELS` 两个键指向同一个 `ControlOutputModel` | AST diff：与 `ControlOutputModel` 只差一处类型标注；`control_head_for` 已在 heads 里分派参数化 | B |
| 2 | **删 `config.control_hook_gate` + `HOOK_GATES`** + 校验 + naming 条目 + 一行测试 | 词汇表只有一个成员，没有任何代码读它；序列化进每个检查点 | B |
| 3 | **删 `control_dense_state_loss_weight`** + CLI 标志 + `run_ts_pipeline` 7 处 + 必需序列化字段条目 | 全仓库无任何损失读它；只有一个"是否被序列化"的测试；验证指标 `fixed_anchor_validation:675` 保留 | A |
| 4 | `fixed_anchor_validation.py:20` 改从 `terminal_state_loss` import `last_reliable_terminal_velocity_target`，删 `components` 的再导出 | 唯一活的归属规则违反：state 路径共享模块伸进 `control/` 拿一个顶层符号 | B |
| 5 | **`STRAIGHT_TORTUOSITY` 与分层标签统一到 `approach_difficulty`**：`compare_control_arms_stratified.py` 改用 `strata_masks`（其掩码原缺 `& ~established`）、`run_ts_runway_hypotheses.py` 改用 `STRATUM_*`；`score_control_arms.py` 的 1.02 **是有意的更严的地板阈值**（"genuinely straight" 参考），改名 `GENUINELY_STRAIGHT_TORTUOSITY` 并与 1.05 显式关联（断言 <），不改数字 | 四处定义、一处掩码错；stratified 读数的 vectored 层会移动——这正是目的 | C |
| 6 | 修错测试：`test_ts_transformer.py:305` 重述 schema（import `DEVELOPMENT_COHORT_SCHEMA`）；`:3253` 重述通道元组；`:382,424,440,2495` `SystemExit` 无 `match`；`test_cta_conditioning.py:182` **无断言**（断言探针批带有限的 `cta_s`）；`test_final_constraint.py:363-378` 绑定本机日期产物、与 5 行后的封闭测试重复（删） | 逐条引用了断言 | C |
| 7 | **recipe 冻结改为字面量**：`control_simple_v1_overrides` 里 `dt_s / seq_len / channels / aircraft_type / random_train_anchor_min_future_s / validation_common_grid_points / position_loss_scale_m / final_time_scale_s` 八个字段钉在模块默认值上 | 改默认值静默重定义 v1/v2/v3，追溯改写"同 recipe 一轴"的配对比较 | A |
| 8 | 五个字节相同的私有助手合并到 `io_utils.py`（`_write_json_atomic`×2、`_file_sha256`×2、`_utc_now`×2、`_sha256_bytes`×2、`_is_cuda_oom`×2）；`fixed_anchor_validation._signed_spread` 改 import `metrics._spread`，`_magnitude_spread` 改名（与 `evaluation/stats.magnitude_spread` 同名不同 schema） | `cross_validation.py` 的 `allow_nan=False` 变体是有意义的差异，做成参数 | A |

## 三、T1 — 删除已废除设计（各一次提交）

| # | 改什么 | 证据 / 风险 | 量 |
|---|---|---|---|
| 9 ✅`0702669` | **closure 跟踪器**：`control/constraints/closure_tracking.py`、`tests/test_closure_tracking.py`、`forecast.py` 的 `tracking_config` / `track_closure_forecasts` / `--closure-track` 路径（≈110 行）、`__main__` 标志、`docs/experiments/closure_p1d_arms.json` | 设计文档 §四 批准；带未修 BLOCKER；`2026-09-06_closure_p1d_tracking_results.zh.md` 加"代码已退役，跟踪代价修正为 +10.5 m"banner。closure 输出本身**保留**作对照臂 | M |
| 10 | **horizon curriculum 的死半边**：`control/training/curriculum.py` ≈150 行、`train.py` 约 10 处 `training_stage` 穿线、两个 config 字段；保留 `close_duration_prefix`（同时把它开头五个不可达的形状 guard 删掉，只留 `allclose`） | 默认 `()`，所有 recipe 钉 `()`，无臂设过；字段在 `REQUIRED_SERIALIZED_CONTROL_FIELDS`——移除是宽松方向，旧检查点仍能加载 | M |
| 11 | **`arc-length-geometry` 目标族**：目标、`fixed-anchor-arc-length-geometry` 选择指标、16 个 `control_arc_*` / `control_terminal_*` 字段、`train.py:2320-2400`、`fixed_anchor_validation.py:110-315`、`components._arc_length_geometry_objective`；随之删 `predicted-detached-time` / `final-time-decoupled`、`CONTROL_GRADIENT_CLIP_POLICIES` 第二值；**保留** `arc_length_geometry.resample_horizontal_arc_length_numpy`（`geometric_metrics` 在用）与三个 `*_metrics_numpy`（验证每轮在用） | 27 个臂文件 0 使用；ENGINEERING_NOTES / OPEN_ITEMS / README 0 提及；唯一 setter 是三个 2026-08 教师运行器；约 20 条测试点名它；2026-08 教师检查点将不能加载（它们早已因单位变更被拒） | M |
| 12 ✅`0898291` | **`control/loss/regularization.py`** + `control_effort_loss_weight` / `control_smoothness_loss_weight` + 两个 CLI 标志 | 所有 recipe 钉 0.0，无臂设过，每批算一次乘零 | S |
| 13 | `control/loss/terminal_clock.py` + `control_terminal_supervision_clock` 轴（随 11 一起，`state-supervision` 策略是 `return result`） | `2026-08-02_dual_clock_terminal_ablation.zh.md` 引用其数字——归档而非删除 | S |
| 14 | `state_position_reference="anchor-relative"`：被自己预注册的规则否决，删或在字段注释写"vetoed" | 只有历史臂文件用 | S |

## 四、T2 — 归档 2026-08 教师机器（一次提交，L）

**归档为一个单元**到 `4dTrajectory/ts_transformer/archive/oracle_teacher_2026_08/`（保留在仓库、不在
import 路径上、README 指向引用它的结果文档 `2026-08-02_oracle_teacher_experiment.zh.md`、
`2026-08-16_control_simple_v1_development.zh.md`）：

- `control/oracle/{shooting,optimization,pretraining,curriculum,cohort,evaluation,imitation,targets}.py`
  （1,410 行）及其测试 `test_control_oracle.py`、`test_oracle_teacher.py`；
- 运行器 `run_ts_control_oracle.py`、`run_ts_oracle_teacher_{audit,optimize}.py`、
  `run_ts_simple_teacher_paired_cv.py`（C 判定"保留"的是它们各自"不互相重复"，B 判定它们的库已死；
  结论：作为**同一个已完成 campaign 的集合**归档，`run_ts_control_basis_oracle.py` 是宽度研究的现役
  继任者）；
- `__main__ --control-teacher-schedules`（零运行器、零臂文件引用）；
- `TransportChartVelocityBackend` 与 `(first-order-lag, transport-chart-velocity)` 注册项；
  `dynamics/inverse.refine_piecewise_constant_schedule`；oracle 版模仿损失（与 `train.control_imitation_mse`
  公式不同——两个"模仿目标"只活一个）。
- **`control/oracle/basis.py` 移出 `oracle/`** 为 `control/basis_fit.py`（它是拟合不是教师，是唯一有
  现役消费者的成员）；`BatchedOracleTeacher` 随归档。
- 同一提交归档 `control/constraints/nominal_residual.py` + `control/guidance_laws.py`（跟踪器删后
  它的唯一消费者是未采用的 hook；`2026-09-06_control_hooks_results.zh.md` 引用其数字——加"代码已归档"）。
  `barrier_filter.py`、`gates.py` 保留（采用的预测期屏障）。
- `scene/features.py` 的序列半边（`neighbours [N_MAX, L, 6]` + 掩码 + `_series_on_grid`）：L4 门不过、
  唯一消费者只读实体/标量半边——删，`scene/__init__.py` 说明。

## 五、T3 — 结构（各一次提交，M）

| # | 改什么 | 理由 |
|---|---|---|
| 15 | `train.py:173-1187` → **`objective.py`**（`target_contract`、`loss_component_names`、全部损失构造、`prediction_loss{,_components}`） | 杀掉 `batching.py:75` 函数内延迟 import 的循环；四个运行器为了一个损失 import 整个训练循环；`batch_contract.py` 存在的理由 |
| 16 | `train.py` 回放 + 验证选择（≈740 行）→ **`validation.py`**；`fit_model`（542 行）留在 `train.py` | train.py 从 3,246 → ≈1,500 |
| 17 | `config.__post_init__`（742 行）拆为 `_validate_vocabulary / _validate_recipe / _validate_control_contract / _validate_ranges`；T1-11 之后 `control_state_loss_grid` 改为由目标**派生**的属性，删六个互推 raise | 三个"轴"其实是一个选择 |
| 18 | `dataset.py` 拆出 **`data_provenance.py`**（122-366，纯哈希；`evaluation_protocol` 为了比较哈希拖进 torch）与 **`splits.py`**（1788-1954） | 纯函数困在 2k 行模块里 |
| 19 | `__main__.main`（714 行）拆为 `cli/<command>.py`，各暴露 `add_cli_arguments()` + `run_cli()`（仓库已有此模式：`approach_clustering/cli.py`）；给 `cli_values` 加与 `run_naming` 同款的 import 时字段断言；把 ~20 个与字段异名的标志改成同名 | 两套 config 表面靠手写映射 |
| 20 | `dynamics/backends.py` 三个近似相同的后端类折成一张 `(endpoint_fn, dense_fn, post_fn)` 表（≈200 → 40 行）；`_refuse_hook` 六处调用消失 | T2 删掉两个注册项之后 |
| 21 | 给 `control_nominal_*` / `control_barrier_*` 八个 hook 增益加 CLI 标志（设计文档 §三.8 要它们进检查点，但今天没有任何入口能设）；七个从未非默认的 `*_scale` 字段降为模块常量 | 15 个字段只能靠手写 `--config-overrides` JSON 设 |
| 22 | `forecast.py` 四个 `*_latent_forecasts` 共用一个 `_latent_forecasts(latents, probabilities, …)`；`latent_dim < 1` / `cta_offset` 的库内重复校验删掉（CLI 边界已校验一次） | 90 % 相同的前导 |

## 六、T4 — 测试与运行器（各一次提交）

| # | 改什么 | 证据 |
|---|---|---|
| 23 | **把 `trajectory_data_process/tests/test_ts_{pipeline,clock_attribution,control_capacity_ceiling,kinematic_ablation,overfit_diagnostic,predictability_report}.py`（1,350 行）搬进 `ts_transformer/tests/`，同一提交修掉 13 条红**（全在 `test_ts_pipeline.py`，fixture 腐烂：`'tuned_parameters' in ('one or more lateral-pass eligibility rosters are missing')`） | `CLAUDE.md` 的命令永远不跑它们，套件绿色不携带信息 |
| 24 | **`tests/support.py`**：`tiny_config` / `synthetic_series` / `fake_provenance` / `dynamics_context` / `terminal_contexts`（8 个助手 2–4 份拷贝，`_fake_data_provenance` 字节相同）；删 25 个文件的 `sys.path` 前言（`conftest.py` 已覆盖）；`test_closure_tracking.py` 为了 import 兄弟测试文件把 `tests/` 塞进 sys.path 的做法随 T1-9 消失 | |
| 25 | **`runner_support.py`**：`parse_airports`×5、`write_json_atomic`×5、`series_digest`×3、`file_sha256`、`write_reports`（三个消融运行器同签名同结构）——16 份字节相同的拷贝 | |
| 26 | **拆 `test_ts_transformer.py`**（6,507 行 217 条）按 C 给出的 19 个连续、主题单一的行段（评估协议 / 通道帧 / 窗口 / state 损失 / 划分 / 锚点 / 验证选择 / control 配置 / control 目标 / … / 端到端 CLI） | 24、25 之后是机械操作 |
| 27 | **运行器 20 → 14**：归档 `run_ts_control_arms.py`（`run_ts_frame_ablation.py` 的 `main` 逐字相同且后者已能跑 control campaign）、`run_ts_flight_model_paired.py`（一次性、结果已进 CLAUDE.md 默认表）、`run_ts_cv.py`（`run_ts_pipeline` 的预设）；`run_ts_oracle_teacher_audit.py` 折入 `optimize --steps 0`（15 行相同的 `TSConfig` 字面量）——后两者随 T2；`README.md:388` 引用的 `run_ts_control_mixture_report.py` **不存在**，删引用。约定写进 `CLAUDE.md`：`run_ts_<subject>.py` = 现役可重跑工具；已完成的一次性驱动进 `archive/`，头一行写引用它的结果文档 | |
| 28 | `test_architecture.py` 的 `docs` 豁免在 T5 完成后去掉 | |

## 七、T5 — `docs/*.py` 迁移（枢纽三个先，其余随手）

`docs/` 内部只有 4 条依赖边，它们标出三个枢纽：

| 脚本 | 消费者 | 去向 |
|---|---|---|
| **`p1_closure_oracle.py`**（437） | **包代码引用它当数据生产者**：`closure_output.py:27`、`config.py:1002`、`__main__.py:254` 都写 "`docs/p1_closure_oracle.py labels`" | `closure_labels.py`（顶层；`fit` + 三个子命令的库部分）+ `run_ts_closure_oracle.py`。**13 个里最高优先** |
| **`compare_frame_arms.py`**（319） | 三个兄弟 import 它的 `load_arm / print_table / _fmt / flight_key`；8 个文档引用 | `arm_readout.py`（包）+ `run_ts_arm_readout.py`（含 `--stratified` / `--paired` 折入 `compare_control_arms_{stratified,paired}.py`；`_sign_test` 进包） |
| **`score_control_arms.py`**（275） | `plot_imitation_design`、`compare_control_arms_stratified` import 它；`CLAUDE.md:153` 与 ENGINEERING_NOTES 引用 | `control/metrics/bank_skill.py` + `run_ts_control_score.py`；**迁入时修 1.02**（T0-5） |
| `phase0_intent_diagnostics.py`（439） | `_manifest` / `_series_for` → `p1_closure_oracle` | 两个助手进包（`intent_explainability` 已收了 population/CV）；五个子命令 → `run_ts_intent_diagnostics.py` |
| `compare_constraint_arms.py`（287） | 10 个结果文档 + 8 个臂文件 `_comment` + OPEN_ITEMS | `run_ts_constraint_readout.py`，import 新 `arm_readout` |
| `measure_procedure_adherence.py`（168） | 零包 import，只依赖 geokit | `run_ts_procedure_adherence.py`（最干净的一个） |
| `relabel_published_categories.py`（136） | README + ENGINEERING_NOTES | `run_ts_relabel_categories.py`（已正确 import `run_naming`） |
| `trace_architecture.py`（461） | 两个教程 html | `run_ts_trace_architecture.py`（文档构建工具，或留下并在 CLAUDE.md 明说） |
| `plot_control_wiggle_diagnosis.py`、`backfill_category_accuracy.py` | 一次性（后者原地改写 `categories.json`，已应用） | `archive/` |
| `plot_imitation_design.py`（207） | **全仓库零引用** | 删（两张 PNG 留在 `docs/figures/`） |

## 八、T6 — 文档卫生（一次提交，全部一行修法）

- `OPEN_ITEMS.md`：头部换成设计文档 §〇 的状态表；P1.d 跟踪器条目、`control_training_review` 条目加"废除"。
- `CLAUDE.md:42-58`：closure 降为对照臂、跟踪器标"已退役（BLOCKER 未修）"；Contracts 加 `latent_dim / latent_prior_components / latent_beta / latent_free_bits_nats` 与 `cta_conditioning`（`z=posterior` / `cta=given` 进 run name 的纪律）；`Current defaults` 表加 `latent_dim=0`、`cta_conditioning=off`；`reanchored-rk4` 的角色写明是 casadi 交叉检查的孪生。
- `README.md`：`## Status`、`## Historical results…`、`## Known gaps` 加"pre-v5 历史"banner（它仍说 control "没有发表过精度结果"、"只训过 KRDU"）；Design 加 closure + latent/CTA 段；与 `CLAUDE.md` 重叠的六节改为一行指针。
- 7 处对 `2026-09-07_scene_join_anchor_design.zh.md` 的死链（含**源码** `intent_conditioning.py`）→ `.discard.md`。
- `2026-09-03_airport_frame_ablation_results.md:1` 的 `30d` 前缀（vim 命令泄漏）。
- `2026-09-06_closure_p1d_tracking_results.zh.md`、`2026-09-07_control_training_review.zh.md` 加取代 banner。
- `ENGINEERING_NOTES.md` 加 closure / latent / CTA 节，或在 `CLAUDE.md` 说明它止于 control 路径。
- `notes_7_20.md`（416 字节、四行问号）删或并入 OPEN_ITEMS。
- `docs/code-health-followups.md` 两条已失真的断言改正（"每个测试文件都有 sys.path 前言"——7 个没有；"运行器要把 docs 放上 sys.path"——零个）。
- `inverse.py` 的核心断言改准确（逆是按飞行模型键，不是 `(model, backend)` 对）；四处 docstring 引用的旧模块名。

## 九、明确保留（审计核查过、不要再翻）

`export.py` 从 `optimization/evaluation_export.py` 导入契约（无重述）；`coordinate_frames / channels`
全部常量来自 `geokit`；`run_naming._DEFAULTS` 从 `TSConfig` 派生并在 import 时断言字段名；测试运行时间
不是问题（最慢 10 s，全是真实的一步训练/rollout）；`run_ts_coordinate_ablation.py` 是契约校验器不是消融
驱动；`run_ts_control_basis_oracle.py` 与 `run_ts_control_oracle.py` 不重复（后者随 T2 归档是因为其
campaign 已结束，不是因为重复）；仓库根 `final_approach/` 与 `closure_geometry` 无重叠；
`reanchored-rk4` 后端保留（优化器的孪生）。

## 十、执行顺序与提交切分

```
现在（L1 跑完前，dev-l2 上）：T0-1…T0-8，每项一次提交 → 套件 → 合入 dev-leg-ctrl（L1 结束后、L2 启动前）
L2 campaign 启动后：T6（文档）→ T1-9…14 → T2 → T4-23/24/25 → T3-15/16 → T5 枢纽三个 → T3-17…22 → T4-26/27/28 → T5 其余
每个工作包：代码 → opus review（只审代码）→ 修 → 套件 → 提交；campaign 之间合入
```

预期收益：`control/` −41 %，config 字段 134 → ≈105，`train.py` 3.2k → ≈1.5k，运行器 20 → 14，
`docs/*.py` 13 → 0，测试套件的退出码重新有意义，四处 `STRAIGHT_TORTUOSITY` 归一。
