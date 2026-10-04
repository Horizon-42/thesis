# 表 1：因 two-tier 重构而无用的实验数据（只列清单，未删除任何文件）

统计日期 2026-10-04。路径相对 `4dTrajectory/outputs/`，除非写明 `前端` 或其他根。大小是 `du` 实测值，日期是目录内最新文件的修改日。

## 判断依据

- 设计 D20（`docs/two_tier/design/outline.md` §3.1）：新设计「不读任何旧版本做的 artefact、executor spec 或 prior（`v1`–`v6`）」。
- 现行的新设计产物只有两个：`instruction_language/v9_20261004` 和 `executor/v14_20261004`（A21 正式产物）。它们**不在本表内**。A25 之后它们也会被取代，那时再单独列。
- 旧版 two-tier 的所有句子、执行器、prior、窗口/交通读数，都建立在旧词表和旧执行器上，新设计不会再读。
- 被**已归档代码**生成的数据放在表 2，本表不重复。

## 删除前必须知道的三件事

1. **磁盘 99 % 满，只剩约 7.4 GB 可用**（`df /`）。本表合计约 **24 GiB**，是最直接的空间来源。
2. **`dev-two-tier` 分支上的 runner 仍把旧路径写成默认值**：`instruction_language/v4_20260924`、`v5_20260926`、`v6_20261002`、`executor/v11_20260927`（例：`experiments/traffic_window_reward.py:67-68`、`prior_free_generation.py:40`）。在新设计合并之前删掉这些目录，这些 runner 在 `dev-two-tier` 上就跑不起来。你说过旧数据无用，所以我只在此提醒，不改变判断。
3. **现在有进程在读新旧数据**：`final_descent_tolerance`（四个 h 值）、`check_live --set-id stage_a_v9_trial`、一个 `pytest` 全量测试，都在跑。它们读的是 worktree `two-tier-v4-a24` 和 `v9`，不在本表内；但删除前请先 `ps` 确认没有读本表目录的任务。

## 1. 旧版 instruction 句子 artefact（3.47 GiB）

| 路径 | 大小 | 日期 | 是什么 | 备注 |
|---|---:|---|---|---|
| `POOLED/instruction_language/v1_20260923` | 0.58 GiB | 09-23 | instruction-v1 句子/信号/标签 | 被 v2 取代 |
| `POOLED/instruction_language/v2_20260924` | 0.59 GiB | 09-24 | instruction-v2 | |
| `POOLED/instruction_language/v3_20260924` | 0.56 GiB | 09-24 | instruction-v3 | 归档 README 说 v3 曾测出执行器延迟 0 s |
| `POOLED/instruction_language/v4_20260924` | 0.58 GiB | 09-25 | v3 词表 + 日期划分 | 前端类型文件 `trainingSample.ts` 注释提到它 |
| `POOLED/instruction_language/v5_20260926` | 0.58 GiB | 09-26 | v5 | `dev-two-tier` 上 12 个 runner 默认读它 |
| `POOLED/instruction_language/v6_20261002` | 0.58 GiB | 10-02 | v6，所有行落在 UTC 偶数秒 | 新设计之前的最后一版；KAUS 留出集的 `spec_from` 来自它 |

## 2. 旧版执行器 spec + 回放（9.28 GiB）

| 路径 | 大小 | 日期 | 是什么 | 备注 |
|---|---:|---|---|---|
| `POOLED/executor/v2_20260924` | 1.72 GiB | 09-24 | 第 2 版 spec + 真值句回放 | 含从数据测出的参数，那条路线已归档（`archive/executor_vocabulary_only_2026_09`） |
| `POOLED/executor/v3_20260924` | 0.36 GiB | 09-24 | 同上 | |
| `POOLED/executor/v4_20260924` | 0.36 GiB | 09-25 | 同上 | |
| `POOLED/executor/v5_20260924` | 2.07 GiB | 09-25 | 同上 | |
| `POOLED/executor/v6_20260924` | 2.03 GiB | 09-25 | 同上 | |
| `POOLED/executor/v7_20260925`、`v8_20260925` | < 1 MiB | 09-25 | 只剩 spec 小文件 | |
| `POOLED/executor/v9_20260926` | 0.35 GiB | 09-26 | | |
| `POOLED/executor/v10_20260926` | 0.35 GiB | 09-26 | | |
| `POOLED/executor/v11_20260927` | 2.03 GiB | 10-03 | v11：目前 `dev-two-tier` 的实时执行器和 12 个 runner 的默认 | 前端 `executor_v11_20260927` 和后端实时执行器读它的 spec |

## 3. 旧版 prior 训练/读数（5.64 GiB，55 个目录中扣掉归到表 2 的 3 个）

旧 prior 全部在旧 v3–v6 词表上训练，新设计要从 v9 重新训练（prior §0.4）。

| 路径 | 大小 | 日期 | 是什么 |
|---|---:|---|---|
| `POOLED/prior/v3_reread_v11_20260927` | 3.0 GiB | 09-30 | 用 v11 执行器重读所有 v3 模型的生成结果（前端 45 个 `generation_v3_reread_v11` 类目的来源） |
| `POOLED/prior/m4_window_20260930` | 476 MB | 10-01 | M4 窗口后训练（`window` 模型） |
| `POOLED/prior/v3_drift_diagnosis_20260926` | 354 MB | 09-26 | 漂移诊断 |
| `POOLED/prior/v3_stage2_clip_20260926` | 203 MB | 09-27 | 阶段 2 clipped 模型（记忆里「augmented = clipped run」） |
| `POOLED/prior/step9_4_small_c_20261003`、`_cp_` | 各 193 MB | 10-03 | 步骤 9.4 小规模试验 |
| `POOLED/prior/v3_rl_20260925` | 128 MB | 09-27 | landing reward 的 RL 训练 |
| `POOLED/prior/step9_4_traffic_20261003` | 105 MB | 10-03 | 步骤 9.4 traffic |
| `POOLED/prior/step8_probe_test_20261002` | 103 MB | 10-02 | 步骤 8 go-around 探针 |
| `POOLED/prior/step9_4_small_w10/w30/w100_20261003` | 各 97 MB | 10-03 | 权重试验 |
| `POOLED/prior/v3_augdata_20260926` 等 40 个较小目录 | 合计约 0.7 GiB | 09-24 → 10-03 | v1/v2/v3 各阶段、augbudget 1–3、reread v9/v10、glidepath/mva/runway 诊断、`airport_embedding_20261003`、`.run` 日志 |

**需要你定的一项**：`airport_embedding_20261003`（52 MB）+ `.run`（64 KB）属于未合并的分支 `dev-airport-embedding`（R46）。它建在 v6 上，按 D20 无用，但那条分支还没合并，所以我没有把它当作确定可删。

## 4. 旧版多机读数（`POOLED/traffic`，0.24 GiB，扣掉 `free_generation_*`）

`census_20260927`、`census_v6_20261002`、`labelled_v2/v6`、`masks_v2/v6`、`interaction_20260928`、`scene_readout*`、`m4_readout`、`m4_passes_readout`、`window_generation*`、`window_val*`、`window_rewind*`、`go_arounds*`、`one_val*`、`step8_small_tests`、`m0_v6.run` 等约 60 个目录。全部建在 v5/v6 词表和 v11 执行器上。

注意：设计文档和读数文档 `docs/two_tier/readouts/` 引用这些数字（例如 `census` 的 0.98 vs 0.60 pairs/h）。outline 说旧版「只有测量值可引用」，所以删除前确认读数文档里已有这些数字。

## 5. 旧版分析读数（`POOLED/analyses`，0.31 GiB，扣掉表 2 的两个）

| 路径 | 大小 | 日期 | 是什么 |
|---|---:|---|---|
| `POOLED/analyses/heading_lead_ablation_20260927` | 296 MB | 09-27 | 航向提前量消融（v5 + v10 执行器） |
| `POOLED/analyses/instruction_distribution_20260922` | 138 MB | 09-22 | 旧词表的分布统计 |
| `POOLED/analyses/turn_change_20260922` | 68 MB | 09-22 | |
| `POOLED/analyses/single_executor_20260927` | 2 MB | 09-27 | 单机执行器一致性检查 |
| `POOLED/analyses/shallow_class_law_20260927`、`other_runway_crossing_20260927` | < 1 MB | 09-27 | |
| `POOLED/analyses/word_frames_20261003`、`final_approach_20261003` | < 1 MB | 10-03 | 读 v6，为下一版设计找证据（vocabulary §9）。**建议保留**：体积小，是新设计的证据 |

## 6. 旧版 two-tier v3 实验树（3.7 GiB）

| 路径 | 大小 | 日期 | 是什么 |
|---|---:|---|---|
| `KRDU/experiments/two_tier_v3_grid_20260918` | 2.61 GiB | 09-19 | 阶段 A 的无 token 执行器网格（lookback × 段长，4×5 格 × 2 种子）。代码（`lockstep.py`、`two_tier_grid_queue`）还在，但那是 v3 设计 |
| `KRDU/experiments/two_tier_v3_a3b_20260919` | 105 MB | 09-19 | A3-b 控制分辨率 |
| `KRDU/experiments/two_tier_v3_bprime_20260920` | 0 | 09-20 | 空目录 |
| `KRDU/experiments/two_tier_v3_bprime_20260921` | 434 MB | 09-21 | B′：指令词表 box-v3 训练/读数 |
| `POOLED/experiments/two_tier_v3_bprime_20260921` | 493 MB | 09-21 | B′ 的五机场版；前端 `box`、`box_v3` 来自它 |
| `KRDU/experiments/cohorts_v7_20260923` | 19 MB | 09-23 | v5 roster 上的开发队列 |
| `POOLED/experiment_predictions/{L60_D20_*,L120_D20_*,generation}` | 约 20 MB | 09-18 → 09-30 | 网格的预测记录 |

## 7. 发布/导出的中间产物（`POOLED/training_publish` 等）

| 路径 | 大小 | 日期 | 是什么 |
|---|---:|---|---|
| `POOLED/training_publish/attitude_20260930` | 563 MB | 10-01 | 旧版 Training 导出（带姿态）的暂存，已发布到前端 |
| `POOLED/rebuild_20260926`、`rebuild_20261002` | < 1 MB | | 旧 artefact 重建的日志/脚本 |
| `POOLED/training_export_v11_20260927`、`training_export_content_binding_20260928` | < 1 MB | | 旧导出的日志/脚本 |

## 8. 前端已发布的旧版 Training 集（`aeroviz-4d/public/data/airports/<ICAO>/training/`）

五个机场（KRDU、KSJC、KMSY、KSMF、KSTL）合计 **0.55 GiB**（KRDU 136 MB，KSTL 120 MB，KSMF 114 MB，KMSY 98 MB，KSJC 94 MB）。KAUS 没有。

| 集 | 每机场大小 | 对应旧版本 |
|---|---:|---|
| `instruction_v1`、`instruction_v2`、`instruction_v3`、`instruction_v3_day_split` | 5–7 MB / 2 MB | instruction-v1…v3 回读 |
| `prior_v3_step1_20260924_full_s1337`、`prior_s1337_val`、`prior_s2024_val` | 2 / 13 / 13 MB | 旧 prior 生成 |
| `executor_v11_20260927` | 1.3 MB | v11 执行器回放 |
| `generation_base/landing_r01–r08/augmented_r01–r08`、`generation_augstart_*`（共 21 个） | 各约 2.4 MB | 三个模型各 8 轮生成 |
| `windows_traffic_r05_*`、`windows_window_r05_*`、`windows_augmented_r07_*`、`traffic_windows_select` | 各 4.5–4.7 MB | 窗口模型 |
| `box`、`box_v3`（归表 2）、`v15_nomerge_noposition`（归表 2） | | 见表 2 |

**手工删目录会弄坏前端**：`training/index.json` 和 `overlays.json` 列着这些集，后端启动时还会预热它们。应当用发布器重写索引，不要只删目录。

## 9. 前端已发布的旧版对比类目（`.../<ICAO>/comparison/`）

| 类目前缀 | 个数 | 大小 | 对应 |
|---|---:|---:|---|
| `experiment_generation_v3_reread_v11_2026*` | 45（五机场各 9） | 346 MB | 旧 prior 的生成重读 |
| `experiment_l60_d20_*`、`experiment_l120_d20_*` | 13（仅 KRDU） | 175 MB | v3 阶段 A 网格 |

## 本表小计（约）

| 块 | 大小 |
|---|---:|
| 1 instruction v1–v6 | 3.47 GiB |
| 2 executor v2–v11 | 9.28 GiB |
| 3 prior | 5.64 GiB |
| 4 traffic | 0.24 GiB |
| 5 analyses | 0.31 GiB（建议保留其中两个小目录） |
| 6 two-tier v3 实验树 | 3.72 GiB |
| 7 发布中间产物 | 0.55 GiB |
| 8 前端 Training 集 | 0.55 GiB |
| 9 前端对比类目 | 0.51 GiB |
| **合计** | **约 24 GiB** |

## 明确不在本表、请勿当作旧数据的

| 路径 | 原因 |
|---|---|
| `POOLED/instruction_language/v9_20261004`、`POOLED/executor/v14_20261004` | 现行 A21 正式 artefact；A25 之后才被取代，且要你同意才删 |
| `POOLED/instruction_language/heldout_kaus_part1_20261003`（87 MB） | KAUS 是留出的测试数据（part_1 已打开）；它用 v6 的 spec 做成，将来需要按新 spec 重建，但这是测试集的开封记录，不要在我这边删 |
| `trajectory_data_process/outputs/harvest*` | 原始数据，不是实验产物 |
| `4dTrajectory/outputs/<ICAO>/{runway,fitted_adsb,runway_cons,shared_references}` | 优化器输出，根 CLAUDE.md 说它们是要重新生成评测的来源，不是 ts 产物 |
| `4dTrajectory/ts_transformer/data/day_split_20260924.json` | 已提交的 90 天划分（C32），新设计继续用 |
| `.claude/worktrees/*` | 各分支的工作目录，`two-tier-v4`（1.7 GiB）是新设计的开发目录 |
