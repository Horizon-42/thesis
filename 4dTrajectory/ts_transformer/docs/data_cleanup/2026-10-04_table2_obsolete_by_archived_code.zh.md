# 表 2：由已归档代码生成、现在无用的老实验数据（只列清单，未删除任何文件）

统计日期 2026-10-04。路径相对 `4dTrajectory/outputs/`，除非写明 `前端`。与表 1 不重叠：凡是旧版 two-tier 词表/执行器/prior 的产物都在表 1，这里只列**生成它们的代码已经移进 `ts_transformer/archive/`** 的数据。

## 判断依据与可信度

每一行按 `archive/<名>/README.md` 逐条核对。「可信度」列：

- **已证实**：归档 README 或设计文档点名了这个目录，或这个类目的 runner 就在归档里。
- **推断**：按目录名、日期和配置名判断它属于某个归档方案，没有 README 点名。动手前请你过目。

「读它的活代码」一列：我对 `4dTrajectory/ts_transformer`、`aeroviz_backend`、`aeroviz-4d/src` 里的非归档代码做了 grep，找不到引用的写「无」。

## 删除前必须知道的事

1. **磁盘 99 % 满，只剩约 7.4 GB。** 「A 已确认」合计约 **22 GiB**，另有 B 块约 3.2 GiB 待你判断。
2. **前端类目是发布出去的**：`comparison/categories.json` 列着它们，要用发布器撤类目再删文件。归档 README 说 `two_tier_v2` 的类目已经撤过；其余还在。
3. 有些目录是 `docs/` 里某份读数的**唯一数字来源**。归档代码的 README 都说「数字在 docs 里」，所以删数据不丢结论，但丢「重新核对的机会」。

## A. 已确认：归档 README 点名的数据

| 归档目录 | 路径 | 大小 | 日期 | 可信度 | 说明 |
|---|---|---:|---|---|---|
| `closure_2026_09` | `KRDU/experiments/closure_p1_20260905` | 4 MB | 09-05 | 已证实 | closure 输出的 P1 oracle |
| | `KRDU/experiments/closure_p1c_20260905` | 1.26 GiB | 09-06 | 已证实 | P1.c closure 解码器训练。归档 README：该输出的训练 09-09 冻结、tracker 09-07 删除 |
| | `KRDU/closure_labels` | 22 MB | 09-06 | 已证实 | `closure_labels_path`，标签写出器已归档 |
| | `KRDU/experiment_predictions/C_pred_2cb773e59089` | 10 MB | 09-07 | 推断 | 前端有 4 个 `experiment_c_pred_*` 类目（180 MB，含 closure-oracle） |
| | 前端 `experiment_c_pred_*`（仅 KRDU） | 180 MB | | 推断 | 4 个类目 |
| `scene_encoder_2026_09` | `KRDU/experiments/scene_phase0_20260905` | 862 MB | 09-05 | 已证实 | 场景编码器的 Phase 0 真值意图上界；L4 门失败，编码器从未建成 |
| | `KRDU/experiment_predictions/C_truth_intent_5d0f61394cee` + 前端 2 个 `experiment_c_truth_intent_*` | 5 MB + 90 MB | 09-07 | 推断 | `truth-*` 意图 oracle（已冻结，只能加载旧配置） |
| `plan_head_2026_09` | `KRDU/experiments/plan_guidance_20260910` | 2.78 GiB | 09-12 | 已证实 | 归档 README 写明此树「留在原处」，因为里面的 `step3d_lockstep_l1` 是规则引导基线、新 gate 要读。那些 gate（`manoeuvre_gates`）后来也归档了，活代码里 grep 不到任何读者，只剩 docs 引用 |
| | `POOLED/experiments/runway_intent_r2_*`、`r2b`、`r2c`、`r2d`、`r31`、`r31_centered`、`r32`、`r3` | 3.87 GiB | 09-13 → 09-14 | 已证实 | 归档 README：`runway_intent_r2*.py`、`r3*.py`、`r31*`、`r32` 移进 `plan_head_2026_09/runway_intent/`。其中 `r2d` 1.83 GiB、`r3` 1.93 GiB 占大头 |
| | `KRDU/experiment_predictions/step3g_*`、`step5b_*`、`POOLED/experiment_predictions/step5b_*`、`*_day_a_expert_*` | 约 40 MB | 09-13 | 已证实 | plan 头 fan4 和按天划分的 plan 专家 |
| | 前端 `experiment_step3g_*`/`step5b_*`（五机场 37 个）+ `experiment_*_day_a_expert`（5 个） | 528 MB + 159 MB | | 已证实 | plan 头的已发布类目 |
| `manoeuvre_codes_2026_09` | `KRDU/experiments/manoeuvre_tok_20260918` | 4.83 GiB | 09-18 | 已证实 | 意图 token 分词器 + 执行器联合训练（段长 × K × 种子） |
| | `KRDU/experiments/two_tier_v3_b_20260919` | 3.39 GiB | 09-20 | 已证实 | 阶段 B 意图 token 的相对门 |
| | `codebooks/`（12 个目录） | 3.5 MB | 09-18 → 09-19 | 已证实 | 学出来的 FSQ 码本，C28 已归档 |
| | `KRDU/experiment_predictions/S60_*` + 前端 6 个 `experiment_s60_*` | 约 50 MB + 237 MB | 09-18 | 已证实 | 意图 token 的预测/类目 |
| `two_tier_v2_2026_09` | `archive/KRDU/two_tier_2026_09`（`two_tier_{feasibility,t0b,t0c,t1a,l1,l1b,l1c,l2,l2c,l2d}`） | 1.76 GiB | 09-16 → 09-17 | 已证实 | 归档 README：「campaign tree 在这里；picker 类目已撤」。这个 `archive/` 目录本来就是为了存它，删前确认你不再需要这些 checkpoint |
| | `KRDU/experiment_predictions/L1_pa_*`、`L1_sf_*`、`L1_tf_*`、`T0c_*` | 约 100 MB | 09-16 → 09-17 | 推断 | 按名称是 two-tier v2 的 L1 / T0c 预测记录。注意 `L1_native32`、`L1_dense32/64` 是一层控制头的 L1（活代码），**不在这里** |
| `closed_loop_sft_2026_09` | `POOLED/prior/v3_sft_20260925` | 260 MB | 09-27 | 已证实 | CAT-K 闭环微调；归档 README 点名，正式运行失败（val 90.2→93.1 %，转弯进近变差） |
| `one_commanded_scene_2026_10` | `POOLED/prior/m4_traffic_20260929`、`m4_passes_20260929` | 各 443 MB | 09-29 → 09-30 | 已证实 | R32 的 M4 第一轮；归档 README 点名保留位置 |
| | `POOLED/traffic/free_generation_20260928`、`free_generation_aug_20260928`、`free_generation_20260928.aborted-20260928T1330Z` + `.sh/.pid/.log` | 约 45 MB | 09-28 | 已证实 | R31 的 M3 自由生成 |
| `heading_reading_2026_09` | `POOLED/analyses/heading_reading_20260924` | 2.34 GiB | 09-24 | 已证实 | 航向读法 H1/H3 比较；README 点名 |
| | `POOLED/analyses/heading_clearance_20260924` | 1.34 GiB | 09-24 | 已证实 | 三种 clearance 规则；README 点名 |
| `oracle_teacher_2026_08` | `KSJC/experiments/oracle_teacher_20260802` | 38 MB | 08-02 | 已证实 | 直接射击 oracle 教师 |
| | `KSJC/experiments/oracle_teacher_20260816_current_manifest`、`POOLED/oracle_teacher_simple_v1_20260816` | < 1 MB + 80 KB | 08-16 | 已证实 | 同一条线的小目录 |
| | `KSJC/experiments/control_simple_v1_20260816`、`control_simple_v1_smoke_20260816` | 95 MB + 126 MB | 08-16 | 推断 | simple-v1 是「教师初始化」对照（intents：`simple-v1 · 教师初始化对照`），教师预训练代码已归档 |
| `flight_model_paired_2026_09` | `KSJC/experiments/flight_model_paired`、`flight_model_paired_seeded_head` | 各 452 MB | 08-19 | 已证实 | `point-mass` 对 `first-order-lag` 的一次性配对比较；结论已写进 `CLAUDE.md` 默认值表（D4） |
| `nominal_law_hook_2026_09` | `KRDU/experiment_predictions/R_nominal_residual_c1848a1b8a3f` + 前端 2 个 `experiment_r_nominal_residual_*`（KRDU、KSJC） | 10 MB + 71 MB | 09-06 | 已证实 | `nominal-residual` 命令 hook；barrier 滤波保留在活代码里 |

**本块合计约 22 GiB**（其中 `plan_guidance` + `runway_intent` + `manoeuvre_tok` + `two_tier_v3_b` + `heading_*` 五项已占约 17 GiB）。

## B. 老师初始化模型的发布产物（体积大，单独列出，请你判断）

这一组的依据是「模型名含 `teacher`」，属于**推断**，不是 README 点名。

| 路径 | 大小 | 日期 | 说明 |
|---|---:|---|---|
| `POOLED/checkpoint_publications/ts_itr_control_simple_v1_teacher_all_airports_seed1337_cb18f034f4af` | 1.40 GiB | 08-16 | 五机场共用的发布 checkpoint |
| `POOLED/ts_itr_control_simple_v1_teacher_all_airports_seed1337` | 49 MB | 08-16 | 训练目录 |
| `POOLED/experiment_predictions/development_{no_teacher,teacher}_current_manifest_seed1337_*` | 各 275 MB（共 550 MB） | 09-30 | 老师 / 无老师配对 |
| 前端 `prediction_ts_itr_control_simple_v1_teacher_*`（五机场各 2 个） | 1.15 GiB | | 已发布类目 |
| 前端 `experiment_development_{teacher,no_teacher}_*`（KSJC 4 个） | 440 MB | | 同上 |

注意：这是 08-16 的五机场共用发布 checkpoint，出自 simple-v1 的教师预训练路线（该路线的代码已归档）；`checkpoint_publications` 目录里只有它这一个 checkpoint 目录。我没有把它列入 A 块合计，等你决定。

## C. 旧版词表的前端可视化集（归档代码做的）

前端 `training/` 里下面几类按读法名（`box-v3`、`box-v2-wedge`、`segment-v13/v14`）属于被归档的 `instruction_vocabulary_2026_09`，属推断。五机场各有，合计约 60 MB：

| 集 | KRDU 大小 | 对应归档 |
|---|---:|---|
| `box` | 6.9 MB | `box-v3` 词表回读，来自 `two_tier_v3_bprime_20260921/vocabulary_box_five_airports` |
| `box_v3` | 5.9 MB | `box-v2-wedge`，同上 |
| `v15_nomerge_noposition` | 9.3 MB | `segment-v14` 真值，09-21 |
| `prior_s1337_val`、`prior_s2024_val` | 各 13 MB | 标题是 `segment-v13` prior，词表读法属于被归档的 segment/box 那一代（按标题推断） |

这些在 `index.json` 里登记；与表 1 第 8 节同样要用发布器重写索引。

## D. 边界情况：运行器已归档，但生成它们的代码路径仍然在

这些数据不是被**已归档模块**生成的，所以我**没有**把它们算进上面两块，只列出来让你知道。

| 路径 | 大小 | 日期 | 情况 |
|---|---:|---|---|
| `KSJC/experiments/wiggle_*`（loss_design 1.1 GiB、conditioning 587 MB、training_budget 450 MB、segment_count 412 MB、combination 344 MB） | 约 2.9 GiB | 08-19 → 08-20 | 驱动它们的 `run_ts_control_arms.py` 已归档（`control_arms_runner_2026_08`），但训练代码（`train`）仍在，结论在 `docs/history/2026-08_control_path/`。前端 `ts_krdu_imitation_design_*`（8 个，约 340 MB）同属这条线 |
| `KSJC/experiments/imitation_v5`、`imitation_replication` | 187 MB + 374 MB | 08-21 | 同一个归档 runner 的 KSJC imitation 权重阶梯 |
| `KRDU/experiments/control_hooks_20260906`、`control_hooks_v2_20260906` | 各 1.04 GiB | 09-05 | 混合：里面有 **barrier**（已采纳，活代码）和 nominal-residual（已归档）的臂。**不能整目录删** |

## E. 不在两张表里的老一代数据（代码仍在，是否保留由你定）

这些是单机一层路径（state / control）的训练数据，对应的代码**没有**归档，所以我没有判为无用：

| 路径 | 大小 | 说明 |
|---|---:|---|
| `KRDU/experiments/` 里其余 48 个一层路径战役（`sf_n3/n4/n6/n7`、`l1…l5`、`a0…`、`b1…`、`final_constraint`、`airport_frame` 等） | 29.1 GiB | 一层控制路径，`CLAUDE.md` 默认值表仍引用它们的结论 |
| `KSJC/experiments/` 其余（`airport_frame`、`final_constraint`、`control_procedure`、`state_v2` 等） | 2.9 GiB | 同上 |
| `<ICAO>/ts_pred_pooled_*_{train,val}`（五机场） | 3.7 GiB | 08-17 的 pooled 一层预测 |
| `POOLED/ts_*`（07 月）、`POOLED/experiments/{openap_direct_*,kinematic_weight,...}` | 1.1 GiB | 07–08 月早期一层路径 |
| `KRDU/experiment_predictions`、`KSJC/experiment_predictions` 其余 | 约 0.4 GiB | 同上 |

如果你的重构也打算淘汰一层路径，请告诉我，我再单独做一张表。

## 本表小计

| 块 | 大小 |
|---|---:|
| A 已确认 | 约 22 GiB |
| B 老师初始化发布产物（待你判断） | 约 3.2 GiB（含前端 1.6 GiB） |
| C 旧词表前端集 | 约 0.06 GiB |
| D 边界情况 | 约 6.5 GiB（含 control_hooks 两个，其中一部分是活的） |
