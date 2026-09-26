# 做完的研究线（代码仍在的）

这里的文档都是**记录**：实验已经跑完，或方案已经放下；有效的结论已进 `../reference/` 的默认值、契约和陷阱（表里写了编号）。
文中的路径、状态、"下一步"都停在它写成的那天，不再改；文件名原样保留，所以旧路径按文件名就能找到。
代码已归档的研究线，文档在 `../../archive/<名>/docs/`；还有 10 份记录暂留在 `docs/` 根目录（原因见 `../README.md`）。
`OPEN_ITEMS_2026-09-18.md` 是单模型路径到 2026-09-18 的状态记录（原 `docs/OPEN_ITEMS.md`），仍开着的决定已移到仓库的
`docs/open-items.md`。

## 2026-07_state_path — 状态输出（纯运动学基线）的调优

| 文件 | 问的什么 → 结论 |
|---|---|
| `2026-07-27_convergence_and_optimization_experiment_guide.zh.md` | 状态路径怎么收敛、调什么 → 实验计划，结果在 07-28 的总结 |
| `2026-07-27_kinematic_weight_epoch_ablation.zh.md` | 运动学一致性权重与轮数 → `kinematic_consistency_loss_weight = 3.0` 进了默认值 |
| `2026-07-27_small_sample_overfit_diagnostic.zh.md` | 小样本能否过拟合（管线是否通） → 通 |
| `2026-07-27_training_test_evaluation.zh.md` | 训练 / 测试评估的做法 → 记录（文中的 evaluation 版本号是当时的） |
| `2026-07-28_optimization_experiment_summary.zh.md` | 状态路径调优的汇总 |
| `coordinate_frame_ablation_results.md` | 2026-07-25 的坐标系消融 → 选 `enu`（D1 的前身） |
| `pooled_training_cross_validation_plan.md` | 五机场合并训练与交叉验证 → 已实现（`cross-validate`） |
| `normalized_time_and_control_output.zh.md` | 归一化时间与控制输出的设计 → 机制仍在，文中文件路径是包按平面分组（2026-09-10）以前的 |

## 2026-08_control_path — 控制输出（有界控制量 + 动力学积分）的建立

| 文件 | 问的什么 → 结论 |
|---|---|
| `2026-07-28_control_prediction_experiment_guide.zh.md` | 控制输出的第一份实验指南 → 被后面的文档取代 |
| `2026-07-30_control_mixture_pause_and_resume_plan.zh.md` | 多专家控制输出 → 代码 2026-08-18 删除 |
| `2026-07-30_direct_duration_ablation.zh.md` | 直接预测时长 → 不采用，代码已删 |
| `2026-07-30_fixed_dt_control_overfit.zh.md` | 固定步长的状态损失网格 → 冻结（D17） |
| `2026-07-30_random_anchor_experiment_plan.zh.md` | 随机锚点训练 → 当时的门没过；后来 A0 / A2b 重新做过（D22、D23） |
| `2026-07-31_deployable_control_training_optimizations.zh.md` | 课程学习等训练手段 → 已删 |
| `2026-08-01_arc24_multiflight_capacity_diagnosis.zh.html`、`2026-08-01_arc_length_geometry_loss_experiments.zh.md` | 弧长几何损失 → 已退役，只剩度量那一半 |
| `2026-08-01_continuous_rollout_acceleration_experiment_guide.zh.md` | 积分提速 → 已实现 |
| `2026-08-01_terminal_state_loss_design.zh.md` | 终端状态损失 → 已退役 |
| `2026-08-02_backbone_capacity_experiment.zh.md` | 骨干容量 → 不是瓶颈 |
| `2026-08-02_control_model_experiment_group_summary.zh.md` | 8 月初控制模型的一组实验汇总 → 其"最佳配方"已被 `simple-v3` 取代 |
| `2026-08-02_dual_clock_terminal_ablation.zh.md` | 双时钟终端监督 → 已删 |
| `2026-08-02_nondimensional_transport_experiment.zh.md` | 无量纲输运后端 → `scaled-transport-chart-velocity` 保留（P8） |
| `2026-08-02_progressive_n_experiment.zh.md` | 渐进增加分段数 → 门没过 |
| `2026-08-02_reference_velocity_consistency_ablation.zh.md` | 参考速度的来源 → 保留 `track-fit`（注意：它带约 7.5 s 未来信息，仓库 `docs/open-items.md`） |
| `2026-08-02_single_qfu_approach_clustering_experiment.zh.md` | 单跑道方向的进近聚类 → 不采用 |
| `2026-08-15_comprehensive_metrics_review.zh.md` | 指标体系 → 状态损失三项、端点权重等进了代码；README 仍引它作数学定义 |
| `2026-08-16_control_simple_v1_development.zh.md` | `simple-v1` 配方 → 被 `simple-v3` 取代 |
| `2026-08-18_control_dynamics_lag_model.zh.md` | 一阶滞后动力学 → D4 |
| `2026-08-24_ksjc_result_labels_explained.md`（及同名 `.html`，html 没有同步后来的横幅） | KSJC 结果标签逐项解释 → 命名语法仍在 `run_naming.py`，目录是当时的 |
| `control_parameter_prediction.zh.md` | 控制路径 2026-08-16 的全貌（调用图、模块表）→ 被取代的记录 |
| `current_architecture.zh.html` | 2026-07-31 的架构快照 → 被 `../reference/prediction_paths.md`、`layout.md` 取代 |
| `RESUME_ksjc_v5_ladder.md` | KSJC 模仿剂量梯子的恢复说明 → 没有恢复，被 w64 复现取代（D3） |

## 2026-09_frames — 坐标系与跑道

| 文件 | 问的什么 → 结论 |
|---|---|
| `2026-09-03_airport_frame_ablation_results.md` | 机场坐标系 → 保留 `enu`（D1、C2）；计划在 `docs/` 根目录 |
| `2026-09-03_runway_frame_experiments_index.zh.md` | 9-03 四份文档的索引 |
| `2026-09-03_runway_hypothesis_expansion.md` | 跑道假设扩展 |
| `2026-09-03_state_v2_anchor_relative_results.md` | 以锚点为原点的状态输出 → 否决（D2） |

## 2026-09_constraints — 最后进近的程序约束

| 文件 | 问的什么 → 结论 |
|---|---|
| `2026-09-04_constraint_methods_survey.zh.md` | 约束方法调研与 P0–P3 顺序 → 已执行 |
| `2026-09-05_control_constraint_design.zh.md` | 控制路径的约束 → barrier 采用（D6）；名义律一半已归档 |
| `2026-09-05_control_penalty_results.zh.md` | 惩罚项 → 不采用（D5） |
| `2026-09-05_final_constraint_results.zh.md` | 走廊作有界输出 → `corridor-bounded`（D2） |
| `2026-09-08_hard_constraints_survey_and_integration_plan.md` | 硬约束接进训练 → H0–H6 没做；§3 的文献调研仍可参考 |

## 2026-09_latent_anytime — 潜在意图、任意时刻预测

| 文件 | 问的什么 → 结论 |
|---|---|
| `2026-09-07_control_training_review.zh.md` | 控制训练审查 → 已被取代（它依赖的引导律已归档） |
| `2026-09-07_l0_control_basis_results.zh.md` | 控制基函数 → `outputs/control/basis_fit.py` |
| `2026-09-07_l1_lowdim_results.zh.md` | 低维控制头 → 32 段无代价（D16） |
| `2026-09-08_straight_in_residual_readout.zh.md` | 直线进近残差在哪 → 沿航迹方向，是时间误差（T22） |
| `2026-09-08_wind_residual_readout.zh.md` | 风解释多少残差 → 不加风 |

设计文档（`2026-09-07_anytime_…`、`2026-09-07_latent_intent_design.zh.md`）暂在 `docs/` 根目录。

## 2026-09_package_reviews — 包的审查

| 文件 | 问的什么 → 结论 |
|---|---|
| `2026-09-07_package_audit_plan.zh.md` | T0–T6 审查计划 → T0–T3、T6 做完，T4 并入 9-09 审查，T5（`docs/*.py`）2026-09-26 做完 |
| `2026-09-09_package_review_bugs_and_architecture.md` | 包审查 → 按平面分组；C-4、C-7、C-9 仍开着（仓库 `docs/open-items.md`） |

## 2026-09_runway_intent — 落地跑道意图

| 文件 | 问的什么 → 结论 |
|---|---|
| `2026-09-13_runway_intent_plan.zh.md` | R0–R3.3 跑完（W1）；R2 / R3 的 runner 已归档到 `archive/plan_head_2026_09/runway_intent/`，R0 / R1 / R1.1 与 `data/runway_context.py` 仍在（先验用它） |

## 2026-09_two_tier_v3 — 两层模型 v3（被 2026-09-23 的指令词设计取代）

| 文件 | 问的什么 → 结论 |
|---|---|
| `2026-09-18_two_tier_plan_v3.zh.md` | v3 总计划 → 被取代；§10 的审计可引用 |
| `2026-09-18_two_tier_plan_v3_A.zh.md` | 阶段 A：无词执行器的 (L, Δ) 网格 → 做完；代码 `manoeuvre/` 仍在（R8） |
| `2026-09-18_two_tier_v3_results.zh.md` | 阶段 A 与编码版阶段 B 的读数 |
| `2026-09-18_two_tier_v3_stage_a_notes.md` | 阶段 A 的交接笔记 |

## reports_2026-09-08 — 2026-09-03 至 09-08 的阶段报告

五份报告汇总的是上面各线的日期文档（坐标系、约束、任意时刻），`2026-09-08_programme_results_and_plan.md` 是最后一版
（它自己说取代了 `…_programme_stage_report.md`）；其 §5 的"下一步"被计划头设计取代。
