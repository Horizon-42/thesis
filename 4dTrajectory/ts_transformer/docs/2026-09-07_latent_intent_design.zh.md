# 隐意图 + 操作参数解码：control 预测的重做（dev 文档，2026-09-07）

> **路径更新（2026-09-07，包审计 T2）：** 本文 L0 行引用的 `control/oracle/basis.py` 现在是 `control/basis_fit.py`；`control/oracle/` 的其余部分已归档到 `archive/oracle_teacher_2026_08/`。


**取代** `2026-09-07_scene_join_anchor_design.zh.discard.md` 作为 control 路径的当前设计。旧文档的
Phase 0 / P0 / P1.a–d 的**测量与产物全部保留并被本文引用**；被废除的是它的方案骨架（把汇入
决策做成 K 个显式锚、由查询解码器输出、几何闭式画路径再由跟踪器飞）。废除理由与新方案见 §一、§二。

分支 `dev-leg-ctrl`。允许大改：本文列出的废除清单是**批准过的**，不需要保留兼容。

---

## 〇、状态表（压缩 context 后从这里继续）

**当前状态（2026-09-07）**：L0、L1 完成；L2（隐意图）、L3（CTA）代码完成并经 review，在 `dev-l2`；L4 前置测量
门不过（场景编码器不建）。**L1 的答案：N=32 免费（1322 vs 1333），轨迹误差损失单独不够（dense 2515）——
L2 的 base = native32 + 教师**。包审计 T0 完成待 review。下一步 = 合入 `dev-leg-ctrl` → L2 campaign 入队
（`l2_latent_arms.json`：L2_gauss / L2_mix4，预测时 `--latent-samples 6 --latent-random 6 --latent-shuffle`）。

| 阶段 | 状态 | 产物 / commit | 门 |
|---|---|---|---|
| L0 操作参数维度 oracle | **完成（2026-09-07）** — 门按字面不过（N=16 为 315–330 m），走"否则"分支：**N\* = 32**（uniform 203 / free 191 m）；N=64 为 91 / 81 m。结果 `2026-09-07_l0_control_basis_results.zh.md` | `control/oracle/basis.py` + `run_ts_control_basis_oracle.py` + 22 项测试；产物 `l0_control_basis_20260907/` | 存在 N\* ≤ 16 使雷达引导 ADE(N\*) ≤ 200 m |
| L1 低维控制头 + 稠密监督（确定性基线） | **完成（2026-09-07）**：native32 全体 ADE 1322 vs 基线 1333（配对胜率 52.6 %，bank skill 0.726 vs 0.728）——**N=32 免费**；dense/无教师 2515/2603，否决触发，wiggle 回归——**轨迹误差损失单独不够**。结果 `2026-09-07_l1_lowdim_results.zh.md` | `l1_lowdim_20260907/`（readout、readout_bank） | 不差于 simple-v3 ✓；参数 257 → 96 ✓；bank skill ✓（native32） |
| L1.b 监督替代教师 | **预注册（2026-09-07）**：① 模仿 0（零代码，排在 L2.d 后）；② 航向率损失；③ ② + bank TV——教师只给坡度命名，L1 未测『速度项单独够不够』；模仿目标与 rollout 不一致也是 z 无利可图的候选原因 | `l1b_supervision_arms.json`、`control_heading_rate_loss_weight`、`control_bank_tv_loss_weight` | 见 §六 L1.b |
| L2 CVAE 骨架（隐意图 z） | **`L2_gauss`（β=1）与 `L2b_beta0p1`（β=0.1）都坍缩，KL 轨迹相同且低于 free-bits 地板——β 不是约束，阶梯停在第一阶；β=0.1 的 top-1 与 native32 打平（雷达引导 FDE −235 m）。死路是后验初始化 → L2.d 热启动后验（`latent_posterior_init_std=0.1`）预注册，campaign `l2_warm_posterior_20260907`（见 §六 L2 末）** | `control/latent.py`、`config` 四字段、`models`/`batch_contract`/`train`/`run_naming`/`forecast`/`export`/`__main__` 接缝、`run_ts_latent_readout.py`、`tests/test_latent_control.py`（21 项，含整链） | 不坍缩 ∧ minADE_K < top-1 ∧ z-oracle 臂 ≤ 1235 m |
| L3 CTA 条件化（交付形态） | **代码完成（2026-09-07，`dev-l2`）**：`cta_conditioning ∈ off \| given`，给定 CTA 直接**成为**时长（不回归），`predict --cta-offset-s` 反事实；7 项测试 | `config`/`control/heads`（CTA token + `final_time` 规则）/`dataset`/`forecast`/`export`/`run_naming`/`__main__`、`tests/test_cta_conditioning.py` | 给真值 CTA 时时长误差 = 0（恒等，按构造）∧ 反事实 CTA 轨迹仍可飞 |
| L4 场景条件（先验吃邻机） | **前置测量完成，门不过（2026-09-07）**：场景实体特征对 d_join / 剩余时长**零增量**（R² 0.37 vs Phase 0 粗上下文 0.38；34.7 vs 35.1 s）；可观测的前机 ETA 与其真实落地时刻相关仅 0.11。场景编码器**不建**（数据平面 review 未发现泄漏或帧/基准错误；HIGH/MEDIUM 项已修，测量成立） | `intent_explainability.py`、`run_ts_scene_explainability.py`；产物 `l4_scene_explainability_20260907/` | KL(q‖p) 下降 ∧ 雷达引导 top-1 改善 |
| L5 先验三臂 / 合并机场 / 多机 | **L5.a 拟合教师已预注册（2026-09-07，用户决定：400 步 + batch 1024 + 网络初值；契约见 §六 L5.a），待 T2 合入后实现**；其余未开始 | `run_ts_control_basis_oracle.py --checkpoint`、`control_imitation_target`、`l5_fitted_teacher_arms.json` | 见 §七 |

**误差预算（KRDU val，雷达引导 497 架，未跟踪，`closure_p1c_20260905`）——本文所有目标都相对它**：

```
C_pred 2197 ──962──> C_truth_intent 1235 ──777──> C_oracle 458 ──458──> 0
              本机看不见的意图        真值(d_join,T)之后        家族+标签+重建
              （L4 的空间）           仍未定的意图（L2 的空间）   的地板
```

对照基线：simple-v3 control 全体 1333 / 雷达引导 2858 / 直线进近 469；closure C_pred 996 / 2197 / 310。

**已知需要修正的旧数字**：P1.d 的"跟踪付出 +9 m"含 8 架跟踪器跳段的航班（review 结论，代码未改），
剔除后 **+10.5 m**。跟踪器在新方案里被废除，不修，但引用时用 +10.5。

---

## 一、为什么重做

旧方案（closure 家族 + 跟踪器 + K 锚查询）在两个交付目标上都不达最优：

1. **搜索空间不是"缩小"，是"换了个笼子"。** closure 把 257 个自由数压到 14，但代价是一个只能表达
   **一次转弯**的 Dubins 家族：3.2 % 的标签不可规范化、盘旋/转场没有表示、家族外航班**掉回 257 维的
   control 头**。以"缩小搜索空间"为目标的设计，其兜底分支是最大的那个搜索空间。
2. **"按构造动力学一致"这句话被挪到了模型外面。** 画出的参考只有 22 % 整条可飞；92 % 是一个手调增益、
   不进 checkpoint、不进 run name、不在训练回路里的跟踪器补回来的（且其最近节点搜索有 BLOCKER 级缺陷）。
3. **意图被放到了输出位置。** K 个 d_join 锚是"规定的意图"，需要人先定义意图是什么、再从它**求出**
   飞行参数。这一步只有三条路：闭式（只在家族内有解）、反馈律（手调增益）、优化（不可微、秒级）。
4. **与论文定位不符。** 交付物是给调度程序做参考，**调度意图应当是隐空间**，模型的输出必须是航迹或
   飞机操作参数，而不是一份意图说明。

## 二、新方案

**学的是"意图 → 操作参数"的条件分布，意图是被发现的隐变量；输出是操作参数，航迹由已有的可微物理
积分得到。"给定意图求飞行参数"这一步不存在——它是学出来的，不是解出来的。**

```
训练时（两条路都在）：
  完整观测航迹（锚点之后） ──→ 推断网络 q(z | 未来)   ──┐   z：8 维（默认），永不是输出
  锚点历史 + 交通场景      ──→ 先验网络 p(z | 上下文) ──┘   ← 962 m 住在这里（L4）
                                                        │
                        ┌───────────────────────────────┴──────────────┐
                        │ 解码器 (z, 锚点状态, 动力学条件, [CTA])        │
                        │      → 操作参数：N* 段控制 + 段时长            │ ← 输出之一
                        └───────────────────┬──────────────────────────┘
                                            ↓
                          现有可微 RK4 rollout（物理，零学习参数）
                                            ↓
                                        4D 航迹                          ← 输出之二

目标： E_q[ 轨迹重建（现有 control 目标，稠密监督）] + β · KL( q(z|未来) ‖ p(z|上下文) )
推理： 只用先验，采样 K 个 z → K 组操作参数 → K 条航迹 + 概率
```

三条约束同时满足：z 是隐的、输出是操作参数与航迹、没有任何一步在求解意图到参数的映射。

### 2.1 为什么这在两个目标上占优

| | 自由数 | 结构 | 家族外航班 | 曲线来源 |
|---|---:|---|---|---|
| control 头 simple-v3 | 257 | 无 | — | 257 个数凑出来 |
| 优化器（无约束，IPOPT） | 25 | 单相 | — | 配点 + 硬约束 |
| closure | 14 | 一次转弯的 Dubins 家族 | **掉回 257** | 家族拟合 + 跟踪器 |
| **本方案** | **≈4N\* + 8（z）** | 低维控制基 + 隐意图 | **不存在** | **可微物理积分** |

- **搜索空间**：由 L0 测量决定 N\*，预期 8–16 → 操作参数 32–64 个数，且**没有兜底分支**（盘旋、S 弯、
  复飞都是同一组控制参数的不同取值）。
- **曲线像不像真的**：曲线是物理积分出来的，不是家族画出来的；可飞性按构造成立（不是 92 %），
  且这次"可飞"确实等价于"形状对"——不会掉进"更平淡的预测器可飞率更高"的陷阱，因为形状来自机制。

### 2.2 为什么这正好是"给调度程序做参考"

调度程序（AMAN 一类）需要的三样东西，本方案直接给出：

1. **到达时刻的分布**，不是一个点：K 个 z 采样 → K 个到达时刻 + 概率。
2. **反事实**："要他晚 40 s 到会飞成什么样" —— CTA 作为解码器的条件输入（L3）。
3. **可交给优化器的参数**：输出本来就是操作参数，可直接做 IPOPT 的 warm start / 可行初值。

论文定位由此确定：**IPOPT 在 4.3–56 s 给出「最优」的那条；本模型在毫秒级给出「现实中大概率发生」
的那条，并给出分布与反事实。** 调度程序需要的是后者。

### 2.3 AR 与扩散在本方案里的位置（都不进 v1）

- **v1 两个都不用**：一次性解码器 + 混合先验。理由：不确定性已被压到 8 维，K 分量混合先验足以表示；
  AR/扩散是为"高维、有序列结构的不确定性"准备的机器。
- **扩散唯一正确的落点 = z 的先验**（8 维隐扩散，两阶段：先用轨迹误差训好 VAE，冻结后在聚合后验上
  拟合扩散先验）。这样去噪目标虽在 z 空间，但 z 的语义已由轨迹空间的重建损失定死——不构成
  "损失打在参数上"的问题。列为 **L5 的第三臂**，与单高斯 / K 混合同臂对照。
- **AR 的落点 = 段级（leg 级）解码**，判据是训练后**逐段误差是否超线性增长**；不超线性就不上。

## 三、契约与不变量（违反会静默算错东西）

1. **z 永远不进记录契约。** 导出的 `source` 可以带 `modeIndex` / `modeProbability`，不带 z。
2. **推断网络 q 只在训练期被调用。** 它随 checkpoint 保存（为了续训），但 `forecast` 永不触碰；
   唯一的例外是显式的 `predict --z-from-posterior`（= z-oracle 上界臂，run name 必须带
   `z=posterior`，与 `intent=truth-…` 同一条纪律）。
3. **CTA 条件化的臂读的是未来。** `cta_conditioning=given` 的 `final_time_error_s` 是恒等检查，
   不是时长结果；这类臂是**交付形态演示，永远不能作为预测精度结果引用**。run name 必须带 `cta=given`。
4. **重建损失在轨迹空间，稠密监督。** 打分点是真值采样时刻（`rollout_control_dense` 的 query），
   不是 N\* 个段端点——控制网格（粗）与打分网格（稠密）分离，这是优化器能用 3 段/leg 的原因。
5. **新损失分量必须进 `loss_component_names`**（否则第一批 `KeyError`，在慢的数据集构建之后）。
6. **锚点状态是 `batch_contract.anchor_state(x, C)`，永远不是 `x[:, -1]`。**
7. **控制量在包内无量纲**（`control/envelope.py` 是唯一来源）；牛顿只出现在 `physical_controls()`
   与 `forecast.py`。
8. **隐变量的容量是 config 字段并进 checkpoint 与 run name**（`latent_dim`、`latent_prior`、
   `latent_beta`）。跟踪器"增益是模块常数、不进 checkpoint"的错误不能重犯。

## 四、废除清单（本分支上直接删/降级，不保留兼容）

| 废除 | 变成什么 |
|---|---|
| `prediction_output="closure"` 的**主线地位** | 保留代码，降级为对照臂（"规定的意图" vs "发现的意图"） |
| `control/constraints/closure_tracking.py`（跟踪器） | 废除（操作参数直接进 rollout，不需要跟踪）；review 的 BLOCKER 不修 |
| K 个 d_join 锚 + 查询解码器（旧 P3） | 废除，被隐变量 + 混合先验严格泛化 |
| `2026-09-07_control_training_review.zh.md` 的 P0/P1（垂直定价、事件锚定、闭环教师） | 废除。它们修的是 64 段 control 头，而该头降级为基线 |
| 64 段固定控制网格 | 由 L0 测出的 N\* 取代 |

**保留不动**：`state` 路径（科学对照，纯运动学基线）、可微 rollout / 包线 / 一阶滞后、
`geometric_metrics`、记录契约、evaluation、campaign 工具、`approach_difficulty`、
P1 标签（`closure_labels.json` 降级为**隐空间探针**，不再是回归目标）。

## 五、复用清单（不重写）

| 已有 | 在新方案里的角色 | 路径 |
|---|---|---|
| 可微 RK4 rollout + 包线 + 一阶滞后 | 解码器后半段，一字不改 | `control/dynamics/` |
| `ControlOutputHead` / `ControlPrediction` | 解码器的输出头（`n_segments=N*`） | `prediction_outputs.py` |
| `rollout_control_dense` | 稠密监督的打分器（现只在预测期用） | `control/dynamics/rollout.py` |
| 逆动力学 `segment_controls` | L0 拟合的初值；模仿项的教师 | `control/dynamics/inverse.py` |
| iTransformer 主干 `encode_features` | 本机历史编码器 | `models.py` |
| 场景数据平面（WIP，未 review） | L4 的输入 | `trajectory_data_process/scene_index.py`、`flight_scenarios/scene_context.py`、`scene/features.py`（`045c233`） |
| Phase 0 的 `intent_conditioning` | z-oracle 与真值意图上界臂 | `intent_conditioning.py` |
| `closure_labels.json` / `profile_labels.json` | **隐空间探针**（z 能否线性解码出 d_join / 时长） | `outputs/KRDU/closure_labels/` |
| IPOPT 优化器 | 同 CTA 下的最优基准 + warm-start 收益 | `4dTrajectory/optimization/` |

## 六、实施步骤（每步：写代码 → opus review → 修 → 实验 → 记录 → 提交）

### L0 — 操作参数维度的 oracle（无训练，纯 numpy；≈1 天）

**问题**：解码器该输出多少个数？64 段 PWC 是从没被测过的默认；优化器用 8 段（无约束）/ 3 段每 leg
（约束）就能把整条飞到终端硬钉；closure 用 14 个数。三者夹出的答案在 8–16，但那是"找一条轨迹"的
证据，不是"拟合观测轨迹"的证据——必须量。

**做法**：库 `control/oracle/basis.py`（`BasisSchedule` / `fit_basis_schedules` /
`inverse_dynamics_seed`，带 11 项测试）+ 运行器 `run_ts_control_basis_oracle.py`。对 KRDU 验证集（Phase 0 的同一批 1404 架，同 config /
同锚点，`_series_for`）：

1. 初值 = `control/dynamics/inverse.py::segment_controls(..., n_segments=N)`（逆动力学采样 + 裁剪）；
2. 以**稠密 rollout 在真值采样时刻的水平/垂直误差**为目标，拟合 N 段控制（有界最小二乘 / L-BFGS-B，
   包线为盒约束）；
3. 两条轴：N ∈ {4, 8, 16, 32, 64} × 段时长 ∈ {均匀, 自由（L 个自由时长，softmax 参数化）}；
4. 报 ADE / FDE / chamfer / Fréchet / 时长误差，按 `approach_difficulty` 分层，并报每航班的
   拟合失败率与包线饱和率。

**门**：存在 **N\* ≤ 16** 使雷达引导 **ADE(N\*) ≤ 200 m**（表示误差要远小于 962 m 的意图不确定性，
否则控制基本身就是瓶颈）。
**否则**：把 N\* 定在满足 200 m 的最小值上（即使 > 16），并在文档里记下搜索空间没有缩小到预期；
若 N=64 仍 > 200 m，说明 PWC 控制基不足以表示真实航迹 → 回到 closure 作解码器，本方案的 L1 改为
"closure 决策向量作为解码器输出"，L2 之后不变。


> **拟合的标定（2026-09-07，256 架、uniform，不是结果）**：这个测量唯一能给出的错误答案是
> "N 不够"，而它来自欠优化，所以先标定优化器再读 ADE(N)。三档固定学习率（400 步，雷达引导 ADE）：
> lr 0.05 → N=8 1236 / N=64 1167；0.01 → 642 / 411；0.002 → 724（71 % 未收敛）/ 141。
> 三点结论都进了代码：(a) **固定学习率不行**——快的地板差、好的没跑完，改成余弦退火到起点的 5 %；
> (b) **最优学习率随宽度移动**（N=8 ≈0.01、N=64 ≈0.002，比值 ≈ 1/N），改成
> `width_scaled_learning_rate(base, N) = base/N`，`--control-learning-rate` 的语义变为"N=1 时的速率"，
> 默认 0.08；手调每个宽度会把宽度与调参混淆，这是宽度研究唯一不能做的事；
> (c) **退火下"最佳步在预算末尾"恒为真**，收敛判据换成 `tail_gain` = 最后 10 % 预算买到的相对改善。
> 标定后（1200 步）：N=8 雷达引导 495 m（tail p50 0.8 %）、N=64 **81 m**（0.9 %）——两点都收敛，
> 对数斜率 ≈ 0.87，外推 N=16 ≈ 270、N=32 ≈ 150。**正式跑的设置由此定为
> `--steps 1200 --control-learning-rate 0.08 --learning-rate-floor 0.05 --batch-size 256`。**

**产物**：`4dTrajectory/outputs/KRDU/experiments/l0_control_basis_<date>/oracle_basis.{txt,json}`，
逐航班拟合参数存 `basis_fit.json`（L2 的隐空间探针之一）。

### L1 — 低维控制头 + 稠密监督（确定性基线；≈2 天）

**勘察结论（2026-09-07）：不需要新代码，L1 是一个 campaign 不是一次实现。** 稠密监督已经存在——
`control_state_loss_grid ∈ native-segment-endpoints | fixed-dt`，`fixed-dt` 就是"在 2 s 真值网格上
打分"，`train._CONTROL_STATE_LOSS_HANDLERS` 已经派发它，L0 的拟合用的就是这条路径。所以两个改动
都是 config：

1. `n_segments = N*`；
2. `control_state_loss_grid = fixed-dt`（其约束：`prediction_output=control`、
   `control_state_supervision_clock=observed`，且 `control_state_objective` 不能是
   `true-time-position`——它要求 native 网格；用 `normalized-mse`，`arc-length-geometry` 另作一臂）。
   simple-v3 冻结了这几个字段，所以 L1 的臂是 `custom` 且逐字段写明。

**臂**（KRDU）：`L1_lowdim`（N\*，dense）、`L1_lowdim_endpoints`（N\*，端点，隔离两个改动）、
对照 `control_procedure_20260905/A_control_v3`（64，端点）。
**门**：`L1_lowdim` 的 pooled 与雷达引导 ADE 不差于 simple-v3（1333 / 2858），直线进近不退；
参数量 257 → ≈4N\*。
**注意**：模仿项权重 64.0 是按 64 段标定的，**换段数必须重标剂量**（1/4/16/64 阶梯，读幅度不读 p）。

> **模仿项的教师（用户 2026-09-07 决定：先跑 L1 主线，看轨迹误差损失够不够）**。L0 顺带量到
> 现在这个教师有多差：它就是 L0 的 `seed`（真值航迹的逆动力学），**照单飞出来离真值 2.5–7.8 km**
> （N=4 7850、N=8 6381、N=16 4095、N=32 2537 m），而同宽度拟合后的控制表飞出来只差 88–433 m。
> 也就是说"完美模仿现在的教师"并不等价于"飞出真值航迹"——这就是复审文档 §三 缺陷 B 的数字形式。
> `basis_fit.json` 是一个严格更好的教师（它按构造复现真值航迹）。**但不进 L1 主线**：主线只用轨迹
> 误差损失，先证明它够不够；若不够，再补跑训练集的拟合（KRDU 训练集约 1 万架，N=32 单臂 ≈15 h，
> 一次性可复用）并作为 `imitation-target=fitted` 对照臂。教师是**宽度专属**的，N\* 一变即作废，
> 所以那份文件必须带 schema 与拟合配置戳（陈旧标签静默训错东西，仓库已踩过一次）。
> 三臂对照届时是：模仿项关掉 / 逆动力学教师（现状） / 拟合教师。

#### L1.b — 监督替代教师（2026-09-07 预注册；用户提议，评估后采纳）

**教师在这里只做一件事：给坡度命名。** 位置是 0 阶量，坡度是 2 阶量，L1 的无教师臂用 6 倍于观测的坡度
能量凑位置（wiggle）。它不解决 962 m 的意图缺口。但 L1 的结论有一处**没有测量**：两个 dense 臂在
fixed-dt 网格下速度项和模仿项都没注册（只在 true-time-position 下注册），所以"速度项单独压不住
wiggle"是未测的。还有一层：模仿项占损失 0.556，而逆动力学目标与 rollout **不一致**（开环飞出去偏
2.5–7.8 km）——解码器被钉在飞不回真值的控制上，位置项与模仿项在打架；这也是 z 无利可图的候选原因，
所以 L2.e 等这里的胜者定下来再做。

**三臂，同 native32 底座（custom、N=32、native 端点、true-time-position、速度项 0.003 照旧）：**
1. `L1b_noteacher`：模仿 0——零代码，L2.d 一结束就排；回答"1 阶速度监督单独够不够"。
2. `L1b_heading_rate`：模仿 0 + **航向率损失** `control_heading_rate_loss_weight`：rollout 在真值时钟段端点
   处的航向率（协调转弯下即 g·tan φ / V，取实际 bank，不是指令）对观测航迹平滑后的 ψ̇；同一个观测转弯
   率，但穿过 rollout 施加、不经逆动力学与滞后反解——**模型一致**，零件更少。预期"等价且更干净"，
   不是新信息；只给 bank 命名，推力/载荷仍靠速度项。
3. `L1b_heading_rate_tv`：② + bank 相邻段总变差惩罚 `control_bank_tv_loss_weight`（结构约束代替损失
   约束的最便宜形式；T1-12 删掉的 effort/smoothness 是"无人使用"，不是"证明无效"——诚实记下）。
**门**：bank skill ≥ 0.70（native32 0.726）、ADE / FDE p50 逐分层不劣于 native32；否决：直线进近 FDE 超
种子噪声。读数同 L1。**决定规则**：若 ① 或 ② 过门，教师整条线降级为对照臂（L5.a 拟合教师保留为上界
臂，回答"教师质量还能买多少"）；若都不过，L5.a 拟合教师成为主线教师。不做的：闭环（贪心）教师——
拟合已降到 2–3 h，贪心是它的近视近似且把纠错瞬变写进目标；双层优化与分布匹配（GAN/MMD）——超出
路线与规模。"少数几次有滚转率限制的机动"作为另一种基，先用 L0 的宽度 oracle 量表示误差再决定。
臂文件 `docs/experiments/l1b_supervision_arms.json`（①）；②③ 实现 + review 后补进同一文件。

### L2 — CVAE 骨架（隐意图；≈1 周）

**实现决定（2026-09-07，与 §九 的命名草案不同）**：隐变量是 **control 输出上的一根轴**（`latent_dim > 0`），
不是新的 `prediction_output`。理由：它就是 control 路径（出有界控制量、经 rollout 积分）加一个 z，
和 `control_dynamics_model` 一样是 control 的轴；这样 config / dataset / forecast 里所有
`== PREDICTION_CONTROL` 的判断按原样成立，零处需要改。run name 把它当作**不同的模型**报出：
`control+z8`（K=4 混合先验时 `control+z8k4`），而不是一个损失编辑。模块在 `control/latent.py`
（每个消费者都是 control 专用，符合 `control/` 的归属规则）。

原草案（保留作对照）：新增 `prediction_output="latent-control"` 与顶层模块 `latent_intent.py`：

- `PosteriorEncoder`：锚点之后的真值轨迹重采样到固定 32 点 `[32, 7]`（图坐标 e/n/u + 速度 3 通道 +
  归一化时间）→ 小 GRU/transformer → `(μ_q, logσ_q)`。**训练期唯一入口**。
- `PriorNetwork`：`(fused_features)` → 混合先验 `(π, μ_k, logσ_k)`，K 由 `latent_prior_components`
  给出（v1: 单高斯 K=1 与混合 K=4/6 两臂）。L4 之后 fused_features 里含场景 token。
- `LatentControlModel`：`z` 拼进融合特征 → 现有 `ControlOutputHead(n_segments=N*)` + `FinalTimeHead`。
- 损失：现有 control 目标（`true-time-position` + 速度 + 模仿 + 时长）作为重建项，加
  `latent_kl` 分量 = 单样本 MC 估计 `log q(z|x) − log p(z|c)`（混合先验无闭式 KL）。
  β 退火 + free bits（`latent_free_bits`）。**`latent_kl` 必须进 `loss_component_names`。**
- 推理：`forecast` 从先验采样 `latent_samples` 个 z（默认 K=6），top-1 = 概率最高的模态走现有记录
  契约，其余写 `modes/` 子目录，`source` 带 `modeIndex` / `modeProbability`。
- `predict --z-from-posterior`：z-oracle 上界臂。

> **建成的形态（2026-09-07）**：
> - `control/latent.py`：`PosteriorEncoder`（输入 = 归一化目标行 `y[B, pred_len, C]` + 真值时长，
>   数据集本来就产出，**零数据管道改动**）、`PriorNetwork`（K 分量对角高斯混合，初始化为 N(0, I)）、
>   `LatentControlModel`（z 拼进融合特征 → 现有控制头；**z 也进时长头**：`softplus(raw + W·z)·scale`，
>   Phase 0 量到雷达引导的散布主要是时序，z 若到不了时长就把最大的变异留给了点估计）、`latent_kl`
>   （K=1 逐维解析 + 逐维 free bits；K>1 单样本 MC + 总量 free bits，诊断用最负责分量的解析 KL）、
>   `with_latent_kl`（`latent_kl` 分量 + 坍缩诊断）。
> - 接缝：`batch_contract.model_forward(..., future=None)` 只把 `(y, final_time_s)` 交给声明
>   `consumes_future` 的模型，且**只有训练步与验证目标步传它**——固定锚点回放（选 checkpoint 的
>   top-1 ADE）和 `forecast` 从不传，后验按构造到不了预测。`train.py` 按 `LatentControlPrediction`
>   类型派发损失；history 每轮多一条 `latent = {kl_nats_per_flight, active_units}`。
> - 推理：`predict` 默认走先验 top-1（记录契约不变）；`--latent-samples K` 把 K 个先验采样各写成
>   `modes/modeNN/` 完整预测目录（`source.modeIndex/modeProbability`）；`--latent-shuffle` 把"每架
>   用别架的 top-1 z 解码"写成 `shuffled/`（`source.latentShuffled`）——坍缩判据。z 永不进记录。
> - 读数 `run_ts_latent_readout.py`：top-1 / minADE_K / minFDE_K / miss rate / 模态 FDE 散布 /
>   shuffled ΔADE，按分层；`--control` 接同 K 的随机隐变量对照臂。
> - **未做**：β 退火（先只用 free bits，坍缩再加）、`--z-from-posterior`（z-oracle 臂）、z 探针
>   R²（需要 `--latent-dump` 诊断文件，不进记录）。验证目标步是随机的（后验采样），所以 checkpoint
>   选择用 `fixed-anchor-common-grid-ade`（先验 top-1 回放，确定性），不要用 `fixed-anchor-objective`。

**臂**：`L2_gauss`（K=1）、`L2_mix4`、`L2_mix6`、`L2_zoracle`（predict-only）。
**门（全部要过）**：
1. **不坍缩**：活跃维数（per-dim KL > 0.05 nat）≥ 3；打乱 z 后 pooled ADE 劣化 > 200 m；
2. **多模态有用**：minADE_6 显著低于 top-1，且优于"K 个随机 z + 同一网络"的对照；
3. **上界成立**：`L2_zoracle` 雷达引导 ADE ≤ 1235 m（不差于真值 (d_join, T) 条件化）。
**诊断（必须进读数脚本，否则坍缩会被读成收敛）**：per-dim KL、shuffle-z ΔADE、K 条散布 vs 真值散布、
**z 对真值 d_join / 时长 / via 位姿的线性可解码性 R²**（用 `closure_labels.json` 当探针，不当目标）。
若 z 解码不出 d_join 但重建很好 → z 吸收的是风与执行噪声，是坏消息，先压 `latent_dim` 再查。

> **L2.c β 标定（2026-09-07 预注册，campaign `l2_latent_20260907` 第一臂出结果后）。** `L2_gauss`（β=1.0）
> **坍缩**：best epoch 136 的 `active_units` 0.0（第 8 轮起恒 0，峰值 0.30 在第 1 轮），KL 0.0016 nat/航班，
> 选择指标 1398 m（native32 1322）。这不是病态而是最优解，**量纲**决定的：重建项是缩放 MSE——位置按
> `(m / position_loss_scale_m = 10 km)²`——962 m 的全部隐藏意图只值 (0.096)² ≈ 0.009 个损失单位，而一个
> 4 路决策要 log 4 = 1.4 nat；β=1 下 KL 是 z 能买到的全部收益的约 150 倍。β 必须是"每 nat 的损失单位"量级。
> 预注册：**十进制阶梯 β ∈ {0.1, 0.01, 0.001}**，同 base（K=1 解析 KL，free bits 0.05/dim 不动，以保持可比；
> 不加退火——终值 β 过高时退火只推迟坍缩），臂文件 `docs/experiments/l2_beta_ladder_arms.json`，
> campaign `l2_beta_ladder_20260907`。读法：每臂 best epoch 的 active_units / KL、shuffled ΔADE、
> minADE_6 vs top-1 vs 同 K 随机对照、top-1 vs native32 逐分层；**选过门 (1) 且 top-1 不劣于 native32
> （种子噪声内）的最大 β**，然后把它带到 K=4 与 z-oracle 臂（`l2_zoracle_arms.json`，predict-only
> `--z-from-posterior`，从阶梯的 checkpoint 出发，门 (3) 在此测）。否决同前。**若没有任何 β 在不损失
> top-1 的前提下过门 (1)**，说明解码器不靠 z 也能解释数据（模仿教师把控制钉在逆动力学目标上），下一个
> 杠杆是后验的输入，不是 β。`L2_mix4`（β=1）按同一论证预计同样坍缩，其读数只作 K=4 的坍缩对照。

> **L2.c 结果（2026-09-07）：β 不是约束——阶梯在第一阶后停止。** `L2b_beta0p1`（β=0.1）的 KL 轨迹与
> `L2_gauss`（β=1）**逐轮相同**（峰值 0.046 vs 0.045 nat/航班，第 8 轮起 active_units 恒 0），且整段
> **低于 free-bits 地板**（8 × 0.05 = 0.4 nat）——KL 惩罚从未生效，任何 β 都分辨不出阶梯。读数：shuffled
> ΔADE +1 m（各分层 −0…+4），minADE_6 与同 K 的 N(0,I) 对照相差 < 2 m（1204.8 vs 1205.7）：门 (1)(2) 以
> 满幅失败。唯一的真发现：**β=1 在拉低点估计，0.1 把它买回来**——top-1 与 native32 六个分层全部打平
> （胜率 49.3–50.6 %，中位 Δ ±7 m；雷达引导 FDE p50 1747 vs 1982），直线进近 FDE p50 671 = 671，否决干净。
> 后两阶（0.01 / 0.001）未跑：它们只会再测一次同一个零。z-oracle campaign 因此也未跑。
>
> **死路在后验本身，不在 β。** `PosteriorEncoder` 初始化时均值 ≈ 0.2、方差 1：z 对解码器是噪声，解码器的
> z 权重在噪声梯度与 weight decay 下衰减到 0，后验随之得不到任何梯度——一个稳定的不动点，与 β 无关
> （惩罚在地板之下本来就是零）。
>
> **L2.d 预注册（2026-09-07）：热启动后验。** 新 config 轴 `latent_posterior_init_std`（默认 1.0 = 坍缩运行
> 的初始化，进 checkpoint、进 run name `q-std=…`）；臂文件 `docs/experiments/l2_warm_posterior_arms.json`
> 钉 0.1：后验 log 方差半边初始化为常数 2·ln 0.1（权重置零），均值半边保持默认——z 从第 0 步起是未来
> 的函数（初始 KL ≈ 1.8 nat/dim，高于地板，惩罚从第 0 步生效），KL 在配置的 β 下把它拉宽——这才是阶梯
> 要测的权衡。同 base（K=1、N=32、simple-v3 监督 + 教师、free bits 0.05/dim），臂只差 β：0.1 与 0.01
> （0.001 是自编码器一端，仅当 0.01 仍坍缩才有意义）。门与选法同 L2.c。**预注册的失败形态**：(a) z 成为
> 未来的自编码器——z-oracle 极好、shuffled ΔADE 巨大、但 top-1 劣于 native32（先验预测不了 z）→ β 太
> 低；(b) 热启动下仍坍缩 → 解码器不靠 z 就能解释数据，下一杠杆是 z 上的辅助意图目标（L2.e：z → 剩余
> 时长 / d_join 的训练期辅助头，标签来自 `truth_duration_s` 与 closure labels），不是再换初始化。
> campaign `l2_warm_posterior_20260907`。

### L3 — CTA 条件化（交付形态；≈2 天）

`cta_conditioning ∈ off | given`。`given` 时把目标到达时刻（训练用真值时长）作为解码器的条件标量。
**臂**：`L3_cta`（predict 时用真值 CTA）+ 反事实扫描（CTA ± 30 / 60 / 90 s）。
**门**：给真值 CTA 时时长误差中位 < 5 s；反事实 CTA 的轨迹逐样本可飞率不低于 `L2` 臂，且
到达时刻误差随 CTA 线性跟随（不是被时长头忽略）。
**契约**：见 §三.3——这类臂永远不能作为预测精度结果引用。

> **建成的形态（2026-09-07）**：`given` 下时长头被**旁路**——`final_time := dynamics["cta_s"]`（`ControlFeatureModel.final_time`
> 是三个 control 模型共用的唯一规则），网络只决定"在那个时刻到达的路径"；CTA 同时作为一个融合 token 进解码器
> （`cta_encoder`）。训练喂真值时长（`dataset.truth_duration_s`，`_dynamics_arrays` 写 `cta_s`）；预测喂真值 +
> `--cta-offset-s`（`forecast._dynamics_batch`），记录带 `source.ctaS / ctaOffsetS`，run name 带 `cta=given`
> （与 `intent=` 同一条"读了未来就必须穿在名字上"的纪律）。隐变量模型下 `latent_duration` 在 `given` 时失效
> ——CTA 就是时长，这是有意的。门里的"时长误差 < 5 s"按构造为 0，所以 L3 真正要读的是**反事实**：
> CTA ± 30/60/90 s 下逐样本可飞率相对 CTA=0 臂不退、且路径几何（chamfer）随 |偏移| 平滑变化而不是崩掉。

### L4 — 场景条件（先验吃邻机；≈2 周）

先把 WIP 数据平面（`045c233`）过 opus review 并补上 P2.d 的可解释性测量，再接：

- `scene/encoder.py`：邻机实体 token（≤ N_max=16，掩码）+ 几何折线 token + 标量 token →
  set/cross-attention → 场景向量；拼进 `fused_features` **只喂先验网络与解码器**。
  **不要把邻机做成 iTransformer 的额外变量 token**——`target_conditioning="channels"` 已经测过，
  主干基本不用协变量 token（只有拍平的时长头用了）。
- 泄漏红线：邻机特征只用 t ≤ t₀ 的样本；邻机落地时间与最终跑道只进 `future_label`，不进特征。

**门**：KL(q‖p) 相对 L2 显著下降（先验变尖 = 场景信息进来了）；雷达引导 top-1 ADE 改善。

> **前置测量结果（2026-09-07，`run_ts_scene_explainability.py`，KRDU 14,418 架，同 Phase 0 人群与 CV 协议）**：
> 先复现了 Phase 0 到小数点（锚点后汇入 6,557 架：d_join R² 本机 0.34 → 粗上下文 0.38 → 真值前机 ETA
> 0.47；剩余时长中位误差本机 35.8 s → 真值前机 26.4 s）。然后**场景数据平面的实体级特征零增量**：
> 跑道使用标量 0.37 / 34.7 s，再加 4 架邻机的静态行 0.36 / 36.1 s。门（0.55 / 28 s）**不过**。
> **诊断**（2,000 架抽样）：数据平面为 96 % 的雷达引导本机找到了"前机"，但它按当前地速估的前机 ETA
> 与前机**真实**落地时刻的相关只有 **0.11**（中位晚 275 s，p10/p90 −90 / +1067 s）——雷达引导的前机
> 自己的剩余时间和本机一样没定，它是同一个排序决定的另一个结果，不是本机能观测到的原因。真值前机 ETA
> 能解释 d_join（0.47）正因为它是**结果**。有信号的只有 `lead_gap_s`（corr −0.44），而它已被本机自身
> 状态覆盖。
> **结论**：在锚点时刻，管制排序决定还没有写进邻机的可观测状态里；一个完美的邻机编码器也拿不回 962 m
> 的大部分。**L4 场景编码器暂不建**；962 m 的交付形态是 L2 的分布（minADE_K + 校准），不是 top-1。
> 若要继续追场景信息，方向是"前机的**预测**剩余时间"（把本机模型用在邻机上，GooDFlight 的 one-then-all）
> 或更早的时间窗，且先在这个协议上量到增量再建模。待数据平面 review 确认特征无缺陷后此结论成为最终。
**预期形状**：minADE_K 基本不变而 top-1 改善——那正是"场景信息在选模态"，是本设计预期的收益形态，
要在结果文档里明说，避免被读成"多模态没用"。

### L5 — 先验三臂 / 合并机场 / 多机（之后）

#### L5.a — 拟合教师（用户决定 2026-09-07：400 步 + batch 1024 + 网络初值，排进 L5）

**动机。** L0 §三.5：现役模仿教师（真值航迹的逆动力学）开环飞出来离真值 2.5–7.8 km，而同宽度、穿过同一
可微 rollout 拟合出的控制表只差 88–433 m——一个严格更好的教师；L1 证明教师不可缺。15 h 的估计来自 L0
的设置（1200 步、batch 256、逆动力学初值，6851 架 → 27 个 batch × 1200 步 rollout 前后向）；L0 的收敛
判据（最后 10 % 预算只买 0.4–1.2 %）说明 1200 步远超所需。**预注册设置：400 步、batch 1024、初值 =
native32 checkpoint 自己的预测**（直线进近已在 445 m 内），预期一臂拟合 2–3 h；拟合覆盖 train + val 两个
划分（val 也拟合，模仿分量在验证上才有定义；test 永不拟合）。

**契约（实现前定）。**
- 拟合器 `run_ts_control_basis_oracle.py` 增 `--checkpoint`：队列 = checkpoint 的划分（`--splits train,val`），
  N / 动力学 / 锚点从 checkpoint 的 config 继承，`--init network|inverse-dynamics`（有 checkpoint 时默认
  network：checkpoint 的确定性前向——隐变量模型取先验 top-1——给出 N×3 初值；总时长仍给定 = 真值时长，
  与 L0 相同）。产物 `basis_fit.json` schema 升 v2：按 `flight_key` 键的控制表（N×3）、总时长、fitADE /
  seedADE / best_step；戳 n_segments、uniform、锚点、checkpoint sha256、config sha256、步数 / lr / floor /
  init。目录不可变，不入 git。
- config 轴 `control_imitation_target ∈ {inverse-dynamics, fitted}`（默认与三个 recipe 字面量都是
  inverse-dynamics）+ `control_fitted_teacher_path`（fitted 时必填，否则拒绝）；进 run name
  （`imit-target=fitted`）；表文件的 sha256 进 checkpoint 的数据来源块，checkpoint 因此说得出它是哪张表
  教出来的。
- 数据集：fitted 下 `reference_control_supervision` 的逆动力学被表查找取代——`reference_controls` =
  表里的控制，权重全 1（拟合覆盖整个视界）；表里没有的航班、N 不符、锚点不符、总时长与
  `truth_duration_s` 不符（> 1e-6 s）→ 数据集构建时**拒绝**并报覆盖数（a of b），不静默回退。
  损失 `control_imitation_mse` 不动。`random_train_anchor` 与 fitted 互斥（表只在固定锚点上有定义）。
- 臂 `docs/experiments/l5_fitted_teacher_arms.json`：base = native32（custom、N=32、simple-v3 监督）+
  fitted；`L5_fitted64`（剂量 64，与现役同）、`L5_fitted16`（16：新教师离真值近一个量级，剂量可能需要
  重标）。对照 L1_native32，L2.d 若过门则加其胜者。
- **门**：top-1 ADE / FDE p50 逐分层不劣于 native32，雷达引导层预期改善（教师从 2.5–7.8 km 变 88–433 m
  是这一层的事）；**bank skill ≥ 0.70**（native32 0.726，地板 0.17 / 天花板 0.70）——拟合只最小化位置，
  控制表可能带 bank wiggle，这是预注册的风险：若 bank skill 掉，修的是**拟合端**（拟合目标加平滑先验），
  不是训练端；否决：直线进近 FDE p50 退化超过种子噪声。读数同 L1（`compare_constraint_arms.py` +
  `score_control_arms.py`），拟合的 wall time 与 fitADE / seedADE 分布写进结果文档。

- 先验三臂：单高斯 / K 混合 / **隐扩散**（两阶段，见 §2.3）。预注册读数：minADE_K、miss rate、
  校准曲线、验证 NLL、**逐层看**（直线进近层三臂应基本相同，差别必须全在雷达引导层）。
  若 K 混合 ≈ 隐扩散，诚实报"在这个数据规模上扩散先验无增量"——这是有价值的负结果。
- 合并 5 机场训练（隐空间与机场无关，42,650 条 vs 单机场 1.4 万，是最大的未动杠杆）。
- 段级 AR：由"逐段误差是否超线性增长"决定。
- 多机联合：先验在到达序列上联合建模。
- IPOPT warm-start 对照：同 CTA 下最优 vs 现实，量迭代数与收敛率。

## 七、风险与预注册否决

1. **后验坍缩**——头号失败，外观与已知的"平淡陷阱"一模一样。诊断见 L2；**诊断必须先于实验落地**。
2. **z 吸收执行噪声而非意图**。信息瓶颈会优先编码方差最大的因素（意图 962 + 777 vs 执行 ≈60），
   所以把 `latent_dim` 压到 8 是对策的一部分；诊断是 z → d_join 的可解码性。
3. **模仿项剂量不随段数迁移**（`CLAUDE.md` 已警告连跨机场都不迁移）。L1 必须重做剂量曲线。
4. **稠密监督改变了目标的定义**，L1 的两臂就是为了隔离它；不隔离会把两个改动的收益混在一起。
5. **本方案不改善 top-1 ADE 的上限**——962 m 是缺失信息。**预注册**：L2/L3 的验收是分布口径
   （minADE_K、校准、可飞率、反事实），**top-1 不退即为通过**；只有 L4 承诺 top-1 改善。
6. **否决**：直线进近层 top-1 FDE 退化超过种子噪声；或 minADE_K 的改善不超过随机 z 对照；
   或 z-oracle 臂打不过 C_truth_intent（说明隐变量的容量或解码器不足，而不是信息不足）。

## 八、读数与复现约定

- 每个分层同时报**两组指标**：时间对齐 ADE/FDE/时长误差 + 时间无关几何（chamfer / Fréchet /
  弧长对齐 ADE，`geometric_metrics`，真值默认 `closed`）。只用一组下的结论不算数。
- 多模态读数新脚本 `docs/compare_latent_arms.py`：minADE_K / minFDE_K / miss rate / 校准曲线 /
  per-dim KL / shuffle-z ΔADE / z 探针 R²，并与 `compare_constraint_arms.py` 同口径打印 top-1 分层。
- 正式 campaign 前工作树必须干净；`git add` 明确路径；看进程用 PID；引用数字只引当前产物。
- 代码在跑实验前用 opus subagent review，文档不送 review。

## 九、名词对照（写代码时的命名）

| 概念 | 代码里的名字 |
|---|---|
| 隐意图 | `latent_intent.py`，config `latent_dim` / `latent_prior` / `latent_prior_components` / `latent_beta` / `latent_free_bits` / `latent_samples` |
| 预测输出类型 | `PREDICTION_LATENT_CONTROL = "latent-control"` |
| 推断网络 / 先验网络 | `PosteriorEncoder` / `PriorNetwork` |
| 稠密监督开关 | `control_supervision_grid ∈ endpoints \| dense` |
| CTA 条件化 | `cta_conditioning ∈ off \| given` |
| z-oracle 上界臂 | `predict --z-from-posterior`，run name `z=posterior` |
