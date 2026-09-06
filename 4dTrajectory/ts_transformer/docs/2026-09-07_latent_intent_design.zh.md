# 隐意图 + 操作参数解码：control 预测的重做（dev 文档，2026-09-07）

**取代** `2026-09-07_scene_join_anchor_design.zh.discard.md` 作为 control 路径的当前设计。旧文档的
Phase 0 / P0 / P1.a–d 的**测量与产物全部保留并被本文引用**；被废除的是它的方案骨架（把汇入
决策做成 K 个显式锚、由查询解码器输出、几何闭式画路径再由跟踪器飞）。废除理由与新方案见 §一、§二。

分支 `dev-leg-ctrl`。允许大改：本文列出的废除清单是**批准过的**，不需要保留兼容。

---

## 〇、状态表（压缩 context 后从这里继续）

**当前状态（2026-09-07）**：L0 完成，**N\* = 32**（96 个操作参数，对照 control 头 257 = 2.7 倍缩减，
不是设想的 10 倍）。**L1 campaign `l1_lowdim_20260907` 在跑**（三臂，约 6–8 h）。用户决定：先看轨迹误差损失
够不够，拟合教师（`basis_fit.json`）留作对照臂。下一步 = L1 读数 → L2 CVAE 骨架的代码。

| 阶段 | 状态 | 产物 / commit | 门 |
|---|---|---|---|
| L0 操作参数维度 oracle | **完成（2026-09-07）** — 门按字面不过（N=16 为 315–330 m），走"否则"分支：**N\* = 32**（uniform 203 / free 191 m）；N=64 为 91 / 81 m。结果 `2026-09-07_l0_control_basis_results.zh.md` | `control/oracle/basis.py` + `run_ts_control_basis_oracle.py` + 22 项测试；产物 `l0_control_basis_20260907/` | 存在 N\* ≤ 16 使雷达引导 ADE(N\*) ≤ 200 m |
| L1 低维控制头 + 稠密监督（确定性基线） | **campaign 在跑（2026-09-07 启动，`l1_lowdim_20260907`）**；无新代码 | 臂 `docs/experiments/l1_lowdim_arms.json`（L1_dense32 / L1_dense64 / L1_native32，对照 A_control_v3） | 不差于 simple-v3；参数 257 → 96；bank skill 必读 |
| L2 CVAE 骨架（隐意图 z） | 未开始 | `latent_intent.py` + 十处接缝 | 不坍缩 ∧ minADE_K < top-1 ∧ z-oracle 臂 ≤ 1235 m |
| L3 CTA 条件化（交付形态） | 未开始 | `cta_conditioning` | 给真值 CTA 时时长误差 < 5 s ∧ 反事实 CTA 轨迹仍可飞 |
| L4 场景条件（先验吃邻机） | 未开始（数据平面 WIP 已在 `045c233`） | `scene/` + 先验网络 | KL(q‖p) 下降 ∧ 雷达引导 top-1 改善 |
| L5 先验三臂 / 合并机场 / 多机 | 未开始 | — | 见 §七 |

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

### L2 — CVAE 骨架（隐意图；≈1 周）

新增 `prediction_output="latent-control"`（`PREDICTION_LATENT_CONTROL`）与顶层模块
`latent_intent.py`：

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

**臂**：`L2_gauss`（K=1）、`L2_mix4`、`L2_mix6`、`L2_zoracle`（predict-only）。
**门（全部要过）**：
1. **不坍缩**：活跃维数（per-dim KL > 0.05 nat）≥ 3；打乱 z 后 pooled ADE 劣化 > 200 m；
2. **多模态有用**：minADE_6 显著低于 top-1，且优于"K 个随机 z + 同一网络"的对照；
3. **上界成立**：`L2_zoracle` 雷达引导 ADE ≤ 1235 m（不差于真值 (d_join, T) 条件化）。
**诊断（必须进读数脚本，否则坍缩会被读成收敛）**：per-dim KL、shuffle-z ΔADE、K 条散布 vs 真值散布、
**z 对真值 d_join / 时长 / via 位姿的线性可解码性 R²**（用 `closure_labels.json` 当探针，不当目标）。
若 z 解码不出 d_join 但重建很好 → z 吸收的是风与执行噪声，是坏消息，先压 `latent_dim` 再查。

### L3 — CTA 条件化（交付形态；≈2 天）

`cta_conditioning ∈ off | given`。`given` 时把目标到达时刻（训练用真值时长）作为解码器的条件标量。
**臂**：`L3_cta`（predict 时用真值 CTA）+ 反事实扫描（CTA ± 30 / 60 / 90 s）。
**门**：给真值 CTA 时时长误差中位 < 5 s；反事实 CTA 的轨迹逐样本可飞率不低于 `L2` 臂，且
到达时刻误差随 CTA 线性跟随（不是被时长头忽略）。
**契约**：见 §三.3——这类臂永远不能作为预测精度结果引用。

### L4 — 场景条件（先验吃邻机；≈2 周）

先把 WIP 数据平面（`045c233`）过 opus review 并补上 P2.d 的可解释性测量，再接：

- `scene/encoder.py`：邻机实体 token（≤ N_max=16，掩码）+ 几何折线 token + 标量 token →
  set/cross-attention → 场景向量；拼进 `fused_features` **只喂先验网络与解码器**。
  **不要把邻机做成 iTransformer 的额外变量 token**——`target_conditioning="channels"` 已经测过，
  主干基本不用协变量 token（只有拍平的时长头用了）。
- 泄漏红线：邻机特征只用 t ≤ t₀ 的样本；邻机落地时间与最终跑道只进 `future_label`，不进特征。

**门**：KL(q‖p) 相对 L2 显著下降（先验变尖 = 场景信息进来了）；雷达引导 top-1 ADE 改善。
**预期形状**：minADE_K 基本不变而 top-1 改善——那正是"场景信息在选模态"，是本设计预期的收益形态，
要在结果文档里明说，避免被读成"多模态没用"。

### L5 — 先验三臂 / 合并机场 / 多机（之后）

- 先验三臂：单高斯 / K 混合 / **隐扩散**（两阶段，见 §2.3）。预注册读数：minADE_K、miss rate、
  校准曲线、验证 NLL、**逐层看**（直线进近层三臂应基本相同，差别必须全在雷达引导层）。
  若 K 混合 ≈ 隐扩散，诚实报"在这个数据规模上扩散先验无增量"——这是有价值的负结果。
- 合并 5 机场训练（隐空间与机场无关，42,725 条 vs 单机场 1.4 万，是最大的未动杠杆）。
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
