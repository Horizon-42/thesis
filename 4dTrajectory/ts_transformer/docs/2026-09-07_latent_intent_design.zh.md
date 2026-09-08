# 隐意图 + 操作参数解码：control 预测的重做（dev 文档，2026-09-07）

> **路径更新（2026-09-07，包审计 T2）：** 本文 L0 行引用的 `control/oracle/basis.py` 现在是 `control/basis_fit.py`；`control/oracle/` 的其余部分已归档到 `archive/oracle_teacher_2026_08/`。


**取代** `2026-09-07_scene_join_anchor_design.zh.discard.md` 作为 control 路径的当前设计。旧文档的
Phase 0 / P0 / P1.a–d 的**测量与产物全部保留并被本文引用**；被废除的是它的方案骨架（把汇入
决策做成 K 个显式锚、由查询解码器输出、几何闭式画路径再由跟踪器飞）。废除理由与新方案见 §一、§二。

分支 `dev-leg-ctrl`。允许大改：本文列出的废除清单是**批准过的**，不需要保留兼容。

---

## 〇、状态表（压缩 context 后从这里继续）

**当前状态（2026-09-07）**：L0、L1 完成；L4 前置测量门不过（场景编码器不建）；L3（CTA）代码完成待跑。
**L1 的答案：N=32 免费（1322 vs 1333），轨迹误差损失单独不够（dense 2515）——L2 的 base = native32 + 教师**。
**L2 已跑四轮**：L2.c（β 阶梯，全坍缩）、L2.d（热启动后验，warm β=0.01 是至今最好的点估计 1214 m）、
L2.e'（free bits 当预算：KL 保住了，top-1 反而变差），以及解释这一切的探针——三臂的**后验均值都坐在先验
均值上**（位移 < 0.2 σ），预算全花在收窄方差上。**下一步 = L2.f campaign**（`l2f_mean_information_arms.json`：
`L2f_anneal` β 退火 40 轮 / `L2f_aux_T` z 上的辅助时长目标，预测时
`--latent-samples 6 --latent-random 6 --latent-shuffle`），代码在 `dev-l2f`；兜底是 L2.d 的 1214 m。

| 阶段 | 状态 | 产物 / commit | 门 |
|---|---|---|---|
| L0 操作参数维度 oracle | **完成（2026-09-07）** — 门按字面不过（N=16 为 315–330 m），走"否则"分支：**N\* = 32**（uniform 203 / free 191 m）；N=64 为 91 / 81 m。结果 `2026-09-07_l0_control_basis_results.zh.md` | `control/oracle/basis.py` + `run_ts_control_basis_oracle.py` + 22 项测试；产物 `l0_control_basis_20260907/` | 存在 N\* ≤ 16 使雷达引导 ADE(N\*) ≤ 200 m |
| L1 低维控制头 + 稠密监督（确定性基线） | **完成（2026-09-07）**：native32 全体 ADE 1322 vs 基线 1333（配对胜率 52.6 %，bank skill 0.726 vs 0.728）——**N=32 免费**；dense/无教师 2515/2603，否决触发，wiggle 回归——**轨迹误差损失单独不够**。结果 `2026-09-07_l1_lowdim_results.zh.md` | `l1_lowdim_20260907/`（readout、readout_bank） | 不差于 simple-v3 ✓；参数 257 → 96 ✓；bank skill ✓（native32） |
| L1.b 监督替代教师 | **180 轮三臂跑完（2026-09-08）——无一过门**：bank skill hr8+TV 0.713 / hr16+TV 0.677 / hr16 0.706（门 0.726），剂量翻倍变差，TV 在 hr16 下反而有害；hr8+TV 仍是最好的（FDE 各分层全优 848/632/1722 vs 864/671/1982，chamfer 161 vs 224，ADE 略劣 1332 vs 1322）。按决定规则教师留任、L5.a 拟合教师成为主线教师候选（其臂排在队列末） | `l1b_full_arms.json`、`l1b_full_20260907/readout_arm*.json` | 见 §六 L1.b |
| L1.c 训练期 LPV 走廊项（L1.b 胜者上） | **跑完（2026-09-08，`l1c_procedure_20260908`）——两训练臂都不过门，09-05 的否决在无教师底座上复现**：λ=2e-4 走廊违反率只降 0.9 点（门 5 点）、雷达引导 ADE +28 m；λ=1e-3 降 4.2 点但雷达引导 +187 m（否决）。同底座的推理期软屏障 hook 全面占优：横向违反率 83 → 44 %、ADE 1332 → 1296、FDE 848 → 721、chamfer 161 → 87、直线 \|xt\| p95 1282 → 55 m，代价 bank skill 0.713 → 0.646。罚项只在垂直轴有稳定收益（垂直违反 56 → 51 %，雷达引导 83 → 67 %），hook 不碰垂直——训练期 LPV 项定案为不用；候选后续：横向 hook + 训练期只垂直项 | `l1c_procedure_20260908/readout.{json,txt}`、`readout_bank.txt` | 见 §六 L1.c 结果 |
| L2 CVAE 骨架（隐意图 z） | **量纲测试臂 L2z_units 跑完（2026-09-08，`l2_units_test_20260908`，180/180 未早停）——量纲论断成立，z 第一次活了**：打乱 z 的 ΔADE **+928 m**（雷达引导 +2444；此前七臂 0–60 m），minADE_6 1021 < N(0,I) 对照 1043（首次），先验总 σ 0.848（L2d 0.58），8 维全活跃；top-1 ADE 1249（native32 1322、L2d 1214），**FDE p50 688 vs 864**（雷达引导 1519 vs 1982，胜率 66–70 %）；位移中位 0.334σ（门 1σ 未过，p90 1.04）——两条候选判据分裂，打乱判据压倒性通过；最佳轮 180 = 预算受限。**L2.g 跑完（2026-09-08 晚）：四道门两种子全过**——ADE 1230 / 1194（native32 1322，种子差 37 m）、FDE p50 656 / 627（864）、打乱 z 代价 +1017 / +935 m、先验 σ 0.87 / 0.90；隐变量线重新成为主线组件候选，见 §六 L2 末 | `l2_units_test_20260908/latent_readout_L2z_units.{json,txt}`、`probe/`、`readout.json` | 见 §六 L2 末 |
| L3 CTA 条件化（交付形态） | **跑完（2026-09-07 晚）**：给定真值到达时刻，全体 ADE 1214 → 841，雷达引导 2643 → 1596（胜率 87 %），时长误差 0 按构造；几何只小改（chamfer 165 → 128）；**反事实扫描跑完（2026-09-08 上午）**：七个偏移的时长误差精确等于偏移；几何随 \|偏移\| 单调（chamfer 232 → 1038 / 1160，路径长度比 0.59 → 1.51）；不对称——提前到更可飞（8 → 45 %），推迟到可飞率归零、以横向偏出千米吸收延时（xt@thr p50 3.2 km） | `l3_cta_20260907`、`l3_cta_counterfactual_20260907` | 见 §六 L3 |
| L3.c 推迟吸收进走廊（只预测） | **跑完（2026-09-08，`l3c_delay_corridor_20260908`）**：hook 把横向偏出压回中线（+90 s：xt@thr p50 3288 → 5 m），偏移 0 处白拿（横向违反率 84 → 43 %、FDE 746 → 582）；时长精确服从（门 2 过）；但完全可飞 0 %（门 3 未过）、任意点违反率 86 %（门 1 分裂）；不可飞 99.9 % 是失速项、77–94 % 在剩余 ≥ 20 km 的外段（不是尾巴、不是蜿蜒）——模型在外段减速过头，延时再往那里压。下一个设计项待定：rollout 速度下限 + 汇入前路径拉长自由度 | `l3c_delay_corridor_20260908/readout.{json,txt}` | 门 2 过，门 1 分裂，门 3 未过；见 §六 L3.c 结果 |
| 直线进近残差分解（测量） | **完成（2026-09-08）**：沿航迹占水平误差² 89–95 %，集中在最后 5 km（p10..p90 ±700 m）；给定真值时长只去偏置（−191 → +8 m）不去散布（RMS 333 → 316）；减速点偏移 p10/p90 ±1.4 km、解释末段方差 22 %（风 10 %）。是速度指令意图，不是动力学。L3.b 减速点 oracle 臂待用户决定 | `run_ts_straight_in_residual_readout.py`、`2026-09-08_straight_in_residual_readout.zh.md` | 无门 |
| L4 场景条件（先验吃邻机） | **前置测量完成，门不过（2026-09-07）**：场景实体特征对 d_join / 剩余时长**零增量**（R² 0.37 vs Phase 0 粗上下文 0.38；34.7 vs 35.1 s）；可观测的前机 ETA 与其真实落地时刻相关仅 0.11。场景编码器**不建**（数据平面 review 未发现泄漏或帧/基准错误；HIGH/MEDIUM 项已修，测量成立） | `intent_explainability.py`、`run_ts_scene_explainability.py`；产物 `l4_scene_explainability_20260907/` | KL(q‖p) 下降 ∧ 雷达引导 top-1 改善 |
| L5 先验三臂 / 合并机场 / 多机 | **L5.a 拟合跑完（2026-09-08，`l5_fitted_teacher_20260907/basis_fit.json`，49 min，train+val 8255 架全覆盖，N=32、网络初值、400 步 × batch 1024）**：val fitADE p50 **106 m**（p90 737、p99 1320），train 108 m；种子（native32 自身输出）542 → 106 m，5 倍；**best step p50 = 400 两个 split 都顶在预算上限，仍在下降**——上界不紧，L5.a 臂读数弱先怀疑步数。L5.a 臂按 2026-09-08 重排降为队列末尾（教师只做上界）；其余未开始 | `run_ts_control_basis_oracle.py --checkpoint`、`control_imitation_target` / `control_fitted_teacher_path`、`control/basis_fit.py` 的表加载器、`docs/experiments/l5_fitted_teacher_arms.json`、`tests/test_fitted_teacher.py` | 见 §七 |

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

**修正（同日）：『速度项单独够不够』其实已经测过**——`CLAUDE.md`：bank 无监督时 skill 0.124，低于平凡
基线，simple-v3 的模仿项把它拉到 0.735；上面的臂 ①（模仿 0 = simple-v2 在 N=32）只会重测这个数，
撤销（"do not re-litigate without new evidence"）。L1 未测的只是 fixed-dt 网格下的组合，与本题无关。

**两臂，同 native32 底座（custom、N=32、native 端点、true-time-position、速度项 0.003 照旧）：**
2. `L1b_heading_rate`：模仿 0 + **航向率损失** `control_heading_rate_loss_weight`：rollout 在真值时钟段端点
   处的航向率（协调转弯下即 g·tan φ / V，取实际 bank，不是指令）对观测航迹平滑后的 ψ̇；同一个观测转弯
   率，但穿过 rollout 施加、不经逆动力学与滞后反解——**模型一致**，零件更少。预期"等价且更干净"，
   不是新信息；只给 bank 命名，推力/载荷仍靠速度项。
3. `L1b_heading_rate_tv`：② + bank 相邻段总变差惩罚 `control_bank_tv_loss_weight`（结构约束代替损失
   约束的最便宜形式；T1-12 删掉的 effort/smoothness 是"无人使用"，不是"证明无效"——诚实记下）。
**门**：bank skill ≥ 0.70（native32 0.726）、ADE / FDE p50 逐分层不劣于 native32；否决：直线进近 FDE 超
种子噪声。读数同 L1。**决定规则**：若 ② 过门，教师整条线降级为对照臂（L5.a 拟合教师保留为上界
臂，回答"教师质量还能买多少"）；若都不过，L5.a 拟合教师成为主线教师。不做的：闭环（贪心）教师——
拟合已降到 2–3 h，贪心是它的近视近似且把纠错瞬变写进目标；双层优化与分布匹配（GAN/MMD）——超出
路线与规模。"少数几次有滚转率限制的机动"作为另一种基，先用 L0 的宽度 oracle 量表示误差再决定。
**筛选结果（2026-09-07，campaign `l1b_supervision_20260907`，60 轮，匹配对照 = 同预算的教师臂）**：

| 臂 | bank skill | 直线参考 bank RMS | 共同剖面份额 | ADE 全体 / 直线 / 雷达引导 | 直线 FDE p50 |
|---|---:|---:|---:|---|---:|
| 对照（教师 64） | 0.709 | 0.39° | 3.1 % | 1459 / 540 / 3083 | 776 |
| hr=1 | 0.516 | 0.47° | **22.3 %** | 1597 / 566 / 3430 | 926 |
| hr=8 | 0.665 | 0.20° | 2.3 % | **1396 / 497 / 2987** | **734** |
| **hr=8 + TV=1** | **0.711** | **0.17°** | 2.5 % | 1404 / 508 / 2991 | 846 |

（地板 0.170、孪生天花板 0.699、观测 bank RMS 0.41°、观测共同剖面 1.8 %。）**hr=8 + TV 过门**，教师线按决定规则
降为对照臂，L5.a 只作上界。剂量是杠杆（1 → 8：0.516 → 0.665），8 是括号上沿不是最优；TV 在损失里只占 2 %、
四臂符号反转皆为 0，"给反转定价"的机制不成立，但它确实把 skill 抬了 0.046、RMS 压到 0.17°——记为"有效、机制未明"。
hr=1 不是"效果弱"而是把模型推进共同坡度剖面（22 %），比无教师的 dense 臂还差。保留意见：过门只赢 0.002、单种子；
直线 FDE 846 是弱点而 hr=8 单独最好（734）。**确认臂**（`l1b_full_arms.json`，180 轮，对照 L1_native32 的 1322 /
0.726 / 671）：hr8+tv1、hr16+tv1、hr16；过门者成为主线底座的监督（隐变量臂在多机场前换底座），都不过则筛选结论翻转、
拟合教师回到主线。

**180 轮确认，第一臂（2026-09-07 夜，`l1b_full_20260907/L1b_hr8_tv1_full`，116/180 早停，对照 L1_native32 的 180 轮）：
分裂而非通过。** bank skill **0.713**（门 0.726，差 0.013）；ADE 各分层略劣（全体 1332 vs 1322，直线 461 vs 445，
雷达引导 2872 vs 2870，胜率 44–48 %）；**FDE p50 各分层全部更好**（全体 848 vs 864，直线 **632 vs 671**，雷达引导
**1722 vs 1982**，胜率 51–61 %）；否决过（直线 FDE 好 39 m）。坡度几何是测过最干净的：直线参考 bank RMS **0.12°**
（native32 0.34°，观测 0.41°）、共同剖面 2.2 %、无反转——但比真实飞机还直，配合雷达引导路径长度比 0.88（native32
0.93）和 chamfer 变差（1040 vs 901）而 FDE 改善 260 m，读法是**抄近路**：用更短更顺的路径到达正确终点，bank skill
是对观测剖面的逐航班相关，过度平滑在这里失分。结论：教师线**不能**据此降级为对照——终点指标赢、路径指标输，
门差 0.013；剂量（hr16 两臂，排在 A0 之后）决定翻不翻。A0-random 臂按预注册仍用这个配方（它是唯一与随机锚点
相容的监督），其 L−1 否决线是"不劣于 native32 超过种子噪声"，10 m 在噪声内。**筛选档**：坍缩 / wiggle 这类定性问题用 60 轮（约 20 min/臂）先筛，过筛的臂再跑满 180 轮出门数（`epochs` 是 config 字段，臂文件直接设；不减航班——减航班改变总体，数字不再可比，而且加速比更小）。

**180 轮三臂定案（2026-09-08，`l1b_full_20260907`，对照 L1_native32 180 轮）：无一过门，剂量问题有了答案。**

| 臂 | 轮数 | bank skill（门 0.726） | 直线参考 bank RMS | ADE 全体 / 直线 / 雷达引导 | FDE p50 全体 / 直线 / 雷达引导 | chamfer |
|---|---|---:|---:|---|---|---:|
| native32 | 180 | **0.726** | 0.34° | 1322 / 445 / 2870 | 864 / 671 / 1982 | 224 |
| hr8 + TV | 116 早停 | 0.713 | 0.12° | 1332 / 461 / 2872 | **848 / 632 / 1722** | 161 |
| hr16 + TV | 142 早停 | 0.677 | 0.12° | 1358 / 448 / 2968 | 921 / 658 / 1801 | 147 |
| hr16 | 180 未早停 | 0.706 | 0.15° | 1344 / **442** / 2940 | 941 / 672 / 2329 | **146** |

读法：hr16 比 hr8 差不是好——剂量 1 → 8 的上升趋势（0.516 → 0.665）没有延续；TV 的符号随剂量翻转（hr8 下 +0.046，hr16 下 −0.029），
说明 TV 只在航向率项欠剂量时有用。三臂的坡度几何都比 native32 更"直"（RMS 0.12–0.15° < 观测 0.41°），路径长度比 0.87–0.88（雷达引导），
chamfer 与雷达引导 FDE 的赢来自同一个抄近路，bank skill（逐航班相关）因此失分。**决定规则落在"都不过"分支：教师留任，L5.a 拟合教师成为
主线教师候选**（其两臂排在队列末）。hr8+TV 保留为 A0.b 的监督（唯一与随机锚点相容的），以及 L1.c 的底座——**这里偏离了 L1.c 预注册
的字面（"都不过则底座回 native32"）**：L1.c 的问题是罚项在没有教师时是否仍付中段代价，只有无教师底座能回答它，hr8+TV 离门 0.013、FDE 全优，
是最接近的无教师底座；在 native32 上只是 09-05 结果的复现。偏离在此声明，读数时带此保留。

**实现（2026-09-07，`dev-l1b`）**：分量名 `heading_rate` / `bank_tv`，与 `velocity` / `imitation` 同在
`true-time-position` 目标下注册；在别的目标下给非 0 权重会被 config **直接拒绝**（不会静默失效）。
CLI `--control-heading-rate-loss-weight` / `--control-heading-rate-loss-scale-dps` / `--control-bank-tv-loss-weight`；
run name 缩写 `hr` / `hr-scale` / `bank-tv`（306 份存档 config 重算命名，0 处变化）。四个臂
`L1b_native32_e60`（**同预算对照臂**，教师 64）/ `L1b_hr1` / `L1b_hr8` / `L1b_hr8_tv1`。

- **目标**（`dataset.reference_heading_rate_supervision`，只在训练期、读未来）：观测 ψ 取自速度通道
  （`states_from_channels`，即建模层的 math-ENU 航向，符号约定与 rollout 同源），`np.unwrap` 后做
  **中心差分**，窗宽是模块常数 `HEADING_RATE_SMOOTHING_WINDOW_S = 10 s`。这个常数是**标称值**：
  半宽 = `int(round(W/2/dt_s))` 个样本，**实际跨度** = `2·half·dt_s`；dt_s = 2 s 时 half =
  round(2.5) = 2（Python 的 .5 向偶数取整），**实际跨度 8 s 而不是 10 s**——引用平滑量时报实际跨度。
  舍入两个方向都会偏（dt_s = 3 s → 12 s、5.05 s → 10.1 s，注释里写明了），只有粗到连一个半宽样本
  都凑不出的 dt_s 被**直接拒绝**（不再静默 `max(1, …)` 兜底）。为什么要平滑：ADS-B 航迹角是量化值，
  单步 2 s 差分是噪声而不是转弯率。再采样到 N 个段端点。权重 = 端点时刻 ≤ 最后一个实测速度时刻——
  与速度项**同一刀，但不是同一组数**（这里是 0/1 硬阶跃，速度项是插值出的逐通道权重）；模仿项用段
  中点，因为控制量是段上的量、转弯率是瞬时量。
- **端点配对靠一条链**：端点取均匀的 `(k+1)·T/N`，它等于 rollout 的 `cumsum(段时长)` 只因为
  `true-time-position` 目标下 config 只允许 uniform 段时长；非均匀分割时两侧第 k 行就不是同一时刻。
  模仿项的中点有完全相同的隐含假设——两处 docstring 现在都写明了。
- **模型侧**（`train.control_heading_rate_mse`）：`aerodynamic_model.torch_dynamics.heading_rate_rad_s`
  直接从 `enu_rhs` 里读 ψ 行，输入是 rollout 的段端点状态和该处的**实际**控制
  （`EndpointControlRollout.actual_controls`：point-mass 下就是指令，一阶滞后下是作动器状态）。
  **更正预注册的一处写法**：RHS 积分的是 `g·n_realized·sin φ /(V·cos γ)`（载荷因子含失速限幅），
  教科书的 `g·tan φ / V` 只在协调平飞（`n = cos γ / cos φ`）下与之相等；代码调用 RHS 本身，
  "按构造模型一致"才是真成立而不是近似成立。尺度 `control_heading_rate_loss_scale_dps = 1.5`
  是读数单位（半个标准转弯率），不是剂量。
- **`bank_tv`**：相邻**指令** bank 的平均 |步长| ÷ 半个 bank 盒宽（`CONTROL_HALF_WIDTH[BANK_INDEX]`）。
  罚指令而不是滞后后的实际 bank——罚后者等于为作动器执行它收到的指令而罚它。**它罚的是「掉头」不是
  「斜率」**：`|x|` 的次梯度是 `sign(x)`，单调段内部两项相消（只有两端被计价，平滑滚入与同幅度的一次
  阶跃等价），而**完全平坦处是驻点**——值 0、梯度也 0，恰好是 `_initialize_control_head` 把投影权重
  清零后每次训练的起点，只有别的项能把它推离。若臂 ③ 读出「TV 没起作用」，先查这一条，再谈剂量小。
- **剂量是标定不是结论**：模仿项的剂量曲线在 ~11.8×–~47× 位置项之间是噪声平台，所以航向率的剂量未知，
  1.0 / 8.0 是**把它夹在中间**，TV 的 1.0 是毫无先验的第一次探针。
- **筛选档的读数必须对同预算对照臂读**：`L1b_native32_e60` = 同底座 + 教师 64（即 L1_native32 跑
  60 轮）。已发表的 native32 数字（1322 m、bank skill 0.726）来自跑满 180 轮且**未早停**的运行，
  是预算受限的上界；拿 60 轮的数去和它比会把「监督方式」和「预算」混在一起。60 轮里早停是**活的**
  （patience 20），臂停在第几轮本身是读数的一部分。命名上：模仿权重为 0 时最近的 recipe 是 simple-v2，
  所以处理臂解析成 `simple-v2+(hr=…)`——读作「simple-v2 + 航向率」，不是「v3 去掉教师」。

#### L1.c — 训练期 LPV 走廊项在 L1.b 胜者上（2026-09-08 预注册；用户决定，优先级提到 L1.b hr16 之后、L3 反事实之前）

**为什么重测（新证据，不是翻旧案）。** 走廊 + 下滑道罚项（`objective.procedure_loss`：跑道轴坐标 hinge²，
横向对 LPV 角锥 |xt| ≤ k·hw(d)、垂直对下滑道窗口，只在观测已建立五边的行上计价，与优化器共用
`fas_geometry`）2026-09-05 在 control 路径上被否决：λ = 1e-3 终点变好（KRDU FDE 1650 → 1549、入口横向 p95
2769 → 2203 m）但中段变差（雷达引导 ADE +658 m），5e-3 让坡度调度坍缩（skill 0.728 → 0.280）。那次的底座是
simple-v3 **带教师 64**（模仿项占损失 87 %）：教师把控制量钉在 rollout 飞不回真值的逆动力学目标上，罚项经动力学
传回的梯度与教师打架——中段代价和坡度坍缩都与"打架"一致。L1.b 拿掉了教师（航向率 + TV 穿过 rollout 本身给坡度
命名，模型一致），罚项在没有教师的底座上还付不付中段代价，**没有测过**。当前所有臂的损失里都没有这一项
（默认全关），交付时走廊只作为推理期 hook。

**底座**：L1.b 决定规则指定的主线监督——现在是 hr8+TV（180 轮幸存者，`l1c_procedure_arms.json` 已按它写）；
若 hr16 某臂过 L1.b 的门，**启动前**把 base 改成那臂（一行，改动记在这里）。若 L1.b 整体不过、教师留任，
底座回到 native32（教师 64），本臂退化为 09-05 结果在 N=32 上的复现——仍然跑（用户优先级），但按复现读。

**臂**（`l1c_procedure_arms.json`，KRDU，180 轮，种子 1337，val 1404，campaign `l1c_procedure_20260908`）：

| 臂 | 做法 | 回答 |
|---|---|---|
| `L1c_lpv_2e-4` | 底座 + λ_lat = λ_vert = 2e-4 固定 | **平价剂量**：09-05 量得 control 上 1e-3 在第 1 轮是位置项的 4.4 倍（state 0.78 倍），1e-3/4.4 ≈ 2.3e-4——上次"平价标定在 control 上不成立"的修正 |
| `L1c_lpv_1e-3` | 底座 + λ = 1e-3 | 与 09-05 同剂量，直接对照：没有教师后 +658 m 的中段代价还在不在 |
| `L1c_base_barrier_infer`（只预测） | 底座检查点 + 软屏障 hook | 可部署的推理期替代（09-06 净收益），同底座上的对照 |
| `L1c_lpv_2e-4_barrier_infer`（只预测） | 平价臂检查点 + 软屏障 hook | 训练期项与推理期 hook 能否叠加 |

对偶步长保持 0（state 上四次发散、ε 不可达，不重测）。control 路径的罚项要求 `control_state_loss_grid =
native-segment-endpoints`，L1.b 底座正是。读数同 09-05：`compare_constraint_arms.py`（分层 ADE/FDE、走廊 /
下滑窗口违反率、入口横向 p95）+ `score_control_arms.py`（bank skill、共同剖面、反转）。

**门（预注册，对同 val 集的底座逐航班配对）**：真值已建立行的走廊违反率**下降 ≥ 5 个点**，且雷达引导 ADE 不劣于
底座超过 **30 m**（control 单种子；state 路径两种子汇总 ADE 差 5–22 m，取其上沿再放宽），且 bank skill ≥ 0.70
（坍缩否决线）。**否决**：雷达引导 ADE 劣 > 100 m（09-05 的失效形态是 +658），或直线进近 FDE 劣于种子噪声。
**决定规则**：过门但 FDE 与违反率都不优于"底座 + 推理期 hook"，训练期项**继续关闭**，hook 仍是交付形态；
优于 hook、或与 hook 叠加有增益，则进入主线配方（B 线的底座随之换）。两个训练臂都触发否决 → 09-05 的结论
在无教师底座上复现，训练期 LPV 项在本路线上定案为"不用"。

**成本**：2 × ~65 min 训练 + 2 次预测 + 2 次只预测 ≈ 2.7 h GPU。**排期**：L1.b hr16 两臂之后立即执行
（用户决定 2026-09-08），先于 L3 反事实扫描、L2 单位测试、L5.a 与 A0.b。

**L1.c 结果（2026-09-08，`l1c_procedure_20260908`，底座 `L1b_hr8_tv1_full`，配对 1404 架）**

| 臂 | 轮数 | 选择指标 | ADE 全体 / 雷达引导 | FDE p50 | chamfer | \|xt\| p95 全体 / 直线 | 横向违反率 | 垂直违反率 | bank skill |
|---|---|---:|---|---:|---:|---|---:|---:|---:|
| 底座 hr8+TV | 116 早停 | 1332 | 1332 / 2872 | 848 | 161 | 2601 / 1282 | 83.3 % | 55.9 % | 0.713 |
| λ=2e-4 | 180 未早停 | 1343 | 1344 / 2900 | 860 | 165 | 2206 / 1291 | 82.4 % | 51.2 % | 0.715 |
| λ=1e-3 | 140 早停 | 1471 | 1471 / 3059 | 899 | 165 | 2111 / 1367 | 79.1 % | 52.9 % | 0.719 |
| **底座 + 软屏障 hook（只预测）** | — | — | **1296 / 2857** | **721** | **87** | 1977 / **55** | **43.8 %** | 55.9 % | 0.646 |
| λ=2e-4 + hook（只预测） | — | — | 1307 / 2884 | 750 | 85 | **1281** / 56 | 45.5 % | 51.2 % | 0.634 |

门（走廊违反率 −5 点 ∧ 雷达引导 ADE ≤ +30 m ∧ bank ≥ 0.70；否决雷达引导 > +100 m）：**λ=2e-4 未过**（违反率只降 0.9 点，ADE +28 m 压线，bank 过）；
**λ=1e-3 未过且触发否决**（降 4.2 点，ADE +187 m）。**决定规则落点：两训练臂都不过——09-05 的结论在无教师底座上复现，训练期 LPV 项在本路线上定案为不用。**
没有教师也一样：罚项经动力学传回控制量的梯度伤中段，剂量小到不伤时也买不到走廊。

读法：(1) **推理期 hook 在同一底座上全面占优**——横向违反率降 39.5 点（门要求的 8 倍）且 ADE / FDE / chamfer 同时变好，直线进近终点 \|xt\| p95
1282 → 55 m，不需要重训；代价是 bank skill 0.713 → 0.646（hook 改了指令路径，逐航班坡度相关失分，与 L1.b "更直反而失分"同源）。
(2) hook 叠在罚项臂上得到最好的全体 \|xt\| p95（1281）但最差的 bank（0.634），两机制不干净叠加。(3) **罚项的稳定收益在垂直轴**
（垂直违反 55.9 → 51.2 %，雷达引导 82.9 → 66.6 %），而 hook 按构造不碰垂直——候选后续（待用户决定，不排队）：横向用推理期 hook、
训练期只加垂直项（L1.d），本次测的"横向 + 垂直"合罚不是正确的切分。

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

> **L2.d 结果（2026-09-07，campaign `l2_warm_posterior_20260907`）。** 热启动奏效：两臂都把 8 个维度活着保持
> 了 50+ 轮（冷启动为 0 轮）。β=0.1 在第 ~87 轮衰亡（best epoch 138：active 0，KL 0.08），且选择指标在 z 死
> 的过程中单调变好（活着的最好 epoch 71：1466 m / 5.85 维；被选的 138：1356 / 0 维）——**确定性 top-1 回放
> 的选择规则奖励 z 的死亡**，这是一个结构性冲突，留作后账。**β=0.01 活到最后**（180 轮未早停，best 175：
> active 2.82、KL 0.17 nat/航班），而且是至今最好的点估计：
>
> | 分层 | native32 ADE | **warm β=0.01** | native32 FDE p50 | **warm β=0.01** | 配对胜率（中位 Δ） |
> |---|---:|---:|---:|---:|---|
> | 全体 | 1322 | **1214** | 864 | **827** | 65.3 %（−44 m） |
> | 直线进近 | 445 | **402** | 671 | **594** | 62.5 %（−24 m） |
> | 雷达引导 | 2870 | **2643** | 1982 | **1851** | 70.4 %（−169 m） |
>
> bank skill **0.729**（八臂最好，直线参考 bank RMS 0.30° 低于观测 0.41°）；chamfer p50 165 vs 224。门：
> (3) 全分层通过，否决干净（直线 FDE 594 < 671）；(1) **险败**——active 2.82 < 3、shuffled ΔADE +19 m
> （雷达引导 +39）< 200：解码器读 z，但读得很淡；(2) **败**——minADE_6 1004 < top-1 1214（真实的 210 m 多
> 模态，散布 325 m），但 **N(0,I) 对照 947 更好**，各分层皆然：随机 z 的 best-of-6 胜过训练先验的 best-of-6。
> 读法：这既不是 (a) 也不是 (b)，是第三种——**z 活着、被用、点估计全面改善，但信息量只有 0.17 nat，
> 先验比 N(0,I) 还窄**。N(0,I) 胜出说明解码器的 z 方向有意义且超出训练分布仍有效，只是 q/p 被压得太集中：
> 量纲论证在 β=0.01 仍成立——每 nat 0.01 个损失单位，仍高于位置项能买到的全部 0.009——z 只能靠模仿项挣，
> 所以信息停在 0.17 nat。**下一杠杆仍是 β，不是先验形状**（0.17 nat 无物可摊）；K=4 混合在 z 携带数 nat
> 之后再上。
>
> **z-oracle（campaign `l2d_zoracle_20260907`，门 (3)）：全败，且都不近。** 用后验 z（各航班自己的未来）解码：
> 冷 β=0.1 雷达引导 2846 → 2829（−17）、暖 β=0.1 2950 → 2949（−1）——死的隐变量给它 oracle z 也不动，与
> shuffled ΔADE ≈ 0 从另一方向互证；**暖 β=0.01 2643 → 2416（−227）、全体 1214 → 1136、直线进近 402 → 402
> （0）**：z 携带的信息真实、形状正确（只在雷达引导/长路径层起作用，已建立的航班无话可说），但只有门
> 所需的约五分之一（缺口 1181 m），且改善的是路径不是终点（FDE p50 827 → 836）——z 学到的是路线形状，
> 不是到达时序。
>
> **机制修正（L2.e' 的第一版预注册作废）。** 五次运行 KL 全部落在 free-bits 地板（8 × 0.05 = 0.4 nat）之下，
> 地板之下 KL 梯度恒为零——**β 在这个区间根本不起作用**，"每 nat 0.01"的量纲论证只在地板之上成立，
> 所以再降 β 测的还是同一个零。地板之下的衰减也不是 weight decay（lr 3e-5 × 2450 步，总衰减 < 1 %），而是
> **低信噪比的自毁**：地板之上的惩罚把后验推到 σ≈1、μ≈0.3 的状态才越过地板，那里 z 对解码器几乎是噪声，
> 解码器的 z 权重在噪声梯度下继续缩，后验因此失去恢复力——0.05 nat/dim 的"免费区"恰好是一个低信噪比区。
> 杠杆是 **free bits 本身当作信息预算**（Kingma 等 IAF 的用法）：1 nat/dim = 8 nat 总预算，正好是意图
> （d_join + T，log 可分辨意图数）的量级；预算之内 KL 零成本，后验可以保持 σ≈0.5、μ≈1（信噪比 2），
> 早期 14 nat 被 β 压到预算处就停，解码器一直有可用的 z。
>
> **L2.e' 预注册（2026-09-07，第二版）：热启动后验、β=0.01、free bits ∈ {0.5, 1.0} nat/dim**，K=1、同 base、
> 跑满 180 轮；臂文件 `docs/experiments/l2e_free_bits_arms.json`，campaign `l2e_free_bits_20260907`；读数同
> L2.d 加 z-oracle。注意 `active_unit_threshold_nats = max(free_bits, 0.05)` 随预算移动：门 (1) 的"活跃"变成
> "KL_dim > 预算"，比现在严——读数同时报逐维 KL 与总 KL。**预注册读法**：(a) KL 停在预算附近（4–8 nat）、
> shuffled ΔADE ≫ 200 m、z-oracle 雷达引导向 1235 m 靠近、minADE_6 < N(0,I) 对照 → 预算是对的，把它带到
> K=4 与 L3；(b) 出现自编码器形态（top-1 劣于 native32）→ 取仍过门 (3) 的最大预算（β=0.01 / fb 0.05 的
> 1214 m 已是兜底）；(c) KL 仍掉回地板之下 → 自毁不是由惩罚触发的，杠杆在解码器的 z 通路（注入方式）。


> **L2.e' 结果（2026-09-07，campaign `l2e_free_bits_20260907`）。** 预算生效：raw KL 停在 2.59（预算 4.0）与 6.60
> （预算 8.0）nat，第 50 轮起平稳不再下滑——(c) 作废。但 **信息越多 top-1 越差**：fb 0.05 / 0.5 / 1.0 的 top-1
> 1214 / 1260 / 1387 m，雷达引导 2643 / 2747 / 3052；shuffled ΔADE +47 / +16 m；minADE_6 仍输给 N(0,I) 对照
> （1069 vs 1012；1238 vs 1184）；bank 不受影响（0.719–0.729）。fb 0.5 仍过门 (3)（胜率 58 %），fb 1.0 不过。
>
> **直接测量后验与先验（`run_ts_latent_probe.py` 的前身，200 架 val，三个 checkpoint）——问题终于看清了**：
>
> | 臂 | q 均值对 p 均值的位移 / p σ（逐维中位） | q σ | p σ | KL/dim |
> |---|---|---:|---:|---:|
> | warm β=0.01 fb 0.05 | 0.05–0.11 | 0.40 | 0.44 | 0.03 |
> | fb 0.5 | 0.10–0.20 | 0.23 | 0.49 | 0.32 |
> | fb 1.0 | 0.11–0.19 | 0.15 | 0.58 | 0.83 |
>
> **三个臂的后验均值都坐在先验均值上**——逐航班位移不到 0.2 个先验 σ。预算全部花在**把后验方差收窄**上
> （KL 的方差项），均值项几乎为零：z 是一个去噪的常数，不携带该航班的任何信息。这同时解释了 shuffled ΔADE
> ≈ 0、z-oracle 只买 227 m、N(0,I) 对照赢（它的样本比先验宽）、以及 top-1 随预算变差（q σ 与 p σ 的差距
> 拉大，训练时解码器见到的 z 与推理时不一致）。机制：训练最初 10 轮 z 是有信息的（KL 14 nat、8 维活跃），
> 预算之上的 β 惩罚首先压缩 KL 的**均值项**（梯度与位移成正比），把各航班的 q 均值拉到 p 均值上；等 KL 落到
> 预算之下时均值信息已经没了，解码器的 z 权重也随之适应了一个近似常数的 z，此后没有任何梯度把均值信息拉
> 回来。free bits 只是让"无信息的窄后验"免费。
>
> **L2.f 预注册（2026-09-07）：让后验均值带上信息的两条路，各一臂，同 base（热启动、fb 0.05、K=1）。**
> 1. `L2f_anneal`：**β 退火** 0 → 0.01，线性 40 轮（`latent_beta_warmup_epochs=40`）。标准疗法：在解码器学会
>    用 z 之前不施加均值项压力，让 z 权重先长大，之后均值信息有重建梯度作为恢复力。此前"退火无用"的判断
>    只对方差项成立，对均值项不成立——诚实记下。
> 2. `L2f_aux_T`：**z 上的辅助意图目标**（训练期头 `latent_aux_duration_weight=1.0`）：从后验样本 z 线性预测
>    归一化剩余时长 (T / final_time_scale_s)，MSE。它直接定义 z 必须携带什么（意图的"何时"一半），给后验均值
>    一个不依赖解码器的恢复力；先验从上下文预测不了 T → 先验必须变宽 → K 个样本才真正散布在可能的到达
>    时刻上。z 仍是隐变量（辅助头只在训练期），符合"意图是隐空间、不是输出"的约束。
> 同一提交把诊断补上：history 的 `latent` 块记逐维 KL、KL 均值项/方差项拆分、后验均值位移（/p σ 的中位）
> ——这个数在第 1 轮就能看出问题，本应一开始就有；`run_ts_latent_probe.py --checkpoint` 读同样的量。
> **门**：后验均值位移中位 > 1 σ（信息在均值里）、shuffled ΔADE > 200 m、minADE_6 < N(0,I) 对照、top-1 不劣于
> native32（1322）；否决同前。K=4 与 L3 的 z 版本都等这里的胜者。L2.d 的 warm β=0.01（1214 m）仍是兜底。
>
> **实现（2026-09-07，`dev-l2f`：`be56088` 诊断、`60dd620` 退火、`6233967` 辅助目标、`63d1fdd` 探针、
> `59640bd` 臂文件）。**
> - **诊断**（`control/latent.py`）：`latent_kl` 改返回 `LatentKL` 记录（`charged` / `per_dimension` /
>   `mean_term_per_dimension` / `displacement_sigma` / 取诊断的那个先验分量的 `prior_mean`、`prior_logvar`；
>   方差项是 `per_dimension − mean_term` 的**余数**，不是第二个闭式——目标收取的那个数必须逐比特不变）。
>   `with_latent_kl` 因此多记五个诊断，`latent_epoch_record` 组装 history 的 `latent` 块：
>   `component_kl_per_dim`（逐维，对航班取平均）、`component_kl_mean_term_nats` /
>   `component_kl_variance_term_nats`（逐航班）——名字里的 component 是要紧的：这三个加起来等于
>   `component_kl_nats_per_flight`（对最负责分量的解析 KL），**不等于**被收取的 `kl_nats_per_flight`
>   （free bits 与混合估计把两者分开）——、`mean_displacement_sigma`（|μ_q−μ_p|/σ_p 的中位；中位不可加，
>   所以是**各 batch 中位数的航班加权平均**，与 `active_units` 同形；用 `torch.quantile(…, 0.5)` 而不是
>   `torch.median`（后者取偶数个数里较小的那个中位），外部读者用 numpy 复算才落在同一个数上）、
>   `active_units_0p05`（固定 0.05 nat 的尺子；`active_units` 保持随预算移动的旧定义——L2.e' 三臂各按自己的
>   阈值计数，门 (1) 因此读不出来）。判定句子（位移 > 1 σ 才算"信息在均值里"）是
>   `control.latent.displacement_verdict` 一处，读出脚本与探针共用同一把尺（`DEAD_MEAN_DISPLACEMENT_SIGMA`）。
>   `run_ts_latent_readout.py --history <run>/history.json` 打印被选 epoch 的这一块（epoch 号读产物自己的
>   `fit_diagnostics.training_objective`）；**`--history` 可以单独用**——第 1 轮就要看的数，那时还没有任何预测。
> - **β 退火**：`latent_beta_warmup_epochs`（默认 0；无 latent 拒绝、负值拒绝；
>   run name `beta-warmup=40`）。第 e 轮（1 起）的有效权重 = `latent_beta · min(1, e/warmup)`，
>   `control.latent.effective_latent_beta` 是唯一写法；每轮的目标是 `replace(config, latent_beta=有效值)`，
>   训练批次与验证遍历**同用它**（`evaluate_validation_airport` 因此显式收下它要打分的 config），
>   所以同一轮的 train / val `latent_kl` 含义相同——与 procedure 惩罚的 λ 同一条规矩。history 里
>   `train_components.latent_kl` 仍是**被收取的**损失项（有效 β × 航班加权均值 KL），`latent` 块里全是未缩放
>   的 nat，旁边多一个 `beta_effective`；checkpoint 选择不变。
> - **辅助意图目标**：`latent_aux_duration_weight`（默认 0；无 latent 拒绝、负值/非有限拒绝、
>   `cta_conditioning=given` 拒绝——CTA 已经把时长交给解码器了；run name `aux-T=1`）。
>   训练期头 `aux_duration: Linear(latent_dim→1)` 只作用在**后验样本**上，只在权重非零时构造（且在
>   `__init__` **最末**构造，别的模块的初始化抽样因此与无此臂完全相同，两臂配对），`decode` 从不调用它，
>   记录里没有任何它的痕迹。损失分量 `latent_aux` = 权重 × 航班加权 MSE（目标 = batch 自己的 `final_time` /
>   `final_time_scale_s`），只在权重非零时注册（`loss_component_names` 与适配器同一个条件、同一个文件）。
>   **读法上的要害（review 提出，写进臂文件）**：这个目标就是后验编码器的**输入之一**（真值时长），所以
>   `latent_aux` 小本身什么也不证明——一个隐维复制一个输入即可。判决必须靠 KL 均值项 / 逐维 KL 动了
>   **并且**下游指标动了（shuffled ΔADE、minADE_6、top-1）。
> - **recipe 把整条隐变量轴钉死**（review 采纳）：`control_simple_v1_overrides` 现在钉住全部 7 个
>   `latent_*` 字段的默认值，于是"named recipe = 确定性对照臂"按定义成立，隐变量运行一律 `custom`
>   （所有隐变量臂文件本来就这么写）。只钉 L2.f 两个杠杆是不自洽的：那样 `simple-v3` 能改 β 却不能改退火。
>   逐产物复算：无一改名、无一改 slug、无一 `from_dict` 失败。
> - **探针**：`run_ts_latent_probe.py --checkpoint LABEL=PATH --split val [--limit N] --out <dir>`——
>   逐 checkpoint 打印参考探针那张表（先验/后验的逐维跨航班 std 与中位 σ、位移的逐维中位与总中位/p90、
>   逐维 KL 及其均值/方差拆分、每航班总量、先验总 std 对 N(0,I)——总 std 由**混合分布自己的矩**算
>   （`Σπμ` 与 `Σπ(σ²+μ²)−E[z]²`，再加 E[z] 的跨航班方差；K=1 退化成分量自身的 (μ, σ²)），因为 K>1 的先验
>   量程主要在**分量之间**；参考脚本的标签这么写但只算了宽度那一半），人群按回放规矩重建
>   （`load_arm` / `cohort_series`，即 `checkpoint_data_provenance` 那条 roster 规则），拒绝 `cta=given`
>   （共用加载器，带 `instrument` 参数所以每个 runner 用自己的措辞）与无 latent 的 checkpoint，产物目录不可变。
>   `--limit N` 是**划分的前缀**不是抽样（KRDU 100 → 200 架之间位移中位动了 25 %），旗标与产物都写明，
>   限量表不得与全量数并列引用。**后验读未来，所以这里的数永远不是预测结果。**
> - **臂文件** `docs/experiments/l2f_mean_information_arms.json`（base = L2.d + fb 0.05 + β 0.01；
>   `L2f_anneal` / `L2f_aux_T`；predict `--latent-samples 6 --latent-random 6 --latent-shuffle`）。
> - **建头时量到的一件事（诚实记下，直接影响门 (1) 的读法）**：合成数据三个种子、配对初始化，辅助目标让 KL
>   的**均值项**三比三变大（102.60→108.06、26.854→26.917、15.48→15.69，轮数越多差越大），但**位移中位**只有
>   二比三变大（种子 11 反而小 0.6 %）。原因是目标的形状——一个标量读出只需要**一个**隐维方向，八维上的
>   中位数看不见它。`L2f_aux_T` 必须同时读 `component_kl_per_dim` 与 `component_kl_mean_term_nats`，
>   并在结果里说明判决靠的是哪个。

> **L2.f 结果（2026-09-08 凌晨，campaign `l2f_mean_information_20260907`）：两臂都败，L2 线的病根定案。**
>
> | 臂 | top-1 ADE | 后验均值位移（σ） | active_0.05 | raw KL（均值项 / 方差项） | shuffled ΔADE | minADE_6 / N(0,I) 对照 | 先验总 σ |
> |---|---:|---:|---:|---|---:|---|---:|
> | L2f_anneal（β 0→0.01，40 轮） | 1318 | 0.131（探针 0.049） | 1.2 | 0.31（0.21 / 0.11） | +8 | 1241 / 1164 | 0.41 |
> | L2f_aux_T（z→T 辅助头，w=1） | 1365 | 0.176（探针 0.083） | 6.0 | 0.70（0.46 / 0.24） | +21 | 1305 / 1160 | 0.27 |
> | 兜底 warm β=0.01 | **1214** | 0.079 | 2.8 | 0.17 | +19 | 1004 / 947 | 0.58 |
>
> 退火：预热期位移升到 0.23，β 到位后回落——解码器学会用 z 之后惩罚照样把均值信息压掉；辅助头：KL 与活跃维数
> 都被抬起来（信息"更多"），但均值位移仍 < 0.2 σ、top-1 反而最差、`latent_aux` 0.006 而下游无一项移动——臂文件预
> 注册的失败形态逐字兑现。探针把七个隐变量臂的共同事实摆在一起：**后验均值永远坐在先验均值上**（位移 0.05–0.18
> σ），KL 的一半是方差项；**学到的先验总 σ 只有 0.27–0.58**（N(0,I) 是 1.0）——这就是五个 campaign 里"随机 z 打败
> 训练先验"的全部原因。机制归结为量纲：损失以 (m / 10 km)² 计，位置项与辅助项能买到的增益都在 0.01 个损失单位
> 量级，而 β=0.01 下一个 nat 就是 0.01；free bits 给出的免费容量被花在收窄方差（永远可得的小增益）而不是让均值依
> 赖未来（需要编码器学到一个解码器能用的映射）。β、free bits、热启动、退火、辅助目标——五个杠杆都动过，没有一个
> 让 q(z|future) 真正依赖未来。
>
> **建议（待用户决定，2026-09-08 早）**：(1) 隐变量线**停止加臂**，warm β=0.01 保留为点估计模型（+8 % 是真实的）；
> 只留一个"量纲测试"臂作为最终判决——`position_loss_scale_m` 10 km → 1 km（位置增益放大 100 倍）、β 0.01、其余同
> 兜底；它若仍不让均值依赖未来，L2 线以负结果收官。(2) 主线转向 B 线（`2026-09-07_anytime_prediction_and_calibrated_eta_design`）：
> L3 已证明"何时"值 373 m，B1 分位数时长头 + B2 conformal 校准 + B3 `cta=self-q` 逐分位数解码，不读未来地交付到达
> 时刻区间与每个分位数一条可飞航迹——调度程序要的正是这个，且不依赖 z。(3) A 线（随机锚点 + 网格选点）今夜
> 出结果，决定多步预测的底座。
>
> **用户决定（2026-09-08 早）**：隐变量线按建议收官——只跑一个量纲测试臂 `L2z_units`（`position_loss_scale_m` 1 km、
> 其余同 warm β=0.01；臂文件 `l2_units_test_arms.json`），无论结果如何不再加臂；warm β=0.01 保留为点估计模型。
> **B 线（校准的到达时刻分布）成为主线，A 线并行**；两者的设计、门与实施顺序在
> `2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`。

**L2 量纲测试结果（2026-09-08，`l2_units_test_20260908/L2z_units`，`position_loss_scale_m` 1000、180/180 轮未早停、最佳轮 180）**

| | L2z_units | L2d_warm_β0.01 | native32 |
|---|---:|---:|---:|
| 打乱 z 的 ΔADE 全体 / 雷达引导 | **+928 / +2444 m** | ≤ 60 m | — |
| minADE_6 vs N(0,I) 对照（全体 / 雷达引导） | **1021 vs 1043 / 2084 vs 2163** | 劣于对照 | — |
| 先验总 σ / 活跃维 | 0.848 / 8 | 0.58 / — | — |
| 后验均值位移中位 σ（p90） | 0.334（1.04） | 0.079 | — |
| top-1 ADE 全体 / 直线 / 雷达引导 | 1249 / 447 / 2660 | 1214 / 402 / 2643 | 1322 / 445 / 2870 |
| FDE p50 全体 / 直线 / 雷达引导 | **688 / 498 / 1519** | 827 / 594 / 1851 | 864 / 671 / 1982 |

损失分量（最佳轮）：state 2.28、imitation 1.10、terminal 0.40、velocity 0.05、latent_kl 0.04——重建项终于与 nats 同量级，这正是量纲论断
预测的效果。读法：(1) **z 第一次真的被用了**：打乱 z 代价 928 m（此前七臂 0–60 m），六个先验样本第一次打败六个标准正态样本，
先验宽度回到接近单位——五轮 campaign 里"训练出的先验输给 N(0,I)"的谜团是先验坍缩到三分之一宽度，改量纲即解。
(2) **两条候选判据分裂**：打乱判据以 4.6 倍通过，位移中位 0.334σ 未过 1σ（p90 过）；"z 是常数"的读法在打乱代价 2444 m 面前不成立，
1σ 门是在两判据一致的臂上标定的。(3) **换来的是 FDE 不是 ADE**：终点比 native32 好 170 m（中位），路径平均持平；L2d 仍持有全体 ADE
（1214 vs 1249）。(4) 最佳轮 = 180 = 预算受限，这些数是下限。**决定（待用户）**：隐变量线不按原议"收官"，而是以量纲配方重开一臂
（更长预算），并把 z 的样本扇形当分布读——它与 C 线"轨迹分布"的目标是同一件事。

**L2.g 预注册（2026-09-08，用户决定：两种子，排队列末尾；`l2g_latent_distribution_arms.json`，campaign `l2g_latent_distribution_20260908`）**：
同 L2z_units 配方（`position_loss_scale_m` 1 km、warm 后验、β 0.01、8 维），300 轮、关早停（patience 300），种子 1337 与 2024（同分割种子）；
predict 同量纲测试（6 先验样本 + 6 标准正态样本 + 打乱）。不再叠加 26× 时长权重（第二种子已推翻其收益）。**门（两种子都要成立）**：
(1) 打乱 z 的 ΔADE > 200 m（z 在用）；(2) minADE_6 < N(0,I) 对照（全体与雷达引导）；(3) FDE p50 优于 native32（864）超过 100 m；
(4) top-1 全体 ADE 不劣于 native32（1322）超过 ~125 m 种子线。**否决**：任一种子门 1 不过（z 又死了）→ 隐变量线永久关闭。
读法：每个数报两种子散布，散布之内的差不算发现。成本 2 × ~100 min GPU + predict。

**L2.g 结果（2026-09-08 晚，`l2g_latent_distribution_20260908`，两臂各 300/300 轮、104 min，最佳轮 258 / 292）——四道门两种子全过，无否决。**

| | 种子 1337 | 种子 2024 | 两种子散布 | 参照 |
|---|---:|---:|---:|---|
| top-1 ADE 全体 / 直线 / 雷达引导 | 1230 / 430 / 2639 | **1194 / 406 / 2579** | 37 / 24 / 60 | native32 1322 / 445 / 2870；L2d 1214 |
| minADE_6 vs N(0,I) 对照（全体） | 994 vs 1030 | 959 vs 1035 | 35 | 门 2 |
| minADE_6 vs 对照（雷达引导） | 2042 vs 2159 | 1995 vs 2223 | 46 | 门 2 |
| **FDE p50 全体 / 雷达引导** | **656 / 1449** | **627 / 1225** | 29 / 224 | native32 864 / 1982 |
| **打乱 z 的 ΔADE 全体 / 雷达引导** | **+1017 / +2702** | **+935 / +2406** | 82 / 296 | 门 1（> 200） |
| 先验总 σ | 0.870 | 0.896 | — | L2d 0.583 |
| 后验位移中位（p90） | 0.336σ（1.04） | 0.312σ（1.02） | — | 1σ 门未过（非本臂判据） |

读法：(1) **量纲论断复现**：两种子全体 ADE 差 37 m（control 种子线 125 m 的三分之一），四道门在两种子上都以数倍余量通过——
不是种子运气。(2) **先验不再坍缩**（0.87 / 0.90 对 1.0；L2d 0.58），六个先验样本在全体与雷达引导上都打败六个标准正态样本，
五轮 campaign 的谜团关闭。(3) **位移中位稳定在 ~0.32σ、p90 ~1.03**，三个臂一致；打乱代价近千米——位移中位这条诊断对本工况失效，
以打乱判据为准（写入 §八 读数约定）。(4) 种子 2024 几乎处处更好且最佳轮 292，300 轮预算对它略有约束；两臂都优于 180 轮的 L2z_units。
(5) 换来的仍是**终点与分布**：FDE p50 627–656（native32 864）、best-of-6 ADE 959–994（top-1 1194–1230）；top-1 ADE 与 L2d 持平。

**决定（待用户）**：隐变量线以 L2.g 配方**重新成为主线组件**——z 是"给定历史下的未来分布"，与 B 线的分位数扇形、C 线的轨迹分布是同一交付物。
下一步候选：(a) 把 z 的 6 样本扇形按 B 线门 3 的协议读（真值落入扇形份额、扇形最近解 chamfer）；(b) CTA 条件化叠在 L2.g 底座上
（给定时间后 z 还剩多少信息，即 L3 在量纲底座上的重跑）；(c) 主线配方 `position_loss_scale_m` 改为 1 km——这是配方改动，需在
native32（无隐变量）上也测一臂，分清"量纲"与"隐变量"各自的贡献。

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
>
> **排期（2026-09-07 晚，用户决定：主线优先）**：`L3_cta` = L2.d 的 warm β=0.01 底座 + `cta_conditioning=given`
> （臂文件 `l3_cta_arms.json`，campaign `l3_cta_20260907`，预测时 `--latent-samples 6`），紧接 L2.e' 之后跑；
> 反事实扫描 `l3_cta_counterfactual_arms.json`（predict-only，±30/60/90 s，campaign `l3_cta_counterfactual_20260907`）
> 从它的 checkpoint 出发。对照 = 同底座无 CTA 的 `L2d_warm_beta0p01`：ADE 逐分层的差就是"知道到达时刻"
> 买到的那一半意图（预期雷达引导层动得最多）。
>
> **L3 结果（2026-09-07 晚，campaign `l3_cta_20260907`）。** 180 轮未早停，时长误差逐轮恒为 0（按构造）。对照同底座无
> CTA 的 warm β=0.01：
>
> | 分层 | 无 CTA ADE / FDE p50 / chamfer | **给定 CTA** | 配对胜率（中位 Δ） |
> |---|---|---|---|
> | 全体 | 1214 / 827 / 165 | **841 / 745 / 128** | 65.5 %（−57 m） |
> | 直线进近 | 402 / 594 / 84 | 392 / 586 / 66 | — |
> | 雷达引导 | 2643 / 1851 / 827 | **1596 / 1180 / 721** | **87.3 %（−896 m）** |
>
> 知道"何时"买到 373 m 的全体 ADE，几乎全在雷达引导层（−1047 m）；直线进近只动 10 m。这是误差预算里"时序
> 一半"的直接测量，也给隐变量线标了尺度：z 至今只抓到 108 m，而单单告知到达时刻值 373 m。几何改善远小于时序
> （chamfer 165 → 128，雷达引导 827 → 721）：模型在"何时"上好得多，"何处"仍是开放的一半；可飞率 8.3 %。
> **注意**：这是 oracle（真值落地时刻），841 m 是完美 ETA 的上界，不是可达的工作点；校准 ETA（B 线）是它的
> 诚实版本。反事实扫描第一次在 −90 s 失败（短航班的 CTA 变负，rollout 拒绝非正段时长）——修复：CTA 低于
> 60 s 最小剩余未来的航班跳过并在 summary 里计数（`cf4bb36`），扫描重排队列末尾。

> **反事实扫描（2026-09-08 上午，`l3_cta_counterfactual_20260907`，同一 L3_cta checkpoint，只预测）**。七个臂把真值
> CTA 平移 −90/−60/−30/0/+30/+60/+90 s；平移后 CTA 低于 60 s 最小剩余未来的航班跳过并计数（−90 s 190 架、−60 s 12、
> −30 s 4、+30 s 2），`compare_constraint_arms.py` 对七臂**交集配对（1214 架）**——所以下表每列都是同一批航班，但比全集
> 略难（偏移 0 在交集上 ADE 929 / FDE 786，全集 841 / 745；与已发表的 L3 数字比较请用全集）。
>
> | 偏移 s | \|Δ时长\| p50 | 可飞率 | chamfer p50 | 路径长度比 | xt@thr p50 | \|xt\| p95 | 垂直违反行 | ADE | FDE p50 |
> |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
> | −90 | **90.0** | 44.9 % | 1038 | 0.59 | −254 | 3615 | 41.3 % | 3039 | 5504 |
> | −60 | 60.0 | 44.2 % | 622 | 0.75 | −350 | 3063 | 41.9 % | 2072 | 3314 |
> | −30 | 30.0 | 30.0 % | 333 | 0.86 | −293 | 2607 | 44.0 % | 1344 | 1685 |
> | 0 | 0.0 | 8.3 % | 232 | 0.98 | −4 | 2598 | 47.3 % | 929 | 786 |
> | +30 | 30.0 | 0.3 % | 364 | 1.12 | 558 | 3166 | 49.4 % | 1165 | 1332 |
> | +60 | 60.0 | 0.0 % | 661 | 1.28 | 1517 | 4903 | 54.4 % | 1672 | 2074 |
> | +90 | **90.0** | 0.0 % | 1160 | 1.51 | 3156 | 8177 | 60.1 % | 2235 | 3047 |
>
> 读法：(1) **CTA 是硬约束**——每一行的时长误差精确等于偏移。(2) **几何随 \|偏移\| 单调且分级**：少给时间就飞更短
> 更直的路，多给就飞更长的路，chamfer 与路径长度比两侧都单调，±90 s 也没有坍缩——反事实在按控制输入起作用。
> (3) **不对称是结论**：提前到时可飞率反而从 8.3 % 升到 44.9 %（路更短更直），推迟到时可飞率归零——模型吸收延时的
> 唯一办法是把路拉长、在阈值处横向偏出（xt@thr p50 3.2 km、p95 8.2 km），这些状态过不了可飞性检查；垂直违反行同向
> （41 → 60 %）。它说明偏移 0 处 8.3 % 的可飞率缺口主要是路形问题不是时序问题，与上面"知道何时买 373 m 而 chamfer
> 几乎不动"一致。**未排除的混杂（待测）**：提前臂的 rollout 更短、行数更少，可飞率按行计的分母随之变小；以及推迟臂
> 的横向偏出是否会被推理期屏障 hook（L1.c 的对照臂形态）压回走廊——这正是"调度程序要求推迟时模型给什么参考"的
> 交付问题，B 线（分位数条件航迹）与 L1.c 的 hook 臂各答一半。产物：`readout.{json,txt}` 与七个 `*_pred_val/`。

#### L3.c — 推迟吸收进走廊（2026-09-08 预注册；用户决定，排在 A0.b 之后、B1_point_matched 之前）

**问题。** 反事实扫描（上表）说明：要求晚到时，模型把延时变成横向偏出走廊（+30/+60/+90 s：xt@thr p50 +558/+1517/+3156 m，
路径长度比 1.12/1.28/1.51，完全可飞份额 0.3/0/0 %）。真实飞机在走廊内只能用减速吸收延时（最后 10 NM 约 20–30 s，受最低进近
速度限制），更多的延时要在汇入五边**之前**拉长路径（trombone / 引导）或等待。调度程序最常要的恰是推迟，所以当前的
CTA 参考在推迟侧不可交付。

**臂**（`l3c_delay_corridor_arms.json`，campaign `l3c_delay_corridor_20260908`，只预测，同一 `L3_cta` checkpoint，KRDU val 1404）：
偏移 0（对照：hook 在 CTA 底座上的代价）、+30、+60、+90 s，每臂都套推理期**软屏障 hook**（09-06 采纳的安全层）。无 hook 的同偏移臂
已在 `l3_cta_counterfactual_20260907`，配对比较。成本 4 次预测 ≈ 20 min GPU。

**读数**：`compare_constraint_arms.py`（参照 = 无 hook 偏移 0）：走廊违反率、终点 |xt| p95、chamfer、路径长度比、分层 ADE/FDE；
每臂 `flyability_report.json` 的完全可飞份额（与反事实同定义）；每臂 `final_time_error_s`（CTA 服从度：hook 改命令后 rollout 可能
提前到，其偏离即"走廊内吸收不了的延时"）；`run_ts_straight_in_residual_readout.py` 对 +60 s 的 hook 臂与无 hook 臂：按剩余距离
分带的地速差（延时是否被减速吸收）。

**门（预注册）**：
1. **走廊**：+60 s hook 臂在真值已建立行上的走廊违反率不高于偏移 0 hook 臂 + 5 个点，终点 |xt| p95 ≤ 偏移 0 hook 臂的 1.5 倍
   （hook 顶住了推迟带来的横向偏出）。
2. **延时去向**：+30 s hook 臂的时长误差在 30 ± 10 s 内（30 s 可由减速吸收）；+60/+90 s 的未吸收部分只测量不设门——
   它就是交付量 X（每架飞机在走廊内可吸收的延时窗口）。
3. **可飞率**：+60 s hook 臂的完全可飞份额 ≥ 无 hook 偏移 0 臂（8.3 %）。
**否决**：偏移 0 的 hook 臂本身若比无 hook 偏移 0 臂的全体 ADE 劣超过 100 m（hook 只在 A_control_v3 上验证过，CTA 底座是新的），
则推迟臂的读数带此保留，L3.c 结论降级为"hook 需先在 CTA 底座上重标"。
**决定规则**：三门过 → 推迟侧的 CTA 参考可交付（延时由减速吸收，走廊内），X 进入 B 线的交付物；门 1 过而门 2 未过 → 延时吸收
需要"汇入前路径拉长"的自由度，作为下一个设计项（训练期版本 = L1.c 的 LPV 项换到 CTA 底座）；门 1 未过 → hook 在推迟工况下
失效，回到训练期约束（L1.c）。

**L3.c 结果（2026-09-08，`l3c_delay_corridor_20260908`，配对 1402 架）**

| 臂 | ADE | FDE p50 | chamfer | 路径长度比 | xt@thr p50 | \|xt\| p95 | 横向违反率 | 垂直违反率 | 完全可飞 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 偏移 0 无 hook（`L3_cta`） | 842 | 746 | 129 | 0.98 | −23 | 2402 | 84.4 % | 43.2 % | 8.3 % |
| 偏移 0 + hook | 840 | **582** | **70** | 0.98 | −1 | **1611** | **43.2 %** | 43.2 % | 8.3 % |
| +30 + hook | 1045 | 891 | 176 | 1.13 | −5 | 2010 | 50.1 % | 45.5 % | 0.3 % |
| +60 + hook | 1451 | 1101 | 426 | 1.30 | **16** | **2364** | 85.9 % | 50.5 % | 0.0 % |
| +90 + hook | 1894 | 1328 | 822 | 1.53 | **5** | 2332 | 98.8 % | 56.2 % | 0.0 % |
| +30 无 hook | 1051 | 1125 | 314 | 1.13 | 524 | 2975 | 96.1 % | 45.4 % | 0.3 % |
| +60 无 hook | 1502 | 1601 | 624 | 1.30 | 1539 | 4737 | 99.4 % | 50.4 % | 0.0 % |
| +90 无 hook | 2000 | 2431 | 1117 | 1.53 | 3288 | 7694 | 99.9 % | 56.2 % | 0.0 % |

门：**门 2 过**（每臂每分位数的时长误差精确等于偏移，延时全部在时间上吸收）；**门 1 分裂**（终点 \|xt\| p95 2364 ≤ 1.5 × 1611 过；
任意点横向违反率 85.9 % vs 43.2 % 未过）；**门 3 未过**（+60 s 完全可飞 0 %，与无 hook 臂逐架相同）；否决未触发（偏移 0 hook 臂 ADE 840 vs 842）。

读法（含两次追加读数）：
1. **hook 做到了它设计的事，且在 CTA 底座上免费**：+90 s 时 xt@thr p50 3288 → 5 m、p95 7694 → 2332 m；偏移 0 处横向违反率 84 → 43 %、
   chamfer 129 → 70、FDE 746 → 582，ADE 不变。它应成为 CTA 参考的默认推理层。
2. **延时的吸收方式与 hook 无关**：两个 +60 臂的逐带地速差几乎相同（10–5 km 减速约 2 m/s），路径长度比同为 1.30；20–10 km 段横向
   RMS 1–3 m——**不是蜿蜒**，是一条更长但平滑的路。
3. **不可飞的原因是失速项（99.9 % 的硬违反），且 77–94 % 的失速样本在剩余 ≥ 20 km 的外段**；阈值后的 rollout 尾巴只占 5 %（+60 臂 22 %），
   按阈值截断后完全可飞 8.3 → 10.5 %（+60：0 → 1.4 %），不是解释。最后 10 km 最低地速 84.6 m/s 高于目标 74.6 m/s——模型不在近端减速，
   它在外段（对雷达引导航班是三边/四边）把速度压到 Cl_required > Cl_max，偏移 0 时已如此（失速样本 5.2 万），+60 s 再加 34 %。
   与直线进近分解里"全体 25–20 km 带地速差 −16 m/s"一致：**模型在外段系统性偏慢**，延时被继续往那里压。
4. 可飞率作为头条指标主要在报告离跑道最远的那段（观测航迹 98.4 % 可飞是地板，模型 8–10 %），引用时须带这句。

决定规则的落点：门 1 终点条款过、门 2 过、门 3 未过，不属于预注册的三种结局中的任何一种——延时既没被减速合法吸收，也没被
推到汇入前，而是压进了模型本来就偏慢的外段。**下一个设计项（待用户决定）**：(a) rollout 的速度下限（V ≥ V_stall(n)·裕度，
包线进动力学而不是只进指标），使外段偏慢不可能发生；(b) 汇入前的路径拉长自由度（trombone），使延时有合法去处；(c) 训练期 LPV 项换到
CTA 底座（L1.c 的 CTA 版）。

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

> **实现（2026-09-07，`dev-l5`）**：拟合器
> `run_ts_control_basis_oracle.py --checkpoint <ckpt> --out <dir> [--splits train,val]
> [--init network|inverse-dynamics] [--steps 400] [--batch-size 1024]`——`--reference` 的宽度研究
> 原样保留（默认仍 600 / 256），两模式共用 `prepare_batch` / `fit_batch` / `restored_fit`，其余互不相干；
> `--splits test` 拒绝，输出目录不可变。产物 `basis_fit.json` schema `ts-basis-fit-v2-teacher`（常量
> `control/basis_fit.py::FITTED_TEACHER_SCHEMA`，写方与读方同一个），按 `flight_key` 键，带 controls /
> total_duration_s / fit_ade_m / seed_ade_m / best_step / split，顶层戳 n_segments、`duration_mode:
> uniform`、anchor_index、checkpoint 与 config 的 sha256、步数 / lr / floor / clip / init / wall time /
> 逐划分覆盖数，末尾打印逐划分 fitADE / seedADE 分位数。`--init network` 的初值是 checkpoint 自己的确定性
> 前向（`batch_contract.model_forward`，隐变量模型取先验 top-1），只取控制量，总时长仍是真值（float64——
> 数据集按 1e-6 s 比对，batch 的 float32 `final_time` 到不了这个精度）。
> 拟合前按真值时长排序再分批：稠密监督按批内最长航班补齐，训练集 p50 183 s / 最长 1454 s，
> 按划分顺序取 batch 1024 会多积分约 2.5 倍飞行秒；排序同时让产物与划分列出的顺序无关（有测试）。
> 训练侧 config 轴 `control_imitation_target ∈ {inverse-dynamics, fitted}` + `control_fitted_teacher_path`
> （CLI `--control-imitation-target` / `--control-fitted-teacher-path`；默认与四个 named recipe 的字面量都是
> inverse-dynamics；两者都进 run name：`imit-target` 与 `teacher=<目录>/<文件>`——两代表是两个运行，
> 与 `closure_labels_path` 同一条规则；728 份含 config 的存量产物重算 0 处改名）。config 先拒绝不自洽的
> 组合：fitted 无路径、有路径非 fitted、非 control 路径、`random_train_anchor`、模仿权重为 0、
> `control_state_objective` 非 true-time-position（`loss_component_names` 只在该目标下注册 `imitation`；
> 该目标又要求 native 网格与均匀时长，所以这一条同时堵住了"均匀表监督可学时长分配"的洞）。
> **表是训练期输入，不是数据集自己加载的东西**：`train.fit_model` 打开一次，交给它监督的 train / val
> 两个窗口集；`evaluate-fit`、`predict --z-from-posterior`、approach-cohort 比较等回放路径各自建窗口集
> 且**不带教师**，删表后照样跑（有测试）。校验共六项，任一不过即拒绝：加载时的 schema 与 `uniform`，
> 构建时的机场（flight key 只在机场内唯一，否则会报成"covers 0 of N"）、N、数据集实际锚点、覆盖率
> （报 a of b，无回退）、每架总时长 1e-6 s。`reference_control_supervision` 的逆动力学被表查找取代，
> 权重全 1；`control_imitation_mse` 不动。表的 path/sha256/N/锚点/机场/航班数进
> `checkpoint_metadata.json` 与 checkpoint payload 的 `fitted_teacher`——**不进 `data_provenance`**：
> 那个对象被 `evaluate-fit` / `freeze-test` 按相等比对。臂文件
> `docs/experiments/l5_fitted_teacher_arms.json`（表跑出来之前两臂会在数据集构建处失败，这是设计）。

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
