# 两重 Transformer（短时控制层 + 长程计划层 + 多机图层）可行性开发报告（2026-09-16）

> 用户提问（2026-09-16）：现在的主要方法或者是 iTransformer 把状态序列作为 token 直接预测未来，或者像
> PatchTST 把每一段作为 token，或者只预测飞行意图；有没有可能做两重 transformer？短时间内用 iTransformer
> 预测控制参数（预测时间短，应该能大幅降低 ADE）；大尺度上把每个时间段作为 token，预测长期飞行计划；
> 大尺度流程再引入 graph 网络，用多机的预测段预测多机交互（优先级靠后）。要求：搜相关研究、给数学推理、
> 写可行性开发报告，可以不止局限于航迹预测。

## 〇、状态表（压缩 context 后从这里继续）

| 项 | 状态 |
|---|---|
| 结论 | **可行，但要改前提。** "短时预测 ADE 大幅降低"是评估协议的性质，不是模型的性质（§3、§4.1）：现役 native32 在 60 s 内的 ADE 已经是 250 m（雷达引导）/ 115 m（直线），整段进近是 2870 / 445 m。两层结构真正能改的是**跟踪/表示项**（§4.3：现役规则制导把真值计划飞成 1847 m，学习型短时控制层的上限约 100–300 m）和**多模态交付**；**从锚点起整段进近的 ADE 受意图信息下界约束**（§4.4：真值汇入点+时长给定也只到 2011 m），任何架构都改不了它。图层对单机 ADE 的前置测量已经不过（L4），它的用武之地是跑道/构型/顺序这类离散决定（R 系列已量到信号）。 |
| 建议 | 按 §7 的 T0 → T1 → T2 做，T3（长历史段 token）与 T4（图层）各自先过一道**信息前置测量**再建。T1 是本方案唯一能"大幅"改数字的地方，且它同时是调度器指派计划的交付形态（v5.4）的更好跟踪器。 |
| 新测量 | `run_ts.py lead_time_error`（`experiments/lead_time_error.py`，`tests/test_lead_time_error.py`）：**误差随提前量**的曲线，逐分层。产物 `4dTrajectory/outputs/KRDU/experiments/two_tier_feasibility_20260916/lead_time_error.{json,txt}`（native32、L3_cta、state_A_threshold_enu 三臂，KRDU val）。 |
| 关键数字（KRDU val，native32，雷达引导 497 架，位移 p50） | 10 s 26 m · 30 s 148 · 60 s 632 · 120 s 1884 · 300 s 3659；ADE[0,60 s] 250 m（均值）；整段 2870。增长指数 α（p50 的 log-log 斜率）10→60 s **1.77**、60→120 s 1.58；直线进近 1.55 / 1.29。 |
| 文献 | 仓库 `docs/literature/hierarchical_prediction/`（32 篇 PDF，README 逐篇笔记 + `download.sh`），四簇：分层航迹预测、多尺度/段 token 时序 transformer、迭代 vs 直接多步预测的误差累积理论、多机图/意图输入的航空论文。§5。三句话：分层买的是模态覆盖不是点误差（MTR、QCNet）；段 token 只在长时域付钱、短时域更差（Crossformer）；本包的 control 路径在理论上是"中间预测器"，两种设定下都不是最好也不是最差（Somalwar 2026），go/no-go 取决于短时模型是否错设——动力学层没错设，意图层错设。 |
| 决定状态 | 用户 2026-09-16："按照计划开始开发和试验，每完成一个节点要 review（opus agent）"。**执行记录与最新状态见 §十**（T0(b) → T0(c) → T1 → T2，分支 `dev-two-tier`，worktree `.claude/worktrees/two-tier`，基于 `dev-two-tier-feasibility` 6e8a6a2；大改动，用户合并）。每个训练 campaign 启动前先写 `docs/experiments/intents.json` 条目（仓库规则）。 |
| commit | 报告 6e8a6a2、runner 8a03408（分支 `dev-two-tier-feasibility`）；之后的提交见 §十。 |

---

## 一、提议是什么，仓库里已经有什么

把提议拆成三层，对照现有代码：

| 层 | 提议 | 仓库里的对应物 | 已测到的数 |
|---|---|---|---|
| **第一层：短时控制层** | iTransformer 从最近的窗口预测**控制参数**，时域短（30–60 s），反复问 | `prediction_output=control`（`outputs/control/`）：iTransformer → 32 段 (推力, 坡度, 过载) + 时长 → RK4 点质量 rollout。但它一次预测**整段剩余进近**，不是短时域。`--horizon-mode window`（60 s 递推）只在 state 路径上测过 | native32：全体 1322 / 雷达引导 2870 / 直线 445 m（L−1 锚点）；window 递推 vs full 一次成形（第一代 state，KRDU 152 架）：60 s 处 656 vs 980，120 s 1247 vs 1358，300 s 4382 vs 3803；入口横向误差递推差 1.5–1.7× |
| **第二层：长程计划层** | 把每个时间段作为 token，预测长期飞行计划 | `prediction_output=plan`（`outputs/plan/`，设计 v5.5）：从 60 s 窗口预测**下一条指令**（下一个 fly-by 定位点 + 到点速度 + "前方无点"）和运行参数（T、V_mid、d_decel、V_final、h_capture），K=4 混合头；确定性制导层每 30 s 再问一次并飞出来（lockstep）。没有"长历史的段 token"——输入仍是 120 s（L=60，dt=2 s）的状态序列 | K=4 头 60 s 锚点：雷达引导 3087 / 直线 749；真值计划的制导上限（lockstep）L−1 1847 / 60 s 2417；整条真值路径（K 个定位点）1159；头的定位点在两次询问之间中位游走 766 m |
| **第三层：多机图层** | 用多机的预测段做 graph，预测多机交互 | 未建。L4 前置测量（2026-09-07）：邻机实体特征对 d_join **零增量**（R² 0.37 → 0.36），可观测的前机 ETA 与其真实落地时刻相关 0.11。R 系列：跑道方向由同时段落地解出（96–99 %），一侧由 per-runway 头 +23–32 点；R3 调度器 0 冲突但不改进时间，R3.1 不确定性调度无增益 | 见 §4.6 |

已经存在的两条测量线直接约束本提议：

- **误差预算（KRDU val 雷达引导，closure 系，2026-09-05）**：`C_pred 2197 →(962)→ C_truth_intent 1235 →(777)→ C_oracle 458 →(458)→ 0`。962 m 是"本机历史里没有的管制意图"，777 m 是给定真值 (d_join, T) 后仍未定的意图，458 m 是族+标签+重建的地板。
- **Phase 0 真值意图上界**（simple-v3，KRDU val）：喂真值汇入点，雷达引导 2858 → 2356；再喂真值时长 → **2011**；几何（chamfer）只从 942 降到 795。

## 二、评估协议先分清（否则结论会互相打架）

三种"ADE"要分开报，后面的数学分别对应它们：

| 协议 | 定义 | 谁用 | 现役数字（KRDU val, native32） |
|---|---|---|---|
| **A. 单锚点整段**（论文主指标） | 在 L−1（首个完整回看，进近入口）一次给出到入口的整条航迹；`ADE = (1/T)∫₀ᵀ e(t)dt` | 论文表格、`python -m evaluation`、所有 arm 比较 | 1322 / 雷达引导 2870 / 直线 445 |
| **B. 滚动再锚（运行时）** | 每 Δ 秒用**真实**新观测重新预测，只给到 Δ 或到下一次询问；`ADE_Δ = (1/Δ)∫₀^Δ e(t)dt` | 管制席位上的实时预测、anytime 曲线（A0）、冲突探测 | Δ=30 s：77 m（雷达引导）/ 41（直线）；Δ=60 s：250 / 115 |
| **C. 计划给定的跟踪上限（oracle）** | 把真值（或调度器指派）的计划喂给下层，看它能把计划飞到离真值多近 | 交付形态（v5.4 指派时间）、分层设计的表示项 | 规则制导 lockstep：1847（雷达引导）/ 283（直线）；32 段控制基拟合：p50 106 m |

"短时预测能大幅降低 ADE"在协议 B 下**已经成立**（250 m vs 2870 m），且对现役模型就成立；在协议 A 下它**不能成立**——A 的 ADE 是对整段积分，哪个模型都不能靠缩短自己的时域来降它。两层结构的价值必须在 A 和 C 上分别量。

## 三、测量：误差随提前量怎么长（新 runner）

`run_ts.py lead_time_error`：对每架航班，把预测状态与观测状态（同一锚点时钟）插到 1 s 网格上，取 3D 位移 `e(h)`；每个提前量 h 只统计预测与真值**都到达 h** 的航班（缺席不计 0、不 hold 最后一点——这与 `evaluation` 的整段 ADE 不同，那个把最后一点 hold 到真值结束）。分层用 `approach_difficulty.strata_masks`。

KRDU val（1404 架），三臂：

| 提前量 h | native32 雷达引导 p50 / 均值 | native32 直线 p50 / 均值 | L3_cta（真值时长给定）雷达引导 p50 | state 基线 雷达引导 p50¹ |
|---:|---:|---:|---:|---:|
| 10 s | 26 / 35 | 13 / 18 | 27 | 593 |
| 30 s | 148 / 197 | 73 / 104 | 155 | 521 |
| 60 s | 632 / 714 | 215 / 282 | 535 | 683 |
| 90 s | 1185 / 1599 | 424 / 512 | 894 | 843 |
| 120 s | 1884 / 2287 | 525 / 648 | 1265 | 1109 |
| 180 s | 2496 / 2966 | 865 / 1243 | 2305 | 1673 |
| 300 s | 3659 / 4343 | — | 1383 | 3425 |
| ADE[0,30 s] 均值 | 77 | 41 | 79 | 813 |
| ADE[0,60 s] 均值 | 250 | 115 | 240 | 847 |
| 整段 ADE 均值 | 2870 | 445 | 1596 | 2599 |
| α（10→60 s / 60→120 s） | **1.77 / 1.58** | 1.55 / 1.29 | 1.66 / 1.24 | 0.08 / 0.70 |

¹ `airport_frame_20260903/A_threshold_enu`（state 输出，2026-09-03 的 val 名册 2104 架，与另外两臂不配对，只作形态对照）。

四条读法：

1. **控制路径的误差从 0 起长，state 路径从 ~400–600 m 起**：state 模型第一步就带着已知的 ~250 m 西北端点平移和节点锯齿（α≈0.1 是地板不是增长）。这是"用控制参数做短时层"的第一个硬理由：rollout 从锚点状态积分，短提前量的误差天然小。
2. **增长是超线性的，α≈1.6–1.8**。物理读法：航向偏差 δψ 给 `e ≈ V·δψ·h`（α=1），转弯率/加速度偏差给 `e ≈ ½·V·δω·h²`（α=2）。α 落在 1.6–1.8 说明 h² 项占主导——短时误差是**转弯（或减速）开始与否**的决定误差，不是稳态航向/速度偏差。雷达引导 60 s 处 p50 632 m 折成等效侧向加速度 ≈ 0.35 m/s²，≈ 2° 坡度的一分钟平均——即"这一分钟里到底转不转"在人群上的平均效应。
3. **给定真值时长（L3_cta）不改短时误差**（60 s 内两臂几乎重合），只在 180 s 以后把误差拉回来（时间钉住了终点）：时长是长程量，短时层拿它没用。
4. **直线进近也是超线性**：末 5 km 的沿航迹散布是减速时机（`2026-09-08_straight_in_residual_readout`：减速点 p10/p90 ±1.4 km，解释末段方差 22 %）。同样是决定，不是动力学。

## 四、数学推理

记锚点 t=0，真值 `x(t)`，剩余时长 T，历史 H₀；一次成形预测 `x̂(t|H₀)`，位移 `e(h)=‖x̂(h|H₀)−x(h)‖`。

### 4.1 短时 ADE 为什么低——按构造，与模型无关

测得 e(h) 在每个分层里到 ~300 s 都单调递增。于是 `ADE_h = (1/h)∫₀ʰ e` 也随 h 递增，对任何模型 `ADE_Δ ≤ ADE_T`（Δ ≤ T）。用幂律 `e(h)=c·hᵅ` 代入：

$$\mathrm{ADE}_\Delta=\frac{c\,\Delta^{\alpha}}{\alpha+1},\qquad
\frac{\mathrm{ADE}_\Delta}{\mathrm{ADE}_T}=\Big(\frac{\Delta}{T}\Big)^{\alpha}\quad(\text{同一模型})$$

Δ=60 s、T≈330 s（p50 时长）、α≈1.6：比值 ≈ 0.065——与测得的 250 / 2870 ≈ 0.087 同量级（整段 ADE 还含 hold 最后一点的部分）。**这就是"短时 ADE 大幅降低"的全部来源**：它由 Δ/T 决定，换架构不换协议不会再多降。论文里能报的模型性质是**固定 h 的 e(h)**（或固定 Δ 的 ADE_Δ）在模型之间的差，不是 ADE_Δ 与 ADE_T 的差。

### 4.2 开环递推（没有新观测）：累积项

把 Δ 时域的模型接成链，第 k 段把自己的输出当历史。设一次成形在**真实**历史上的误差 ε=e(Δ)，模型对历史扰动的敏感度（Lipschitz 常数）为 L，则

$$e_k\le \varepsilon+L\,e_{k-1}\;\Rightarrow\;
e_k\le \varepsilon\,\frac{L^{k}-1}{L-1}\;(L\ne1),\qquad e_k\le k\varepsilon\;(L=1).$$

与一次成形 `e(kΔ)=c(kΔ)ᵅ=kᵅ·ε` 比：L=1 时链式赢当且仅当 `k < kᵅ`，即 α>1 时**恒赢**，优势 `k^{α−1}`（α=1.6、k=5：2.6×）。但链式赢的两个条件在这个数据上都不满足：

- **L>1**。rollout 从锚点状态积分，锚点位置/航向的偏差整段带着走，L ≥ 1 + (速度/航向敏感度)·Δ。第一代 state 路径实测（README，record 口径，KRDU 152 架）：60 s 递推链在 60 / 120 / 300 s 处 656 / 1247 / 4382 m，一次成形 980 / 1358 / 3803 m——链从第 1 段到第 5 段增长 6.7×，按上式反解每 60 s 一段的有效 **L≈1.14**（接近线性累积 k·ε），而那个模型的一次成形只按 k^0.84 长，所以 5 段处链已经输了，入口横向差 1.5–1.7×。对 α≈1.6–1.8 的控制模型，若 L≈1 且每段 ε 保持真历史上的值，链式本可赢 k^{α−1}≈2.6×（k=5）；输在下一条。
- **链上的 ε 不是真历史上的 ε**。第 k 段的历史是自己画的平滑输出，里面没有"转弯要开始了"的迹象；决定型误差在每段被**重复**而不是被平均（各段误差完全相关，`e_k = Σ` 而不是 `√k·ε`）。README 记的现象正是这个："递推出来的路径平滑保守、可飞（73.7 %）、不在飞机去的地方"。

所以"短时层 + 递推"本身不产生信息；它把超线性的 α 换成线性的 k 和一个 L^k 的惩罚。

### 4.3 两层分解：把累积项变成计划项

设第二层给出计划 `p`（下一条指令 + 运行参数，或一串段 token），`x_p(t)` 是把计划 p **完美执行**出来的航迹，`p*` 是真值自己的计划（标签）。三角不等式：

$$e(h)\;\le\;\underbrace{\|\hat x(h)-x_{\hat p}(h)\|}_{e_{\rm track}}
\;+\;\underbrace{\|x_{\hat p}(h)-x_{p^*}(h)\|}_{e_{\rm plan}}
\;+\;\underbrace{\|x_{p^*}(h)-x(h)\|}_{e_{\rm repr}}.$$

- `e_repr`：计划**语言**的表达上限（真值计划完美执行 vs 真值）。
- `e_plan`：预测计划而非真值计划的代价。
- `e_track`：第一层对自己计划的跟踪误差。

**第一层是闭环跟踪器**这一点是分解的全部意义：若第一层每 Δ 从自己的 rollout 状态重新问、计划固定，且它向计划**收缩**（每步把与计划的偏差乘以 ρ<1，像制导律那样），则

$$e_{\rm track}(h)\le \frac{\varepsilon_1}{1-\rho}\quad\text{与 }h\text{ 无关},$$

累积项消失，整段误差 ≈ `e_plan + e_repr + O(ε₁)`。这就是分层把 §4.2 的 `L^k` 变成常数的机制（也是 MPC 与 DAgger/DaD 一类"在自己状态上训练"的机制——v5.2 已经在 plan 路径上量到：不在自己飞出的窗口上训练，每 30 s 再问一次的头 5391 m；训练在滚动窗口上（share 0.75）→ 3641 m）。

仓库里三项各自的现值（KRDU val，雷达引导）：

| 项 | 现值 | 来源 |
|---|---|---|
| `e_repr + e_track`，规则制导（下一条指令语言，lockstep） | **1847 m**（L−1）/ 2417（60 s） | 设计 §12.5 |
| `e_repr + e_track`，规则制导，整条真值路径（K 定位点） | 1159 m | §12.3 |
| `e_repr + e_track`，32 段控制（学习型语言，逐航班拟合） | **p50 106 m**（p90 737） | L5.a `basis_fit` |
| 端到端（K=4 计划头 + 规则制导） | 3087（60 s）/ 3143（L−1） | §12.8 |
| 一次成形控制模型（无分层） | 2870（L−1） | L1 native32 |

读法：现役分层（plan 路径）比不分层（control 路径）**更差** 270 m，差在 `e_plan`（头的定位点 chamfer ~1.4 km、两次询问间游走 766 m）和 `e_repr+e_track`（规则制导把真值计划飞成 1.8 km：拐角按固定半径、速度按固定律，"一半航班在 85 m 内，另一半 ~900 m"）。**学习型第一层能改的是后者**：从 1.2–1.8 km 拉向控制基的 100–300 m。这是两层方案里唯一能"大幅"改的数，而它属于协议 C，是 oracle（读了真值计划），其交付价值在于**调度器指派的计划**（v5.4 已交付指派时间，代价正是制导层丢掉的 700 m chamfer）。

### 4.4 计划项的信息下界：为什么协议 A 改不了多少

`e_plan` 的期望不低于"从 H₀ 估计 p" 的 Bayes 误差；换架构不换输入，这个下界不动。量化：

- Phase 0：喂真值 (d_join, T)，雷达引导 2858 → 2011（−847）。这 847 m 是任何只看本机 120 s 历史的模型**拿不到**的（它需要管制排序决定）。
- 误差预算：962 m 意图 + 777 m 给定意图后仍未定的部分。

于是协议 A 下两层模型的雷达引导 ADE 下界约 **2.0 km**（真值意图上界）；从 2870 出发，可争取的最多 ~600–850 m，其中一部分本来就是种子噪声（控制路径 ~125 m）。直线进近：现值 445，制导上限 194–283、控制基 ~106，减速点 oracle 估计 400 → 330——可争取 ~100–250 m。**这是可行的、但不是"大幅"的**，而且要用两个种子读。

### 4.5 每一层该"迭代"还是"直接"

多步预测理论（Marcellino–Stock–Watson 2006；Ben Taieb 2012；Chevillon 2007；2025 "Learning with Imperfect Models"）的一致结论：**一步模型正确设定（状态是 Markov 的）时迭代更有效；错设（缺状态、非 Markov）时直接多时域更好**。套到两层：

- **第一层的对象是动力学**：给定控制，点质量状态是 Markov 的、模型是精确的（RK4 内部已经在 0.5 s 步长上迭代）。第一层**迭代**（rollout + 每 Δ 再问）是对的。
- **现役 control 路径在这个理论里是"中间预测器"**（单步模型用多步损失训练，Somalwar 2026，§5.3）：设定正确时次于单步、错设时次于直接——两种情形下都居中。两层拆分就是把它按设定正确/错设拆开，各取所长。
- **第二层的对象是管制意图**：相对本机状态**非 Markov**（缺的变量是排序/指令序列），按理论应当**直接**多时域、多模态（MTR / FlightBERT++ 的形态：K 个意图 query 一次给出，不做段级自回归）。段级自回归 M 段累积 `M·ε₂`（L₂≈1），直接预测付 `ε₂'(M)`，本数据上递推与一次成形的交叉在 3–5 段（§4.2）——M ≥ 3 时直接更好。v5 的"一次一条指令、每 30 s 再问"其实就是"直接预测下一个 token，滚动"，与理论一致；把它推广成"直接预测下 M 个段 token 的 K 个模态"是第二层的合理形态，而不是长自回归。

### 4.6 图层：先验条件

图层 `g(p̂₁…p̂_N, scene)` 只有当**别机在 t₀ 的可观测状态携带本机计划的信息（超出 H₀）**时才有增量。已量：

- L4（KRDU 14,418 架，同 Phase 0 协议）：d_join R² 本机 0.34 → 粗上下文 0.38 → 4 架邻机实体行 **0.36**（零增量）→ 真值前机 ETA 0.47。可观测前机 ETA 与其真实落地时刻相关 **0.11**：前机的剩余时间和本机一样没定，它是同一个排序决定的**另一个结果**，不是本机能观测到的原因。
- R3.1：按 ETA 不确定性调度，无增益；"落地时间误差在每架飞机自己的 ETA 里，不在飞机之间"；真值在 99.3–100 % 的相邻对上保持间隔最小值——联合约束几乎总被边缘分布满足，图学不到多少。
- 有信号的地方：跑道**方向**（同时段落地 96–99 %）、平行跑道**一侧**（R1 +23–32 点，`r11_lift`）、构型日（KSJC 关闭日）。这些是离散决定，且已经有树模型在做。

因此：图层对单机 ADE 的前置门 = "把**前机真值计划 token**当输入的 oracle 臂（像 Phase 0 那样）能把雷达引导 ADE 配对改善 ≥ 300 m"；不过就不建。若过，再用"一机然后全体"（one-then-all：邻机用同一模型预测的 token）替换真值。图层对**跑道/构型/顺序**是 R 系列的自然后继（树 → 集合编码器），门沿用 §14（KSJC 关闭日 ≥ r11_lift）。

## 五、文献（`docs/literature/hierarchical_prediction/`，32 篇 PDF；README §1 逐篇笔记、§2 跨簇模式、§3 反例、§4 未取得的 4 篇）

### 5.1 簇 A：分层 / 两段式航迹预测——分层买的是模态覆盖，不是点误差

- **QCNet**（CVPR 2023，Table 5）：同一模型两种时域。Argoverse 2（6 s）：一次成形 1 段 minFDE₆ 2.10 → 3 段 2 s 的分段提议 2.02 → 再加整段一次成形 refinement **1.90（−13.6 %）**；3 段 → 6 段逐位相同；Argoverse 1（3 s）几乎不动（0.92 → 0.89，作者：3 s 内场景不怎么变）。读法：段数过 3 无益，收益在"提议 + 整段一次 refinement"，且只在长时域出现——第二层 M=3 段够用，第一层不是"更多段"。
- **MTR**（NeurIPS 2022，Table 3，WOMD 8 s）：flat latent query minADE 0.6829 / mAP 0.2633；静态意图 query 0.7036（**minADE 更差**）/ mAP 0.3059；全栈 0.6697 / 0.3437。**MTR++** Table 5：flat latent embedding 的 minADE 0.6564 是五种解码器里**最好**的。→ 意图分层买的是覆盖（mAP），不是点误差；与 §4.4 "962 m 的交付形态是分布"一致。
- **TNT / DenseTNT**（Argoverse 3 s）：目标点 → 轨迹两段式，TNT 0.73/1.29 vs MultiPath 0.80/1.68；DenseTNT minFDE −19 %、minADE 不变。**C2F-TP**（NGSIM/highD 5 s）：粗到细去噪把 FDE 降 34–60 %，ADE 只降 2–19 %。→ 两段式改的是**终点**，不是整段平均：协议 A 里 FDE 与 ADE 要分开读。
- **MultiPath++**（Table 4）：学习型 anchor 2.305/0.978 vs 静态 k-means anchor 2.99/1.22；控制量输出 vs 原始坐标：误差相同（2.319/0.987 vs 2.305/0.978），运动学不可行率 → **0.00 %**。→ 与本包 control 路径"可飞按构造"一致，控制量参数化不付精度代价。
- **ThreeStepHT**（行人，JTA 4.8 s）：加社交层 ADE 1.35 → 1.02（−24 %）。行人群里交互是近邻驱动的；终端区里交互经管制员中转、L4 量到零增量——同一架构结论不能搬。
- HierarchicalIL（驾驶）无同架构 flat 对照，分层贡献未分离；**PlanT** 的价值在 token 设计（每段路线一个 6 属性有向框 token，固定个数、长度截断）——第二层"段 token"的具体形态可以照抄。

### 5.2 簇 B：多尺度 / 段 token 时序 transformer——段 token 只在长时域付钱，短时域反而变差

- **Crossformer**（ICLR 2023，Table 2，ETTh1 MSE）：段 token（DSW）+ 层级 encoder-decoder（HED）：τ=24 0.373 → **0.406（更差）**，τ=48 0.456 → 0.493（更差），τ=168 0.947 → 0.614，τ=720 1.086 → 0.841（−23 %）；段长 4 → 24 只在 τ ≥ 168 降 MSE。→ 段 token 属于第二层（我们整段 300 步），第一层（30 步）用它会更差。
- **Pathformer**（ICLR 2024）：多 patch 尺寸 {2…32} + 按输入路由，vs PatchTST −8.1 % MSE，去掉路由是最大的消融损失。**TimeMixer**（ICLR 2024）：vs PatchTST ETTh1 0.447 vs 0.516；**交换尺度间混合方向** 0.390 → 0.412——信息流的方向与"有没有尺度"同样重要（我们的方向：粗 → 细条件化；细 → 粗只在训练时经滚动窗口）。
- **Scaleformer**（ICLR 2023）：粗到细迭代 refinement −5.5 … −38.5 %，但作者明说迭代多尺度会 "runaway error propagation"，跨尺度归一化只为止住它。→ 分层不是把累积换成计划误差就完事，它**新开一条跨尺度的累积通道**（第二层的错进第一层的条件）——§7 风险 2 的计划 dropout 为此而设。
- **Pyraformer** 自己的最好解码器是**一层全连接**；Crossformer / Scaleformer 都早于 PatchTST / iTransformer 且未与之比较；Pathformer 的表里 Pyraformer ETTh1-720 0.963 vs PatchTST 0.495——多尺度线被 flat patch 模型超过，之后只赢 8 %。

### 5.3 簇 C：迭代 vs 直接、误差累积理论——公式

- **Ross & Bagnell 2010** Thm 2.1：`E_{s∼d_π*}[e_π̂(s)] ≤ ε ⟹ J(π̂) ≤ J(π*) + T²ε`（紧）。**DAgger** Thm 2.2：ε 在学习者**自己**的分布上量、专家 cost-to-go 优势 ≤ u，则 `J ≤ J* + uTε`。Kääriäinen 的精确计数 `T/2 − (1−(1−2ε)^{T+1})/(4ε) + 1/2 = Θ(T²ε)`。
- **DaD**（Venkatraman 2015）Thm 1：一步误差 ε、Lipschitz L ⟹ `‖M̂(x̂_T) − x_{T+1}‖ ≤ ε Σ_{t=0}^{T} Lᵗ`：L>1 指数、L=1 O(T)、L<1 O(ε)（但预测衰到均值）；Thm 3：Õ(T) 轮后 O(T·ε_N)。不改架构：单摆 rollout RMSE −45 %、cartpole −57 %、直升机 −43 %。→ §4.2 的界与 §4.3 "在自己状态上训练"的出处；v5.2 的滚动窗口训练（5391 → 3641）就是 DaD 的一轮。
- **Somalwar 等 2025/2026**（arXiv 2504.01766 / 2603.23465）：设定正确时**单步 ≻ 中间 ≻ 直接**（统计效率，差距随 H 线性增长，闭环里单步 rollout 的 LQR 代价更低）；错设时次序**完全反转** `B(MS) ≤ B(I) ≤ B(SS)`。"中间" = 单步模型用多步损失训练——**正是本包的 control 路径**（段控制头穿过 RK4 rollout 训练）：两种情形下都不是最好也不是最差。→ 文献给的 go/no-go 不是"分层是否更好"，而是"**短时模型对这份数据是否错设**"。§4.5 的答案：动力学层设定正确（Markov、精确），意图层错设（缺排序变量）——所以拆成两层各取其优。
- **Marcellino–Stock–Watson 2006**（171 个美国宏观月度序列）：h=24 时迭代在四种滞后规则下**全胜**，直接/迭代 MSFE 比的 90 分位 > 1.2；直接只在一步模型滞后短且错设时赢（0.86），**把滞后加长到 p=12 后优势归零（1.00）**。→ §7 T0(c) 的理论理由：先加长回看（120 s → 10 min）量信息，长回看可能直接消掉"需要直接多时域"的理由。**Chevillon 2007** §8.1：直接法成立的错设是"未察觉的单位根、短时域被忽略的残差相关、结构突变"；平稳过程里收益随时域迅速消失。
- **Ben Taieb 2012**（NN5，H=56）：MIMO 21.92 < DIRMO 22.36 < REC 24.22 < DIR 24.48 ≪ DirRec 44.97；MIMO 在 12 种配置里全胜。**Stratify**（1080 次实验）：五种经典策略统计上不可区分，最优策略随数据集变。
- **FlightBERT++**（AAAI 2024，8,643 条 ADS-B，步长 20 s，MDE km，h = 20 s / 1 / 3 / 5 min）：自回归 Transformer 0.65/1.21/2.85/4.50，直接 Transformer-Seq2Seq 0.67/0.74/1.28/2.01，直接 FlightBERT++ 0.22/0.39/0.97/1.54。**直接法在最短时域付 3 %，从 1 min 起赢回**；1 → 15 步的增长自回归 ×6.9、直接 ×3.0。**WTFTP+**（Nat. Commun. 2026，同语料，NM）5 min 处与 FlightBERT++ 相同（1.06），发表的 −40 % 是对自己的自回归前身。→ 与本包 README 的 window vs full 交叉点（120–300 s）同形。

### 5.4 簇 D：多机交互与意图输入

- **SIA-FTP**（Nat. Commun. 2024，口语指令进预测，同语料同骨干）：MDE 0.15/0.29/0.77/1.22 vs 无指令 0.16/0.33/1.03/1.55 → **−6 / −12 / −25 / −21 %**（20 s / 1 / 3 / 5 min）；5 min 高度 MAE −44 %；作者解释 20 s 处无增益："指令在第一个时域内可能尚未执行"。Table 2 消融：删掉整个指令预训练阶段仍得 0.16/0.30/0.80/1.27——收益在**有指令**，不在训练法。→ 与 Phase 0（真值汇入点：雷达引导 ADE −17 %、几何 −15–20 %）和 §3 读法 3（时长对 60 s 内无用）同形：意图值钱在 ≥ 1 min。
- **Trajectron++**（已在 `control_normalization/`）：动力学积分 FDE −3 %、NLL 大改善；**给定已知的未来计划（机器人自车轨迹）FDE @4 s 2.09 → 1.50（−28 %）**；交互项本身从未消融。→ 协议 C（指派计划的跟踪）在文献里的对应数字。
- **B-STAR / S-STGCNN / FPG-SLSTM / TrajAirNet**（航空多机）：**没有一篇分离出交互项**（无同模型单机消融）；前三篇的形态是"先分类飞行模式再按模式建联合模型"——与本仓库"先跑道/意图头，再按跑道专家"同形（R 系列）。Neurocomputing 2024 的"本地历史意图"是**检索**而非预测（按落地时间与方向筛历史邻居）——用检索给第二层计划的最近类比（`jepa/` 评估 §4.2 也提过）。
- **AgentFormer / Scene Transformer**：联合损失 / 联合隐变量在**联合指标**上 −12–26 %（minSADE/minSFDE）；Scene Transformer 说 marginal → joint 只是损失的一个开关。→ 图层若建，它的指标是联合的（顺序、对间间隔），不是单机 ADE——§4.6 与 §8 第 2、5 条。

### 5.5 文献对 §4 的修正与加强

1. §4.2/§4.3 的界有出处（DaD Thm 1、DAgger Thm 2.2），且 DaD 的经验数字说明"在自己状态上训练"本身值 40–57 %——T1 的滚动窗口训练是**必需**的，不是可选。
2. §4.5 的"第一层迭代、第二层直接"被 Somalwar 的"中间预测器"框架精确化：现役 control 路径已经是"中间"，两层拆分是把它分成"设定正确的单步（迭代）"与"错设的直接（多模态）"。
3. 新增风险（§7 6–8）：跨尺度误差传播（Scaleformer）；段 token 在短时域有害（Crossformer τ=24）；分层可能让 top-1 更差（MTR）。
4. T0(c) 有了理论理由（MSW：长滞后消掉直接法的优势）。
5. 图层的目标函数必须是联合指标；航空文献没有一篇量过交互项本身——本仓库的 L4 前置测量在这个领域里是少见的直接测量，其结论（零增量、前机 ETA 相关 0.11）值得写进论文。

## 六、可行性结论与预注册的门

| 问题 | 结论 | 门（KRDU val，两个种子，配对） |
|---|---|---|
| 短时控制层单独能否"大幅降 ADE"（协议 A） | **否**。ADE_Δ 低是协议性质（§4.1）；开环递推不产生信息（§4.2） | 无门；改口径 |
| 学习型第一层能否把**真值/指派计划**飞得比规则制导近（协议 C） | **能，且是最大的可改项**（1847 → 数百米） | G1：雷达引导 ADE@L−1 < 1000 m、直线 < 200 m；完全可飞 ≥ 95 %；走廊内建立 ≥ 制导层的 88 % |
| 两层端到端在协议 A 上是否好于一次成形 | **可能，幅度 ≤ 600–850 m（雷达引导）、≤ 250 m（直线）**，下界 ~2.0 km | G3：雷达引导 ≤ 2745（2870 − 125）且直线 ≤ 415（445 − 30）；完全可飞不低于 plan 路径；建立 ≥ 94 % |
| 第二层用长历史段 token 是否有信息 | **未知，先量**（现在只看 120 s） | G2：coarse token 的 10 min 历史 vs 120 s，雷达引导 ADE@共同锚点改善 > 125 m（两种子）；不过则第二层 = 现役 plan 头 |
| 第二层多模态 | **是交付形态**（962 m 的意图不是 top-1 能给的） | 沿用 §12.8：minADE_4 vs 5 km 环对照；覆盖 vs 宽度 |
| 图层（单机 ADE） | **门未过，不建** | 前置：前机真值计划 token oracle 配对 ≥ 300 m |
| 图层（跑道/构型/顺序/间隔） | **值得**，R 系列后继 | §14 门 |

## 七、开发方案（每步：写代码 → opus review → 修 → 实验 → 记录 → 提交）

复用清单（不重写）：`outputs/control/`（头、包络、rollout、监督）；`outputs/plan/{labels,extractors,rolled,forecast}`（计划 token、`TruthQueue`、`rolled_history`、`fly_lockstep`）；`outputs/dynamics/*`、`outputs/constraints/*`（走廊屏障、速度下限）；`experiments/plan_rolled_windows`（滚动窗口表）、`plan_oracle`（lockstep 仪器）、`anytime_curve`（协议 B）、`lead_time_error`（本文）；B 线分位数时长头 + 校准；`run_naming`、`intents.json`。

| 步 | 内容 | 门 / 产物 | 成本 |
|---|---|---|---|
| **T0(a)** 误差随提前量 | **已做**（§3） | `two_tier_feasibility_20260916/lead_time_error.*` | 0 |
| **T0(b)** 现役控制模型的链式敏感度 L | predict 侧：native32 从 L−1 起每 Δ=60 s 用自己的 rollout 状态续窗口（`rolled_history` 的机制）再预测，读 e(120/180/300) 对一次成形；反解 L | 量 §4.2 的 L；决定第一层是否必须用滚动窗口训练（v5.2 的答案是必须，这里再在控制路径上量一次） | CPU，1 天 |
| **T0(c)** 长历史信息测试 | `history_ablation` 在控制路径：seq_len 60 / 150 / 300（dt 2 s）于共同锚点 `max(L)−1`，两种子；或 30 s 段 token 的 10 min 变体 | G2 | 6 臂 × ~65 min GPU ≈ 6.5 h |
| **T1** 学习型第一层（真值计划下） | 新的 ControlOutput 变体：输入 = 60 s 窗口 + 计划 token（`targets_from_labels` 的向量 + 下一定位点，**训练时以 0.5 概率遮掉**——计划不可靠时退回窗口）；输出 = Δ=60 s 内 N₁=4–8 段控制；在**滚动窗口表**上训练（share 0.75）；`fly_lockstep` 里把 `PlanGuidance` 换成这个头做跟踪器，走廊屏障照旧合成 | G1；`step_t1_*` 产物；`intents.json` 条目 | 2 臂 × 2 种子 × ~2 h GPU ≈ 8 h；代码 ~1 周 |
| **T2** 端到端（K=4 计划头 → 第一层） | 第二层沿用现役 K=4 混合头；报协议 A / B / C 三张表；fan：K 个模态各飞一条 | G3；§12.8 的 fan 读数 | 训练同 T1；1 周 |
| **T3** 第二层长历史段 token（**仅当 T0(c) 过 G2**） | PatchTST 式 30 s 段 token（每段：航向/速度/高度/转弯率/剩余路程），10 min 历史 → 计划头编码器；直接预测下 M 个段 token 的 K 模态（非自回归） | G2 复测 + G3 | 1–2 周 |
| **T4** 图层（**仅当前置 oracle 过**） | 先做前机真值计划 token 的 oracle 臂；过了再做 one-then-all 的集合/图编码器；对跑道/构型/顺序另按 R 系列门 | §4.6 的门 | 前置 1 天；编码器 2 周 |

风险与否决（预注册）：

1. **种子线**。控制路径 pooled ADE 种子差 ~125 m、直线 FDE/时长小得多；任何 G1/G3 读数必须两种子。
2. **计划 dropout 的副作用**：第一层学会忽略计划（遮得太多）或过度依赖（遮得太少）——在 T1 里各测 0.3 / 0.5 / 0.7，读"遮掉计划后 e(30)/e(60) 的退化"。
3. **第一层在自己状态上的分布漂移**：v5.2 的教训（DAgger 第一轮无益、教它命名越界的定位点）——用真值策略的滚动窗口表，不加自己的状态。
4. **协议混报**：论文只能把协议 A 作主表，B 作 anytime 曲线，C 作交付上限并注明 oracle；一句"ADE 从 2870 降到 250"是错的。
5. **T3 的信息门不过**：则第二层没有"长程"可言，方案退化为"学习型跟踪器 + 现役计划头"，仍值得做（G1 的价值独立于 G2）。
6. **跨尺度误差传播**（Scaleformer 的 "runaway error propagation"）：第二层的错计划进入第一层的条件后被忠实地飞出来——比无条件的一次成形更差是可能的结局。T2 里除了 dropout，还要读"计划遮掉 vs 不遮"的配对差，负值即触发。
7. **段 token 在短时域有害**（Crossformer τ=24：0.373 → 0.406）：第一层永远用原始 2 s 状态序列，段 token 只进第二层。
8. **分层让 top-1 更差**（MTR 0.6829 → 0.7036）：G3 读 top-1 与 minADE_K 两个数，只有 minADE_K 改善而 top-1 退化时，交付形态改为分布，不算失败也不算通过。

## 八、不止航迹预测：这两层还能拿去干什么

第二层的输出是**指令语言**（下一个定位点、到点速度、时间、汇入），第一层是**能飞的短时执行器**。这两样东西在本仓库里已有的或可直接接的用途：

1. **4D 调度 / CTA 交付**（已建 v5.4/v5.5）：调度器把指派的时间与汇入写进计划 token，第一层把它飞出来。T1 的价值直接落在这里：规则制导丢掉的 700 m 拐角误差、"一半航班 ~900 m"的那一半。
2. **间隔 / 冲突概率**：两机的计划 token 放到进近时钟上（`inference/runway_schedule`，FAA 7110.65BB 的最小值），用 B 线的校准 ETA 分位数与 K 模态 fan 给出每对的失距概率。R3 的结论（ETA 误差 ≈ 最小值本身）说明它该做成**不确定性显示**，不是确定的冲突清单。
3. **管制指令识别 / 推荐 / 符合性监视**：第二层就是"下一条 vector 是什么"。文献里指令是**输入**（口语指令进预测，Nature Comms 2024）；这里指令是**输出**——预测指令与实际机动的分歧 = 非标准指令或未执行的告警；接上 ATC 语音/CPDLC 数据后成为 conformance monitor。
4. **复飞 / 异常机动的实时检测**：第一层的短时残差 e(30) 超出校准带（雷达引导 p90 374 m、直线 189 m）即告警：复飞、等待、中断进近。这是协议 B 的直接副产品，现役 native32 就能出。
5. **跑道构型 / 进场顺序预测**：图层真正有信号的任务（§4.6），R 系列的后继；顺序分布 + 一侧概率给排序器做先验。
6. **流量 / 容量仿真（数字孪生）**：第二层作为进场流的**生成模型**——给定构型采样整队的计划，第一层飞出来，得到吞吐量/延误分布；前端已能动画 fan（R2d 的 28 个变体）。
7. **程序设计反馈**：计划 token 的抽取器（`outputs/plan/extractors`）就是"TMA 实际怎么飞"的自动标注：各跑道的下风长度、三边转弯距离、减速点分布——对着 PANS-OPS/OCS 的程序几何做统计，是论文数字孪生一章的材料。

## 九、名词对照

| 本文 | 仓库 |
|---|---|
| 第一层 / 短时控制层 | `prediction_output=control` 的 60 s 变体（T1）；`outputs/control/` |
| 第二层 / 计划层 / 计划 token | `prediction_output=plan`，`outputs/plan/labels.TARGETS`，`PlanOrder`，K=4 `plan_fan_components` |
| 滚动窗口表 | `run_ts.py plan_rolled_windows`，`plan_rolled_share` |
| lockstep / 再问 | `outputs/plan/forecast.fly_lockstep`，`LOCKSTEP_S`=30 |
| 协议 A / B / C | `python -m evaluation` 整段；`run_ts.py anytime_curve` + `lead_time_error`；`plan_oracle --policy truth` / `--assign-time truth` |
| 误差随提前量 e(h)、α | `run_ts.py lead_time_error`，`p50_log_log_slope` |
| 意图上界 | Phase 0 `intent_conditioning=truth-*`（FROZEN，只作 oracle） |
| 图层前置 oracle | 新臂：前机真值计划 token 作 `intent_conditioning` 类输入 |

## 十、执行记录（2026-09-16 起；压缩 context 后从这里继续）

分支 `dev-two-tier`（worktree `.claude/worktrees/two-tier`，数据目录是指向主树的软链接；大改动，**用户合并**，完成后报告能否 `git merge --ff-only`）。每个节点：写代码 → 单测 → opus review（只审代码）→ 修 → 提交 → 实验（opus 队列 agent 串行跑，按 PID 看）→ 读数 → 记录 → 发布（带 intent）。实验原则：只用 val，不碰 test；单机场 KRDU（均衡采样器未被行使，读数里写明）；正式 campaign 从**单独的 runs worktree**（固定在已提交的 commit）启动，开发 worktree 可以脏。

### 10.0 状态表

| 步 | 状态 | 产物 / commit | 关键数字 |
|---|---|---|---|
| T0(a) | 完成（§3） | 8a03408 | — |
| T0(b) 链式敏感度 L | **完成并发布**（§10.4；KRDU picker 4 个类别 `…@chain-60s` / `…@chain-30s`，native32 与 A0b，165 类 0 错误）：开环链式在两臂上都不赢；native32 的累积主要是锚点分布外 | ec9323e；`outputs/KRDU/experiments/two_tier_t0b_20260916/{chain_60s,chain_30s,identity_check.json,lead_time_error.*}` | 雷达引导整段 ADE：native32 一次成形 2870 → 链 60 s 9893 / 30 s 11930；A0b 3839 → 5585；L：native32 1.2–1.7，A0b 0.5–0.95 |
| T0(c) 长历史信息 | **完成：G2 不过（两种子）**（§10.6）：238 s 回看对 118 s，雷达引导配对 ΔADE 均值 −13.9 / +9.7 m（门要求改善 ≥ 125 m），直线 +11.4 / −7.7 m；L90 剂量臂同样无方向。T3 不建。发布交给 opus agent（进行中） | runs worktree @ f5b2531，4 h 28 min，6 臂全 completed；`outputs/KRDU/experiments/two_tier_t0c_20260916/{readout_s1337,readout_s2024}.{txt,json}`、`g2_paired.json` | cohort 1371（雷达引导 468 / 直线 901），全部配对 |
| 契约合入与重构 | **进行中（§十一）**：sf-n7 已合入 `82e27e8`；重构计划 §11.2–11.3，T1a/T2 待其提交后改用航迹角契约重发 | §11.0 | — |
| T1 学习型第一层 | T1.1/T1.2 已提交 d9f40d9（opus review ×2）；T2 review 中发现并修正的 lockstep 捕获高度规则在 7d24238；T1.3 臂 `docs/experiments/t1a_plan_tracker_arms.json`（p50 ×2 种子 + p0）+ intents；**排队：J6 = T0(c) 之后训练（runs worktree @ 7d24238）→ `tracker_lockstep`（receding / one-shot / receding-no-plan）→ `python -m evaluation` → 读 G1** | campaign `outputs/KRDU/experiments/two_tier_t1a_20260916`（`lockstep_30s`, `lockstep_30s_noplan`） | — |
| T2 端到端 | **代码已提交 7d24238**（§10.5，opus review ×2）：`tracker_lockstep --plan-head step3g_fan4_head`；**排队：J7 = J6 之后**，同一批 T1a checkpoint，读 G3 | `…/two_tier_t1a_20260916/{lockstep_30s_head,lockstep_30s_head_noplan}` | — |
| T3 | **按现有 harvest 不可建**（§10.2：10 分钟历史不存在） | — | — |
| T4 | 未开始（前置 oracle） | — | — |

### 10.1 T0(b) 设计：`run_ts.py chain_sensitivity`

问题：§4.2 的链式界 `e_k ≤ ε + L·e_{k−1}` 在控制路径上 L 是多少，链式（开环，用自己的 rollout 续窗口再问）在 60 / 120 / … s 处比一次成形好还是差。

- 臂：`native32`（`l1_lowdim_20260907/L1_native32`，固定 L−1 训练）+ `A0b_path_uniform`（`a0_random_20260907/A0b_lr_objective_path_uniform`，全锚点随机训练）。第二臂是必须的：native32 只在 L−1 训练，链上每次再问都在它**没见过的锚点**，L 会把"锚点分布外"与"自身历史漂移"混在一起；随机锚点臂只剩后者。
- 每架航班（KRDU val，锚点 a₀ = L−1）三种预测，同一 checkpoint、同一前向代码（`forecast_approaches`）：
  1. **一次成形**：a₀ 处预测（跑完后另行与存档 predict 目录逐航班比对，作身份检查）。
  2. **链式**：第 k 段在锚点 a_k = a₀ + kΔ/dt 上，对"观测轨迹到 a₀ + 已飞的链式行（插到 2 s 网格）"组成的**滚动序列**再问（窗口、锚点状态、作动器初值都走模型自己的 predict 路径——作动器初值由滚动行反演，与训练时从观测回看反演同一规则），保留前 Δ 秒；最后一段整段保留。某段预测在 Δ 内结束（预测落地）则链在那里结束。
  3. **真实历史再锚**（协议 B）：观测序列在 a_(k−1) 处预测，读它在提前量 Δ（即绝对 kΔ）处的位移 ε_k（锚点在观测序列上可用才有；k=1 就是一次成形）。
- 读数（按 `strata_fixed_at_anchor` 在 a₀ 处分层，逐航班配对）：k = 1…links+1 的 kΔ 处 e_one、e_chain、ε_k 的 p50/均值与 n（最后一段的前 Δ 也计分）；L 的两个估计——配对中位数 `(e_chain(kΔ) − ε_k)/e_chain((k−1)Δ)` 与过原点最小二乘斜率；链/一次成形之比。`--write-records` 把一次成形与链式写成 predict 形状的目录（`python -m evaluation`、`lead_time_error`、发布器都能直接读）。
- 默认 Δ = 60 s、K = 5 段；另跑 Δ = 30 s（lockstep 的再问周期）。
- 读法（预注册）：L_med > 1.05 且链在 120 s 处 p50 比一次成形差 → T1 **必须**在滚动窗口表上训练（与 v5.2 在 plan 路径上的结论同向）；L_med ≤ 1 → 模型对自身历史漂移是收缩的，T1 的滚动窗口训练降为可选但仍保留（v5.2 的证据）。无论哪个结果都不改 T1 的结构，所以 T0(b) 不阻塞 T1 开发。

### 10.2 T0(c) 修订：数据里没有 10 分钟历史

测量（2026-09-16，KRDU val 1404 架，`usable_series` 在共同锚点 L−1 处）：序列时长 p10/p50/p90 = 256 / 296 / 636 s（到达切片从 25 km 起，首次接收中位只早 45 s）。

| seq_len | 回看 | 共同锚点 | val 可用 |
|---:|---:|---:|---:|
| 60 | 118 s | 59 | 1404（100 %） |
| 90 | 178 s | 89 | 1401 |
| 120 | 238 s | 119 | **1382（98.4 %）**；按固定锚点的建序列规则（review 修正后，三臂共用）为 **1371（97.6 %）**，L=60/90/120 逐航班相同 |
| 150 | 298 s | 149 | 701（49.9 %） |
| 300 | 598 s | 299 | 250（17.8 %，全是长雷达引导） |

所以 §7 的"10 min 段 token 变体"与 seq_len 300 在现有 harvest 上**做不了**（剩下的 18 % 是按结果挑出来的长航班，不是一个可比的 cohort）；T3 所说的"长历史"要先扩 harvest 的接收窗口（数据面改动，用户决定，记入 OPEN_ITEMS）。T0(c) 改为：

- 臂：L ∈ {60, 90, 120} × 种子 {1337, 2024}，**共同锚点 119**（切片起点后 238 s），配方 = native32（`l1_lowdim_arms.json` 的 `L1_native32`：simple-v3 + N=32 + native 端点 + imitation 64 + velocity 0.003，lag 动力学）。6 臂 × ~65 min GPU。
- 需要的代码：`TSConfig.anchor_floor_index`（默认 0；`default_anchor = max(seq_len − 1, floor)`，于是训练 cohort、固定锚点、predict、验证全部跟着走——`history_ablation` 已有的 `minimum_anchor_index` 管道只在 fit_model 内部，train/predict CLI 不传）；命名 `anchor-floor=`；把仍然手写 `seq_len - 1` 当"固定锚点"的 runner 改成读 `default_anchor`。
- 门 G2（不变）：L=120 vs L=60 在共同锚点处雷达引导 ADE 配对改善 > 125 m，两个种子都过；L=90 给剂量方向。注意这是 238 s vs 118 s，不是 10 min vs 2 min。

### 10.3 T1 设计（2026-09-16 定稿，未开建）：计划给定下的学习型跟踪器（协议 C）

**问题**：给定真值计划（协议 C，读未来，oracle），学习型控制层每 30 s 在自己飞出的行上再问，能否把计划飞得比规则制导近——G1：雷达引导 ADE@L−1 < 1000 m、直线 < 200 m，两种子；完全可飞 ≥ 95 %；建立率 ≥ 规则制导的 88 %（`plan_guidance` 设计 §12.5：lockstep 1847 / 283 m，建立 87.9 % / 99.8 %；leg form 1492；整条真值路径 1159）。对照：L3_cta（一次成形、只给真值时长）雷达引导 1596 m——只给时长的一次成形已经比规则制导近，所以 G1 真正要证明的是"计划 token + 再问"再往下拿走 ≥ 600 m。

**决定（在规则内自定，理由写明）**：

1. **形态 = 滚动时域（receding），不是 §7 表里写的"固定 60 s、N₁ 段"头**。每次询问仍预测到入口的整段 32 段控制，但**只飞前 30 s**，然后在自己的滚动行上再问。理由：(a) 计划层给出剩余时间 T，用现成的 `cta_conditioning=given` 让 rollout 时长 = 计划的 T——这同时消掉了 duration head 的 ~125 s 地板（近跑道再问时它会把控制拉长，固定 60 s 头也要单独解决）；(b) 零改动 spine（数据集目标网格、目标函数、验证指标、记录契约都不动）；(c) 飞出去的部分就是用户提议的"短时控制"。固定 Δ 头留作 **T1b**：只有 T1a 没过 G1、且差在每次询问的前 30 s 跟踪（读每步 e(30 s) 对滚动真值）时才建。
2. **计划 token**：plan 路径的目标向量在锚点处的读数（`outputs/plan/labels.targets_from_labels`，与 plan 头的标签同一定义）——14 个值按 `SCALES` 缩放、14 位有效掩码、"前方无定位点"标志、1 位"计划在场"标志，共 30 维，作为与 CTA token 并列的一个融合 token 进 `ControlFeatureModel`。新配置轴 `plan_conditioning ∈ {off, truth-next}`（名 `plan=truth-next`，读未来）与 `plan_conditioning_dropout`（名 `plan-drop=`；训练时逐样本以该概率把 token 置零、在场位 0——计划不可靠时退回窗口，§7 风险 2）。标签抽取实测 0.3 ms/锚点（KRDU val 1000 次），逐抽样现算，不缓存。
3. **训练**：随机锚点（`remaining-path-uniform`，`l1-share 0.5`，A2b 的配方：heading-rate 8、bank TV 1、velocity 0.003，**无模仿项**——随机锚点 + 模仿是已知性能悬崖），`cta=given`，180 epoch 不早停（patience 180），选择指标 fixed-anchor common-grid ADE（anchor-grid 指标在 `cta=given` 下被拒）。**不先上滚动窗口表**：跟踪器在离开真值轨迹的状态上没有定义好的监督目标（plan 头的标签是"那里该下的指令"，控制层的标签是"从那里怎么飞"，真值不给）；先读 T1a 在 lockstep 里是否漂移，再决定 T1c（目标 = 最近真值行起的真值未来，时间平移）。
4. **部署 / 读数** `run_ts.py tracker_lockstep`（T0(b) 链式机器的推广，共用 `rolled_series` / `cut_at_lead` / 位移定义）：从 L−1 起每 30 s 在滚动序列上再问；每次询问的 CTA 与计划 token 来自在 L−1 处建好、按时间顺序读取的 `TruthExpert`（`fly_lockstep` 的真值策略同一个对象）；飞前 30 s；在入口 on-final 过线处截断（`cut_at_threshold_crossing`）、或一次询问的预测在 30 s 内结束、或时间上限 1.5× 真值时长。变体（同一 checkpoint）：计划在场 / 计划遮掉（在场位 0）/ 一次成形（不再问）。写 predict 形状的记录 → `python -m evaluation`（可飞、建立）+ `lead_time_error`。
5. **臂（T1a）**：`T1a_plan_p50_s1337`、`T1a_plan_p50_s2024`（门）、`T1a_plan_p0_s1337`（不遮，读风险 2 的"过度依赖"）。~70 min × 3 GPU。

**实施步骤**（每步 opus review → 提交）：T1.1 配置轴 + token + 上下文行 / forecast / probe 键一致 + 测试；T1.2 `tracker_lockstep` runner（把链式共用件移到一处）+ 测试；T1.3 arms + intents → 启动 → 读 G1。

### 10.4 T0(b) 结果（2026-09-16，KRDU val 1404 架，队列 agent 于 runs worktree @ ec9323e 运行）

产物 `4dTrajectory/outputs/KRDU/experiments/two_tier_t0b_20260916/`：`chain_60s/`、`chain_30s/`（`chain_sensitivity.{txt,json}` + `records/<臂>/{oneshot,chain_*}`）、`identity_check.json`、`lead_time_error.{json,txt}`。耗时 2 min（Δ60）/ 3 min（Δ30），GPU。**身份检查**：native32 一次成形记录与存档 `L1_native32_pred_val` 逐航班 ADE/FDE/时长 |Δ| = 0（1404/1404）。

整段 ADE 均值（`lead_time_error` 口径，全体 | 直线 | 雷达引导，m）：

| 记录集 | 全体 | 直线 | 雷达引导 |
|---|---:|---:|---:|
| native32 一次成形 | 1322 | 445 | 2870 |
| native32 链 Δ60（5 次再问） | 3840 | 485 | **9893** |
| native32 链 Δ30（10 次再问） | 4563 | 488 | **11930** |
| A0b 一次成形 | 1782 | 628 | 3839 |
| A0b 链 Δ60 | 2404 | 633 | **5585** |

位移 p50（m），雷达引导，Δ60：

| 提前量 | native32 一次 / 链 / ε | L 中位 / lsq | A0b 一次 / 链 / ε | L 中位 / lsq |
|---:|---|---|---|---|
| 60 s | 632 / 632 / 632 | — | 1995 / 1995 / 1995 | — |
| 120 s | 1887 / 1930 / 779 | 1.69 / 1.71 | 2027 / 2998 / 2028 | 0.49 / 0.53 |
| 180 s | 2497 / 3605 / 1200 | 1.26 / 1.27 | 3602 / 4886 / 2071 | 0.92 / 0.95 |
| 300 s | 3658 / 11257 / 1392 | 1.33 / 1.24 | 5605 / 5740 / 1472 | 0.75 / 0.85 |

Δ30 雷达引导（native32）：30 s 148 · 60 s 605（ε 159，L 2.81/2.33）· 120 s 2917（ε 185）· 300 s 14753（ε 313）。直线进近两臂链式只输 30–70 m 整段 ADE。

读法：

1. **开环链式不产生信息，两臂都输**（§4.2 的预期）。native32 的 L > 1（1.2–2.8）且随询问变密更糟：它只在 L−1 训练，链上的每次再问都在没见过的锚点上，又喂自己平滑的历史——两个分布外叠加。
2. **随机锚点训练把 L 压到 1 以下**（A0b 0.5–0.95：模型对偏离的历史有收缩，雷达引导链/一次成形在 300 s 处 0.89–1.22，而 native32 3.0–4.2）——所以累积主要是**锚点分布外**，不是"自己的历史"本身；但 A0b 自己的短时误差大（60 s 处雷达引导 p50 1995 m 对 native32 632），链式整体仍输。
3. **协议 B 的 ε 很小**：native32 在真实历史上逐 30 s 再锚，30 s 提前量的位移 p50 雷达引导 148–313 m——"再问能降误差"只在有**新观测**时成立。
4. **对 T1 的决定（预注册读法 §10.1）**：native32 触发"L_med > 1.05 且 120 s 处链式更差"。T1a 的设计已经用随机锚点训练（A0b 显示这一项把 L 压到 1 以下）且每次询问都重新给真值计划 token（外部信息，抵消漂移），所以**T1a 按原设计先建**；若 T1a 在 lockstep 里漂移（每步 e(30 s) 相对滚动真值上升），T1c（离轨状态上的时间平移真值目标）升为必做。

### 10.5 T2 设计（2026-09-16）：计划头替换真值计划（协议 A）

`run_ts.py tracker_lockstep --plan-head <plan checkpoint>`：同一个 lockstep、同一个 T1 跟踪器，只把计划来源从 `TruthPlans`（每架一个 `TruthExpert`，按时间顺序读）换成 `HeadPlans`——计划头在**同一段滚动历史**上的预测，经 `order_from_prediction`（各参数夹到范围内，到达时间下限 `T_MIN_S`）得到 `(Operating, Instruction)`，再走真值走的同一条 `targets_at` → `plan_token`；CTA = 头的 T。于是 T1 与 T2 是同一个仪器、换一个输入，差值就是 `e_plan`（§4.3）。

- 计划头：KRDU 交付头 `plan_guidance_20260910/step3g_fan4_head`（K=4 混合，seq_len 30，滚动窗口训练 share 0.75，§12.8：60 s 锚点雷达引导 top-1 3087 m）；取 top-1 权重分量（`prediction_rows`），不做 fan。
- **捕获高度的有效位用同一条可观测规则**（review 2026-09-16）：训练标签在"锚点起一直在五边上"（`join_at_anchor`，读未来）时把 `h_capture` 置未定义，而在 a₀ 建好的 `TruthExpert` 会把 a₀ 的值一直带到后面的询问——T1 自己的 lockstep token 在后续询问上就和训练分布不一致。现在两种来源在**首次询问之后**共用 `on_final_capture`（此刻 `on_final_pose` 在五边上 → 未定义）；真值的首次询问就是标签本身。实测 KRDU val 100 架 a₀：修正前头来源 `h_capture` 有效率 1.00 对真值 0.40（60/100 掩码不同）；修正后 0.39 对 0.40（1/100 不同）。其余运行参数位两边都是 1，指令组有效率 0.39 / 0.38，T p50 187 / 181 s（|ΔT| 中位 11 s）。所以 T2 − T1 ≈ e_plan。
- 头的契约在读任何轨迹前核对：dt、坐标系、输入通道与跟踪器一致，回看不长于跟踪器锚点，机场覆盖；产物记头的 sha256。
- 读未来的地方都去掉：时间上限改为 `cap_factor × 第一次询问的到达时间`（真值来源下 = 真值时长，头来源下 = 头自己的 T）；头在任何它训练过的航班上被拒（逐航班对照头的 train 名册）。**跑道仍是已知的**（包内通例：阈值锚定坐标系）。
- 门 G3（§6）：雷达引导整段 ADE ≤ 2745 m 且直线 ≤ 415 m（对 native32 一次成形 2870 / 445，两种子）；读 top-1，fan 不作交付。同时报 `receding-no-plan`（只给头的 T）与 `one-shot`（头的计划只在 L−1 给一次）。
- 冒烟（native32 + 头，30 架，CPU 33 s）：机制跑通；native32 本身不读 token / CTA，所以数字只是链式本身（雷达引导 12 km），不代表 T2。

### 10.6 T0(c) 结果（2026-09-16，KRDU val 1371 架，共同锚点 119，队列 agent @ f5b2531）

| 种子，分层 | L60 | L90 | L120 |（ADE 均值 / FDE p50 / chamfer p50，m）
|---|---|---|---|
| s1337 雷达引导（468） | 2929 / 2361 / 1058 | 2938 / 2399 / 1078 | 2915 / 2451 / 1077 |
| s1337 直线（901） | 131 / 148 / 85 | 126 / 160 / 71 | 143 / 166 / 85 |
| s2024 雷达引导 | 2917 / 2224 / 1011 | 2921 / 2212 / 1068 | 2926 / 2275 / 1064 |
| s2024 直线 | 124 / 124 / 61 | 112 / 140 / 55 | 116 / 151 / 59 |

配对 ΔADE（L120 − L60，均值 / 中位，负 = 长回看更好）：雷达引导 −13.9 / −16.5（s1337）、+9.7 / +10.3（s2024）；直线 +11.4 / +2.4、−7.7 / +1.8；L120 更好的航班占 39–52 %。L90 − L60 同样在 ±12 m 内无方向。**G2 不过**（读法：改善 = −配对均值，门 125 m；直线不变差 = 配对均值 ≤ 30 m，`g2_paired.json` 记规则）。结论：在 25 km 切片可得的历史范围内，比 118 s 更长的观测历史没有控制路径能用的进近信息；段 token 长历史层（T3）不建，与 §10.2 的数据约束一致。

## 十一、合入航迹角契约（sf-n7）与控制契约重构（2026-09-16 起；压缩 context 后从这里继续）

**为什么。** T1a 若用推力分数契约，G1 的"完全可飞 ≥ 95 %"几乎必不过（δ 孪生只有 0.4 % 完全可飞，航迹角契约 97.7 %，sf 设计 §15）。用户 2026-09-16："合并进来，然后做实验；不要直接合，要审核它的实现；要保证模块化，结构上的简洁高效；两种不同的动力系统应该可以直接切换，而不是胡乱打补丁。"

### 11.0 状态表

| 步 | 状态 | commit / 产物 |
|---|---|---|
| M0 合并 sf-n7 | **完成**：8 个文本冲突两边都留；一处语义冲突（`tracker_lockstep.ask_row` 未把契约传给 `dynamics_arrays`）当场修。ts 743 + 525、aerodynamic_model 140 全过 | `82e27e8` |
| 架构审查 | **完成**（opus，只读 sf-n7 @ ccd71cb）：见 11.1 | — |
| 金标准 | **已采集**：合并后代码上 CPU 急切模式 153 项（4 种定律 × 端点/稠密/hook 三种引擎 + 梯度；N4_twin / N3 / N6 / N7 真实 checkpoint 各 3 架 val 的预测、导出记录、损失分量、教师、初始作动器、probe、名字、契约串、饱和标签）；复跑 0 差 | scratchpad `golden_contracts.py` + `golden_merge_82e27e8.pt` |
| R1 物理层 | **建成，opus review 完**：前向（端点/稠密/hook，四种定律，543 个张量）逐位不变；review 发现运动学构造顺序变了导致点质量与推力分数的梯度有舍入差 → 恢复原顺序后逐位不变；参数列 dtype 统一由消费方指定、几何读数检查形状、删多余的 `_step`；新增 `aerodynamic_model/tests/test_torch_lag_laws.py`（三种定律端点金标准、每类编译入口飞自己的定律、批量几何读数） | 未提交 |
| R2 契约行 + 消费方 | **建成，opus review 完**：无阻断/中等问题；T1a 形状的配置（路径角 + heading-rate 8 + 随机锚点 + plan token + lockstep + 导出）在合成数据上端到端跑通；89 个可构造的存量 config 名字/slug/身份串全部不变。低级问题已修：记录段字段一处定义（`export.CONTROL_SEGMENT_FIELDS`）且测试查契约列名不与之冲突；`Forecast` 不变式加形状对齐；记录写契约名改为"非默认即写"；`concatenate` 拒绝拼接相对锚点空速的 speed-command 指令；删 `lag_control_law` 别名与 heads 的第三份 `CONTROL_NAMES`；导入期检查改 `raise RuntimeError`；inverse 文档串更正 | 见下 |
| R3 测试 + 金标准 + 全套 | **金标准全量 153 项**：除预期的两类外逐位不变——比力族（SF/SC/PA）梯度 float64 相对 ≤ 5e-14（共享一个阻力张量，重训 N3/N6/N7 不再逐位复现），PA 记录里的解算过载 ≤ 1e-16 相对（cos γ 改为 √(1−sin²γ)）。**CUDA**（§11.3 第 4 步，16 架 × 8 段，三种定律）：每个版本自身运行间逐位确定，但新旧代码的编译核不同，端点/稠密/梯度差 ≤ 1.6e-13 相对（推力分数也在内）——GPU 上重预测存量 run 与其记录只差舍入，GPU 上重训任何存量 run 不再逐位复现（与升级 torch 同类；对照一律同代码孪生）；修完后 ts 704 + 573、aerodynamic_model 154 全过，金标准不变；新增 `tests/test_control_contracts.py`、PA 下 heading-rate 项测试 | 未提交 |
| T1a / T2 重发 | **J6 训练中**（2026-09-16 22:32 起，runs worktree @ 77e1372，队列 agent；预计 7–8 h）：冒烟 1 epoch（全 KRDU，train 6851 / val 1404）所有损失项有限，heading-rate 项 train 0.978 / val 1.276（非零），梯度裁剪 13/14 步，53 s/epoch（航迹角契约 + 随机锚点，180 epoch 无早停 → 每臂约 2.3–2.7 h）。之后 lockstep → evaluation → G1；J7（`--plan-head`）→ G3 | campaign `outputs/KRDU/experiments/two_tier_t1a_20260916` |

### 11.1 审查结论（sf-n7 原样不满足要求）

公式本身只定义一次（`path_angle_load_factor`、`specific_force_thrust_n`、`drag_force_n`），物理是对的；结构不是：
- **契约是一个字符串，行为散在约 15 个模块、4 张平行查表里**（`envelope._CONTRACTS` 盒子、`backends._LAG_CONTROL_LAWS` 定律、`inverse` 的教师 `==` 链、`strategy` 的身份串 `==` 链；另有 `forecast` 的 `isinstance` 链、`diagnostics` / `speed_floor` 的字符串比较、`config` 5 条手写拒绝、`run_naming` 的 slug 表），没有任何东西检查这些表的键一致。
- **物理层每种定律一整套复制**：RHS、RK4 步、上下文解包、两个 CUDA 编译入口、rollout 入口，4 × 6 = 24 个函数加 4 份手写 step_context 偏移，约占 `torch_lag_dynamics.py` 994 行里的 500 行。编译缓存按代码对象分确实需要"每定律独立代码对象"，但 RHS / RK4 / 解包不需要复制。
- **每个 RK4 stage 重复算**：比力定律密度、气动系数、阻力各 2 次，航迹角定律几何 3 次；CPU 急切模式每次 RHS 比推力分数慢 39 % / 65 %（B=512 实测）。
- **潜在错读**：heading-rate 项、barrier、trombone、速度下限都把第 3 列当过载读（航迹角契约下是弧度），靠 config 拒绝挡住；速度下限按 `== specific-force` 选分支，航迹角契约若解禁会静默走推力分数分支。
- `cut_rows` / `concatenate`（本分支 T0(b)/T1 新增）不切契约的指令字段——合并后的语义冲突，重构里一并消掉。
- 已失败的速度指令契约（N6）全量接线约 250 行生产代码。**决定：保留为注册表里的一行**（N6 两个 checkpoint 仍可加载；已告知用户），它特有的两件事（锚点空速进 step_context、教师输出相对锚点空速）变成定律参数与契约行上的一个布尔，不再在调用处分支。

### 11.2 目标结构

**物理层**（`aerodynamic_model/`，只依赖 torch）：
- `torch_dynamics.py`：`FlightCondition(speed, sin γ, mass, density)` 与 `geodetic_flight_condition(states)`；`flight_aerodynamics(condition, load, aero) -> (stalled, drag)`——系数与阻力的唯一入口。
- `torch_transport_chart_dynamics.py`：`transport_chart_rhs` 拆成 `transport_chart_kinematics(state, frame)`（几何、速度基、`FlightCondition`）+ `transport_chart_rate(kinematics, thrust, bank, load, aerodynamics, aero)`；`transport_chart_rhs` = 两者组合（点质量行照旧调用，逐位不变）。三个 `transport_chart_*_thrust/load` 辅助函数与 `_chart_speed_altitude_mass` 删除。
- `torch_lag_dynamics.py`：**一个定律协议**。每个定律是冻结 dataclass：`PARAMETERS`（进 step_context 的列名）、`parameters(max_thrust, initial_states)`、静态 `resolve_load(condition, actual, params)`、静态 `resolve_thrust(condition, actual, drag, max_thrust, params)`，以及两个 2 行的编译入口（每类独立代码对象，这是 torch.compile 缓存要求）。**一个** `_lag_rate`（运动学一次、气动一次、定律两次调用、actuator ODE）、**一个** RK4、**一个**解包（布局 `[frame 4 | τ 3 | T_max 1 | 定律参数 k | scale 10]`，四种定律现有布局恰好都是它的特例）、一个分派。几何读数（记录、heading-rate 项）走 `law.geodetic_load(...)` / `law.geodetic_controls(...)`，与 RHS 调同一对静态方法。
- 为逐位复现：航迹角定律的 cos γ 仍由 sin γ 开方得到（N7 的写法）；比力推力复用本 stage 已算的阻力张量（值与原来两次计算逐位相同）；**不**化简成 `V' = g(n_x − sin γ)`。记录里航迹角契约的过载从 `cos(γ)` 改为定律自己的 `√(1−sin²γ)`，差在 1e-16 相对量级（记录不重写，只是今后导出的末位可能不同）。

**契约层**（ts 侧）：
- `config.py`：一张纯数据表 `CONTROL_PARAMETERIZATION_SCOPES`，每个取值一行：slug、能否走点质量行、能否用 fitted 教师、建好了哪些 hook 成员（及不支持的理由）。词表 `CONTROL_THRUST_PARAMETERIZATIONS` 从表派生；5 条手写拒绝变成 3 条读表的检查；**heading-rate 的拒绝删除**。
- `outputs/envelope.py` 的 `ControlContract` 一行持有该契约的全部行为：名字、单位、盒子、中性值、`law`、`teacher`（逆动力学列，numpy）、`relative_to_anchor_speed`、`identity_suffix`（TF/SF 为空，N6/N7 为现存串原文）、`saturation_labels`（TF 保留历史 `thrust_N`）、`record_command_columns`（TF `()`、SF/SC `(0,)`、PA `(0, 2)`）、`longitudinal`（速度下限选反演用）。导入时断言：注册表键 == config 词表。
- 消费方只调契约行：`backends`（删 `_LAG_CONTROL_LAWS` / `lag_control_law`）、`inverse`（删 `==` 链与 `anchor_relative` 分支）、`strategy.target_contract` 与 `record_fields`、`diagnostics.saturation_labels`、`control/forecast`（记录换算对四种定律同一条路径，TF 的 `physical_controls` 特例删掉）、`speed_floor`（按 `longitudinal` 查表，导入时断言 config 允许 speed-floor 的契约都有实现）、`run_naming`（slug 读 config 表）。
- `Forecast`：`longitudinal_commands` / `longitudinal_parameterization` / `vertical_commands` 三个字段换成 `commands [N,3]`（契约单位的已飞指令）+ `control_parameterization`；导出按 `record_command_columns` 循环写列；`cut_at_threshold_crossing`、`cut_rows`、`concatenate` 与 `controls` 一起切 `commands`。记录格式不变（金标准比对 JSON）。
- **heading-rate 项**：预测侧的过载改为 `contract.law.geodetic_load(states, actual, …)`。TF/SF/SC 下就是第 3 列原样（逐位不变）；航迹角契约下是回路解出的过载——γ* 通过升力对转弯率有一个小梯度，这是正确的物理（与 δ 契约下过载对转弯率的作用同类）。
- **不在本次范围**（记入 code-health follow-ups）：barrier / trombone / 速度下限在航迹角契约下仍拒绝（组合是单独的设计，§14.4）；plan guidance 里重写的路径回路、五份滞后补偿反演、三处 cos γ 写法。

### 11.3 次序与门

1. **R1 物理层** → `golden_contracts.py compare --physics-only` 必须 0 差 → aerodynamic_model 测试改到新 API → opus review（后台，只看物理层代码）。
2. **R2 契约层 + 消费方**（与 R1 review 并行，文件不重叠）。
3. **R3** 新测试：注册表完整性（契约键 == config 词表 == scope 表；每个契约行字段自洽）、航迹角契约下 heading-rate 项等于用回路过载显式算出的 ψ 行、`cut_rows`/`concatenate` 切 `commands`、速度下限实现覆盖；**全量金标准 0 差**；ts 全套 + aerodynamic_model 全套 → opus review → 修 → 复核 → 提交。
4. CUDA 编译路径：GPU 空出后跑一次 CUDA 上的小批量对比（编译后代码对象变了，可能有 ULP 级漂移；对比而非假设）。

### 11.4 T1a / T2 的调整（R 系列提交后）

- 臂 `docs/experiments/t1a_plan_tracker_arms.json`：`control_thrust_parameterization=specific-force+path-angle`，其余不变（A2b 配方：heading-rate 8、bank TV 1、随机锚点、剩余路径均匀、l1 份额；plan token；CTA given）。campaign 名与 intents 同步改（旧名下没有任何产物）。
- 队列 agent 的 J6 / J7 换到新 commit；J6 冒烟先跑 1 epoch 核对 heading-rate 项非零、有限。
- G1 / G3 门不变（绝对阈值）。
