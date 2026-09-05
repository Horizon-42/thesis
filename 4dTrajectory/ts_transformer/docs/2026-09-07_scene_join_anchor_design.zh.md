# 场景编码 + 汇入锚多模态的 control 预测（dev 文档，2026-09-07）

承接 `2026-09-07_control_training_review.zh.md` §七：pooled ADE 的 60–76 % 来自雷达引导航班，它们的误差是"三边飞多远、何时汇入五边"这个管制决定，本机 120 s 历史里没有这个信息，单点回归只能输出各种决定的平均。本文给出把**交通上下文**放进输入、把**汇入决策**做成多模态输出的完整方案：意图、当前方案的缺陷、设计、架构、实现步骤、风险与参考文献。§五 的 Phase 0 上界实验已做完（2026-09-05），结果改写了优先级——见 §〇 与 §五。

## 〇、进度与状态（压缩 context 后从这里继续）

**当前状态（2026-09-05）**：Phase 0 完成并提交；**P0（几何指标进标准读数）完成并提交**，Phase 0 四臂与 hook v2（KRDU + KSJC）已回填；**走向待用户决定**（§五 P1 之后的顺序：推荐 C 先修输出侧，再 B 改决策变量继续）。没有在跑的 campaign。磁盘只剩 4.0 GB——**下一个 campaign 前先清理**（`clean_pipeline_data.py --dry-run` 或删旧实验树）。

| 阶段 | 状态 | 产物 / commit | 关键数字（KRDU val，雷达引导 497 架） |
|---|---|---|---|
| Phase 0 机制：`intent_conditioning` ∈ none / truth-join / truth-join-lead / truth-join-duration（真值汇入点 + 可选真值前机 ETA / 真值剩余时长，作为只输入的常量协变量通道） | 完成 | `57d3e2b`, `def1cc5`；`intent_conditioning.py`、`config.py`（`INTENT_*`）、`FlightSeries.lead_landing`、`series_conditioning(..., anchor=)`、`--intent-conditioning`；测试 `tests/test_intent_conditioning.py` | 顺带修了两个既有阻塞 bug：控制训练回路/auto-batch 把 `x[:, -1]`（含协变量的整行）当锚点状态；配方覆盖检查拒绝新字段 |
| Phase 0 campaign `4dTrajectory/outputs/KRDU/experiments/scene_phase0_20260905/`（臂 `docs/experiments/scene_phase0_arms.json`，与 `control_procedure_20260905/A_control_v3` 配对） | 完成 | `readout.txt/.json`（四臂）、`diagnostics_arms.txt`、`diagnostics_population.txt`；结果文档 `2026-09-05_scene_phase0_results.zh.md` | ADE：基线 2858 → 仅汇入点 2356 → +前机 ETA 2364（无增量）→ +剩余时长 **2011**；时长误差 39 → 22 → 20 → 5 s；时间无关几何误差 chamfer 942 → 791 → 801 → 850 m（**不改善**）；直线进近 469 → 458/442/434 |
| Phase 0 诊断 `docs/phase0_intent_diagnostics.py`（residual / sensitivity / template / context / timing；经 opus review） | 完成 | 同上 | 真值路径 + 朴素速度剖面 1308 m；真值汇入 + trombone + 真值时序 1688 m；真值路径 + 常速 2813 m；上下文对 d_join 的 R²（锚点后才汇入 6,557 架）：本机 0.34 → +因果计数 0.38 → +真值前机 ETA 0.47；知道 d_join 后剩余时长误差 35.8 → 22.8 s |
| 门的判定 | 未达且门定错量级 | `2026-09-05_scene_phase0_results.zh.md` §四–§八 | 1.5 km 对"只知汇入点"的 oracle 在时间对齐指标下不可达；决策变量应含时间；输出侧（控制头 + rollout 按给定意图画几何）是先于场景编码的瓶颈 |
| P0 几何指标读数：`geometric_metrics.py`（chamfer / 离散 Fréchet / 弧长对齐 ADE / 路径长度比 / 沿路径滞后 / 时长误差；真值 `closed` = 观测行闭合到阈值，`observed` 复现 Phase 0）进 `compare_constraint_arms.py` / `compare_frame_arms.py`（`--geometry-truth`） | 完成 | `6867512`；`tests/test_geometric_metrics.py`（23 项）；回填 `scene_phase0_20260905/readout_geometry.*`、`control_hooks_v2_20260906/readout_geometry.*`（KRDU + KSJC）；结果写进两份结果文档 | 雷达引导：chamfer 942 → 806 / 795 / 847，Fréchet 3100 → 2488 / 2468 / 2615，弧长 ADE 2175 → 1857 / 1775 / 1900——汇入点值 15–20 % 几何，**时长臂的几何不比 O_join 好**（其 ADE 增益全是时序）；`observed` 真值复现 Phase 0 chamfer 差 ≤ 2 m；**state 输出的锯齿折线使弧长参数化失效**（相邻节点航向反转 > 90° 的份额中位 0.50，control 臂与真值为 0；长度比 ≈ 2 对 1.01）——每航班按反转份额 ≤ 5 % 判"折线是航路"，arc-ADE / lag 只在这些航班上聚合并打印份额，块内合格 < 95 % 打印 n/a，不做平滑；state campaign 上弧长族因此基本全是 n/a，可读的几何是 chamfer + Fréchet |
| P1 输出侧几何封闭（走向 C） | 待决定 | — | 见 §五 P1 |
| P2 数据平面（原 Phase 1）| 未开始 | — | 验收指标已改（§五 P2） |
| P3 场景编码 + (d_join, T) 锚解码器（原 Phase 2/3） | 未开始 | — | top-1 目标已改（§一） |
| P4 生成式对照等（原 Phase 4） | 未开始 | — | 不变 |

**恢复工作时的约定**（来自 memory 与本轮经验）：正式 campaign（`run_ts_frame_ablation.py`，默认 formal）要求干净工作树——先 commit；代码在跑实验前用 opus subagent review，文档不送 review；`git add` 明确路径，不用 `-A`；看进程用 PID 不用 `pgrep -f`；子进程 stdout 到日志是块缓冲，`epoch` 行会滞后几分钟，不是卡死；引用数字只引当前产物；判"预测是否建立五边"用成员门（`hard_on_final` + `stays_mask`），k=0.5 真值门会被已知的 250–350 m 终点平移饱和成"从不"。

**复现命令**：见 `2026-09-05_scene_phase0_results.zh.md` §九。

## 一、意图

**要预测的东西**：一架进场飞机从锚点（进入 25 km 切片后 120 s）到过阈值的 4D 轨迹，现在由 control 模型给出——64 段常值控制 + 时长，经可微点质量 rollout 得到路径，动力学一致。

**要改的东西**：让模型知道决定汇入位置的外部信息（同跑道前机、排队、跑道使用），并把汇入位置这个离散的决定显式地建模成 K 个候选（多峰），而不是一个平均。

**成功的定义（预注册；2026-09-05 按 Phase 0 修订，修订前的原文见 §7.4 之前的版本历史 `git log`）**：
- 每个分层同时报**两组数**：时间对齐的 ADE/FDE/时长误差（4D，交付的量）与时间无关的几何误差（chamfer、Fréchet、弧长对齐 ADE；§五 P0）。只用其中一组下的结论不算数。
- 雷达引导分层，top-1：Phase 0 量到"完美 (d_join, T) + 现解码器"= 2011 m、真值路径 + 朴素速度剖面 = 1308 m，所以 **top-1 ADE 的目标改为 < 2.0 km（从 2858），几何误差 chamfer < 600 m（从 942）**；minADE_K（K ≤ 6）< 1.7 km 保留。
- 时长误差（雷达引导中位）：< 25 s（从 39；仅汇入点已到 22，是可实现的量级）。
- 直线进近分层：top-1 ADE/FDE 不退（这一层现在已经不错，方案不能用它换雷达引导层）。
- 汇入锚分类：top-1 准确率显著高于按先验的猜测；概率校准（预测概率 0.8 的锚命中率 ≈ 0.8）。
- 动力学一致性、走廊约束、推理时屏障过滤器：每个模态都保留。
- 否决：直线进近 top-1 FDE 退 > 种子噪声；或 minADE_K 的改善全部来自"K 条里总有一条碰巧近"（对照：K 条随机锚 + 同一网络）；或几何误差不动而只有时序改善（那是时长头的功劳，不是场景/多模态的）。

**交付物形状**：K 条带概率的物理轨迹。现有评估（一条轨迹一个记录）用 top-1；新增一个多模态读数（minADE_K、miss rate、锚准确率、校准）。下游若要一条，取 top-1 或按概率加权；若要风险评估，用全部 K 条。

## 二、当前方案与缺陷

**现在的架构**（`control/heads.py`、`batch_contract.model_forward`）：iTransformer 把本机的 6 个通道各当一个变量 token（60 个时间点嵌入成一个 token），加一个动力学条件 token；`ControlFeatureModel` 把 (enc_in + 1) 个 token **拍平**成一个向量，过 MLP，再由 `ControlOutputHead` 一次性回归 64 段控制、`UniformDurationControlHead` 回归总时长。输入只有本机；输出只有一条。

缺陷，按实测：

1. **看不见决定路径的变量。** 三边长度由与前机的间隔决定；模型只看本机。KRDU 雷达引导航班 ADE 2858 m、KSJC 2066 m，占 pooled 误差的 76 % / 60 %；所有约束/hook 实验在这一层最多改善 6 %。
2. **单点回归对多峰未来输出平均。** vectored 分层 FDE 中位 1.5–2 km 的形状就是"平均意图"；`docs/2026-09-03_runway_hypothesis_expansion.md` 里 K=2 跑道假设已经显示"选对模态"的价值（真实姐妹跑道 oracle 拿回 79 m 中位 FDE），汇入位置的模态数远多于 2。
3. **变量 token 的框架装不下可变数量的实体。** `target_conditioning="channels"` 的消融测到：把目标坐标做成协变量 token 后注意力主干基本不用它们（只有拍平的时长头用了，−10 % 时间 MAE）。邻机是可变数量、有各自历史的实体，需要的是实体级的注意力，不是变量级。
4. **拍平 + 单头的解码器没有"以决策为条件"的机制。** 要输出 K 条互斥的候选，解码器必须以候选为查询；现在没有查询这一层。
5. 目标函数的四处问题（复审文档 §三）独立于本文，可以在同一套训练里并行修，但它们只动百米级。

## 三、设计方案

### 3.1 问题形式化

以本机为中心的场景预测。记锚点时刻 t₀，本机的阈值锚定坐标系（跑道轴 d 沿航向道向上游、xt 向右）。

- 输入 X = {本机历史 H_ego, 邻机历史 {H_j}, 静态几何 M, 标量上下文 c}，全部只用 t ≤ t₀ 的信息。
- 决策变量 g = 汇入距离 d_join（本机建立五边航向道时离阈值的距离；真值由 `final_approach_geometry.truth_final_gate` 给出，即 readout 的 `gate_start_d_m`），可选地扩为 (d_join, 进入侧)。
- K 个锚 {a_k}：d_join 的 K 个代表值，从训练集真值 d_join 的直方图聚类（k-means 或分位数），按机场分别做；直线进近集中在 FAF 内外一档，雷达引导分布在 10–25 km。
- 输出：对每个锚 k，概率 p_k、连续的汇入距离修正 δ_k（d̂_join,k = a_k + δ_k，意图的连续值）和一条控制序列 u_k（64 段 + 时长）；u_k 经现有 rollout 得到轨迹 τ_k。
- 训练：k* = argmin_k |d_join − a_k|；损失 = L_rank(p; k*) + λ_g·|δ_{k*} − (d_join − a_{k*})| + λ·L_control(u_{k*})。L_rank 用 Plackett–Luce 排序损失（TrajFlow）替代单纯交叉熵：按各锚与真值的距离排序，让概率的相对大小可校准；L_control 是 simple-v3 的现有项（位置、速度、模仿、时长）只算在被选中的模态上（winner-takes-all）。意图作为显式的辅助目标（IMPACT 的做法：意图标签自动生成、与轨迹联合训练），不只是选锚。
- 邻机的意图也预测：对每架邻机 j，用它自己的历史给出 d_join 的分布 q_j（同一锚集），真值来自邻机自己的航迹（辅助监督，不影响推理契约）。这是到达序列进入模型的显式通道：本机的汇入点受前机汇入点约束（GooDFlight 的"one-then-all"：先独立估计每架的目标分布，再按交互裁剪）。
- 推理：K 条轨迹 + p + d̂_join。

### 3.2 输入表示

**本机历史**：同现在（60 × 6，图坐标位置与速度），加锚点控制状态（已有 `anchor_controls`）。

**邻机**：锚点时刻在同一机场 TMA（25 km 切片半径内或更大，取 40 km）内且 t ≤ t₀ 有样本的航班，来自 `tracks/manifest.json` 的名册（所有 outcome：进场、离场、过境；按名册，不 glob——仓库不变量）。每架一个实体，特征：
- 时序：过去 120 s（与本机同网格，缺失处掩码）在**本机坐标系**里的相对位置和速度（6 通道）；
- 静态：类别（进场落哪条跑道 / 离场 / 过境）、是否与本机同跑道、是否已建立五边（用同一门控定义）、当前离阈值距离与估算 ETA（按当前地速）、在到达序列里的次序。
- 数量可变，上限 N_max = 16（按距离与相关性排序截断），掩码。

**静态几何（"地图"token）**：
- 五边航向道折线：d ∈ [0, 30 km] 每 1 km 一个点（xt = 0），带 d 的位置编码——锚就落在它上面；
- FAF / IF / 阈值点、下滑道角、跑道航向；
- CIFP 的 STAR / 进场程序航段折线（`trajectory_data_process` 的 procedures 输出），作为可能的飞行走廊；
- 平行跑道的航向道折线（KRDU 05L/05R、KSJC 30L/30R）——让模型看到"邻机在另一条跑道上"。

**标量上下文**：同跑道上一次落地距今、本机前面的进场架数（按 ETA 排序）、同时段到达率、时段（小时、工作日）；风（METAR，Phase 4）。

**泄漏红线**：任何邻机特征只用 t ≤ t₀ 的样本；邻机的"落地时间"不能作为特征（它是未来）；ETA 只能由当前状态估算。测试里专门检查。

### 3.3 编码器

VectorNet / Wayformer 的形状，以本机为中心：

1. **实体编码器**（参数共享）：每个时序实体（本机、每架邻机）的 [60 × 6 + 静态属性] → 一个小 Transformer 或 GRU → 一个 token（d_model）；每条折线（航向道、程序航段、平行跑道）→ PointNet 式的逐点 MLP + max-pool → 一个 token；标量上下文 → MLP → 一个 token。
2. **场景注意力**：所有 token 拼成集合（本机 1 + 邻机 ≤ 16 + 折线 ≤ 8 + 标量 1），做 2–4 层自注意力，或更省的形式：本机 token 作为 query 对其余做 cross-attention（Wayformer 的 early fusion / MTR 的局部注意力都可以，先用最简单的全自注意力）。位置编码用本机坐标系里的相对位置（对折线用其中点）。
3. **邻机意图 token**：每个邻机 token 经一个共享的意图头得到 q_j（K 维分布），把 q_j 的嵌入加回该邻机的 token（意图增强的实体表示）；同时把同跑道邻机按估算 ETA 排序的次序作为位置编码——到达序列的结构显式进入注意力。
4. 输出：场景编码 Z（全部 token，含意图增强的邻机 token）与本机 token z_ego。

这一层替代 iTransformer。本机历史本身用 iTransformer/PatchTST 编码仍可（当成实体编码器的一种），但 60 个点不需要它的重装备。若想保留现有主干走一条更低风险的路，MAIFormer（T-ITS 2026）的做法可以直接套：把 N 架邻机的通道当作 N·F 个变量 token 做掩码多变量注意力，再加一层 agent attention（每架邻机一个注意力分数，可解释）；它没有意图 token 和锚解码器，但作为 Phase 2 的编码器足够。

### 3.4 解码器：汇入锚查询

MTR 的"意图查询"/DenseTNT 的目标候选：

1. **锚查询**：K 个可学习 query，每个加上锚的嵌入（d_join 值的位置编码 + 所属侧）。
2. **交叉注意力**：query 对场景编码 Z 做 2 层 cross-attention（其中邻机 token 已带意图，query 能直接"看到"前机大概在哪里汇入），再加一层 query 间的自注意力让模态互相排斥。
3. **每个 query 的头**：
   - 得分头 → logit_k（排序损失训练）；
   - 意图头 → δ_k（汇入距离的连续修正）；
   - 控制头 → 64 段控制 + 时长（复用现有 `ControlOutputHead` / `UniformDurationControlHead` 的结构，输入换成 query 特征）；参数化输出再积分这一形式有外部证据（ASCENT 的消融：预测航向/俯仰/速度再积分优于直接回归位置），我们的控制序列 + 动力学 rollout 是它的更强形式；
   - → 现有 rollout（`rollout_control_endpoints` 训练、稠密 rollout 推理）。
4. **可选的几何封闭**（`2026-09-04_procedure_constraints_design.zh.md` 的 P2）：汇入之后的路径不由网络出，按航向道 + 下滑道 + 减速剖面用逆动力学闭式给出（`control/dynamics/inverse.py`）；网络只出汇入前的控制。这把多模态严格限制在决策量上，代价是末段的个体差异（真实航班末段也有 ±60 m 的散布）被抹掉。建议 Phase 3 先不做封闭，把它作为一个对照臂。

推理时每个模态都可以套屏障过滤器（已采用的推理时安全层）。

### 3.5 损失与训练细节

- 得分：Plackett–Luce 排序损失（按各锚到真值 d_join 的距离给出目标排序），必要时加交叉熵项；若真值与两个锚的距离都在容差（1 km）内，两者并列。
- 意图辅助项：本机 δ_{k*} 的 L1；邻机意图头对每架邻机的 CE（真值 = 邻机自己的 d_join 所在锚；只对有真值的邻机算）。
- 回归：simple-v3 的全部项只在 k* 上算；其余模态不回传（WTA）。这是 MultiPath / MTR / ASCENT 的标准做法，避免模态坍缩。
- 训练成本：回归只需一条 rollout（k*），得分头不需要 rollout，所以训练时 rollout 次数不变；推理 K 次。
- 类别不平衡：直线进近多（KRDU 64 %、KSJC 81 %），锚的先验分布倾斜；交叉熵可加类别权重，或按 d_join 分层采样。
- 与复审文档 P0 的关系：垂直定价、阈值平面事件项可以同时开，但要作为独立臂，否则分不清收益来源。

### 3.6 推理、导出、评估

- `forecast` 对每个模态生成一条预测；导出：top-1 走现有记录契约（`evaluation` 的一条轨迹一个记录，不改契约），其余模态作为同一航班的附加记录写在子目录 `modes/`，`source` 里记 `mode_index`、`mode_probability`、`join_anchor_m`。
- 读数脚本 `docs/compare_multimodal_arms.py`：minADE_K / minFDE_K、miss rate（最好模态 FDE > 2 km 的份额）、softmAP 与 brier-minFDE（驾驶基准的标准，概率参与评分）、锚分类准确率与校准曲线、d̂_join 的误差、top-1 的分层 ADE/FDE（与现有 `compare_frame_arms.py` 同口径），并加一个 2 min 视界的分层读数作为与 ASCENT/MAIFormer 等外部数字的量级锚点。
- 对照：K 条随机锚 + 同一网络（排除"多试几次总有一条近"）；oracle 锚（给真值 k*）的上界。

## 四、架构（放在仓库的哪里）

按 `ts_transformer/CLAUDE.md` 的归属规则（只有所有消费者都是 control 专用的模块才进 `control/`；场景编码 state 输出也能用），布局：

```
trajectory_data_process/            数据平面
  scene_index.py                    tracks/manifest.json → 按机场的时间索引 (flight_key, t_first, t_last, outcome, runway)，缓存 JSON，带 SHA 契约
flight_scenarios/                   数据 → 建模的接缝（已有：垂直基准、身份、速度契约）
  scene_context.py                  给定 (机场, t₀, 本机 flight_key) → 邻机列表（只含 t ≤ t₀ 的样本）+ 标量上下文；泄漏检查在这里
4dTrajectory/ts_transformer/
  scene/
    features.py                     邻机/折线/标量 → 张量（本机坐标系），掩码，N_max
    geometry_tokens.py              航向道折线、FAF/IF、程序航段、平行跑道 → 折线张量（从 final_approach_geometry 与 procedures 输出）
    anchors.py                      d_join 直方图 → K 个锚（按机场），持久化进 checkpoint
  models/
    scene_encoder.py                实体编码器 + 场景注意力
  control/
    decoder/join_anchor.py          锚查询 + 交叉注意力 + 得分头 + 控制头（复用 heads.py 的头）
    loss/multimodal.py              CE + WTA 组装（复用 components.py 的项）
  prediction_outputs.py             MultimodalControlPrediction(logits, controls[K], durations[K])
  dataset.py                        窗口带邻机张量与掩码；batch() 组装
  batch_contract.py                 model_forward 传场景张量
  forecast.py / export.py           K 个模态 → 记录（top-1 走现有契约）
  docs/compare_multimodal_arms.py   读数
  config.py                         prediction_output="control-multimodal"，scene_* 与 anchor_* 字段，recipe
```

数据流：

```
tracks/manifest.json ─→ scene_index（时间索引，缓存）
arrivals/manifest.json ─→ FlightSeries（本机） ─┐
                                               ├→ scene_context(t₀) → 邻机 + 标量 ─→ scene/features → 张量
CIFP procedures / runway_thresholds ─→ geometry_tokens ─┘
       ↓
scene_encoder → Z ─→ join_anchor decoder → {(logit_k, u_k)} ─→ rollout(u_k*) 训练 / rollout(u_k ∀k) 推理
       ↓
loss/multimodal（CE + WTA·simple-v3）             forecast/export（top-1 记录 + modes/）→ evaluation + compare_multimodal_arms
```

不变量：邻机只按名册取（不 glob）；所有邻机样本 t ≤ t₀；坐标系仍是本机的阈值锚定 `enu`（跑道假设扩展的机制不变）；锚随 checkpoint 持久化（换机场重算）。

## 五、实现步骤（每步有测试与决策门）

### 5.0 按 Phase 0 结果重排的优先级（2026-09-05）

排序依据：Phase 0 量到（i）4D 误差的主项是沿路径的时序，其次是几何；（ii）现解码器拿到真值 (d_join, T) 仍只到 2.0 km、几何不动；（iii）粗特征上下文对 d_join 的可解释性弱（R² 0.38，真值前机 ETA 0.47）。所以先把"读数能分辨几何与时序"做好（P0），再修"给定意图 → 几何"的输出侧（P1），之后场景编码（P2/P3）的收益才有兑现空间。每步：写代码 → opus review → 修 → 实验 → 记录 → 下一步。

**P0 — 几何指标进标准读数（完成，2026-09-05，`6867512`）。** 在 `docs/compare_constraint_arms.py`（及 `compare_frame_arms.py` 的表）里为每个分层、每臂加三列时间无关几何误差 + 一列时长误差，与现有 ADE/FDE 并列：
- chamfer（对称最近点均值，100 m 重采样；`phase0_intent_diagnostics.py::chamfer_m` 已有，搬成单一来源）；
- 离散 Fréchet 距离（顺序敏感，不被"铺满区域"的路径骗）；
- 弧长对齐 ADE（在飞完路程的同一比例处比位置——保序、去速度）；
- 时长误差中位与沿路径提前/滞后（真值弧长分数处的时间差）。
真值用观测行（post-anchor `observed_states`）还是含拟合尾的 supervision 行要写明（两者差 6 s / ~400 m，Phase 0 的 chamfer 用了观测行）。回填 Phase 0 四臂与 hook v2 campaign 的表。门：无（读数工具）；验收 = 单元测试（合成路径的已知距离）+ Phase 0 数字复现。

> **做成了什么（2026-09-05）**：`geometric_metrics.py` 是单一来源（诊断脚本的 chamfer 也从这里取）。
> 真值默认 **`closed`**：导出的状态文件只有观测行，没有 4D 指标所用的拟合尾，所以在 `true_final_time_s`
> 处补阈值节点（直线闭合，KRDU 中位 379 m / 6 s、KSJC 82 m / 2 s，读数首行打印），两族指标在同一点、
> 同一时刻结束；`--geometry-truth observed` 复现 Phase 0 的 chamfer（差 ≤ 2 m，重采样含终点）。
> 表列：chamfer p50 / Fréchet p50 / arc-ADE / len ratio / abs Δdur p50 / lag p50（Δdur 按导出状态
> 的时钟，horizon-capped 航班与 `final_time_error_s` 的时长头不同）。**review 发现 state 输出的锯齿折线
> 让弧长参数化膨胀约 2 倍**（相邻节点航向反转 > 90° 的份额中位 0.50、每架都 > 0.05；control rollout
> 与观测真值为 0、真值最大 0.008；长度比 ≈ 2 对 1.01），故每航班带 `reversal_share`，≤ 5 % 才进弧长族
> （`arc_family_valid`），块内只在这些航班上聚合 arc-ADE / lag 并在 JSON 里写 `arc_family_share`，
> 表里份额 < 100 % 时跟着打印、< 95 % 打印 n/a，不平滑（用长度比做门被 review 否决：带内中位仍会把带外
> 航班平均进去，且 control 臂上长度比偏离是真实误差不是伪影）。state campaign 上弧长族因此基本全是
> n/a——那里可读的几何是 chamfer + Fréchet。结果：`2026-09-05_scene_phase0_results.zh.md` §三b、
> `2026-09-06_control_hooks_results.zh.md` "几何指标回填"——汇入点值 15–20 % 几何，时长臂零几何增益；
> 软屏障是唯一雷达引导层几何也改善的 hook。P1 的验收（chamfer < 500 m）按 `closed` 真值读。

**P1 — 输出侧：几何封闭的解码器（走向 C；≈1–2 周；待用户决定后开始）。** 设计 §3.4 的可选项升为主线：网络预测决策量 (d_join, T) 与一个速度剖面（例如沿路程的分段线性地速，或 64 段的速度目标），路径由几何闭式给出——锚点状态 → trombone / Dubins 到 (d_join, 0) → 航向道 + 下滑道（`phase0_intent_diagnostics.py::template_path` 是原型），再按速度剖面给时间；可选再经 rollout 做动力学一致化。
- 先用**真值** (d_join, T) 验收输出侧（oracle 臂，KRDU）：目标雷达引导 ADE < 1.5 km（真值时序下模板已 1.7 km，学到速度剖面应更低）、chamfer < 500 m；若做不到，说明几何族本身不够（三边位置/下风延伸要作为第三个决策量），在这里加，不进场景。
- 然后让网络自己预测 (d_join, T)（无真值），与 simple-v3 基线、O_join_duration 上界一起读四组数；直线进近分层用"straight"分支，必须不退。
- 风险：几何族对非标准引导（转场、盘旋）不适用——统计份额（Phase 0 模板构造计数：雷达引导 497 架里 trombone 305、Dubins 181、直线 2、已飞过汇入点 9），给不适用的航班保留现有控制头作为后备分支。

**P2 — 数据平面（原 Phase 1；≈2–3 天）。** 内容不变（`scene_index`、`scene_context`、`scene/features`、泄漏测试），**验收指标改**：邻机位置/ETA/离阈值距离等实体级特征对 (d_join, T) 的可解释性要明显高于 Phase 0 的粗特征基线（d_join R² 0.38 → 目标 ≥ 0.55；剩余时长中位误差 35.8 s → 目标 < 28 s；`phase0_intent_diagnostics.py context/timing` 是基线脚本）。达不到就说明上下文的价值在别处（跑道使用、离场），先量再建模。

**P3 — 场景编码 + 锚解码器（原 Phase 2/3；≈2 周）。** 内容不变，两处改：锚落在 **(d_join, T)** 的联合直方图上（或 T 为主、d_join 为辅），不再只 d_join；top-1 目标按 §一 修订（< 2.0 km），多模态的价值用 minADE_K 与校准报告，不承诺 top-1 低于 2 km。解码器输出接 P1 的几何封闭（每个模态一条闭式路径 + 速度剖面），控制头 + rollout 作为对照臂。

**P4 — 生成式对照、风、合并机场、渲染、报告（原 Phase 4）。** 不变。

**不再做**：只给汇入点的 oracle 臂（结论已清楚）；用 1.5 km 作为意图信息的门；把 top-1 < 2 km 当成场景编码的承诺；用 k=0.5 真值门判预测是否建立五边。

### 5.1 原计划（Phase 0–4 原文；编号映射 Phase 1→P2，Phase 2/3→P3，Phase 4→P4）

**Phase 0 — 上界实验（1 天，不改架构）。** 用现有的协变量 token 机制把真值 d_join 和真值前机 ETA 喂给 simple-v3，训一臂 KRDU。读 vectored ADE。门：若 < 1.5 km（从 2858），说明意图信息值这么多，继续；若几乎不动，问题在别处，停。

> **结果（2026-09-05，`2026-09-05_scene_phase0_results.zh.md`）**：真值 d_join → 2356 m（−17 %，时长误差 39 → 22 s）；+ 真值前机 ETA 无增量；+ 真值剩余时长 → 2011 m（−30 %），但时间无关的路径误差始终不改善。门未达，且门定错了量级：真值路径配朴素速度剖面就有 1.3 km，真值汇入点配粗糙几何与完美时序 1.7 km。结论：决策变量应含时间；输出侧（控制头 + rollout 按给定意图画几何）是先于场景编码的瓶颈。走向（A 停 / B 改为 (d_join, T) 继续 / C 先修输出侧）待定，推荐 C 再 B。

**Phase 1 — 数据平面（2–3 天）。** `scene_index` + `scene_context` + `scene/features`：
- 测试：泄漏（构造一架邻机只在 t > t₀ 有样本，必须不出现；ETA 不用落地时间）；确定性与缓存契约；名册一致性（邻机 flight_key 必须在 tracks manifest 里）；坐标系（邻机在本机图坐标里的位置与 `runway_axes` 一致）。
- 测量：d_join 对上下文的可解释性——用简单模型（梯度提升或线性）从标量上下文预测 d_join，看 R²/分类准确率。这是不训神经网络就能看到"上下文值多少"的第二道门。

**Phase 2 — 最小改动臂（2–3 天）。** 保留 iTransformer，按 MAIFormer 的方式扩展：邻机通道作为额外变量 token（掩码多变量注意力）+ 一层 agent attention，加标量上下文 token，加 K 类汇入头（排序损失）+ 锚嵌入条件的控制头。目的：用最少的代码验证"上下文进得去、锚分得开"，并得到一个可解释的 agent attention 读数（哪架邻机在决定本机的汇入）。门：锚准确率 > 先验；vectored top-1 ADE 有改善。若主干不用邻机 token（注意力权重/消融），直接进 Phase 3。

**Phase 3 — 场景编码器 + 汇入锚解码器（1–2 周）。**
- 模型：`scene_encoder`、`join_anchor` 解码器、多模态预测输出、损失。
- 测试：邻机置换不变性与掩码不变性；WTA 只回传选中模态（梯度检查）；K 条 rollout 与单条逐比特一致；导出契约（top-1 记录与现有评估兼容，modes/ 记录字段完整）；锚持久化与换机场重算。
- 实验（KRDU 后 KSJC，simple-v3 的损失剂量不变）：臂 = 基线、Phase 2 臂、Phase 3 臂（K = 4 / 6）、随机锚对照、oracle 锚上界；读数 minADE_K、top-1 分层、锚准确率与校准。
- 预注册否决：直线进近 top-1 退；minADE_K 的改善不超过随机锚对照。

**Phase 4 — 收尾与生成式对照。** （i）生成式解码器对照臂：同一场景编码、同一锚条件，用流匹配（TrajFlow 式单次 K 模态，或 Diffusion Policy / Streaming Flow Policy 式对控制序列的条件生成，一步蒸馏）替代确定性控制头，rollout 不变；逐段噪声（Diffusion Forcing）作为它的变体。这是"查询式 vs 生成式"在我们数据上的直接对照——ASCENT 在 TrajAir 上量到查询式更好，我们的数据要自己量。（ii）几何封闭对照臂（3.4 可选项）；每个模态套屏障过滤器；风（METAR）；合并机场训练；CZML 多模态渲染（可选）；报告。

**不做**：把多模态写成对 K 条都回归的损失（会坍缩成平均）；把邻机的落地时间当特征；把锚做成罚项；在训练回路里放 hook。

## 六、风险与开放问题

1. **邻机的未来也未知。** 前机的 ETA 只是估计；真正决定间隔的是管制员对前机未来的判断。模型看到的是前机的现状——与管制员看到的一样，这是公平的，但上限不是 100 %。
2. **K 的选择与锚的稳定性。** 锚按机场从数据聚类；K 太小回到平均，太大分类难、每类样本少。先 K = 4 与 6 两臂。
3. **类别不平衡。** 直线进近占多数，锚的先验倾斜；分类头容易只输出"直接汇入"。类别权重或分层采样，并看校准。
4. **计算成本。** 训练时回归一条 rollout，与现在相同；场景编码器的成本随邻机数线性；推理 K 条 rollout（K ≤ 6，可接受）。
5. **跨机场泛化。** 锚、程序几何、跑道布局都按机场；合并训练时机场/跑道作为类别 token。
6. **评估契约。** 现有评估一条轨迹一个记录；多模态读数是新脚本，top-1 保持可比。要在报告里同时给 top-1 与 minADE_K，避免用 minADE_K 讲"更准了"。
7. **意图之外的不确定性。** 即便汇入点对了，三边上的具体路径仍有个体差异；minADE_K 的剩余误差要分层看（汇入前 / 汇入后）。

## 六b、扩展到多机航迹预测

本设计把多模态放在决策层（汇入点），这正是它能扩到多机的原因。三级：

1. **多目标边际预测**（每架各自 K 条）：把场景中每架飞机轮流当本机。效率与坐标各改一处——场景编码改成查询中心式（QCNet，CVPR 2023：相对位置编码，场景编码一次、全体目标复用），编码在机场坐标系（`airport-enu`）+ 相对编码，解码时每架换回自己跑道的阈值锚定系做 rollout（2026-09-03 的消融证明解码要阈值锚定）。§3.3 的邻机意图头 q_j 已经在预测每架邻机的汇入锚，接上各自的控制头与 rollout 即是多目标输出。工程改动，不改设计。
2. **联合预测**（N 架的未来一致）：同跑道到达流的耦合几乎全在决策层——次序与间隔决定各自的汇入点，之后由物理决定。联合模态 = 一个"次序 + 各自 d_join"的分配，不是 N 条轨迹的笛卡尔积。做法：one-then-all 的联合汇入分布（已在 §3.1）+ 一个显式**序列头**（同跑道进场的落地次序）；采样一个联合分配后每架以自己的锚为条件解码并 rollout；一致性由训练时成对的间隔/落地时刻顺序损失（尾流间隔下限、与序列头一致）和推理时对联合分配的可行性检查（间隔低于最小值的组合剔除）保证。评估换联合口径：joint minADE / mAP（WOMD 交互赛道）、预测轨迹间的间隔违反率、落地次序准确率。这是设计已有部件的组合，加两个头和一个损失。
3. **全场景生成**（TMA 内全部飞机十分钟的联合演化）：自回归 token 模型（SMART、MotionLM）与 Diffusion Forcing 一类（MAGNet）的主场——逐时间步、逐飞机生成，管制次序成为生成过程的一部分。对应仿真/容量类问题而非单架精度；若要做，沿用本文的场景编码与物理 rollout，把解码器换成场景生成器，作为第三阶段。

## 七、2024–2026 的进展与本方案的修订（文献核对，2026-09-07）

§三 的骨架来自 2019–2022 的驾驶预测文献。对照 2024–2026 的工作重新审视一遍，结论：**骨架不换（场景编码 + 汇入锚查询解码 + WTA + 物理 rollout），吸收五处新做法，把"生成式解码器"作为对照臂而不是主线。** 依据如下。

### 7.1 各类方法现在的位置

| 家族 | 代表（年） | 现状 |
|---|---|---|
| 查询式多模态解码器（锚/意图查询 + 分类 + WTA） | MTR（2022）→ MTR v3（2024 WOMD 运动预测第一）；IMPACT（RA-L 2026，WOMD 无激光雷达方法第一，交互预测 softmAP +10 %）；**ASCENT（2026-03，航空）** | 驾驶与航空基准上仍是第一梯队；ASCENT 用 2 层自注意力编码 + K 个可学习模式查询 + WTA + 参数化输出（航向、俯仰、速度 → 积分成位置），在 TrajAir 上 minADE₅ 0.35 km、40 s 历史 + K=20 时 0.19 km，**优于同基准的扩散模型 GooDFlight（0.29 km）**，V100 上 17 ms |
| 扩散式轨迹生成 | MID（2022）、MotionDiffuser（2023）、**GooDFlight（2025，航空）**、TopoDiffuser（2025） | 多样性好、可控（分类器无关引导）；代价是多步去噪、需要更多数据；GooDFlight 把"目标估计"与"轨迹生成"两阶段分开——目标先用混合高斯 + 图注意力做联合分布，再用 10 步扩散生成动作序列并积分 |
| 流匹配（单步/单次多模态） | MoFlow（CVPR 2025）、**TrajFlow（IROS 2025，WOMD SOTA 宣称）**、FlowS（2026-04，一步生成） | 把扩散的采样成本压到一次前向；TrajFlow 单次前向出 K 个模态 + Plackett–Luce 排序损失做概率校准——在"K 条一次出"这点上与查询式解码器殊途同归 |
| 自回归 token 模型 | MotionLM（2023）、SMART（NeurIPS 2024）、AMP（2024）、TrajTok（2025 Sim Agents） | 强在多智能体联合仿真与长时程闭环（Sim Agents 赛道），不是单目标预测精度的第一 |
| AR + 扩散混合（Diffusion Forcing 一系） | Diffusion Forcing（NeurIPS 2024）、MAGNet（2025-12，多人运动）、Epona（2025，驾驶世界模型：MST 编码 + TrajDiT 规划 + VisDiT 生成帧）、Causal/Rolling/Self Forcing（2025–2026，视频） | 强项是**流式/超长序列生成和世界模型**：逐 token 不同噪声水平，能无限外推、能做部分观测的补全。对我们的问题——一个离散决定（汇入点）+ 之后由物理决定的路径、总长 64 段——它的长处用不上，代价（训练数据量、采样成本、评估）用得上 |
| 扩散/流策略（动作序列生成） | Diffusion Policy（2023）、Streaming Flow Policy（2025）、One-Step Flow Policy（2026）、A2A（2026） | 与我们的输出（控制序列）同形：以观测为条件生成动作块；它们说明"对控制序列做生成式建模"是成熟路线，可以作为解码器的替代实现 |
| 航空侧 | MAIFormer（TITS 2026：多智能体倒置 transformer，仁川进场，2 min 视界，单输出）；Neurocomputing 2024（意图作为模型无关的附加条件，来自局部历史相似航班）；TartanAviation 概率学习（2024） | 航空侧刚开始做多智能体与多模态；视界都是 2 min（我们 5–10 min，更难）；没有人把到达序列/间隔当输入 |

### 7.2 对"Multimodal Autoregressive Diffusion Transformer"这条路线的判断

它是 Diffusion Forcing 家族（每个 token 独立噪声水平的因果 transformer，AR 与全序列扩散的统一）在多模态条件下的形态（Epona、MAGNet 是最近的实例）。适合：视界不定长、要流式外推、要同时生成多种模态（图像 + 轨迹）、要在生成中途接受新观测。我们的任务：视界由到达时刻封闭、输出是一个决策 + 一段物理轨迹、训练集 1.4 万航班/机场。三个不匹配：（i）多模态的来源是一个离散决定，查询式解码器直接把它建成分类，扩散/AR 要靠采样去"发现"它，样本效率更低，且 ASCENT 已在航空基准上量到这一点；（ii）AR 逐段生成的优势是闭环——每段看到自己的状态——但我们的动力学 rollout 已经把"状态一致性"保证了，AR 再做一遍是重复；（iii）数据量：扩散模型在 TrajAir（1 万多条 GA 航迹）上被轻量 transformer 超过，我们的数据量同级。所以它不进主线；但它的一个子思想值得留：**逐段的噪声水平 = 逐段的不确定性**，在 Phase 4 可以作为"生成式解码器对照臂"实现（见 7.3 第 5 条）。

### 7.3 修订：吸收进 §三 的五处

1. **参数化输出 + 积分（ASCENT）**：它预测航向/俯仰/速度再积分成位置，消融证明优于直接回归 xyz。我们的控制序列 + 可微 rollout 就是这条路的更强形式（多了动力学、包线、执行器滞后）——保留，并在报告里引用它作为外部证据。
2. **意图作为条件、意图标签可以自动生成（Neurocomputing 2024、IMPACT 2026）**：IMPACT 给 Waymo/Argoverse 自动标注行为意图并联合训练，交互预测第一。我们的 d_join 真值就是自动标注的意图——§3.1 的分类头不只是选锚，还是一个可解释的意图预测；训练时作为辅助任务同时回归 d_join 的连续值。
3. **邻机的意图也预测，并让意图之间交互（GooDFlight 的"one-then-all"）**：先给每架飞机（本机 + 邻机）独立估计汇入/目标分布，再用图注意力按意图冲突裁剪成联合分布。这正对应到达序列：前机的汇入点约束本机的汇入点。§3.3 加一个"邻机意图头"（同一 d_join 锚集，只作辅助监督，真值来自邻机自己的航迹），§3.4 的锚查询对邻机意图 token 做交叉注意力。
4. **概率校准用排序损失（TrajFlow 的 Plackett–Luce）**，替代/补充交叉熵；评估加 softmAP / brier-minFDE（驾驶基准的标准），与 minADE_K 一起报。
5. **生成式解码器作为对照臂（Phase 4）**：同一场景编码 + 同一锚条件下，用流匹配（TrajFlow 式单次 K 模态，或 Diffusion Policy 式对控制序列的条件生成，一步蒸馏）替代确定性控制头，rollout 不变。这是"查询式 vs 生成式"在我们数据上的直接对照，也是把 AR-扩散思想收编的位置：若做逐段噪声（Diffusion Forcing），就是这一臂的变体。

另加一条低风险的中间路线：**MAIFormer 式扩展现有主干**——把 N 架邻机的 3 通道当作 N·F 个变量 token、加一层 agent attention，保留 iTransformer。它是 Phase 2 的更自然实现（比"协变量 token"更贴合主干的归纳偏置），Phase 3 若时间不够可以停在这里加锚解码器。

### 7.4 对成功标准的修订

航空侧可比的外部数字：ASCENT/GooDFlight 在 TrajAir 上 120 s 视界 minADE₅ 0.19–0.35 km；MAIFormer 2 min 视界。我们的视界 5–10 min、场景是塔台管制下的商业进场，不可直接比，但给出了两个尺度：（i）短视界（2 min）上我们应当接近它们的量级——建议在报告里加一个 2 min 视界的分层读数作为外部锚点；（ii）minADE_K 与 top-1 的差就是"意图不确定性"的量，ASCENT 从 K=5 到 K=20 minADE 从 0.35 降到 0.19 km，说明这个差在航空数据上很大——与我们 §7.1 的误差预算一致。

## 八、参考文献

**2023–2026（现状，本方案的直接依据）**

- Prutsch, Schinagl, Possegger. *ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace.* arXiv 2603.16550, 2026-03. — 航空侧 SOTA：模式查询 + WTA + 参数化输出，优于扩散模型。
- Yang, Liu, Chen, Cheng, Shi, Zou. *GooDFlight: Goal-oriented Diffusion Model for Flight Trajectory Prediction.* 2025（TrajAir）。— 目标估计（one-then-all）+ 目标条件扩散生成动作序列。
- Yoon, Lee. *Multi-Agent Inverted Transformer for Flight Trajectory Prediction (MAIFormer).* arXiv 2509.21004，IEEE T-ITS 2026。— 多智能体倒置 transformer，仁川进场。
- *Aircraft trajectory prediction in terminal airspace with intentions derived from local history.* Neurocomputing 615, 2024。— 意图作为模型无关条件。
- Sun et al. *IMPACT: Behavioral Intention-aware Multimodal Trajectory Prediction with Adaptive Context Trimming.* RA-L 2026（arXiv 2504.09103）。— 自动标注意图 + 联合训练 + 上下文裁剪，WOMD 交互预测第一。
- Shi, Jiang, Dai, Schiele. *MTR++* / MTR v3（2024 WOMD 运动预测第一）。— 查询式解码器的现役形态。
- *TrajFlow: Multi-modal Motion Prediction via Flow Matching.* IROS 2025（arXiv 2506.08541）。— 单次前向 K 模态 + Plackett–Luce 排序损失。
- Fu et al. *MoFlow: One-Step Flow Matching for Human Trajectory Forecasting.* CVPR 2025；*FlowS: One-Step Motion Prediction via Local Transport Conditioning.* 2026-04。— 一步生成。
- Chen et al. *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion.* NeurIPS 2024；Maluleke et al. *MAGNet: Multi-Agent Motion Generation via Diffusion Forcing.* 2025-12；*Epona: Autoregressive Diffusion World Model for Autonomous Driving.* 2025。— AR + 扩散混合（7.2 的判断对象）。
- Wu et al. *SMART: Scalable Multi-agent Real-time Motion Generation via Next-token Prediction.* NeurIPS 2024；Seff et al. *MotionLM.* ICCV 2023。— 自回归 token 模型。
- Chi et al. *Diffusion Policy.* RSS 2023；*Streaming Flow Policy* 2025；*One-Step Flow Policy* 2026。— 控制序列的生成式建模。
- Abdel Madjid et al. *Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions.* arXiv 2503.03262, 2025。— 综述。

**2019–2022（起源，§三 的形状）**

- VectorNet（CVPR 2020）；TNT（CoRL 2020）、DenseTNT（ICCV 2021）；MultiPath（CoRL 2019）、MultiPath++（ICRA 2022）；MTR（NeurIPS 2022）；Wayformer（ICRA 2023）；Scene Transformer（ICLR 2022）、HiVT（CVPR 2022）；Rupprecht et al. 多假设/WTA（ICCV 2017）；Social LSTM（2016）、Social GAN（2018）。

**领域**：ICAO Doc 4444 间隔标准；AMAN/到达管理文献（前机 ETA 与排队解释 d_join 的依据）；`docs/literature/runway_assignment/README.md`。
