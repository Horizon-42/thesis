# 场景编码 + 汇入锚多模态的 control 预测（dev 文档，2026-09-07）

承接 `2026-09-07_control_training_review.zh.md` §七：pooled ADE 的 60–76 % 来自雷达引导航班，它们的误差是"三边飞多远、何时汇入五边"这个管制决定，本机 120 s 历史里没有这个信息，单点回归只能输出各种决定的平均。本文给出把**交通上下文**放进输入、把**汇入决策**做成多模态输出的完整方案：意图、当前方案的缺陷、设计、架构、实现步骤、风险与参考文献。实现之前先按 §五 的 Phase 0 做上界实验，不通过就不动架构。

## 一、意图

**要预测的东西**：一架进场飞机从锚点（进入 25 km 切片后 120 s）到过阈值的 4D 轨迹，现在由 control 模型给出——64 段常值控制 + 时长，经可微点质量 rollout 得到路径，动力学一致。

**要改的东西**：让模型知道决定汇入位置的外部信息（同跑道前机、排队、跑道使用），并把汇入位置这个离散的决定显式地建模成 K 个候选（多峰），而不是一个平均。

**成功的定义（预注册）**：
- 雷达引导分层：minADE_K（K ≤ 6）相对基线 ADE 下降 ≥ 40 %（KRDU 2858 m → < 1700 m）；top-1 ADE 不差于基线。
- 直线进近分层：top-1 ADE/FDE 不退（这一层现在已经不错，方案不能用它换雷达引导层）。
- 汇入锚分类：top-1 准确率显著高于按先验的猜测；概率校准（预测概率 0.8 的锚命中率 ≈ 0.8）。
- 动力学一致性、走廊约束、推理时屏障过滤器：每个模态都保留。
- 否决：直线进近 top-1 FDE 退 > 种子噪声；或 minADE_K 的改善全部来自"K 条里总有一条碰巧近"（对照：K 条随机锚 + 同一网络）。

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
- 输出：对每个锚 k，概率 p_k 和一条控制序列 u_k（64 段 + 时长）；u_k 经现有 rollout 得到轨迹 τ_k。
- 训练：k* = argmin_k |d_join − a_k|；损失 = CE(p, k*) + λ·L_control(u_{k*})，其中 L_control 是 simple-v3 的现有项（位置、速度、模仿、时长）只算在被选中的模态上（winner-takes-all）。可选：对 |d_join − a_k| 在容差内的锚给软目标。
- 推理：K 条轨迹 + p。

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
3. 输出：场景编码 Z（全部 token）与本机 token z_ego。

这一层替代 iTransformer。本机历史本身用 iTransformer/PatchTST 编码仍可（当成实体编码器的一种），但 60 个点不需要它的重装备。

### 3.4 解码器：汇入锚查询

MTR 的"意图查询"/DenseTNT 的目标候选：

1. **锚查询**：K 个可学习 query，每个加上锚的嵌入（d_join 值的位置编码 + 所属侧）。
2. **交叉注意力**：query 对场景编码 Z 做 2 层 cross-attention（也可以带一层 query 间的自注意力，让模态互相排斥）。
3. **每个 query 的头**：
   - 得分头 → logit_k；
   - 控制头 → 64 段控制 + 时长（复用现有 `ControlOutputHead` / `UniformDurationControlHead` 的结构，输入换成 query 特征）；
   - → 现有 rollout（`rollout_control_endpoints` 训练、稠密 rollout 推理）。
4. **可选的几何封闭**（`2026-09-04_procedure_constraints_design.zh.md` 的 P2）：汇入之后的路径不由网络出，按航向道 + 下滑道 + 减速剖面用逆动力学闭式给出（`control/dynamics/inverse.py`）；网络只出汇入前的控制。这把多模态严格限制在决策量上，代价是末段的个体差异（真实航班末段也有 ±60 m 的散布）被抹掉。建议 Phase 3 先不做封闭，把它作为一个对照臂。

推理时每个模态都可以套屏障过滤器（已采用的推理时安全层）。

### 3.5 损失与训练细节

- 分类：交叉熵，目标 k*；若真值 d_join 与两个锚的距离都在容差（1 km）内，用软目标。
- 回归：simple-v3 的全部项只在 k* 上算；其余模态不回传（WTA）。这是 MultiPath / MTR 的标准做法，避免模态坍缩。
- 训练成本：回归只需一条 rollout（k*），得分头不需要 rollout，所以训练时 rollout 次数不变；推理 K 次。
- 类别不平衡：直线进近多（KRDU 64 %、KSJC 81 %），锚的先验分布倾斜；交叉熵可加类别权重，或按 d_join 分层采样。
- 与复审文档 P0 的关系：垂直定价、阈值平面事件项可以同时开，但要作为独立臂，否则分不清收益来源。

### 3.6 推理、导出、评估

- `forecast` 对每个模态生成一条预测；导出：top-1 走现有记录契约（`evaluation` 的一条轨迹一个记录，不改契约），其余模态作为同一航班的附加记录写在子目录 `modes/`，`source` 里记 `mode_index`、`mode_probability`、`join_anchor_m`。
- 读数脚本 `docs/compare_multimodal_arms.py`：minADE_K / minFDE_K、miss rate（最好模态 FDE > 2 km 的份额）、锚分类准确率与校准曲线、top-1 的分层 ADE/FDE（与现有 `compare_frame_arms.py` 同口径）。
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

**Phase 0 — 上界实验（1 天，不改架构）。** 用现有的协变量 token 机制把真值 d_join 和真值前机 ETA 喂给 simple-v3，训一臂 KRDU。读 vectored ADE。门：若 < 1.5 km（从 2858），说明意图信息值这么多，继续；若几乎不动，问题在别处，停。

**Phase 1 — 数据平面（2–3 天）。** `scene_index` + `scene_context` + `scene/features`：
- 测试：泄漏（构造一架邻机只在 t > t₀ 有样本，必须不出现；ETA 不用落地时间）；确定性与缓存契约；名册一致性（邻机 flight_key 必须在 tracks manifest 里）；坐标系（邻机在本机图坐标里的位置与 `runway_axes` 一致）。
- 测量：d_join 对上下文的可解释性——用简单模型（梯度提升或线性）从标量上下文预测 d_join，看 R²/分类准确率。这是不训神经网络就能看到"上下文值多少"的第二道门。

**Phase 2 — 最小改动臂（2 天）。** 现有主干 + 上下文协变量 token + K 类汇入头（分类损失）+ 锚嵌入条件的控制头。目的：用最少的代码验证"上下文进得去、锚分得开"。门：锚准确率 > 先验；vectored top-1 ADE 有改善。若主干再次不用协变量 token（可用注意力权重或消融测），直接进 Phase 3。

**Phase 3 — 场景编码器 + 汇入锚解码器（1–2 周）。**
- 模型：`scene_encoder`、`join_anchor` 解码器、多模态预测输出、损失。
- 测试：邻机置换不变性与掩码不变性；WTA 只回传选中模态（梯度检查）；K 条 rollout 与单条逐比特一致；导出契约（top-1 记录与现有评估兼容，modes/ 记录字段完整）；锚持久化与换机场重算。
- 实验（KRDU 后 KSJC，simple-v3 的损失剂量不变）：臂 = 基线、Phase 2 臂、Phase 3 臂（K = 4 / 6）、随机锚对照、oracle 锚上界；读数 minADE_K、top-1 分层、锚准确率与校准。
- 预注册否决：直线进近 top-1 退；minADE_K 的改善不超过随机锚对照。

**Phase 4 — 收尾。** 几何封闭对照臂（3.4 可选项）；每个模态套屏障过滤器；风（METAR）；合并机场训练；CZML 多模态渲染（可选）；报告。

**不做**：把多模态写成对 K 条都回归的损失（会坍缩成平均）；把邻机的落地时间当特征；把锚做成罚项；在训练回路里放 hook。

## 六、风险与开放问题

1. **邻机的未来也未知。** 前机的 ETA 只是估计；真正决定间隔的是管制员对前机未来的判断。模型看到的是前机的现状——与管制员看到的一样，这是公平的，但上限不是 100 %。
2. **K 的选择与锚的稳定性。** 锚按机场从数据聚类；K 太小回到平均，太大分类难、每类样本少。先 K = 4 与 6 两臂。
3. **类别不平衡。** 直线进近占多数，锚的先验倾斜；分类头容易只输出"直接汇入"。类别权重或分层采样，并看校准。
4. **计算成本。** 训练时回归一条 rollout，与现在相同；场景编码器的成本随邻机数线性；推理 K 条 rollout（K ≤ 6，可接受）。
5. **跨机场泛化。** 锚、程序几何、跑道布局都按机场；合并训练时机场/跑道作为类别 token。
6. **评估契约。** 现有评估一条轨迹一个记录；多模态读数是新脚本，top-1 保持可比。要在报告里同时给 top-1 与 minADE_K，避免用 minADE_K 讲"更准了"。
7. **意图之外的不确定性。** 即便汇入点对了，三边上的具体路径仍有个体差异；minADE_K 的剩余误差要分层看（汇入前 / 汇入后）。

## 七、参考文献

轨迹预测的场景编码与多模态（自动驾驶，本方案的直接来源）：

- Gao, Sun, Zhao, Shen, Anguelov, Li, Schmid. *VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation.* CVPR 2020. — 实体/折线向量化 + 全局图注意力（§3.3 的实体编码器与场景注意力）。
- Zhao et al. *TNT: Target-driveN Trajectory Prediction.* CoRL 2020；Gu, Sun, Zhao. *DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets.* ICCV 2021. — 先预测目标点再补轨迹（§3.4 的汇入锚 = 目标候选）。
- Chai, Sapp, Bansal, Anguelov. *MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction.* CoRL 2019；Varadarajan et al. *MultiPath++.* ICRA 2022. — 锚 + 分类 + 残差回归，winner-takes-all（§3.5）。
- Shi, Jiang, Dai, Schiele. *Motion Transformer with Global Intention Localization and Local Movement Refinement (MTR).* NeurIPS 2022. — 意图查询做多模态解码（§3.4 的锚查询）。
- Nayakanti et al. *Wayformer: Motion Forecasting via Simple & Efficient Attention Networks.* ICRA 2023. — 场景 token 的早期融合与以自车为中心的注意力（§3.3）。
- Ngiam et al. *Scene Transformer: A Unified Architecture for Predicting Multiple Agents' Trajectories.* ICLR 2022；Zhou et al. *HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction.* CVPR 2022. — 多智能体的联合/层次编码（邻机数多时的替代）。
- Rupprecht et al. *Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses.* ICCV 2017. — 多假设 + WTA 训练的理论依据。
- Alahi et al. *Social LSTM.* CVPR 2016；Gupta et al. *Social GAN.* CVPR 2018. — "社会上下文"进入轨迹预测的起点。

航空侧（需要在写论文时逐一核对适用范围）：

- Zeng, Chu, Xu, Liu, Quan. *Aircraft 4D Trajectory Prediction in Civil Aviation: A Review.* Aerospace 2022. — 综述，含 TMA 预测的误差量级与方法谱系。
- Pang, Xu, Liu. *Data-driven trajectory prediction with weather uncertainties: A Bayesian deep learning approach.* Transportation Research Part C 2021. — 不确定性建模与外部条件（天气）输入。
- Liu, Hansen. *Predicting Aircraft Trajectories: A Deep Generative Convolutional Recurrent Neural Networks Approach.* arXiv 2018. — 生成式（多峰）航迹预测的早期工作。
- 与本项目已有文档的关系：`2026-09-04_constraint_methods_survey.zh.md`（约束方法）、`2026-09-03_runway_hypothesis_expansion.md`（K=2 的跑道模态实验）、`docs/literature/runway_assignment/README.md`（跑道分配的阅读清单，其中的到达序列/间隔文献与本文 §3.2 的上下文特征直接相关）。

领域知识：管制员按尾流/雷达间隔与到达序列引导三边长度（ICAO Doc 4444 的间隔标准；AMAN/到达管理的文献，如 Eurocontrol 的 AMAN 报告）——这是"前机 ETA 与排队数解释 d_join"的依据，Phase 1 的测量会给出它在数据上的强度。
