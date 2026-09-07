# 随观测演进的预测 + 校准的到达时刻分布：调度交付形态的三条主线（dev 文档，2026-09-07）

> 写于 2026-09-06 晚，接在 `2026-09-07_latent_intent_design.zh.md` 之后；本文的 base、分层、否决规则
> 全部沿用它，不重复定义。分支 `dev-leg-ctrl`，HEAD `dfa7e48`。

**本文回答的问题**：L2 之后，哪些工作不依赖"还没找到的意图输入"，却能直接改变交付给调度程序的东西。
三条：**A. 随观测时间演进的预测**（预测在飞行过程中如何变准，以及"何时可信"本身作为交付物）；
**B. 校准的到达时刻分布**（分位数 + conformal 校准，再由 CTA 条件解码器给每个分位数一条可飞航迹）；
**C. 控制参数空间的条件扩散 + rollout 引导**（2026-09-07 补入：把 A 的前缀条件、B/L3 的 CTA 条件和
多模态采样统一到一个采样器里，替代 L2 的 CVAE，而不是叠加在它上面）。
三条都不承诺 top-1 ADE 改善；它们改变的是**读数的维度**（时间轴、区间轴、样本轴），不是点估计。

---

## 〇、状态表（压缩 context 后从这里继续）

**当前状态（2026-09-07）**：A0 / B0 的**代码已建成**（分支 `dev-a0`，命令见 §〇.1）；B0 已在现有产物上
跑出读数（§〇.2），A0-fixed 的回放是 GPU 队列里的作业，未跑。其余各项仍为设计稿。A0 / B0 用现有
checkpoint 与现有预测目录即可做，排在 L2.e′ 之后、L3 campaign 之前或并行（B0 是纯 CPU 读数，A0 只推理）。

| 阶段 | 状态 | 产物 / commit | 门 |
|---|---|---|---|
| A0 重锚曲线（测量） | **两臂都回放了（2026-09-07 夜，§2.4b）**：fixed 曲线单调、warm 逐 bin 优于 native32、closure 8→6 km 崩溃；random 臂 L−1 惨败（早停第 10 轮）但 chamfer 六 bin 全优——选择指标只看 L−1 → A1 提前 | `anytime_a0_20260907/`、`anytime_a0_random_20260907/` | 门 1 过、门 2 12–16 km、门 3 被时长地板挡住 |
| A1 流式评估协议 | 未做 | `evaluation_protocol` 新增按剩余路程分 bin 的锚点网格；`compare_constraint_arms.py` 增列 | 无门，是读数 |
| A2 候选重加权（预测期滤波） | 未做 | `forecast.py` 新增 `reweighted_mode_forecasts`；`predict --stream-dt` | 同锚点下劣于无状态版即否决 |
| A3 学习的递归先验 | 未做，取决于 A2 | `control/latent.py` 先验网络吃上一轮后验 | 仅当 A2 有增益 |
| B0 时长误差分布（测量） | **完成（2026-09-07，`dev-a0` `fe84e76` + 测试 `5f408df`）**，读数见 §〇.2 | `run_ts_eta_error_readout.py`（新 runner）+ `tests/test_eta_error_readout.py`（7 项）：读现有 `summary.json` 的 `final_time_error_s` 与 `fde_m`，按 `strata_masks` 分层报 \|Δt\| p50/p80/p90 与带号 p10/p50/p90 | 无门，定区间宽度的量级 |
| B1 分位数时长头 | 未做 | config 轴 `duration_head ∈ point \| quantile`；`prediction_outputs.QuantileFinalTimeHead` | 中位数不劣于点估计头（种子噪声内） |
| B2 split-conformal 校准 | 未做 | `calibration.py`（新，顶层）；校准表进 `checkpoint_metadata.json` | 校准后 80 % 区间实测覆盖 ∈ [0.76, 0.84] |
| B3 分位数条件航迹 | 未做 | `predict --cta-from-quantiles`：CTA 来自自身分位数头，不读未来 | 每条分位数航迹可飞率不低于 top-1；真值落在扇面内的份额 ≥ 名义覆盖 |
| B4 区间宽度随剩余时间（A × B） | 未做 | A1 网格 × B2 区间 | 交付物本身：冻结点曲线 |
| C0 拟合表检索上界（测量） | 未做，前置 = L5.a 拟合表 | `run_ts_schedule_retrieval.py`（新）：按锚点状态取最近 K 条拟合控制序列各自 rollout | minADE_16 < native32 top-1；记忆上界 = 真值航班自己的拟合序列 |
| C1 控制空间条件扩散 | 未做 | `control/diffusion.py`，config 轴 `control_sampler ∈ none \| diffusion`，run name `control+dif` | minADE_16 < top-1 且 < 同 K 的无条件采样对照；直线进近 top-1 不退 |
| C2 rollout 引导（前缀 / CTA / 走廊） | 未做，取决于 C1 | `predict --diffusion-samples K --guide-prefix-s --guide-cta-s` | 前缀越长样本散布单调收窄；CTA 引导的到达时刻误差 ≤ `cta=given` 的恒等检查 + 5 s |
| C3 = A2 的扩散形态 | 未做，取决于 C2 与 A0 | 锚点网格 × 前缀引导 | 同 A2 的门，对照 A2 的重加权版本 |

**参照数字（KRDU val，全部来自当前产物）**：

| 臂 | 全体 ADE | 雷达引导 ADE | 直线进近 ADE | 时长误差 |
|---|---:|---:|---:|---:|
| native32（L1，control base） | 1322 | 2870 | 445 | — |
| L2.d 热启动 β=0.01（当前最好点估计） | 1214 | 2643 | 402 | — |
| closure C_pred | 996 | 2197 | 310 | — |
| simple-v3 control（Phase 0 基线 A） | 1333 | 2858 | 469 | 39 s |
| + 真值汇入点（O_join） | — | 2356 | — | 22 s |
| + 真值汇入点 + 真值剩余时间 | 1005 | 2011 | — | 5 s |

Phase 0 的另外三条测量决定了本文的方向：雷达引导层 4D 误差的主项是**时序**；`corr(剩余路程, 时长) = 0.83`
而 `corr(d_join, 时长) = 0.29`；因果上下文对 d_join 的 R² 只到 0.38，L4 场景实体特征零增量。

### 〇.1 A0 / B0 的运行命令（2026-09-07 建成）

A0-fixed，§2.2 的三个臂一次跑完。三个 checkpoint 都已实测能在**非默认锚点**回放（KRDU val，CPU，
锚点 87–260 对 L−1 = 59）。

```bash
conda activate aeroviz
E=4dTrajectory/outputs/KRDU/experiments
python run_ts_anytime_curve.py \
    --checkpoint L1_native32=$E/l1_lowdim_20260907/L1_native32/checkpoint.pt \
    --checkpoint L2d_warm_beta0p01=$E/l2_warm_posterior_20260907/L2d_warm_beta0p01/checkpoint.pt \
    --checkpoint C_pred=$E/closure_p1c_20260905/C_pred/checkpoint.pt \
    --out $E/anytime_a0_20260907 --split val --bins-km 20,16,12,8,6,4,2 --min-future-s 60
```

**规模与代价**：3 臂 × 7 bin × 1404 架 ≈ 2.9 万次「前向 + dense rollout」。合批是**按锚点**做的
（同一次 `forecast_approaches` 只吃一个锚点），而真实数据上锚点几乎两两不同，**有效批量 ≈ 2.7**，
`--batch-size` 基本不起作用。实测 control 臂在 CPU 上约 **76 min / 臂**（1404 架 × 7 bin），
closure 臂便宜约两个数量级（它的前向没有 rollout）。先用 `--limit N` 冒烟（产物会标 SMOKE TEST，
覆盖率分母取实际建成的航班数），确认无误再跑全量。

`--command-hook barrier --hook-saturation soft` 与 `predict` 同义，用于给「已采用的交付形态」画曲线；
`cta_conditioning=given` 与 `intent_conditioning=truth-…` 的 checkpoint 一律拒绝（前者时长即真值，
后者的真值汇入点/前机落地时间在**每个锚点都重新读一次未来**，曲线会变成「oracle 收敛得多快」）。

**closure 在 A0-fixed 里是对照，不是竞争者**：它离开 L−1 后分布外代价最大。同一批 KRDU val 航班上，
12 km 处 closure 的雷达引导 ADE ≈ 6.7–8.1 km、时长误差 ≈ 172–176 s，而 native32 是 1128 m / 63 s——
closure 在 L−1 处的 996 m 是它的强项，重锚后不是。读它是为了量分布外代价，不是为了比谁准。

**§2.3 的网格与 `--min-future-s 60` 在近端互相矛盾**：2 km 处真值只剩约 27 s（75 m/s），4 km 处约
53 s，都在 60 s 地板之下，所以按字面这两个 bin 对绝大多数航班是空的。读数会把它们打成
`n=0 / cov 0.00 / partial`（不会静默），且 §2.4 的门 2 只问 s > 4 km，不依赖它们——但 **s_freeze
就只能在 ≥ 6 km 的 bin 上定**；若曲线到 6 km 仍 > 30 s，要定出 s_freeze 必须以更低的地板重跑
（例如 `--min-future-s 20`），并把地板写在结果里，因为它改变的是 bin 的人口而不只是范围。

**时长头有 ~125 s 地板**（包级已知陷阱），所以剩余路程小到一定程度后 **\|Δt\| 反而随着接近跑道上升**：
L1_native32 的雷达引导 \|Δt\| p80 从 12 km 的 64 s 涨到 4 km 的 141 s。读数因此每格都打印
`pred T p50`，freeze 段也把这个地板写出来；s_freeze 不存在时打印「not reached in the bins read」，
不打印裸的 never。

B0，同三个臂的 `_pred_val` 目录，纯读数（秒级，CPU）：

```bash
python run_ts_eta_error_readout.py \
    L1_native32=$E/l1_lowdim_20260907/L1_native32_pred_val \
    L2d_warm_beta0p01=$E/l2_warm_posterior_20260907/L2d_warm_beta0p01_pred_val \
    C_pred=$E/closure_p1c_20260905/C_pred_pred_val \
    --json $E/b0_eta_error_20260907/eta_error.json
```

> §〇.2 的数字是用上面这条命令去掉 `--json` 读出来的（写产物的那一步要在主工作树跑，
> `4dTrajectory/outputs` 在 dev 工作树里是只读软链）。**引用这些数字之前，产物
> `b0_eta_error_20260907/eta_error.json` 必须先由上面的完整命令写出来**——本仓的规矩是只引当前产物。

### 〇.2 B0 的读数（KRDU val，1404 架，2026-09-07 当前产物，覆盖 1404/1404）

其中 \|Δt\| = \|`final_time_error_s`\|，秒；带号为负 = 预测偏早。

| 臂 | 分层 | n | \|Δt\| p50 | p80 | p90 | 带号 p10 | p50 | p90 | FDE p50 | p80 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native32 | 全体 | 1404 | 14.9 | 39.9 | 63.8 | −42.3 | +3.5 | +38.3 | 864 | 2142 |
| native32 | 直线进近 | 904 | 9.9 | 20.3 | 26.5 | −17.7 | +3.6 | +21.7 | 671 | 1139 |
| native32 | 雷达引导 | 497 | 39.3 | 72.5 | 87.1 | −74.8 | +2.2 | +67.0 | 1982 | 4831 |
| L2.d β=0.01 | 全体 | 1404 | 14.1 | 39.3 | 63.7 | −41.0 | +3.4 | +37.8 | 827 | 2027 |
| L2.d β=0.01 | 直线进近 | 904 | 9.4 | 18.9 | 25.6 | −17.7 | +3.4 | +20.3 | 594 | 1041 |
| L2.d β=0.01 | 雷达引导 | 497 | 39.2 | 68.6 | 82.8 | −70.7 | +3.8 | +68.3 | 1852 | 4517 |
| closure C_pred | 全体 | 1404 | 9.9 | 32.3 | 57.3 | −35.0 | −1.4 | +28.8 | 11 | 774 |
| closure C_pred | 直线进近 | 904 | 6.3 | 12.0 | 15.7 | −13.5 | −0.8 | +10.7 | 12 | 477 |
| closure C_pred | 雷达引导 | 497 | 33.7 | 65.8 | 81.5 | −74.3 | −7.1 | +56.6 | 11 | 2403 |

四条结论，都直接决定 B1–B2 的形状：

1. **按分层校准是必须的，不是精细化**：雷达引导层的 \|Δt\| p80 是直线进近层的 3.6–5.5 倍
   （65.8–72.5 s 对 12.0–20.3 s）。合并校准会给直线航班一个荒谬的宽区间——§3.2 的前提在数据上成立。
2. **B 不是一开始就死的，但余量只有一倍**：§三的否决线是雷达引导层区间宽度中位数 > 120 s；
   当前 p80 是 65.8–72.5 s，一个覆盖 80 % 的对称区间宽度就已是这个数的两倍量级。
   B2 之后贴着否决线是可能结局，这正是 §八 风险 4。
3. **区间不能以 0 为中心**：带号 p50 是 +3.4/+3.5 s（native32、L2.d 偏晚）与 −1.4 s（closure 偏早），
   而直线进近层的 p10/p90 落在 ±13–22 s，即分布本身近似对称但**有偏移**。分位数头（B1）
   而不是「点估计 ± δ」是对的形态。
4. **closure 的 FDE p50 = 11 m 不是精度**：它按构造把路径画到跑道头，终点误差因此几乎为零，
   而它的 ADE 仍是 996 m。这条正是「两组指标一起读」的样例，FDE 单独读会得出相反结论。

---

## 一、为什么是这三条

误差预算（`latent_intent_design` §〇）：C_pred 2197 → C_truth_intent 1235 的 962 m 是本机在锚点看不见
的引导意图。三轮 L2 已证明 z 只能带到其中约五分之一；L4 的门没过。在找到携带意图的输入之前，任何
架构的 top-1 都封顶在同一处。

但"看不见"是**锚点时刻**的性质，不是航班的性质：

1. **意图随飞行逐步暴露。** 评估锚点固定在 L−1，即 25 km 到达切片起点后 120 s（`seq_len 60 × dt 2 s`），
   是全程信息最少的时刻。飞机飞过入口正横后，每多飞 10 s 不转三边，就排除掉一批更短的汇入距离；
   转向三边后剩余几何几乎确定，只剩速度剖面。所以对 (d_join, T) 的后验会随观测自然收窄，**不需要任何
   新输入**。这一点现在没有被测量过：每架飞机在评估里只有一个锚点、一个 ADE。
2. **调度程序要的是分布与时间轴，不是一个点。** AMAN 的排序冻结点太早牺牲效率，太晚序列反复变动；
   它需要的量是"落地前多少分钟，到达时刻能信到 ±30 s"。这个量在本包的读数里不存在。
3. **T 的条件分布可学，d_join 的不可学。** 剩余路程与时长相关 0.83；时长头目前是点估计（`FinalTimeHead`，
   `softplus(raw) × scale`），整个包里没有分位数、区间或校准代码（grep 空）。给 T 一个校准的区间，
   再用 L3 已建成的 `cta_conditioning=given` 把每个分位数解成一条按构造可飞的航迹，多模态就不必靠
   z 采样。

三条的关系：B 给出某个时刻的到达时刻区间；A 给出这个区间随时间怎么收窄；C 是把 A 的前缀条件与 B 的
CTA 条件放进同一个采样器的方式，只在 L2 的 CVAE 承载不了意图时启动（§4.4）。B4 = A × B 是最终交付物。

---

## 二、A. 随观测演进的预测

### 2.1 定义

同一架飞机在剩余路程 s 处的预测 P(s)：以该处为锚点、其前 120 s 为回看窗口的一次预测。
交付曲线 E(s) = 某指标在剩余路程 s 处的分层统计（雷达引导层 ADE、时长误差 p50 / p80、B 的区间宽度）。

**推理形态**（三级，逐级需要更多代码，逐级由前一级的门决定）：

- **A1 无状态流式**：每个周期（默认 10 s）以"现在"为锚点从头预测一次。现有代码已能做：锚点是
  `dataset` 的参数，`forecast.default_anchor` 只是默认值。产物是每架飞机的一串预测。
- **A2 候选重加权**：保留上一周期的 K 条候选（L2 的 `modes/`、或 L5.a 拟合教师表的最近邻检索），
  用这 10 s 实际飞过的路径给每条打分（候选航迹前 10 s 与观测的 ADE），偏离的降权，再以新锚点重新
  解码。实质是意图上的粒子滤波，粒子 = 候选。**只在预测期，不改训练**。
- **A3 学习的递归先验**：`PriorNetwork` 多吃上一轮的后验参数 (μ, logσ) 作为输入，训练时用同一航班相邻
  锚点的序列来教。需要改训练与数据集；只有 A2 有增益且 A2 的增益随周期递增时才做。

### 2.2 一个已知的陷阱：固定锚点训练的模型在别的锚点是分布外

所有当前 checkpoint 都在 L−1 训练（`FixedAnchorTrajectoryWindows`）。直接在更靠后的锚点回放，
输入分布（离入口更近、更低、更慢、常已建立）与训练分布不同。A0 因此必须是**两臂**：

- **A0-fixed**：现有 checkpoint（native32、L2.d warm β=0.01、closure）在锚点网格上回放。读数含分布外效应。
  三者都已实测能在非默认锚点回放（§〇.1），其中 **closure 是分布外代价的对照臂**，不是精度竞争者。
  `cta_conditioning=given` 与 `intent_conditioning=truth-…` 的 checkpoint 被 runner 拒绝：
  两者都读未来，且后者在每个锚点都重读一次。
- **A0-random**：`random_train_anchor=True` 训练的臂。`CLAUDE.md` 记录了随机锚点 + 模仿教师是性能悬崖
  （逐样本逐轮重算逆动力学），且拟合教师表按锚点绑定、closure 标签拒绝随机锚点。**唯一与随机锚点相容
  且已通过筛选的监督是 L1.b 的无教师组合**（hr=8 + TV=1，bank skill 0.711 vs 教师对照 0.709，ADE 不劣）。
  A0-random 的臂 = native32 + hr8 + tv1 + `random_train_anchor=True`（`random_train_anchor_min_future_s`
  = **20 s**，不是 60：60 s 契约下训练集 6851 架里有 3 架没有任何可用锚点——它们 L−1 之后的真值不足 60 s，而固定
  锚点策略接受任何正的未来——`train()` 拒绝静默丢弃；20 s 让训练队列与所有固定锚点臂完全相同，且正是曲线最需要的
  近跑道区间。重锚时的 `--min-future-s 60` 是另一个量：哪些 bin 可读，不是哪些锚点被训练）。这也是 L1.b 的一个独立用途。

两臂的差 = 分布外代价；曲线的形状从 A0-random 读，与当前 base 的可比性从 A0-fixed 读。
臂名由每个 checkpoint 自己的 `random_train_anchor` 决定并写进它自己的块（`A0-fixed` / `A0-random`），
一次运行可以同时含两臂，读数会说明；分布外的告诫只对 fixed 臂打印。
**注意（2026-09-07 实测）**：`random_train_anchor=True` 的训练当前**跑不起来**——
`RandomAnchorTrajectoryWindows.__init__()` 不接受 `train.fit_model` 传入的 `fitted_teacher`
（`TypeError`，先于任何 epoch）。A0-random 臂开工前必须先修这个。

### 2.3 锚点网格

按**剩余路程**分 bin 而不是按时间：剩余路程是从真值算的协变量（`approach_difficulty.remaining_path_m`；
逐样本形式 `remaining_path_profile_m`，2026-09-07 建成，协变量自己也从它读，所以两者不可能变成一个名字
的两种定义），与现有 NEAR / FAR 分层同源，且对不同速度的航班可比。网格 {20, 16, 12, 8, 6, 4, 2} km，每架飞机在每个
bin 取剩余路程最接近 bin 值的样本为锚点；锚点前不足 120 s 或锚点后剩余不足 60 s 的 bin 记为空，
读数打印每个 bin 的航班数（有界覆盖必须声明）。分层沿用 `strata_masks`，**分层标签按 L−1 锚点算一次
并固定**，否则一架飞机在 8 km 处"已建立"就从雷达引导层消失，曲线变成幸存者曲线。

### 2.4 A0 的门（预注册）

1. 雷达引导层 ADE(s) 随 s 减小单调下降（允许种子噪声 ±22 m 的逆序）。**读的是 ADE 中位数**
   （`ade_p50_m`，本包读重尾误差的惯例；均值与 p95 同时报出但不进门），而且是**配对**读：
   相邻两个 bin 只在两边都出现的航班上比较，读数打印配对航班数 n。bin 的人口不同，
   不配对的两个中位数之差混了「模型变准」与「这个 bin 的航班更容易」。
2. 存在 s\* > 4 km 使雷达引导 ADE(s\*) < 1.5 km（Phase 0 的门，在 L−1 处不可达；这里问它在哪里变得可达）。
3. 时长误差 p80(s) 首次 < 30 s 的 s 记为 s_freeze；报告它，不设门。

**否决**：A0-random 在 L−1 处劣于 native32 超过种子噪声，说明随机锚点训练损害了 base，A2 / A3 都在
一个更差的 base 上做，先停。

### 2.4b A0 结果（2026-09-07 夜，`anytime_a0_20260907` 与 `anytime_a0_random_20260907`）

**A0-fixed**（三个 L−1 训练的 checkpoint 回放，1404 架，bin 20/16/12/8/6/4/2 km，`--min-future-s 60`）：native32 与
warm β=0.01 的雷达引导 ADE p50 曲线单调（门 1 过），warm 在每个 bin 都优于 native32（20 km 处 −274 m，6 km 处
差别消失）；门 2 两者都在 16 km（1382 / 1301 m）；closure `C_pred` 远端最准（12 km 处 FDE p50 26 m）但 8 → 6 km
崩溃（雷达引导 ADE 613 → 3990 m，门 1 败）。**门 3（s_freeze）三臂都读不出**：时长头 ~125 s 地板使 |Δt| 越近跑道越
大（6 km 处各臂 p50 80–143 s，而剩余只有约 170 s）——近端时序通道无信息，是 B 线的问题不是曲线的问题。
2 km bin 全空（60 s 地板），4 km 覆盖 0.30 标 partial；20/16 km 覆盖 0.35/0.37，直线进近层远端几乎为空。

**A0-random**（`A0_random_hr8_tv1`：随机锚点 + hr8+TV，20 s 训练契约，队列与固定臂完全一致）：训练 30 轮早停，
最好 epoch 10；**L−1 否决惨败**（ADE 2949 vs 1322，FDE p50 3784 vs 864，胜率 6.8 %）。但回放时（与 native32
逐 bin 配对）：

| s km | A0_random 雷达引导 ADE p50 | native32 | chamfer p50 A0_random / native32 |
|---:|---:|---:|---|
| 20 | 2622 | 2708 | 2124 / 2584 |
| 16 | 1847 | 1382 | 1325 / 1439 |
| 12 | 756 | 585 | 426 / 652 |
| 8 | 377 | 385 | 326 / 850 |
| 6 | 298 | 338 | 606 / 1051 |

**chamfer 六个 bin 全优（−114…−524 m）**，8 km 以内 ADE 也优，门 2 从 16 km 推进到 12 km——一个只用了 17 % 预算、
在 L−1 惨败的 checkpoint，在整段进近上画出的几何比固定臂好。读法：`fixed-anchor-common-grid-ade` 只在 L−1 打分，
随机锚点模型在那里最不擅长，选择规则把它冻在第 10 轮；它真正改善的量（全程几何）对选择指标不可见。16 km bin
的退化（+465 m ADE、FDE +1530）是真实的，尚无解释。**决定**：(i) `A0_random_hr8_tv1_p180`（关早停）测停止规则
的份额；(ii) **A1 提前实现**——按锚点网格的验证指标 `anchor-grid-common-grid-ade`（L−1 + 16/12/8/6 km 五个锚点集
的 common-grid ADE 等权平均），第三臂 `A0_random_hr8_tv1_grid` 用它选 checkpoint。

### 2.5 A2 的门

同一锚点网格上，A2 的雷达引导 ADE(s) 与时长误差不劣于 A1（无状态）；候选权重的熵随 s 单调下降
（权重没有收敛就是打分没有信息）。A2 用 L2.d warm β=0.01 的 `modes/`（K=6）与 L5.a 拟合表最近邻各一臂。

---

## 三、B. 校准的到达时刻分布

### 3.1 分位数时长头（B1）

config 轴 `duration_head ∈ point | quantile`（默认 `point`，进 checkpoint 与 run name：`T=q5`）。
`QuantileFinalTimeHead` 输出 τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9} 五个分位数，单调性由累积 softplus 保证
（q_0.1 = softplus(r_0) × scale，q_{k+1} = q_k + softplus(r_{k+1}) × scale），损失为五个 pinball 之和，
替换 `final_time` 分量（**同名分量，`loss_component_names` 不变**）。中位数 q_0.5 走现有的 `final_time_s`
契约，其余四个写入 `source.durationQuantilesS`。control 路径的 rollout 用 q_0.5 作为时长。

### 3.2 split-conformal 校准（B2）

CQR：在校准集上算 conformity score = max(q_lo − T, T − q_hi)，取 (1−α) 分位数 δ_α，区间 [q_lo − δ, q_hi + δ]。
**校准集不能是 test split**（实验原则），也不能是训练集：把 val 按 `split_seed` 对半分，A 半校准、B 半
报覆盖率，然后交换，报两次的平均。δ_α 表（α ∈ {0.2, 0.5}）与校准集航班数写进 `checkpoint_metadata.json`；
`predict` 读它，无校准表的 checkpoint 只输出未校准分位数并在 `source` 标 `calibrated: false`。
按分层分别校准（直线与雷达引导的 δ 差一个量级，合并校准会给直线航班一个荒谬的宽区间）；分层标签
由 L−1 的真值协变量决定，与读数一致。

### 3.3 分位数条件航迹（B3）

L3 建成的 `cta_conditioning=given` 让给定 CTA 直接成为时长；目前 `forecast._dynamics_batch` 里的 CTA
是真值时长 + `cta_offset_s`，读的是未来。B3 加 `predict --cta-from-quantiles`：CTA 依次取自身的
q_0.1 … q_0.9（校准后），每个分位数写一个 `quantiles/qNN/` 完整预测目录，`source.ctaQuantile`
记 τ。**这是第一个不读未来的 CTA 臂**，run name 记 `cta=self-q`，与 `cta=given` 严格区分。
读数：五条航迹的扇面是否覆盖真值路径（真值 chamfer 到扇面的最近一条 ≤ 到 top-1 的份额），
每条分位数航迹的可飞率不低于 top-1（rollout 按构造保证，这里是核对而不是门）。

### 3.4 B 的门（预注册）

1. B1：q_0.5 的时长误差与点估计头相比在种子噪声内；ADE 不劣（时长变了 rollout 就变）。
2. B2：校准后 80 % 区间的实测覆盖 ∈ [0.76, 0.84]，50 % 区间 ∈ [0.45, 0.55]，两个分层各自成立。
3. B3：真值时长落在 [q_0.1, q_0.9] 内的航班里，真值路径到分位数扇面的 chamfer 中位数低于到 top-1 的。

**否决**：分位数头让直线进近层 top-1 FDE 退化超过种子噪声（同 L2 的否决）；或校准后区间宽度的中位数
在雷达引导层超过 120 s，此时区间没有调度意义，说明 T 的不确定性不是校准能收的，回到 A。

---

## 四、C. 控制参数空间的条件扩散 + rollout 引导

### 4.1 为什么放在控制参数上，而不是航迹上或 z 的先验上

`latent_intent_design` §2.3 把扩散的落点定为 z 的 8 维先验（两阶段：先训 VAE，再在聚合后验上拟合扩散）。
L2.c / L2.d 之后这个前提变了：z 本身只带 0.17 nat，先验比 N(0,I) 还窄，在它上面拟合扩散是给一个几乎
没有信息的变量换分布形状。航迹空间（300 × 6）的扩散则丢掉"按构造可飞"，那是 control 路径存在的理由。

剩下的落点是 **96 个操作参数本身**（32 段 × 3 个无量纲控制 + 时长）：

- 训练集现成：L5.a 的拟合教师表（`FITTED_TEACHER_SCHEMA = "ts-basis-fit-v2-teacher"`，
  `control/basis_fit.py`）每架飞机一条通过同一 rollout 把真值复现到 88–433 m 的控制序列。去噪目标就是它。
  这与 L5.a 的模仿项是同一张表的两种用法：模仿项拿它当回归目标，扩散拿它当样本。
- 96 维是低维扩散，8 千架的训练量够用；不再有 z、KL、free bits，后验坍缩按构造不存在。
- **rollout 可微，引导梯度可以穿过物理积分打在控制参数上**，这是本项目独有的条件，其他轨迹扩散
  文献只能在航迹空间做引导。

### 4.2 形态

- **去噪网络**：条件来自现有骨干的融合特征（`ControlFeatureModel.fused_features`，含锚点状态与动力学
  条件），输入为加噪的 96 维向量（控制按 `control/envelope.py` 的包络归一到 [−1, 1]，时长按
  `final_time_scale_s`），输出 x0 预测。损失 = 去噪 MSE；可选加一项 x0 经 rollout 后的航迹损失（稠密监督，
  与 L1 同网格），权重是一个 config 轴。
- **top-1 保持是确定性头**（native32 或 L2.d 的解码器，同 base 不动），扩散只提供分布。这样 top-1 的门
  和否决与前面所有臂同口径，扩散的贡献只从 minADE_K、覆盖率、散布读。
- **采样**：K 个样本各自 rollout，写 `samples/sNN/` 完整预测目录，`source.sampleIndex`；读数沿用
  `run_ts_latent_readout.py` 的 minADE_K / 散布 / 与同 K 无条件采样对照的规则（对照 = 去掉条件特征的
  同一网络，替代 L2 的 N(0,I) 对照）。
- **引导（C2）**：每个采样步对 x0 预测做一次 rollout，加引导梯度 −η ∇_x0 J：
  - 前缀项 J_prefix = rollout 前 τ 秒与已飞观测的 ADE（A 线的条件形态；τ = 锚点后已过的时间）；
  - CTA 项 J_cta = (T − CTA)²（L3 / B3 的条件形态，不再把时长硬替换）；
  - 走廊项 = 现有 barrier hook 的几何（`control/constraints/barrier_filter.py`），只在最后几步加。
  引导只在包络内做（每步 clamp 回 [−1, 1]），强度 η 按前缀 ADE 标定；**每次报引导样本时同时报无引导
  样本**，否则读不出引导做了什么。
- **与 A 的合并（C3）**：锚点网格上每个 s 处，前缀引导的样本散布就是 A2 想用重加权得到的东西，
  不需要保留上一轮候选，也不改训练。

### 4.3 门（预注册）

1. **C0**（无训练，前置 = 拟合表跑出来）：按锚点状态（`approach_difficulty` 的协变量 + 锚点通道）取最近
   K=16 条拟合序列各自 rollout，minADE_16 < native32 top-1；同时报"真值航班自己的拟合序列"的 ADE，
   那是这张表作为记忆的上界（Auto-JEPA 式检索的天花板），扩散过不了它就没有理由存在。
2. **C1**：minADE_16 显著低于 top-1，且低于同 K 的无条件采样对照；样本散布与真值散布同量级（雷达引导层
   真值散布见 L2.d 读数 325 m）；直线进近 top-1 不退（top-1 是确定性头，这一条是核对）。
3. **C2**：前缀 τ ∈ {0, 10, 30, 60} s 下样本散布单调收窄；CTA 引导后到达时刻误差不高于 `cta=given` 的
   恒等检查 + 5 s；引导样本可飞率 = 100 %（rollout 保证，核对）。

**否决**：C0 的检索已经过 C1 的门（minADE_16 与扩散相差在种子噪声内）→ 扩散只是检索的昂贵版本，
交付用检索；或引导后的样本在包络边界饱和的份额 > 10 %（引导在推控制出分布）。

### 4.4 与 L2 的关系

替代，不是叠加：C1 过门后 `latent_dim` 保持 0，L2 的 CVAE 保留为对照臂。只有 L2.e′ 仍把 z 卡在
1 nat 以下时才启动 C；若 L2.e′ 让 z 带上 4–8 nat 且 minADE_6 过门，C 降为 L5 的第三臂。

---

## 五、B4：冻结点曲线（交付物）

A1 的锚点网格 × B2 的区间：W_80(s) = 校准后 80 % 区间宽度在剩余路程 s 处的分层中位数。
交付两条曲线和一个数：雷达引导层与直线进近层的 W_80(s)，以及 s_freeze = W_80 首次 < 60 s（±30 s）的
剩余路程，换算成剩余时间报出。这个数就是"调度程序可以锁死这架飞机次序的时刻"，是论文对
AMAN 的直接回答，也是 IPOPT 最优解给不了的东西。

---

## 六、契约与不变量（新增，违反会静默算错）

1. **分层标签在 L−1 锚点算一次并固定**，所有锚点网格上的读数用同一标签。否则曲线是幸存者曲线。
2. **每个 bin 打印航班数**；bin 内航班少于该分层的 50 % 时该点标 `partial`，不进门。
3. **A0 的两臂必须同时报**；只报 A0-fixed 的曲线会把分布外代价读成"越近越准"。
4. **校准集永远不是 test，也不是训练集**；δ 表随 checkpoint 元数据走，没有表就不声称校准。
5. **`cta=self-q` 与 `cta=given` 是两种臂**，run name 必须区分；只有前者可作为预测结果引用。
6. **分位数是时长的，不是航迹的**；扇面覆盖率是读数不是覆盖保证，文档与 README 不得写成后者。
7. 预测记录仍锚定在 `t=0` = 锚点样本；流式输出的共享时钟靠 `source.anchorTimeS`，与 CZML 规则一致。
8. **扩散样本永远不是 top-1**；记录契约走确定性头，样本只在 `samples/` 下，`source.sampleIndex` 标注。
9. **引导样本必须与无引导样本成对报出**；引导强度 η 与前缀长度进 `source`，不进 run name（它们是
   predict 参数，与 `--cta-offset-s` 同级）。
10. **扩散训练集是拟合表，受它的六项覆盖检查约束**（宽度、锚点、cohort、机场、N、真值时长）；一张外来
    表按现有规则被拒绝，扩散不得绕过 `require_cover`。

---

## 七、实施顺序与工作量

| 序 | 项 | 依赖 | 工作量 | 需要 GPU |
|---|---|---|---|---|
| 1 | B0：现有 `summary.json` 的时长误差分层分布，定区间量级 | 无 | 半天 | 否 |
| 2 | A0-fixed：`run_ts_anytime_curve.py` + 三个现有 checkpoint 回放 | 无 | 1 天 | 预测用，轻 |
| 3 | A0-random：native32 + hr8 + tv1 + 随机锚点，1 臂 180 轮 | L1.b 确认臂结果 | 训练 ~1 晚 | 是 |
| 4 | B1 + B2：分位数头、校准模块、测试 | 无 | 2 天 | 1 臂训练 |
| 5 | B3：`--cta-from-quantiles` | B2、L3 代码 | 1 天 | 预测用 |
| 6 | A2：候选重加权 | A0 过门、L2.d modes 或 L5.a 表 | 2 天 | 预测用 |
| 7 | B4：曲线与 s_freeze 读数 | A1、B2 | 半天 | 否 |
| 8 | A3：递归先验 | A2 有增益 | 1 周 | 是 |
| 9 | C0：拟合表检索上界 | L5.a 拟合表（排队中的 GPU 任务） | 1 天 | 否 |
| 10 | C1：`control/diffusion.py` + 测试 + 1 臂 | C0 过门、L2.e′ 结果 | 4 天 | 是 |
| 11 | C2：三种引导 + 成对读数 | C1 过门 | 2 天 | 预测用 |
| 12 | C3：锚点网格 × 前缀引导，对照 A2 | C2、A0 | 1 天 | 预测用 |
| 13 | 后端接口：`aeroviz_backend/http_server.py` 按时间戳查询预测；前端扇面随时间收窄 | B4 或 C3 | 之后 | 否 |

1、2、4、9 互不依赖，可以并行（9 只等拟合表）；C 的启动条件见 §4.4；每步：写代码 → opus review（只 review 代码）→ 修 → 实验 → 记录 → 提交。
正式 campaign 前工作树干净，`git add` 明确路径，看进程用 PID。

---

## 八、风险

1. **分布外代价可能压过信息增益**（A0-fixed 曲线不单调）。这不是否决，是 A0-random 存在的理由。
2. **随机锚点训练损害 base**：L1.b 无教师组合在随机锚点下未测过。否决规则见 §2.4。
3. **分位数头与 rollout 的耦合**：时长变了每条航迹都变；B1 的门要求 ADE 不劣，若劣化超过种子噪声，
   改为分位数头只作输出、rollout 仍用点估计头（两个头并存，各自的损失分量）。
4. **雷达引导层的区间可能宽到没有调度意义**（p80 > 120 s）。这是 B 的否决条件，也是本文最可能的失败
   形态；失败时它仍是一个结果：说明冻结点必须由 A 的曲线给出，而不是由某个时刻的区间给出。
5. **A2 的打分依赖候选真的不同**（L2 的 spread 325 m，minADE_6 1004）；候选若都一样，重加权无物可选。
   拟合表最近邻是它的对照。
6. **扩散在 8 千架上记忆而不是泛化**：C0 的检索上界就是这个风险的读数，扩散与检索打平即否决。
7. **引导把控制推出分布**：包络饱和份额 > 10 % 否决；引导只作用于 x0 预测并 clamp，不作用于噪声样本。
8. **拟合表自带 wiggle**（L5.a 的已知风险：拟合只最小化位置）；扩散会忠实复现它。修在拟合端（平滑先验），
   不在扩散端，与 L5.a 的规则一致；bank skill 按 `score_control_arms.py` 的地板与天花板读。

---

## 九、名词对照

| 概念 | 代码里的名字 |
|---|---|
| 锚点网格 / 重锚曲线 | `run_ts_anytime_curve.py`，产物 `anytime_a0_<date>/`，读数键 `remaining_path_bin_m` |
| 随机锚点臂 | `random_train_anchor=True` + L1.b 监督（`control_heading_rate_loss_weight=8`, `control_bank_tv_loss_weight=1`） |
| 候选重加权 | `forecast.reweighted_mode_forecasts`，`predict --stream-dt 10` |
| 分位数时长头 | config `duration_head ∈ point \| quantile`，`prediction_outputs.QuantileFinalTimeHead`，`source.durationQuantilesS` |
| 校准 | `calibration.py`（顶层，被 `forecast` 与读数共用），`checkpoint_metadata.json["conformal"]` |
| 分位数条件航迹 | `predict --cta-from-quantiles`，目录 `quantiles/qNN/`，run name `cta=self-q` |
| 冻结点 | `s_freeze`，读数键 `freeze_remaining_path_m` / `freeze_remaining_time_s` |
| 拟合表检索 | `run_ts_schedule_retrieval.py`，读数键 `retrieval_minade_k` / `memory_ceiling_ade` |
| 控制空间扩散 | `control/diffusion.py`，config `control_sampler ∈ none \| diffusion`、`diffusion_steps`、`diffusion_trajectory_loss_weight`，run name `control+dif` |
| 采样与引导 | `predict --diffusion-samples K [--guide-prefix-s τ] [--guide-cta-s T] [--guide-corridor]`，目录 `samples/sNN/`，`source.sampleIndex` / `guidance` |
