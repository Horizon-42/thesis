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
跑出读数（§〇.2），A0-fixed 的回放是 GPU 队列里的作业，未跑。**B 线三项（B1 / B2 / B3）的代码也已建成**
（分支 `dev-b1`，`9f4149a` / `585b0e1` / `ebdb00c`，命令见 §〇.3），臂文件
`docs/experiments/b1_quantile_arms.json` 已预注册，**尚未训练任何一条**。其余各项仍为设计稿。
A0 / B0 用现有 checkpoint 与现有预测目录即可做，排在 L2.e′ 之后、L3 campaign 之前或并行
（B0 是纯 CPU 读数，A0 只推理）。

| 阶段 | 状态 | 产物 / commit | 门 |
|---|---|---|---|
| A0 重锚曲线（测量） | **fixed 两臂 + random 三臂都回放了（§2.4b/c）**：random 训练在第 8–10 轮后于每个锚点集退化——机制是 LR 调度器挂在停滞的选择指标上（第 60 轮 lr 9e-7）+ 锚点按时间均匀采样偏向近跑道 | `anytime_a0_20260907/`、`anytime_a0_random*_20260907/`、`a0_random_20260907/` | 门 1 过、门 2 12–16 km、门 3 被时长地板挡住；random 的 L−1 否决三臂皆败 |
| A0.b 两条机制的修正（§2.4c） | **代码完成并过评审（2026-09-07，分支 `dev-a0b`：`12d35ce`、`23cae12`、`965077a`、`f8a1726`，评审修正见 §2.4c 实现）；两臂未训练** | `lr_plateau_metric ∈ {selection, objective}`（调度器读验证目标而不是选择指标；β 预热或 λ 对偶步下被拒）+ `random_train_anchor_sampling ∈ {uniform, remaining-path-uniform}`（在航班自身剩余路程区间上均匀抽、取最近可用锚点；**评审否掉了先抽层的写法**，分层只留作记账）+ 每轮 `train_anchor_sampling.remaining_path_strata` 与 `_population` 计数；臂 `A0b_lr_objective` / `A0b_lr_objective_path_uniform`；测试 `tests/test_lr_plateau_metric.py`（11）+ `tests/test_random_anchor_sampling.py`（27） | 读法见 §2.4c；两个默认值都是今天的行为，四条固定锚点路径逐字节等价、935 个产物重算命名 0 变化 |
| A1.a 网格选点指标（A1 的前置：先让随机锚点臂能被正确选点） | **代码完成（2026-09-07，分支 `dev-a1`）；臂未跑** | `anchor_grid.py`（网格的唯一定义：bin、60 s 地板、逐航班锚点规则、L−1 固定分层规则；runner 改为 import 它）+ `checkpoint_selection_metric=anchor-grid-common-grid-ade`（五个锚点集的 common-grid ADE 均值，`history.json` 每轮记 `validation_anchor_grid` 块）+ 臂 `A0_random_hr8_tv1_grid`；测试 `tests/test_anchor_grid.py`（7）+ `tests/test_anchor_grid_selection.py`（17）+ `tests/test_frame_ablation_runner.py`（3） | 无门，是选点规则；旧指标逐位不变（4 臂 × 2 轮 × 232 个 history 浮点全等） |
| A0.pub 重锚预测的可视化交付（曲线之外的那一半） | **完成（2026-09-07，分支 `dev-publish2`）**：曲线与记录同一次前向；KRDU val `--limit 300` 三个 checkpoint × 12/8/6 km 共 9 个类别已发布，浏览器核对过（预测从进近中段起画，不在 25 km 切片起点） | `run_ts_anytime_curve.py --write-records` → `<out>/records/<label>/<bin>km/`（`export.write_batch`，`summary.json` 多一个 `anytime` 块，schema `ts-anytime-records-v1`）+ `publish_ts_experiment_trajectories.py` 按 bin 建类别（key `…_a12km_val`、picker id `<run>@12km`、group = 记录 campaign、标签写明 bin 与航班数）；产物 `anytime_records_20260908/`（317 MB，回放 2 min 30 s）；CZML 侧**无需修改**（早已按 `source.anchorTimeS` 平移，§六 7），新增测试钉住晚锚点 | 无门，是交付形态；12 km 覆盖 244/300，8/6 km 300/300 |
| A1 流式评估协议 | 未做 | `evaluation_protocol` 新增按剩余路程分 bin 的锚点网格；`compare_constraint_arms.py` 增列 | 无门，是读数 |
| A2 候选重加权（预测期滤波） | 未做 | `forecast.py` 新增 `reweighted_mode_forecasts`；`predict --stream-dt` | 同锚点下劣于无状态版即否决 |
| A3 学习的递归先验 | 未做，取决于 A2 | `control/latent.py` 先验网络吃上一轮后验 | 仅当 A2 有增益 |
| B0 时长误差分布（测量） | **完成（2026-09-07，`dev-a0` `fe84e76` + 测试 `5f408df`）**，读数见 §〇.2 | `run_ts_eta_error_readout.py`（新 runner）+ `tests/test_eta_error_readout.py`（7 项）：读现有 `summary.json` 的 `final_time_error_s` 与 `fde_m`，按 `strata_masks` 分层报 \|Δt\| p50/p80/p90 与带号 p10/p50/p90 | 无门，定区间宽度的量级 |
| 风读数（测量，B0 的旁证） | **完成（2026-09-08）**：直线进近速度残差对塔台顶风斜率 +0.62 ± 0.13 m/s/(m/s)、时长斜率 −2.0 ± 0.5 s/(m/s)，但 R² 0.10；终点沿航迹误差与风无关（R² 0.01）。**不开风臂**，见 `2026-09-08_wind_residual_readout.zh.md` | `run_ts_wind_residual_readout.py` + `l1_lowdim_20260907/wind_readout.json` | 无门；定风的效应量 |
| B1 分位数时长头 | **跑完（2026-09-08，`b1_quantile_20260907/B1_quantile`，`73d829f`，164 轮早停）**：对预注册参照 native32 全分层更好（ADE 1282 vs 1322，直线 420 vs 445，雷达引导 2805 vs 2870，胜率 56 %）；对 L2d（隐变量臂，仅作上下文）劣 68 m；\|Δt\| p50 全体 11.5 s / 雷达引导 38.8 s，MAE 条款待读；变动超种子噪声 → 预注册的对照臂 `B1_point_matched` 触发，已加入臂文件待跑 | `b1_quantile_arms.json`、`readout_vs_native32.json` | 门 1 ADE 条款过，MAE 条款待读；见 §3.5 |
| B2 split-conformal 校准 | **跑完（2026-09-08）**：部署覆盖率 B1 80 % **0.745 未过**（门 0.76–0.84）/ 50 % 0.450 过（压线）；B3 0.775 / 0.477 两档过；直线进近 80 % 宽度 28–32 s，雷达引导 **142–146 s → 否决线（120 s）触发**；镜像半区覆盖率一致高出部署值 5–10 点（远超二项噪声），半区种子探针进行中 | `B*_calibration/eta_calibration.{txt,json}` | 门 2 B1 半过、B3 过；否决触发；见 §3.5 |
| B3 分位数条件航迹 | **跑完（2026-09-08，`B3_quantile_cta`，`77e6d3a`，129 轮早停）**：扇形内航班里最近分位数航迹的 chamfer 优于 top-1 的份额 **70.9 %**（107 vs 137 m；直线 53 vs 75；雷达引导 704 vs 949），在扇形内份额 0.84；`cal.hit*` 为样本内，不作覆盖率读 | `quantile_fan.{json,txt}` | 门 3 过；见 §3.5 |
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
12 km 处 closure 的雷达引导 ADE ≈ 6.7–8.1 km、时长误差 ≈ 172–176 s，而 native32 是 1128 m / 63 s（**已过时**：这是 closure 修复 `4ecfb69` 之前的第一次回放；当前产物 `anytime_a0_20260907/anytime_curve.json` 2026-09-08 复核为 closure 雷达引导 ADE 均值 3649 / p50 621 m、FDE p50 402 m、|Δt| p50/p80 17/185 s，native32 771 / 585 m、FDE p50 1343 m、|Δt| 45/85 s——结论不变：closure 的 ADE 均值/p95 与 |Δt| p80 离开 L−1 后最差，FDE p50 按构造最好）——
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

### 〇.3 B1 / B2 / B3 的运行命令（2026-09-07 建成，KRDU）

**校准这一步在 train 和最终 predict 之间，不是训练的一部分**：第一次 predict 写出的记录标
`calibrated: false`（不是错，是未校准），校准表写进 checkpoint 自己的 `checkpoint_metadata.json`
之后重跑 predict，记录才带 `source.durationIntervalS`。

```bash
conda activate aeroviz
E=4dTrajectory/outputs/KRDU/experiments
C=$E/b1_quantile_20260907

# 1) 两臂训练 + 第一次预测（未校准）
python run_ts_frame_ablation.py \
    --arms 4dTrajectory/ts_transformer/docs/experiments/b1_quantile_arms.json \
    --campaign $C --airport KRDU --split val

# 2) 校准：半 A 拟合部署的 δ，半 B 量覆盖率；表写进各自的 checkpoint_metadata.json
#    （--split test / train 被拒绝；--limit 的冒烟表边车会拒，除非 --allow-smoke-table）
for ARM in B1_quantile B3_quantile_cta; do
  python run_ts_eta_calibration.py --checkpoint $C/$ARM/checkpoint.pt --out $C/${ARM}_calibration
done
#    门 2 的数在 $C/<ARM>_calibration/eta_calibration.txt 的 DEPLOYED 块里，不在逐分层行里。

# 3) 重跑预测：这一次记录带校准区间；B3 同时画出扇面
D=trajectory_data_process/outputs/harvest/KRDU/arrivals
TS=4dTrajectory/ts_transformer/__main__.py
python $TS predict --checkpoint $C/B1_quantile/checkpoint.pt \
    --data $D/manifest.json --eligibility-roster $D/lateral_pass_eligibility.json \
    --airport KRDU --split val --device auto --output-dir $C/B1_quantile_pred_val
python $TS predict --checkpoint $C/B3_quantile_cta/checkpoint.pt \
    --data $D/manifest.json --eligibility-roster $D/lateral_pass_eligibility.json \
    --airport KRDU --split val --device auto --output-dir $C/B3_quantile_cta_pred_val \
    --cta-from-quantiles          # 加 --interval-endpoints 才另画 a20lo / a20hi 两条

# 4) 读数：门 1 走 B0 的读数（|Δt| 分层分布），门 3 走扇面读数
python run_ts_eta_error_readout.py \
    B1_quantile=$C/B1_quantile_pred_val \
    B3_quantile_cta=$C/B3_quantile_cta_pred_val \
    --json $C/eta_error.json
python run_ts_quantile_fan_readout.py --arm $C/B3_quantile_cta_pred_val \
    --json $C/quantile_fan.json
```

门 2 不需要单独的读数：`run_ts_eta_calibration.py` 的产物 `eta_calibration.txt` 每个 α 先打印
逐分层的 δ / 覆盖率 / 宽度（外加镜像的稳定性核对），再打印 **`DEPLOYED` 块**——每架保留半航班按
它真正拿到的 δ 打分，总体一行加按 `自然分层 -> 实得分层` 的分组。**覆盖率永远在没有拟合该 δ 的
那一半上量**，且**门读的是 DEPLOYED 那一行**。扇面读数里的 `cal.hit` 在 val 臂上是样本内的
（同一批航班既参与校准又被打分），读数会标 `cal.hit*` 并指回这里。

### 〇.4 opus review 后的修正（2026-09-07）

review 复核了位级等价（26,038 个叶子）、0 改名与全部测试，并提了 15 条。落地的实质改动四条，都写在
本文对应小节里：**(1)** 加 `deployed` 块并把 §3.4 门 2 改成读它——逐分层 δ 只量自己分层的成员，而部署
会让被拒分层的航班落到合并 δ 上，合成对照里那一组的实测覆盖是 **0.167** 而合并行是 0.794；
**(2)** 覆盖率一律降级为“实测的交叉半覆盖率，不是有限样本保证”，因为校准集就是选点集（耦合走的是
航迹指标，不是时长残差），带保证的读数预注册为 freeze-test 阶段 test split 上的一次读数；
**(3)** 不再部署两半 δ 的平均，改为教科书式的 split：半 A 拟合 = 部署，半 B 量覆盖，镜像只作稳定性
核对；**(4)** 表带自己的队列（机场 / `--limit` / `smoke_test`），冒烟表默认被边车拒绝。
其余为：表核对 `DURATION_QUANTILES` 镜像、扇面读数标注样本内的 `cal.hit`、B1 的 pinball/点估计项
量级差写进臂文件并预注册条件第三臂 `B1_point_matched`、校准端点改为 `--interval-endpoints` 开关并
补上非正 CTA 拒绝的测试、只拟合优先级用得到的分层、区间被 δ 翻转时报错、共形层级 > 1 时报错、
`run_naming` 的不可哈希值、扇面读数按 α 而不是位置取区间、`_NEW_RUN_VOCABULARIES` 的拒绝测试。

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

**这条否决在第一臂上触发了，而它触发的方式暴露了一个更早的错误：选点指标本身（A1，2026-09-07 建成）。**
第一臂 `A0_random_hr8_tv1` 的 L−1 ADE 是 2949 m 对 native32 的 1322 m，看上去是彻底失败；但同一个
checkpoint 在重锚网格上**每一个锚点的几何都比固定臂好**（chamfer p50 −114…−524 m），且 ≤ 8 km 处 ADE 也更好。
原因不是训练，是选点：`fixed-anchor-common-grid-ade` **只在 L−1 这一个锚点上给验证集打分**，而那正是
随机锚点模型最不专门化的锚点。于是模型一旦开始在整个锚点区间上分摊容量，L−1 分数就停滞，patience 20
在第 30 轮触发、第 10 轮的权重被冻结——**指标对这个臂要改善的东西是盲的**。

因此：**随机锚点臂的选点指标是网格，不是 L−1**。`checkpoint_selection_metric=anchor-grid-common-grid-ade`
把同一个 common-grid ADE 在**若干个锚点集**上各算一次并取均值——L−1 加
`anchor_grid.VALIDATION_ANCHOR_GRID_KM`（16 / 12 / 8 / 6 km）中**这批数据能覆盖的那些 bin**，
每架飞机在每个 bin 上取自己剩余路程最近且可锚（回看足 120 s、锚点后真值 ≥ 60 s）的样本。
**每个锚点集等权**（把所有 (航班, 锚点) 对合并会按覆盖率给远端 bin 加权，正好在雷达引导
航班多的臂上把远端 bin 稀释掉）；**每个集的 ADE 只在拥有该锚点的航班上取平均**（缺该 bin 的航班是缺席，
不是记 0），再按机场取宏平均——与 `fixed-anchor-common-grid-ade` 同一口径，所以 L−1 那一项就是它的值。

**覆盖率闸门（2026-09-07 复核后加）**：候选 bin 不等于会被选点的 bin。KRDU val（1404 架、60 s 地板）
实测覆盖率为 **20 km 33.7 %、16 km 37 %、12 km 78.7 %、8 / 6 km 99.7 %**，中位航班在 L−1 处只剩
**13.4 km**。所以 20 km 直接不入候选，4 / 2 km 在 60 s 地板下部分到全空也不入候选；而 16 km 虽是候选，
在这批数据上会被闸门 **丢弃**（低于 `anchor_grid.PARTIAL_COVERAGE` = 0.5，与 A0 判读 `partial` 的是
同一个常数），打印通知并记进 `validation_anchor_grid.dropped_bins`。否则等权的五分之一会来自
「飞得远的那部分航班」这个子群。**幸存的 bin 是数据的性质、不是模型的性质**，同一 split 上每条臂的
选点集完全相同，因此值可比；幸存不足两个则整个指标被拒（那不是曲线，是 L−1 加一个陪衬）。

**L−1 仍然每轮记录**（`validation_anchor_grid.fixed_anchor_common_grid_ade_m`，与另一指标选的是同一个数），
所以上面这条否决照旧可读——它只是不再决定保留哪一轮。代价：选点这一段约 **4–5×**（每个幸存 bin 一次
额外的 deployable replay；L−1 那一次是复用的，不重算），KRDU 规模下每轮约 **+12–15 %**，180 轮的臂
多花 7–9 分钟。

**拒绝**：`cta_conditioning=given` 与 `intent_conditioning != none` 在这个指标下被 `TSConfig` 拒绝——
网格在每个 bin 锚点都会把 oracle 重读一次（与 A0 runner 拒绝这两种 checkpoint 是同一条理由），
按它选轮次等于按「oracle 收敛得多快」选。

### 2.4c A0-random 第二轮（2026-09-08 凌晨，`a0_random_20260907` 的 `_p180` 与 `_grid`）

两臂都跑满 180 轮：`_p180`（关早停，L−1 选点）最好 epoch **仍是 10**（2949.5，之后 3388 → 3088 平台）；`_grid`
（网格选点，16 km bin 覆盖 37 % 被丢弃并记录，剩 L−1 + 12/8/6 km）最好 epoch **8**（网格均值 1100：L−1 2990、12 km
722、8 km 390、6 km 298）。逐锚点集看：L−1 与 12 km 在第 8–10 轮后**退化**（12 km 722 → 1324，接近翻倍），8 km 从第 3 轮
的 314 退到 406，只有 6 km 一直改善（291 → 218）。两个预注册解释都不成立：不是停止规则（关了早停也在第 10 轮），
也不是选点指标能救的（没有好的晚期 checkpoint 可选）。

**history.json 给出了机制**（`_grid` 臂）：

| epoch | lr | train loss | val 目标 | val state | 选择指标（网格均值） |
|---:|---:|---:|---:|---:|---:|
| 8 | 3.0e-5 | 0.884 | 1.147 | 0.264 | **1100** |
| 15 | 3.0e-5 | 0.466 | 0.836 | 0.150 | — |
| 20 | 1.5e-5 | 0.487 | 0.786 | 0.139 | 1384 |
| 30 | 7.5e-6 | 0.414 | 0.765 | 0.131 | — |
| 60 | 9.4e-7 | 0.424 | 0.707 | 0.102 | 1289 |
| 100 | 2.9e-8 | 0.487 | 0.705 | 0.101 | — |
| 180 | 1.5e-8 | 0.521 | 0.706 | 0.102 | 1285 |

1. **验证目标一直在改善到第 60 轮**（1.147 → 0.707，val state 0.264 → 0.102），而选择指标（common-grid ADE）在第
   8–10 轮之后变差——目标（段端点、真值时钟）与读数（稠密网格 ADE）在随机锚点模型上**分道**。
2. **`ReduceLROnPlateau` 挂在选择指标上**：指标一停滞，学习率从第 20 轮起被逐次腰斩，第 60 轮 9e-7、第 100 轮 3e-8
   ——模型从第 30 轮起实际上不再训练，被冻在指标停滞时的状态；固定锚点臂的指标改善 100 轮以上，所以从没暴露。
3. **锚点采样按时间均匀**：每架飞机每轮抽一个样本，抽样对样本（≈对时间）均匀。**逐航班看这不算偏斜**——
   一架飞机的可用锚点从 index 59 起、中位 84 个，按构造中位就落在 100.8，实测抽中中位 108，两者一致，所以
   「抽中锚点中位 107」本身不是证据。**偏斜在汇总层**：每架飞机无论长短都只贡献一次抽样，而近跑道飞得慢、
   每公里的样本数多于 25 km 处，所以把所有航班合起来看，抽样分布比**存下的可用锚点总体**更靠近跑道。实测
   （KRDU val 全体 1404 架、20 s 契约下 181,906 个可用锚点、200 轮）：抽中 < 6 km 占 **34.3 %**，总体只占
   **25.1 %**；抽中 ≥ 20 km 占 **17.9 %**，总体占 **32.9 %**。6 km 集持续改善、L−1 与 12 km 退化，与这个方向一致。

**A0.b 预注册（待用户决定）**：两臂，同 `_grid` 配方（随机锚点 20 s、hr8+TV、网格选点、patience 180）：
(i) `lr_plateau_metric=objective`（新 config 轴：调度器读验证目标而不是选择指标；固定锚点臂默认值不变，位级等价）；
(ii) (i) + 锚点按剩余路程均匀采样（`random_train_anchor_sampling=remaining-path-uniform`；预注册时写的是「先抽 bin
再在 bin 内抽样本」，评审实测该写法把训练锚点推向跑道，已按下面的实现改为在航班自身剩余路程区间上均匀抽；同一
`eligible_random_train_anchors` 契约）。读法：L−1 / 12 km 集不再在第 10 轮后退化、最好 epoch 晚于 60；
三臂回放对照。**否决**同前（L−1 不劣于 native32 超过种子噪声）——目前三个随机锚点臂的 L−1 都在 2949–2990，
远超种子噪声，说明这条线的底座还没成立。

**实现（2026-09-07，分支 `dev-a0b`：`12d35ce` 调度器轴、`23cae12` 采样轴、`965077a` 两个臂；臂未训练）**：

- (i) **`lr_plateau_metric ∈ {selection, objective}`**（`config.py`；默认 `selection` = 今天的行为，四条具名
  配方都以字面量钉住）。`objective` 下 `ReduceLROnPlateau` 步进的是**每轮记录已经写下的那个宏平均验证目标
  `val_loss`**——从本轮已有的两个数里选一个，不另算第三个——**选点规则不变**（保留哪一轮仍由
  `checkpoint_selection_metric` 决定）。`checkpoint_metadata.json` 的 `lr_scheduler.metric` 写明这一轮按哪个数
  停；run name 记 `lr-metric=objective`。测试 `tests/test_lr_plateau_metric.py`（8 项，其中 spy 逐轮断言
  `scheduler.step` 收到的值就是 history 里对应的那一列）。
- (ii) **`random_train_anchor_sampling ∈ {uniform, remaining-path-uniform}`**（默认 `uniform`；没有
  `random_train_anchor` 时被拒）。抽法：把该航班这一轮的抽样值**均匀放在它自己可用锚点的剩余路程区间
  `[min, max]` 上，取最近的可用锚点**（`anchor_strata.remaining_path_uniform_offset`）——**每公里等权**，
  而不是每样本等权；抽样值取自与 `uniform` 同源的逐航班逐轮 sha256（前 8 字节 / 2^64），
  `sampling_version` = `per-flight-hash-v3-remaining-path-uniform`，随 `training_anchor_contract` 进 checkpoint。
  取「最近」而不是「夹逼」，是因为可用锚点不是等距格点（飞得快的段稀、被 eligibility 去掉的地方缺），且只有
  一个可用锚点的航班也必须能返回它。**可用性契约不变**（`eligible_random_train_anchors` + 20 s），两个策略
  存下的锚点集合逐位相同，只有每轮抽中的那一个不同。
- **两条抽样律的实测对照**（KRDU val 全体 1404 架、181,906 个可用锚点、200 轮，脚本按本节配方重建队列）：
  「总体」= 存下的可用锚点按层的份额，后两列 = 抽中锚点按层的份额。

  | 剩余路程层 | 总体 | `uniform` | `remaining-path-uniform` |
  |---|---:|---:|---:|
  | < 2 km | 3.8 % | 5.4 % | 4.4 % |
  | 2–4 km | 10.7 % | 14.5 % | 13.4 % |
  | 4–6 km | 10.6 % | 14.4 % | 13.2 % |
  | 6–8 km | 10.3 % | 13.8 % | 13.1 % |
  | 8–12 km | 17.7 % | 23.4 % | 24.5 % |
  | 12–16 km | 8.5 % | 7.4 % | 7.3 % |
  | 16–20 km | 5.5 % | 3.2 % | 3.2 % |
  | **≥ 20 km** | **32.9 %** | **17.9 %** | **21.0 %** |
  | 均值 / p50 / p90 | 17.2 / 11.2 / 40.8 km | 12.4 / 8.3 / 32.3 km | 13.4 / 8.9 / 35.3 km |

  新律把抽样**推向总体**：≥ 20 km 17.9 → 21.0 %，< 6 km 34.3 → 31.0 %，承载 12 km 锚点集的 8–16 km 段
  30.8 → 31.8 %。**被否掉的写法**（评审实测，700 架 KRDU val）：先按层等概率抽再层内抽，方向是**反的**——
  均值 12.2 → 8.0 km、p90 31.8 → 14.7 km、≥ 20 km 16.9 → 4.1 %、8–16 km 31.6 → 27.4 %，因为网格把近端切成四个
  2 km 层，而远端是**一个**跨 20–123 km、装着 33 % 锚点的开口层，等权分层等于给它八分之一的概率。所以
  **分层只留作记账**：`history.json` 每轮的 `train_anchor_sampling.remaining_path_strata` 记实际抽中的按层计数，
  旁边 `_population` 记它们抽自的总体，**两个策略都记**——只看抽中份额说明不了过采样，只有对着总体才读得出来；
  抽中之和 = 该轮航班数，总体之和 = 存下的可用锚点数。run name 记 `anchors=remaining-path-uniform`。
  测试 `tests/test_random_anchor_sampling.py`（27 项）。
- **分层的值下沉到叶子模块**：`anchor_grid` 读 `dataset`，而 `dataset` 需要同一批公里数来抽锚点，所以
  `DEFAULT_ANCHOR_GRID_KM`、分层边界/标签、`remaining_path_strata`、抽样律搬进 `anchor_strata.py`（无 torch、
  无 `dataset`），`anchor_grid` 原样 re-export（同一批对象，identity 测试照旧），`dataset` 直接 import。
  边界由 `tests/test_import_boundaries.py` 钉住。
- **`objective` 的两条拒绝**：`latent_beta_warmup_epochs > 0`（β 逐轮变，目标本身在被重新加权）与
  `procedure_loss_dual_step > 0`（λ 逐轮更新，同一条航迹每轮定价不同）下 `lr_plateau_metric=objective` 被拒——
  这两种情况里「目标的平台期」不是模型的平台期，调度器会直接把学习率砍穿正在爬升的 schedule。
- **默认位级等价（实测，对 `4943724`）**：control / latent / state / closure 四条固定锚点路径的 2 轮合成训练
  history **逐字节相同**；随机锚点 `uniform` 臂只多出 `remaining_path_strata` / `_population` 两个键，342 个
  数值叶子（含 `sample_sha256`）全部不变，即 uniform 的抽样本身未被触碰。**重算命名**：`4dTrajectory/outputs`
  下 935 个带 config 的产物（199 history.json、255 summary.json、193 fit_evaluation.json、
  100 experiment_manifest.json、95 campaign config.json、93 其它；判据 = 任何含顶层 `config` 对象的 JSON，
  外加 campaign 的 `config.json` 覆盖集按新 run 重算）name / slug / 是否可加载 **0 个变化**。测试套件 680 → **719** 全绿。
- **臂**：`docs/experiments/a0_random_arms.json` 追加 `A0b_lr_objective` 与 `A0b_lr_objective_path_uniform`
  （都在 `_grid` 配方上）。dry-run 通过，两条新 slug 尾部 `…_8-more_7cfd2b7f` / `…_9-more_48f49b41`。**未训练**。

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

> **实现（2026-09-07，`9f4149a`）**：`DURATION_QUANTILES` 只在 `config.py` 定义一次，头、pinball
> 损失、记录字段、`calibration.py` 与两个读数都从那里读。初始化把 **q_0.5 放在点估计头的起点**
> （对 `DURATION_MEDIAN_INDEX + 1` 个增量各取一份并反解 softplus），所以只差 `duration_head` 的
> 两条臂只差“学到了什么”。`ControlFeatureModel.duration()` 是时长与分位数的**唯一**规则——一次
> 前向只读一次头，因为训练时 dropout 是活的，读两次就是两个答案。`cta_conditioning=given` 下时长
> 仍是 CTA，**点估计头保持惰性、分位数头照常训练**（B3 要用它自己的分位数解码）。
> **两条拒绝**：不在 control 输出上（state / closure 没有这个头）；以及 **`latent_dim > 0`**——
> z 走的是“平移点估计头那一个 logit”，累积 softplus 头有五个；更要命的是训练解的是后验样本，
> 那五个分位数就成了 p(T | z ~ q(z | 本机自己的未来)) 的分位数，B2 会把一个条件在答案上的区间
> 当成 p(T | history) 去校准。宁可拒绝，不做近似。命名的 `q5` 取自 `len(DURATION_QUANTILES)`。

### 3.2 split-conformal 校准（B2）

CQR：在校准集上算 conformity score = max(q_lo − T, T − q_hi)，取 (1−α) 分位数 δ_α，区间 [q_lo − δ, q_hi + δ]。
**校准集不能是 test split**（实验原则），也不能是训练集：把 val 按 `split_seed` 对半分，A 半校准、B 半
报覆盖率，然后交换，报两次的平均。δ_α 表（α ∈ {0.2, 0.5}）与校准集航班数写进 `checkpoint_metadata.json`；
`predict` 读它，无校准表的 checkpoint 只输出未校准分位数并在 `source` 标 `calibrated: false`。
按分层分别校准（直线与雷达引导的 δ 差一个量级，合并校准会给直线航班一个荒谬的宽区间）；分层标签
由 L−1 的真值协变量决定，与读数一致。

> **实现（2026-09-07，`585b0e1`，opus review 后改于 §〇.4）**：`calibration.py`（顶层，无 torch）+
> `run_ts_eta_calibration.py`。本文没写死、代码里写明的决定：
> (i) δ 取**有限样本**分位数 `⌈(n+1)(1−α)⌉ / n` 而不是朴素的 (1−α) 经验分位数（KRDU 半分 n ≈ 700
> 时 δ 变化远小于一秒）；n 小到该分位数 > 1 时**共形答案是 +∞**，代码报错而不是悄悄取最大分数；
> (ii) **δ 不取两半的平均**：半 A 拟合的 δ **就是部署的那个**，半 B 是它没见过的、用来量覆盖率的一半，
> 公布的覆盖率就是这一次测量；半 B 拟合、半 A 打分的镜像作为**稳定性核对**并排打印，绝不平均进去
> （平均出来的 δ 拟合用到了它随后被打分的每一架航班，没有任何数字在量它）。半分规则（哈希 + seed）
> 写进表里；
> (iii) `INTERVAL_STRATUM_PRECEDENCE = 直线进近 → 雷达引导 → 全体`，取它所属且**有 δ** 的第一个，
> 记录写出 `source.durationIntervalStratum`。它是**优先级**不是查表，原因是**落空**而不是重叠：
> 直线进近与雷达引导按构造互斥（`strata_masks` 用同一个曲折度切开），但它们不覆盖全体——已建立的
> 雷达引导航班两者都不在——而且因过薄被拒的分层没有自己的 δ；两种情况都落到合并分层；
> (iv) 某一半不足 `MIN_CALIBRATION_FLIGHTS = 30` 的分层**被拒绝**（连同它的航班数记进 `refused_strata`），
> 只有**合并分层**也不足时整次校准失败——那时没有可落的地方。**只拟合优先级用得到的三个分层**：
> 给一个记录永远不会读的分层算 δ，是往产物里放一个没人量的数字。
> 表是 `checkpoint_metadata.json` 的 **边车**（键 `conformal`），**绝不进 `data_provenance`**：
> `evaluate-fit` / `freeze-test` 按相等比较那个对象，放进去会让此后每次回放都报“manifest 变了”。
> 表按 `checkpoint_sha256` 绑定 checkpoint，别的权重留下的表**报错**而不是悄悄拿来加宽区间。
> runner 只读时长头（`forecast.duration_quantile_predictions`：一次前向、无 rollout、无 CTA），
> 这既是它秒级跑完的原因，也是它可以校准 `cta=given` checkpoint 的原因
> （`load_arm(..., refuse_cta_given=False)`，唯一被允许的仪器；`intent=truth-…` 仍被拒，那个 oracle
> 在历史窗口里）。表还带自己的**队列**（`airports` / `limit` / `smoke_test`）：`--limit` 的冒烟表
> 会被边车拒绝，除非显式给 `--allow-smoke-table`；`predict` 打印表的机场，记录里写
> `source.durationIntervalCohort`。
>
> **`deployed` 块（HIGH-1，review 后加）**：逐分层的 δ 是在**该分层自己的成员**上量的，但部署不是这样
> 走的——航班拿的是优先级里第一个它属于且有 δ 的分层，所以一个分层被拒的航班拿到的是**合并 δ**，
> 而合并那一行完全没说它盖不盖得住这些航班。合成对照复现了这个失败：合并行 0.794，落到合并 δ 的
> 18 架雷达引导航班实测覆盖 **0.167**，部署总体 0.756。`deployed` 块把每架保留半的航班按它**真正会
> 拿到**的 δ 打分，总体一行 + 按 `自然分层 -> 实得分层` 分组（落空的那组打 `*fell through`）。
> **§3.4 的门 2 读的是这个数**。
>
> **它不是保证，文档一律这样写**：校准集就是这个 checkpoint 选点用的那个 split（LR 调度、最好轮次、
> 早停都踩验证指标），把 val 对半分并不能修——两半都参与过选点。所以公布的是**实测的交叉半覆盖率**，
> 不是有限样本保证。耦合是弱的，也值得点名：选点读的是**航迹**指标（common-grid ADE），不是这些 δ
> 所属的时长残差。**预注册**：带保证的那次读数，是 `freeze-test` 阶段在 test split 上的**一次**保留
> 测量，在所有实验决定敲定之后；在那之前本包任何界面都不把覆盖率写成 guaranteed。

### 3.3 分位数条件航迹（B3）

L3 建成的 `cta_conditioning=given` 让给定 CTA 直接成为时长；目前 `forecast._dynamics_batch` 里的 CTA
是真值时长 + `cta_offset_s`，读的是未来。B3 加 `predict --cta-from-quantiles`：CTA 依次取自身的
q_0.1 … q_0.9（校准后），每个分位数写一个 `quantiles/qNN/` 完整预测目录，`source.ctaQuantile`
记 τ。**这是第一个不读未来的 CTA 臂**，run name 记 `cta=self-q`，与 `cta=given` 严格区分。
读数：五条航迹的扇面是否覆盖真值路径（真值 chamfer 到扇面的最近一条 ≤ 到 top-1 的份额），
每条分位数航迹的可飞率不低于 top-1（rollout 按构造保证，这里是核对而不是门）。

> **实现（2026-09-07，`ebdb00c`）**：`predict --cta-from-quantiles` 只接受**同时**满足
> `cta_conditioning=given` 与 `duration_head=quantile` 的 checkpoint——**训练时时长仍由给定 CTA 驱动，
> 分位数头是并排训练的一个目标**；预测时驱动 rollout 的是模型自己的 q_τ，这就是“不读未来”的全部含义。
> 与 `--cta-offset-s`、`--z-from-posterior` 互斥（前者是真值的平移，后者读未来）。
> 目录是 `quantiles/q10…q90/`，**top-1 记录就是 q50 那一次解码**（不重解一遍）；有校准表时另加
> `a20lo` / `a20hi` 两个目录，即 α=0.2 区间的两个端点（`source.ctaInterval`）。
> `cta_conditioning` 因此多了第三个取值 `self-q`：它是**预测期的标签**，
> `CTA_CONDITIONINGS_AVAILABLE` 不许任何新训练选它，`predict` 把它盖在写出的 config 上，
> 于是 run name / `summary.mode` / 记录三处都说 `cta=self-q`，而训练目录仍诚实地写 `cta=given`。
> 扇面里唯一可能不可飞的是**校准端点**（δ 比它加宽的区间还宽时 lo 会到 0），此时整批**报错**，
> 既不夹紧也不静默跳过：各叶子航班不同的扇面就不是扇面，读数是逐航班配对的。
> 读数 `run_ts_quantile_fan_readout.py --arm <pred_dir>` 逐分层给：真值时长落在 [q10, q90] 的份额、
> 落在校准区间的份额、两个中位宽度，以及门 3——真值路径到**五条里最近一条**的 chamfer 对到 q50 的
> chamfer（在“真值时长落在扇面内”的子集上判门，全体一行并排打印）。**几何那一列是读数不是覆盖保证**
> （§六 6）。

### 3.4 B 的门（预注册）

1. B1：q_0.5 的时长误差与点估计头相比在种子噪声内；ADE 不劣（时长变了 rollout 就变）。
2. B2：**部署覆盖率**（`run_ts_eta_calibration.py` 读数里的 `deployed` 块：每架保留半航班按它自己的
   分层优先级**真正拿到**的 δ 打分，落空的那组也在内）80 % 区间 ∈ [0.76, 0.84]，50 % 区间 ∈ [0.45, 0.55]。
   **不是**逐分层那几行——那些各自只量自己分层的成员，而因过薄被拒的分层会把航班送到合并 δ 上，
   合并那一行对这些航班一无所知（合成对照：合并行 0.794，落空组 0.167，部署总体 0.756）。
   这是**实测的交叉半覆盖率，不是有限样本保证**：校准集也是选点集；带保证的读数在 freeze-test 阶段
   的 test split 上做一次。
3. B3：真值时长落在 [q_0.1, q_0.9] 内的航班里，真值路径到分位数扇面的 chamfer 中位数低于到 top-1 的。

**否决**：分位数头让直线进近层 top-1 FDE 退化超过种子噪声（同 L2 的否决）；或校准后区间宽度的中位数
在雷达引导层超过 120 s，此时区间没有调度意义，说明 T 的不确定性不是校准能收的，回到 A。

---

### 3.5 B1–B3 结果（2026-09-08，`b1_quantile_20260907`，KRDU val 1404，单种子）

| 臂 | 提交 | 轮数 | 选择指标（fixed-anchor common-grid ADE） |
|---|---|---:|---:|
| native32（预注册参照） | — | 180 | — |
| L2d_warm_beta0p01（隐变量臂，仅上下文） | — | — | 1213.4 |
| `B1_quantile` | `73d829f` | 164/180 早停（最佳 144） | 1281.8 |
| `B3_quantile_cta` | `77e6d3a` | 129/180 早停（最佳 109） | 1014.2（给定 CTA） |

两臂提交不同（B3 在主树一次文档提交造成的脏树拒绝后重启），训练路径无差异。

**门 1**（对 native32：q50 时长误差在 25.9 s MAE 的种子噪声内，且全体 ADE 不劣于 1322 m）：ADE 条款**过**——B1 全分层更好
（ADE 1282 / 420 / 2805 vs 1322 / 445 / 2870，FDE p50 860 / 595 / 1935 vs 864 / 671 / 1982，配对胜率 55.6–55.8 %）；MAE 条款待读
（\|Δt\| p50 全体 11.5 s、直线 7.3 s、雷达引导 38.8 s）。第一次读数误以 L2d 为参照得出"全分层劣 68 m"，已更正；对 L2d 的劣势只作上下文。
ADE 变动（−40 m）超过种子噪声，按预注册触发对照臂 `B1_point_matched`（点估计头 + `final_time_loss_weight` 26.0），分辨"分位数头本身"
与"时长项重了 26 倍"。

**门 2**（部署覆盖率，80 % ∈ [0.76, 0.84]、50 % ∈ [0.45, 0.55]）：

| 臂 | α | 部署覆盖率 | 宽度中位 s | 直线 / 雷达引导宽度 s | 镜像半区覆盖率 |
|---|---|---:|---:|---:|---:|
| B1 | 0.2 | **0.745 未过** | 33.5 | 28 / **142** | 0.839 |
| B1 | 0.5 | 0.450 过 | 16.5 | 14 / 74 | 0.557 |
| B3 | 0.2 | 0.775 过 | 38.2 | 32 / **146** | 0.823 |
| B3 | 0.5 | 0.477 过 | 18.4 | 15 / 76 | 0.548 |

**门 3**（B3 扇形几何）：过——扇形内航班（84 %）里最近分位数航迹的 chamfer 优于 top-1 的份额 70.9 %（107 vs 137 m）。

**否决触发**：雷达引导 80 % 区间中位宽度 142 / 146 s（扇形读数 149 s）> 120 s。直线进近 FDE 条款干净（595 vs 594）。
读法：直线进近的 80 % 区间 ~30 s 宽且覆盖率合规，对调度程序可用；雷达引导 ±70 s 是真实的不确定性（控制器何时转向，
同 L2/L4 的 962 m 意图），校准只是如实报出，不是校准能收窄的——与"CTA 为脊柱"一致：雷达引导的时间应由调度程序指定
（B3 给定 CTA 后门 2 / 门 3 全过）。是否按否决字面"退回 A"：待用户决定。

**半区不可交换**：镜像覆盖率在两臂两档一致高出部署值 5–10 点（n≈700 时二项 SE ≈ 1.5 点，3–6σ，单向）。半区规则固定用训练种子
且每次运行重写部署表；`dev-b2a`（`212ec50`，已合并）给 runner 加了 `--half-seed` + `--readout-only`（非默认种子只允许只读，
探针表在 sidecar 处无条件拒绝），五种子只读探针（1337 = 部署规则的对照，精确复现；2024、7、99、31337）：

| 臂 / α | 部署覆盖率（五个切分） | 五切分均值 | 镜像−部署 差（五切分） |
|---|---|---:|---|
| B1 / 0.2 | 0.745 / 0.791 / 0.805 / 0.793 / 0.818 | **0.790** | −0.091 / −0.040 / +0.006 / −0.023 / +0.036 |
| B1 / 0.5 | 0.450 / 0.480 / 0.533 / 0.497 / 0.469 | **0.486** | — |
| B3 / 0.2 | 0.775 / 0.795 / 0.811 / 0.802 / 0.811 | **0.799** | 均值 −0.007 |
| B3 / 0.5 | 0.477 / 0.480 / 0.517 / 0.484 / 0.486 | **0.489** | 均值 −0.034 |

**差随种子变号**——是切分假象不是队列差：机制是反相关（把容易的航班分给 A 则 δ_A 偏小，B 欠覆盖、A 镜像过覆盖），
切分间的差散布 sd ≈ 0.05，约为单个覆盖率二项 SE（0.015）的 3 倍；部署切分（1337）恰是五个里最不利的一个（约 1.4 sd 低）。
**门 2 对 B1 α=0.2 的裁决依赖切分**：部署切分 0.745 在带外，其余四个（0.791–0.818）在带内，五切分均值 0.790 在带内；
B3 五个全在带内。**事后修正（如实标注）**：门 2 按"五切分均值 ± 切分 sd"读，部署值随其切分噪声一并引用——按此 B1 门 2 两档
均过（0.790、0.486）。部署 δ 仍是文档规则那一份（单切分，半区 A），不因探针改动。雷达引导宽度五切分均值 144 s（B3 149 s），
否决不变。

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
3b. **网格只有一个定义**（`anchor_grid.py`）：runner 画曲线的 bin / 地板 / 逐航班锚点规则，与
   `anchor-grid-common-grid-ade` 选点用的是同一批对象（测试 `runner.bin_anchor is
   anchor_grid.bin_anchor`）。两份"今天恰好一致"的网格会让"曲线变好了"和"这一轮是按曲线选的"
   变成关于不同锚点的两句话。
3c. **随机锚点臂用网格选点，固定锚点臂不必**（§2.4）。L−1 指标对随机锚点臂是盲的——它只在那个臂
   最不专门化的锚点上打分——但它每轮仍被记录（`fixed_anchor_common_grid_ade_m`），因为 §2.4 的
   否决要读它。**选点指标进 run name**（`select=…`，`META_FIELDS`），按网格选出来的是另一个 run。
3d. **覆盖率不足的 bin 不进选点均值**：闸门与 §六 2 用同一个 `PARTIAL_COVERAGE`，丢弃的 bin 连同
   它的覆盖率写进 `dropped_bins`。**幸存集是 cohort 的性质**，同一 split 上各臂一致——否则「A 臂
   的均值比 B 臂低」可能只是两条臂在不同的 bin 上取平均。
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
| 锚点网格 / 重锚曲线 | `anchor_grid.py`（网格本身）+ `run_ts_anytime_curve.py`，产物 `anytime_a0_<date>/`，读数键 `remaining_path_bin_m` |
| 网格选点指标 | config `checkpoint_selection_metric=anchor-grid-common-grid-ade`，`anchor_grid.VALIDATION_ANCHOR_GRID_KM`，`history.json` 的 `validation_anchor_grid` 块，run name `select=anchor-grid-common-grid-ade` |
| 随机锚点臂 | `random_train_anchor=True` + L1.b 监督（`control_heading_rate_loss_weight=8`, `control_bank_tv_loss_weight=1`） |
| 候选重加权 | `forecast.reweighted_mode_forecasts`，`predict --stream-dt 10` |
| 分位数时长头 | config `duration_head ∈ point \| quantile`，`prediction_outputs.QuantileFinalTimeHead`，`source.durationQuantilesS` |
| 校准 | `calibration.py`（顶层，被 `forecast` 与读数共用），`checkpoint_metadata.json["conformal"]` |
| 分位数条件航迹 | `predict --cta-from-quantiles`，目录 `quantiles/qNN/`，run name `cta=self-q` |
| 冻结点 | `s_freeze`，读数键 `freeze_remaining_path_m` / `freeze_remaining_time_s` |
| 拟合表检索 | `run_ts_schedule_retrieval.py`，读数键 `retrieval_minade_k` / `memory_ceiling_ade` |
| 控制空间扩散 | `control/diffusion.py`，config `control_sampler ∈ none \| diffusion`、`diffusion_steps`、`diffusion_trajectory_loss_weight`，run name `control+dif` |
| 采样与引导 | `predict --diffusion-samples K [--guide-prefix-s τ] [--guide-cta-s T] [--guide-corridor]`，目录 `samples/sNN/`，`source.sampleIndex` / `guidance` |
