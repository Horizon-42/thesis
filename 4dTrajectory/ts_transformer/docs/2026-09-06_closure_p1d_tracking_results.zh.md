# P1.d (b)：closure 参考 + 点质量 rollout 跟踪（KRDU，2026-09-06）

承接 `2026-09-06_closure_p1c_results.zh.md`：closure 解码器画出的轨迹是几何族成员，不满足飞行方程，干净极线下只有 22 % 整条全可飞。用户决定走 (b)：网络出"要做什么"（closure 参考），动力学负责"怎么飞"（点质量 rollout 在制导律下跟踪参考）。本文只量两个数：**跟踪付出多少 ADE**，**整条全可飞率回到多少**。产物：`closure_p1c_20260905/*_tracked_pred_val/`，读数 `readout_p1d.txt/.json`（`compare_constraint_arms.py`，闭合真值），`flyability_report.json`。

## 一、做了什么

- `control/constraints/closure_tracking.py`：`ClosureTracker` 是一个 command hook（`needs_reference=False`），每段开始取 rollout 自身状态，在参考路径上找最近节点：横向 = L1 律（对参考局部航向的横偏与航向差）+ 参考曲率前馈 atan(V²κ/g)；垂直 = 下滑道律跟踪参考高度，参考局部坡度当"下滑道"（限幅 ±0.3）；推力 = PI 速度保持（积分项从锚点隐含推力起、按 `control_nominal_speed_gain`；比例项 0.3；参考加速度前馈 ±0.5 m/s²）+ 沿路径误差（参考此刻应到的弧长 − 飞机所在弧长，0.02/s，±10 m/s）折进目标速度，目标速度不低于 1.15 × 失速速度。增益封顶 1/Δt。
- `forecast.track_closure_forecasts`：决策向量 → 重建 → 参考路径 → 一阶滞后动力学（唯一跑 hook 的后端；`tracking_config` 由 closure 配置派生）rollout，时长 = 参考时长，64 段均分；记录带飞出的控制量（走导出的控制分支）、`source.closureTracked`、`commandHook=closure-tracker`。`predict --closure-track`，可与 `--closure-from-labels` 同用。
- 顺带修了解码器的一处缺陷：高度剖面第一个节点是最小二乘值，画出的轨迹在第一个 50 m 节点内跳了几十米（合成航班上坡度 13 m/m），跟踪器的垂直律读成 −86° 的期望航迹角。现在高度剖面从锚点高度出发、偏移线性衰减到阈值。修前的读数在 `attempt1_readouts/`。
- 臂（predict-only，`docs/experiments/closure_p1d_arms.json`）：C_pred_tracked、C_truth_intent_tracked、C_oracle_tracked。

## 二、两个数

**跟踪付出的 ADE（均值，m；括号里是跟踪比未跟踪好的航班份额）**

| 臂 | 全体 | 直线进近 (904) | 雷达引导 (497) |
|---|---:|---:|---:|
| C_pred → tracked | 996 → 1005（+9） | 310 → 311（+1） | 2197 → 2222（+25） |
| C_truth_intent → tracked | 595 → 611（+16） | 212 → 217（+4） | 1235 → 1244（+9） |
| C_oracle → tracked | 183 → 256（+72） | 27 → 58（+31） | 458 → 554（+97） |

对参考本身的几何偏差：chamfer 中位 8 → 30 m（全体）、雷达引导 722 → 760；FDE 中位 11 → 45 m（沿路径滞后 +0.1 s，跟踪器把参考时长按段跟住了）。**跟踪的代价 ≤ 100 m，对可部署的 C_pred 是 9 m——远小于场景信息的 960 m。**

**整条全可飞率（干净极线；观测航迹 98.4 %，control 基线 0.1 %）**

| 臂 | 未跟踪 | 跟踪后 | 剩余违反样本（跟踪后） |
|---|---:|---:|---|
| C_pred | 22.5 % | **92.5 %** | bank 580、stall 31、thrust_over_max 17 |
| C_truth_intent | 22 % | **90.1 %** | bank 880、stall 182、thrust 119、load 59 |
| C_oracle | 23.4 % | **88.2 %** | bank 1409、stall 430、thrust 176、load 261 |

逐样本可飞率 99.9 %（观测 99.8 %）。跟踪前的违反（thrust 1010、bank 787）是家族的跳变；跟踪后 bank 580 是 L1 律在 CSC 交界处一段内的坡度指令（最大 45°，包线也是 45°，临界样本），stall 31 来自末段。

## 三、读法

1. **两个数都达到目的**：closure + 跟踪 = 准确度基本不变（C_pred 全体 1005 vs 基线 1333，雷达引导 2222 vs 2858），整条全可飞率 22 % → 92 %，与观测的 98 % 差 6 个百分点，而 control 头 + rollout 是 0.1 %。这是"网络出决定、动力学负责执行"的验证：动力学按构造满足（rollout 出的每个状态都是飞行方程的解），可飞性检查失败的只剩制导律的指令幅度。
2. **交付形态定为 closure + 跟踪**：记录同时带几何族的意图（决策向量）与动力学一致的轨迹和控制量，与 control 路径用同一记录契约。
3. **剩余 8 %** 的路：bank 违反集中在转弯交界（L1 律在 3 km 前视下对曲率跳变的响应），把 L1 前视距离或坡度率限制进 config 可再压；stall 在末段——目标速度已有 1.15 倍失速裕度，剩下的是参考本身末段太慢的航班（标签速度剖面来自脏构型的真值）。这些是调参，不是结构。
4. **限制**：跟踪器的增益是模块常数（P1.d 的测量设置），不在 checkpoint 或 run name 里；跟踪只在预测时做，训练回路不变；单种子、只有 KRDU。

## 四、下一步

- P1 完成。跟踪器增益进 config（`closure_tracker_*`）与 run name，KSJC 复现，第二个种子——归入 P4 的"其他机场 / 种子"。
- P2 数据平面按计划开始；验收口径：C_pred → C_truth_intent 的 960 m（跟踪后 1005 → 611 全体、2222 → 1244 雷达引导）。

## 五、复现

```bash
# after closure_p1c_20260905 has C_pred / C_truth_intent checkpoints
python run_ts_frame_ablation.py --arms 4dTrajectory/ts_transformer/docs/experiments/closure_p1d_arms.json \
    --campaign 4dTrajectory/outputs/KRDU/experiments/closure_p1c_20260905 --airport KRDU
K=4dTrajectory/outputs/KRDU/experiments; C=$K/closure_p1c_20260905
python 4dTrajectory/ts_transformer/docs/compare_constraint_arms.py A=$K/control_procedure_20260905/A_control_v3_pred_val \
    O_join_duration=$K/scene_phase0_20260905/O_join_duration_pred_val C_pred=$C/C_pred_pred_val C_pred_tracked=$C/C_pred_tracked_pred_val \
    C_truth_intent=$C/C_truth_intent_pred_val C_truth_intent_tracked=$C/C_truth_intent_tracked_pred_val \
    C_oracle=$C/C_oracle_pred_val C_oracle_tracked=$C/C_oracle_tracked_pred_val --json $C/readout_p1d.json
```
