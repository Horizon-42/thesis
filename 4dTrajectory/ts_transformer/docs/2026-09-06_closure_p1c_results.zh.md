# 闭式解码器 P1.c 结果（KRDU，2026-09-06）

承接 `2026-09-07_scene_join_anchor_design.zh.md` §五 P1（走向 C：先修输出侧）。P1.a/b 定了家族（F3 经由点 Dubins）与剖面（K=4 慢度 / 高度节点）；P1.c 把它做成第三种 `prediction_output="closure"`，在 KRDU 上与 simple-v3 基线、Phase 0 的真值意图臂配对比较。campaign：`4dTrajectory/outputs/KRDU/experiments/closure_p1c_20260905/`（读数 `readout_geometry.txt/.json`，`compare_constraint_arms.py`，闭合真值，两族指标）。

## 一、门（预注册，设计文档 §五 P1.c）

- C_truth_intent（closure + 真值 (d_join, T) 作输入通道）：雷达引导 ADE < 1.5 km，chamfer 中位 < 500 m。
- C_pred（只看本机历史）：不差于 simple-v3 基线，直线进近层不退。

## 二、做了什么

- **c1 标签**：`p1_closure_oracle.py labels` 对名册全体 9720 架（openap-direct 队列）拟合 F3 几何（汇入点 = 航向道入口，经由点规范化）与 K=4/8 剖面；有效 96.8 %（canonical 97.9 %，残差 ≤ 1 km 98.1 %）；雷达引导 3609 架 F3 弧长对齐误差中位 209 m。文件 `closure_labels_20260905.json`（schema v2）。
- **c2 接入**：`closure_output.py` + 十处接缝（`8d03fe1`、`d18cca8`、`e75d31c`）。决策向量 14 个数：d_join、经由 (d, xt, cos Δψ, sin Δψ)、5 个慢度节点（时长 = 其积分）、4 个高度节点（阈值钉 0）。损失 = 逐组 L1（几何 / 时序按 60 s 尺度 / 高度），只对有效标签的航班算；推理闭式重建（`via_dubins` → 退到 CSC → 直线），速度 = 切向 × 地速。两轮 opus review。
- **c3 campaign**：C_pred、C_truth_intent（`intent_conditioning=truth-join-duration`）各训 180 轮（每轮 2–2.5 s——纯回归，无 rollout；验证目标 0.589 / 0.478），C_oracle 从标签直接重建（同一 val 划分）。
- **一次解码器规则修正**：第一次 C_pred 的直线进近层灾难（ADE 6260、路径长度比 1.88、时长误差 143 s）——标签把直线进近的经由点放在锚点上，预测偏几十米、航向偏零点几度，Dubins 就绕一整圈（837 条过长路径里 827 条）。规则：经由点落在锚点一个转弯半径内不算决策，直接画到汇入点的 CSC（记录 `closureConstruction=csc-via-at-anchor`；C_pred 1404 架里 899 架走这条）。修后直线进近 ADE 310。修前的读数留在 `attempt1_readouts/`。

## 三、结果（验证集 1404 架，逐航班配对；闭合真值）

| 臂 | 全体 ADE | 全体 FDE 均值 / 中位 | 直线进近 ADE (904) | 雷达引导 ADE (497) | 雷达引导 FDE 均值 / 中位 | 雷达引导 chamfer | Fréchet | 弧长 ADE | \|Δdur\| 中位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A simple-v3 | 1333 | 1650 / 908 | 469 | 2858 | 2949 / 1971 | 942 | 3100 | 2175 | 39.2 |
| O_join_duration（真值意图 + 控制头） | 1005 | 1317 / 896 | 434 | 2011 | 2073 / 1472 | 847 | 2615 | 1900 | 5.4 |
| **C_pred**（closure，本机） | **996** | 615 / 11 | **310** | **2197** | 1208 / 11 | 722 | 2621 | 1378 | 33.7 |
| **C_truth_intent**（closure + 真值意图） | **595** | 229 / 9 | 212 | **1235** | 295 / 7 | **492** | 2033 | 879 | 10.7 |
| C_oracle（标签重建 = 家族上界） | 182 | 28 / 7 | 26 | 455 | 48 / 16 | 180 | 1180 | 467 | 1.0 |

配对（vs A，雷达引导）：C_pred ADE 优于 70 %（中位 −447 m）、FDE 89 %（−1243 m）、chamfer 80 %（−258 m）；C_truth_intent ADE 92 %（−1421 m）、chamfer 86 %（−430 m）。

**门：两个都过。** C_truth_intent 雷达引导 ADE 1235 < 1500、chamfer 492 < 500（刚过）；C_pred 每个分层的 ADE / FDE / chamfer 都好于基线，直线进近 469 → 310。

## 四、读法

1. **输出侧确实是瓶颈。** 同样的真值 (d_join, T)，控制头 + rollout 给 2011 m（O_join_duration），闭式解码器给 1235 m，chamfer 847 → 492、Fréchet 2615 → 2033——意图变成几何了，而 Phase 0 的结论"几何不动只有时序改善"在这里反过来：C_truth_intent 的 |Δdur| 10.7 s 不如 O_join_duration 的 5.4 s，赢在几何。
2. **只看本机的 closure 就已好于基线**（全体 996 vs 1333，雷达引导 2197 vs 2858），FDE 中位 11 m（路径按构造在阈值结束，时长也大致对）。剩下的雷达引导 ADE 2.2 km 是意图（经由点误差中位 1.8 km、汇入点 1.7 km、时长 34 s）——这是场景编码（P2/P3）要填的：C_pred → C_truth_intent 的 960 m。
3. **家族上界 455 m** 说明 F3 + K=4 剖面对雷达引导航班够用；C_truth_intent → C_oracle 的 780 m 是给定意图后经由点 / 速度剖面的回归误差，可以在 P1 内继续压（更大的头、更多轮、意图输入的表示）。
4. **可飞性要读差值**（`flyability_report.json`，干净极线）：逐样本可飞率 closure 99.8 %（C_oracle 99.3 %）= 观测 99.8 % > control 基线 93.3 %；整条全可飞率 closure 22 %、观测 98 %、**control 基线 0.1 %**（O_join_duration 3 %）。control 臂的违反几乎全是 stall（5.9 万样本——rollout 末段飞得太慢）；closure 的违反是每条几个样本：bank（C_pred 796 / C_oracle 2991：CSC 弧–直交界的曲率跳变）、thrust_over_max（约 1000：慢度节点处的加速度跳变）、load_factor_high（280–380）、stall（190–950）。所以 closure 不是可飞性的退步，而是家族有两处可修的跳变——P1.d：转弯过渡（回旋线 / 坡度率限制）与剖面平滑，或事后 rollout 一致化。
5. 直线进近层的 csc-via-at-anchor 规则是解码器规则而非家族改动：标签在锚点上的经由点本来就不是决策；C_oracle 与 C_truth_intent 的构造计数一致（517 via-dubins / 887 csc）。

## 五、下一步（写回设计文档 §五）

- P1 结束：closure 解码器成为输出侧的主线（control 头作对照）。P1.d：可飞性——判据已量到（bank 与 thrust_over_max 的逐样本跳变），做回旋线过渡 / 坡度率限制与剖面平滑，或事后 rollout 一致化，目标整条全可飞率从 22 % 接近观测的 98 %；后备分支的份额（标签无效 3.2 %）。
- P2 数据平面按计划；验收改用 closure 的口径：C_pred → C_truth_intent 的 960 m 雷达引导 ADE 是场景信息的价值上限。

## 六、复现

```bash
python 4dTrajectory/ts_transformer/docs/p1_closure_oracle.py labels --airport KRDU \
    --reference 4dTrajectory/outputs/KRDU/experiments/control_procedure_20260905/A_control_v3_pred_val \
    --out 4dTrajectory/outputs/KRDU/closure_labels/closure_labels_20260905.json
python run_ts_frame_ablation.py --arms 4dTrajectory/ts_transformer/docs/experiments/closure_p1c_arms.json \
    --campaign 4dTrajectory/outputs/KRDU/experiments/closure_p1c_20260905 --airport KRDU
python run_ts_frame_ablation.py --arms 4dTrajectory/ts_transformer/docs/experiments/closure_p1c_oracle_arms.json \
    --campaign 4dTrajectory/outputs/KRDU/experiments/closure_p1c_20260905 --airport KRDU   # after C_pred/checkpoint.pt
K=4dTrajectory/outputs/KRDU/experiments; C=$K/closure_p1c_20260905
python 4dTrajectory/ts_transformer/docs/compare_constraint_arms.py A=$K/control_procedure_20260905/A_control_v3_pred_val \
    O_join_duration=$K/scene_phase0_20260905/O_join_duration_pred_val C_oracle=$C/C_oracle_pred_val \
    C_truth_intent=$C/C_truth_intent_pred_val C_pred=$C/C_pred_pred_val --json $C/readout_geometry.json
```
