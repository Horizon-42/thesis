# ts 模型里的程序约束：怎么实现（设计与前置测量，2026-09-04）

回答"如果要在 ts 模型中实现 procedure constraint，应该怎么做"。先量了观测数据到底满足优化器的哪几族约束（脚本 `docs/measure_procedure_adherence.py`，只读，每 3 架取 1 架，按落地时间均匀），再给实现方案。没有训练任何模型。

## 结论

1. **优化器的程序约束里，只有"五边"那一段和数据一致**：从建立点到入口的 LPV 角走廊 + 下滑道窗口。IAF/过渡段、FAF 上游的汇入窗口、逐点定位盘都是规范性的：两个机场 **0.0 %** 的观测航班经过过 RNAV(GPS) 程序的任何离轴 IAF；KRDU 4–15 %、KSJC 17–62 % 的航班在 FAF 以内才建立，而优化器要求汇入点严格在 FAF 上游。把这些放进学习模型的损失，等于要求模型去拟合数据里不存在的路径。
2. 所以 ts 模型的"程序约束"应定义为 **final-segment 约束**，分两层：**训练期软约束**（损失里的 hinge 项，跑道尺度）+ **推理期硬约束**（投影，或以预测为参考的 casadi 跟踪求解）。**不要做成输入通道**：实验 1 的 H3 已经测过，主干对目标几何输入不敏感（只有时长头读它）。
3. 这个损失项正好是 `2026-09-03_krdu_nw_endpoint_bias.md` 和 state-v2 文档要的"跑道尺度的终点横向项"：它只在最后 10–18 km 生效，直线进近航班的 250 m 平移会被它看见，雷达引导航班的公里级误差它不碰。两件事一个实验做。
4. 几何全部在入口锚定的图里表达（沿跑道轴的 d、横向 xt、相对 LTP+TCH 的高度差），`enu` 帧就是这个坐标系，`target_chart = 0`，跑道航向 = `scenario.target.psi`。这回答了"引入约束后是否还该以跑道为原点"：是，而且更该。

## 1. 优化器的约束族与数据一致性

`approach_constraints` + `collocation/optimizer.py` 按 leg 分相，逐节点加不等式行：

| 行族 | 内容 | 观测数据是否满足 |
|---|---|---|
| `_terminal_pin_rows` | 末节点钉在目标 | 是（评估的横向门是跑道半宽，观测 99.6 % 通过） |
| `_leg_path_rows`（final leg） | LPV 角走廊 `|xt| ≤ k·hw(d)`，`hw(d) = cw·(d + d_GARP)/d_GARP`；下滑窗口 `h ∈ [h_GP − 60, h_GP + 120]`，`h_GP = TDZE + TCH + d·tan GPA`，只在 `d ≤ d_FAF` 生效 | **是**，见下表：建立之后 87–99 % 的采样在 k=0.5 走廊内，90–99 % 在窗口内 |
| `_fac_alignment_rows` | 航向偏离航道 ±30°（FAF 上游）/ ±10°（FAF 内） | 建立之后成立；ts 的状态输出没有航向变量，可由速度通道算 |
| `_fac_join_rows` | 汇入点在 `[d_FAF + L/5, d_FAF + max_offset]`，严格 FAF 上游 | **否**：KRDU 05R/23L 15 %、KSJC 30R 32 %、12R 62 % 的航班在 FAF 以内建立 |
| `_prefaf_fix_rows` | 经过 FAF 前一个定位点的 k·RNP 盘 | **否**：0 % 经过离轴 IAF；程序前段是"航向引导到五边" |
| box legs 的 floor / 下降梯度 | 逐 leg 台阶高度下限、梯度上限 | 没飞这些 leg，无从判断 |
| ψ 走廊、终端坡度 5° | 求解器的局部最优防护 | 不适用 |
| 速度限制 | CIFP 未提取（open item） | 无数据源 |

## 2. 前置测量：观测航班对五边约束的满足率

每 3 架取 1 架（KRDU 4,812、KSJC 3,694），tail = 沿航向进入 FAF 距离以内、离入口 > 300 m 的采样。`d_join` = 从哪个沿航向距离起航迹一直留在 k=0.5 走廊里直到入口。

| 机场/跑道 | n | 经过离轴 IAF ≤ 1 NM | FAF 处已建立 | 走廊 k=0.5 采样 / 全程 | 下滑窗 −60/+120 采样 / 全程 | ±22 m 采样 / 全程 | d_join p10/p50/p90 km | 在 FAF 上游建立 |
|---|---:|---:|---:|---|---|---|---|---:|
| KRDU 05L | 1,030 | 0.0 % | 97.6 % | 98.7 % / 97.3 % | 98.9 % / 95.5 % | 95.1 % / 52.1 % | 11.8 / 17.5 / 23.7 | 96.8 % |
| KRDU 05R | 493 | 0.0 % | 86.2 % | 88.6 % / 85.4 % | 91.8 % / 89.2 % | 83.4 % / 48.5 % | 8.6 / 15.4 / 23.2 | 84.6 % |
| KRDU 23L | 1,079 | 0.0 % | 85.8 % | 87.1 % / 85.3 % | 90.5 % / 85.9 % | 76.4 % / 39.9 % | 7.9 / 16.5 / 24.4 | 85.0 % |
| KRDU 23R | 2,210 | 0.0 % | 96.4 % | 97.6 % / 96.4 % | 98.5 % / 96.6 % | 93.4 % / 68.9 % | 11.6 / 18.4 / 23.2 | 96.0 % |
| KSJC 30L | 3,188 | 0.0 % | 83.0 % | 82.7 % / 82.9 % | 85.1 % / 79.3 % | 74.6 % / 25.3 % | 9.9 / 24.0 / 24.1 | 82.8 % |
| KSJC 30R | 311 | 0.0 % | 72.0 % | 80.5 % / 69.1 % | 86.3 % / 77.8 % | 62.6 % / 13.8 % | 1.8 / 24.0 / 24.1 | 67.8 % |
| KSJC 12R | 190 | 0.0 % | 37.9 % | 54.8 % / 37.9 % | 69.1 % / 42.6 % | 54.5 % / 16.8 % | 6.8 / 7.2 / 15.3 | 37.9 % |

读法：
- KRDU 的 d_FAF 是 10.0–10.3 km，多数航班在 15–18 km 建立，所以"FAF 内的五边"约束对 85–97 % 的航班是真约束；KSJC 30L 的 FAF 在 15.1 km，d_join 中位数 24 km 是切片起点，说明这些航班进 25 km 环时已经在五边上。
- 下滑窗口的 −60/+120 m 是可以当硬约束的（全程满足 78–97 %）；评估门的 ±22 m 不行（全程满足只有 14–69 %），它是入口穿越处的判据，不是路径约束。KSJC 30L 的 p95 高度差 +1020 m 来自那 17 % 没建立的航班，它们在"FAF 以内"却还在下降转弯。
- 这些数字是学习模型的**底线**（flyability 的读法一样）：约束项训练出来的满足率要和观测的这一行比，不是和 100 % 比。

## 3. 实现：三层

### 3.1 共享几何，定义一次

- CIFP 文档 → `LpvFinalSpec` 的桥（`_lpv_spec` / `build_constraint_segments`）现在住在 `aeroviz_backend/procedure_segments.py`。ts 不能 import backend，所以把这段桥下移到 `approach_constraints`（它已经 numpy/casadi 双后端）或 `flight_scenarios`（数据→建模的 seam，本来就解析 `runway_thresholds.json`）。backend 保留请求侧的 `ProcedureConstraint`，只改 import。
- `approach_constraints.mathx` 加 torch 分派（`fabs/sqrt/atan2/fmax/if_else` 五个函数按类型选 `torch.*`），走廊和下滑道公式就能原样进损失，优化器、评估、损失三处用同一份代码，不再有镜像。
- ts 侧新模块 `procedure_geometry.py`：按 `(airport, runway)` 从 `procedure-details/index.json` 解析 RNAV(GPS) 文档，输出每航班一行常量：`d_faf_m, gpa_rad, tch_m, course_width_m, d_garp_m, psi_runway`。KRDU 已知值：d_FAF 9,975–10,338 m，GPA 3.0°，TCH 16.8–17.5 m。
- 在 `enu` 帧下（`target_chart = 0`，目标高度 = LTP + TCH，与 `flight_scenarios.runway_target` 一致）公式退化成三行：
  ```
  u = (cos ψ, sin ψ)            # 入航方向，ψ 为 scenario.target.psi（数学 ENU 约定）
  d  = −(e·u_e + n·u_n)         # 沿航道离入口的距离，上游为正
  xt =  e·u_n − n·u_e           # 横向
  u_GP(d) = d·tan(GPA)          # 图坐标里的下滑道高度（u 通道）
  hw(d)   = cw·(d + d_GARP)/d_GARP
  ```
- 这行常量进 batch 的方式和 `conditioning` 一样（`TrajectoryWindows` 每航班一行），但**只喂损失，不进模型输入**（batch 元组第 6 位的 `dynamics` 字典已经是"给损失和 rollout 用、不给模型看"的位置，state 路径今天传 None，可以放这里；`unpack_batch` 不用改）。

### 3.2 训练期软约束（state 输出，即 state-v3 候选）

新损失分量 `procedure`（必须加进 `STATE_LOSS_COMPONENT_NAMES`，否则第一批就 KeyError）：

```
lat  = relu(|xt_pred| − k·hw(d_pred))² / s_lat²
vlo  = relu((u_GP(d_pred) − below) − u_pred)² / s_v²      # 低于下滑道，危险侧，below = 60 m
vhi  = relu(u_pred − (u_GP(d_pred) + above))² / s_v²      # above = 120 m
gate = 1[ d_truth ≤ d_join_flight ]  ×  1[ d_pred > 0 ]    # 只在真值已建立的行上生效
procedure = mean over rows( gate · (lat + vlo + vhi) )
```

- **门控用真值的 d_join，不用 d_FAF**：训练时每架航班的建立距离可以从真值算（上表的 d_join），推理不需要它。用 d_FAF 会把 15–62 % 的航班在还没建立的行上拉进走廊，和监督目标打架。
- **尺度是跑道尺度**：`s_lat ≈ 100 m`（走廊在入口的半宽 107 m），`s_v ≈ 30 m`。现在的位置项尺度是 10 km，直线进近 300 m 横向偏差每点只有 9e-4，这就是它看不见 NW 偏差的原因。
- hinge 只罚出界，不罚走廊内的位置，所以它不会把航迹拉向中线（那是位置项的事），只是把走廊变成"墙"。
- 权重标定照 imitation 项的做法：先在收敛的 A 臂检查点上算未加权项的值，按位置项的倍数定 1/4/16/64 阶梯，读数看幅度不看 p。
- 预注册否决：雷达引导层两个种子退化（同 state-v2 规则）；预期该项对雷达引导层几乎中性，因为门只开在最后 10–18 km。
- 起点的连续性项（`2026-09-03_state_v2_anchor_relative_results.md` 的下一候选）是独立的一项，同一轮可以一起做，但要分臂。

### 3.3 推理期硬约束

两档，从便宜到贵：

1. **几何投影**：`forecast._POSTPROCESSORS` 里在 `truncate_at_threshold` 之后加 `project_onto_final(forecast, spec)`：从预测路径最后一次进入 k=1 走廊的行起，把 `xt` 夹进 `±k·hw(d)`、`u` 夹进下滑窗口，终点钉到目标。零训练，必然满足，是"约束能收回多少终点误差"的天花板；代价是折角，flyability 会变差，所以它是消融臂和部署里的兜底，不是主方案。
2. **casadi 跟踪求解**（README 的 route 2）：预测给参考路径 + 时长，优化器用它做初值和目标（跟踪参考 + 控制代价），程序约束行照常加，输出既可飞又满足约束。**需要新目标项**：现在的 `collocation/optimizer.py` 只有 min-time 和 fixed-time 控制代价两种目标，没有跟踪参考的项；且要在隔离子进程里跑（casadi 不线程安全）。这是"预测器 + 优化器"真正的结合方式，也是后期多航班调度里优化器该占的位置，但它是新工作，单独立项。

### 3.4 控制输出路径

同一个 hinge 可以直接套在控制路径的 rollout 状态上（`control_prediction_loss_components` 已经有 `terminal_state_loss` 的跑道分解），动力学可飞 + 约束软满足是最完整的组合。但控制检查点全部过期（2026-08-18 单位变更），且 state 是可测的基线，所以**先在 state 上做**，控制路径复用同一模块。

## 4. 实验设计（预注册）

- 臂：A（基线，已有）；A + 投影（不训练）；A + `procedure` 项，权重 1/4/16/64 中取两档；可选 A + `procedure` + 起点连续性项。两机场 × 两种子，val split，`run_ts_frame_ablation.py --informal` 的 arms JSON。
- 读数（`compare_frame_arms.py` 已有的分层配对读数之外）：入口横向 miss（05/23 各自的符号）、第一步跳变、走廊 / 下滑窗满足率对比上表的观测底线、评估门通过率（横向半宽 + 垂直 ±22 m）、`horizonCapped` 数。
- 成功标准：直线进近 FDE 改善量 ≥ 种子噪声（A 臂 5–22 m）且雷达引导层不退化；走廊满足率接近观测底线而不是 100 %（超过底线说明把真实的偏差也压平了，和 flyability 的 blandness 一个道理）。

## 5. 不做的

- 目标几何 / 程序几何作为输入通道（H3 已否）。
- IAF 过渡段、FAF 上游汇入窗口、定位盘作为损失（数据里 0 %）。
- ±22 m 作为硬窗口（观测全程满足 14–69 %）。
- 速度约束（CIFP 未提取，见 `4dTrajectory/CLAUDE.md` open items）。

## 6. 复算

```
conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/measure_procedure_adherence.py KRDU KSJC --stride 3
```
