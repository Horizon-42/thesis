# 弧长几何损失：最简基线与单航迹容量实验

日期：2026-08-01

## 1. 实验边界

本轮只研究按水平弧长对齐的局部状态误差，不使用走廊、曲率、DTW 或第二次
dynamics rollout。所有容量实验固定使用同一条锁定的 outer-train 航迹：

`KSJC:ASA956_30L_a1f7fb_20260719T014743Z`

训练和回放都只打开这一条 outer-train 航迹。outer-val 和 outer-test 只参与锁定划分
的身份计数，没有读取其轨迹值。所有结果中的
`test_policy.outer_test_tracks_opened` 均为 `false`。

共同设置：iTransformer、N=64、transport-chart-velocity dynamics、0.5 s 积分、
factorized duration、1000 epochs、学习率 1e-4、终端位置权重 1.0、终端速度权重
1.0、control effort/smoothness 权重 0。

## 2. 对齐方法

预测曲线由 anchor 和 64 个 control 段末端状态组成。参考曲线由同一 anchor、固定
2 s 参考点和精确终端点组成。

两条有序曲线分别计算水平累计弧长，再把各自弧长归一化到 0 到 1，并采样 64 个
等间隔弧长位置。同一个插值方案同时重采样位置和 ENU 速度，损失比较的是相同弧长
进度下的位置、水平速度和垂直速度，不比较相同物理时刻下的状态。

位置仍按训练集 state scale 归一化后计算 SmoothL1。水平速度先计算 E/N 两分量误差
的模，再除以显式水平速度尺度；垂直速度使用 U 分量绝对误差并除以独立尺度。只有
两侧原始速度节点都具备真实监督的弧长采样点才参与局部速度项；尾部拟合占位速度和
人为补出的终端速度不会被当作监督目标。

该对齐允许飞机以不同速度通过相同几何路径，因此与可变 control 段时长不冲突。
终端位置、终端速度和最终时间仍由独立项约束。

## 3. 解耦结构

- `arc_length_geometry.py`：负责共享的弧长插值方案、位置/速度重采样、可靠性掩码和
  诊断指标。
- `control_loss_components.py`：通过 objective registry 组合弧长局部状态、终端状态和
  最终时间项。
- `train.py`：通过 validation-selection registry 选择与训练 recipe 对应的 checkpoint。
- `config.py`：每个公开模式、权重和尺度都有显式序列化字段；旧派生 checkpoint 缺字段时
  直接拒绝并要求重建，不提供兼容回退。
- CLI 和 pipeline 把局部速度权重与尺度纳入 artifact identity 和用户可见标签。

当前只保留一个可切换的弧长模式：

1. `arc-length-geometry`：在同一水平弧长进度上监督归一化位置、水平局部速度和垂直
   局部速度，并保留独立的终端位置、终端速度及最终时间约束。

`arc-length-distance` 和 `arc-length-geometry-path` 是本轮已经完成的失败消融，现已从
配置、CLI、训练 registry 和 validation selection 中移除。下面仍列出它们的历史结果，
用于说明为什么没有保留这些模式；历史输出目录不代表当前仍支持对应 recipe。

## 4. 单航迹结果

| 模式 | 关键权重 | best epoch | common-grid ADE | FDE | 终端速度误差 | 弧长几何/距离结果 |
|---|---:|---:|---:|---:|---:|---:|
| 既有 terminal-state | dense=0.25 | 999 | 2993.184 m | 59.857 m | 1.58 m/s（旧报告） | 未记录 |
| arc normalized | geometry=0.25 | 953 | 8692.543 m | 48.868 m | 2.768 m/s | normalized loss 0.6044；水平均值 9305 m |
| arc normalized | geometry=0.75 | 866 | 3194.271 m | 10.560 m | 0.109 m/s | normalized loss 0.0798；水平均值 2369 m |
| arc 3D distance | geometry=0.25 / 1000 m | 903 | 7417.249 m | 10.803 m | 0.373 m/s | 3D 距离均值 6872 m |
| arc normalized + path | geometry=0.75，path=0.25 | 768 | 5410.584 m | 13.482 m | 3.975 m/s | 3D 距离 6728 m；长度比 1.471 |
| arc position + local velocity | geometry=0.75，horizontal velocity=0.25，vertical velocity=0.25 | 966 | 7682.184 m | 14.458 m | 0.434 m/s | 水平速度 MAE 93.230 m/s；垂直速度 MAE 15.937 m/s |

各次完整结果位于：

- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_arc_length_geometry_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_arc_length_geometry_g0p75_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_arc_length_distance_s1000_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_arc_length_geometry_g0p75_path0p25_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_arc_length_state_velocity_g0p75_hv0p25_vv0p25_single_flight_overfit_1000`

## 5. 结论

最好的新模式仍是尚未加入局部速度项时、geometry=0.75 的 normalized arc loss。它显著
改善 FDE 和终端速度，但 common-grid ADE 仍为 3194 m，未超过既有 terminal-state 的
2993 m。

把局部速度直接并入同一个 `arc-length-geometry` 后，1000 epoch 的总 loss 从 605.057
降到 4.921，但 common-grid ADE 退化到 7682 m。FDE、最终时间和终端速度仍然较好，
而弧长对齐水平/垂直速度 MAE 分别停在 93.230 m/s 和 15.937 m/s。这证明速度项确实
进入了优化目标，却没有让单次长时域 shooting 找到正确的中段控制解。因此本轮仍不
进入五机场 train/val 实验。

失败不是因为 loss 没有下降。几种模式都能把总 loss 大幅压低，但仍能形成“终点正确、
中段走错”的控制解：

- normalized SmoothL1 在数公里误差区间内梯度过早减弱；提高权重有帮助但不充分。
- 直接米制距离产生更强梯度，却没有消除单次长时域 shooting 的坏局部解。
- 总路径长度只约束一个全局标量。长度接近参考不代表局部方向正确；对应消融停在
  1.471 倍参考长度，并与终端速度/几何项发生梯度竞争。
- 直接监督 E/N/U 局部速度比单位切向更强，但在当前权重下速度项主导了剩余 loss，仍未
  拟合成功；终点正确、中段走错的退化解依然存在。

## 6. 下一步建议

本轮不再增加 loss 模式。代码只保留直接修改后的 `arc-length-geometry`，失败消融仅留
实验记录。若之后继续研究该方向，应优先在同一模式内检查局部速度权重、归一化尺度及
分阶段启用速度项，而不是继续堆叠新的公开 objective。任何调整仍先在同一 outer-train
航迹上做容量实验；只有明显优于 2993 m 的既有单航迹 ADE，才运行五机场 train/val。
outer-test 继续保持冻结。
