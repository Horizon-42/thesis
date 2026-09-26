# 参考速度一致性消融实验（2026-08-02）

## 1. 目的

本实验检查一个不依赖显式进近意图的基础问题：训练数据中的位置和速度是否来自足够一致的参考状态。如果节点速度和相邻位置差分不一致，控制头需要同时满足互相竞争的位置、速度监督，而连续动力学又要求两者由同一个状态演化产生。

本实验只改变参考速度通道 `edot/ndot/udot` 的构造方式。以下内容全部保持不变：

- ENU 位置参考；
- 2 s 监督网格；
- 0.5 s RK4 最大积分步长；
- transport-chart-velocity dynamics；
- 64 个 factorized control segments；
- arc-length-geometry 2+4 loss；
- iTransformer backbone、优化器、curriculum、梯度裁剪和随机种子。

开发阶段只加载 outer-train 和 validation 航迹。outer-test 只读取清单中的身份计数，不加载轨迹值、不生成预测。

## 2. 解耦设计

速度参考实现在独立模块 `reference_velocity.py` 中，数据集只根据配置调用该模块，不在数据加载主流程中堆叠模式分支。

新增三种显式模式：

1. `track-fit`：保留原有上游轨迹拟合速度，作为基线。
2. `position-difference`：当前节点使用前一个 2 s 区间的位置差分；首行使用首个区间初始化。
3. `smoothed-position-difference`：使用当前节点及之前最多 4 个位置点做局部直线拟合，以拟合斜率作为当前速度；首行同样使用首个区间初始化。

后两种模式在任意固定 anchor 处都不使用 anchor 之后的位置。只有实测前缀的速度会被重建；零权重拟合尾占位行和单独监督的拟合终端状态保持原值，避免跨越稀疏 mask 做错误差分。

`reference_velocity_source` 被写入完整 checkpoint config 和 control recipe。旧 checkpoint 缺少该实验定义字段时必须重新生成，不做静默兼容升级。

## 3. 单航迹筛选

### 3.1 静态一致性诊断

在 outer-train 航迹 `KSJC:DAL1260_30L_a05f31_20260716T224959Z` 上，以相邻位置斜率和两个节点速度均值的分量 RMSE 作为积分一致性诊断：

| 速度参考 | 位置/速度 RMSE (m/s) | 端点后向差分 RMSE (m/s) | 节点速度加速度 p95 (m/s²) |
|---|---:|---:|---:|
| track-fit | 4.8987 | 4.9212 | 0.9602 |
| position-difference | 4.1389 | 0.0000 | 13.8720 |
| smoothed-position-difference | 5.2894 | 4.9597 | 2.2973 |

直接差分只小幅改善梯形积分意义下的一致性，却产生明显更抖的速度；因果平滑降低了差分的高频抖动，但没有在该诊断上超过 track-fit。

### 3.2 179.9 s 单航迹过拟合

三组实验使用同一航迹、seed 和 1000 epoch 配方：

| 速度参考 | 最佳回放 loss | common-grid ADE (m) | common-grid FDE (m) | 终端速度误差 (m/s) |
|---|---:|---:|---:|---:|
| track-fit | 2.2031 | 313.11 | **1.42** | 7.67 |
| position-difference | 1.9209 | 315.22 | 12.62 | **0.69** |
| smoothed-position-difference | **1.7750** | **236.69** | 3.58 | 1.81 |

直接差分显著改善终端速度，却牺牲 FDE。因果平滑将 ADE 降低 24.4%，同时保留较小的绝对 FDE，但仍未在所有指标上超过基线。

### 3.3 386.7 s 困难航迹复核

在 outer-train 航迹 `KSJC:ASA956_30L_a1f7fb_20260719T014743Z` 上比较 track-fit 与筛选出的平滑候选：

| 速度参考 | common-grid ADE (m) | common-grid FDE (m) | 终端速度误差 (m/s) |
|---|---:|---:|---:|
| track-fit | **4511.44** | 14.93 | 0.72 |
| smoothed-position-difference | 5046.54 | **8.11** | **0.034** |

平滑候选改善了终点位置和终端速度，但中途几何进一步变差。这说明单航迹收益依赖航迹形态，必须由 validation 决策。

## 4. KSJC all-QFU train/validation 实验

正式对照使用：

- 2207 条 outer-train 航迹；
- 505 条 validation 航迹；
- OpenAP direct fleet；
- iTransformer，batch 512，learning rate 3e-5；
- 60/120/240 s horizon curriculum，每阶段 10 epoch；
- 180 epoch 上限，patience 20；
- fixed-anchor arc-length-geometry checkpoint selection。

两份结果配置除 `reference_velocity_source` 和说明文本外完全相同。

下表 ADE/FDE 是 `fit_evaluation.json` 的 native normalized-endpoint 指标；checkpoint selection objective 则是在 64 点固定物理时间网格上计算的 arc-length-geometry 选择值。两类指标不混用。

| validation 指标 | track-fit | 平滑位置差分 | 相对变化 |
|---|---:|---:|---:|
| ADE (m) | 1130.67 | **1066.48** | 改善 5.7% |
| FDE (m) | **1202.94** | 1367.12 | 恶化 13.6% |
| cross-track p95 (m) | 3313.22 | **2746.32** | 改善 17.1% |
| altitude p95 (m) | 692.65 | **662.77** | 改善 4.3% |
| final-time MAE (s) | **20.84** | 21.25 | 恶化 2.0% |
| checkpoint selection objective | **31.4126** | 35.2837 | 恶化 12.3% |

两组最佳 checkpoint 都出现在 epoch 179。平滑候选降低了中途 ADE、横向偏差和高度尾部误差，但终端位置损失更高，最终使既定选择目标变差。

## 5. 结论

1. 当前不能用 `smoothed-position-difference` 替换 `track-fit` 基线。按既定 validation 选择规则，track-fit 仍为胜者。
2. 只替换速度、保持原始位置不变，不足以得到统一改善。它改变了几何与终端状态之间的优化折中：平滑候选偏向更好的中途路径，track-fit 偏向更好的 FDE。
3. `position-difference` 在单航迹阶段已暴露明显速度抖动和 FDE 退化，因此没有进入昂贵的正式 train/validation 阶段。
4. 如果以后继续该方向，应联合构造平滑位置与其导数，或显式优化满足积分约束的参考状态；不应继续堆叠更多只覆盖速度通道的启发式滤波器。

当前 baseline 保持：`reference_velocity_source=track-fit`。

## 6. 产物

- 单航迹 track-fit：`4dTrajectory/outputs/KSJC/ts_control_reference_velocity_track_fit_DAL1260_1000/overfit_result.json`
- 单航迹直接差分：`4dTrajectory/outputs/KSJC/ts_control_reference_velocity_position_difference_DAL1260_1000/overfit_result.json`
- 单航迹平滑差分：`4dTrajectory/outputs/KSJC/ts_control_reference_velocity_smoothed_position_difference_DAL1260_1000/overfit_result.json`
- 长时域平滑复核：`4dTrajectory/outputs/KSJC/ts_control_reference_velocity_smoothed_position_difference_ASA956_1000/overfit_result.json`
- 正式 track-fit：`4dTrajectory/outputs/KSJC/experiments/reference_velocity_20260802/a_track_fit/fit_evaluation.json`
- 正式平滑差分：`4dTrajectory/outputs/KSJC/experiments/reference_velocity_20260802/b_smoothed_position_difference/fit_evaluation.json`
