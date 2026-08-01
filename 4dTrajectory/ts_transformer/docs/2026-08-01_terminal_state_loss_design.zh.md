# Terminal-state control loss 设计（2026-08-01）

## 目标

新 loss 面向“以正确状态到达跑道入口”，替代原先近似只优化较差一项的
`smooth-max(ADE, FDE)`。它把沿途拟合、终点位置和终点速度拆成独立组件：

```text
total loss =
    dense state
  + terminal position
  + terminal velocity
  + final time
  + control effort
  + control smoothness
```

该模式名为：

```text
control_state_objective = terminal-state
checkpoint_selection_metric = fixed-anchor-terminal-state
```

旧的 `normalized-mse` 和 `physical-criteria` 模式继续保留，未改变数值语义。

## 默认初始配方

```text
dense state weight        = 0.25
terminal position weight  = 1.0
terminal velocity weight  = 1.0
terminal position scale   = 100 m
terminal velocity scale   = 10 m/s
```

两个 terminal 权重均为 dense 权重的4倍。配置校验要求 terminal position 和 terminal
velocity 权重严格大于 dense state 权重，避免新模式被配置成另一个沿途平均误差模式。

这些尺度只用于统一 loss 数值量级，不改变 dynamics 内部状态方程，也不是本轮暂缓的动力学
无量纲化。

## 组件定义

### Dense state

在现有固定2秒监督网格上，对六个标准化通道计算带 mask 的均方误差：

```text
E, N, U, Edot, Ndot, Udot
```

真实观测行监督六通道；拟合尾段只监督位置，速度权重为零。连续的 fitted-tail 权重会真正进入
加权平均，不再像旧 physical ADE 那样只判断权重是否大于零。

```text
dense contribution = 0.25 * normalized fixed-dt six-channel MSE
```

### Terminal position

预测的最后一个动力学 segment endpoint 与目标终点比较三维位置距离：

```text
position distance = norm([E error, N error, U error])
terminal position contribution = position distance / 100 m
```

默认权重已经包含在上式的 `1.0` 中。该项对应终点位置/FDE语义，但在日志中单独命名为
`terminal`，避免把速度混入标准 FDE 名称。

### Terminal velocity

预测终点的三维 chart velocity 与最后一个可靠观测速度比较：

```text
velocity distance = norm([Edot error, Ndot error, Udot error])
terminal velocity contribution = velocity distance / 10 m/s
```

目标选择规则是显式的：

1. 在当前 fixed-dt supervision prefix 中寻找最后一个三个速度通道权重都大于零的行；
2. 拟合尾段速度权重为零，因此其 placeholder 永远不会成为目标；
3. 如果 anchor 后没有任何可靠速度行，使用观测 anchor 速度。

这保证每条航迹都有 terminal velocity 监督，同时不把训练用的拟合占位速度当真值。日志组件名为
`terminal_velocity`。

### Final time

保持既有独立总时长监督：

```text
final time contribution = ((predicted time - true time) / 600 s)^2
```

observed-clock state rollout 与部署时 predicted clock 的区别没有被隐藏。新 checkpoint selection
仍使用预测总时长进行完整部署口径回放。

### Control regularization

保留既有两个独立模块：

```text
control_effort     = 0.001 * normalized control magnitude MSE
control_smoothness = 0.01  * adjacent normalized control-change MSE
```

它们不是终点准确性目标。后续若重新启用非均匀 duration，应单独把它们升级为 duration-weighted
effort 和 physical-time control-rate，而不把该变化混入首轮 terminal-state loss 消融。

## 训练与选模对齐

训练按航迹计算上述组件，再执行机场宏平均。新选模策略在固定 `L-1` anchor 上使用预测总时长和
64点 common physical-time grid，计算：

```text
selection =
    0.25 * common-grid normalized dense state MSE
  + 1.0  * common-grid FDE / 100 m
  + 1.0  * terminal velocity error / 10 m/s
```

每个机场先得到自己的 selection，再对机场等权平均。scheduler、early stop 和 retained best
checkpoint 都使用这个 selection。配置禁止 `terminal-state` 搭配旧 ADE 或 smooth-max selection，
防止训练和选模目标再次错位。

## 解耦结构

| 模块 | 职责 |
|---|---|
| `fixed_dt_control_loss.py` | backend-neutral fixed-dt rollout 与六通道 normalized MSE |
| `control_loss_components.py` | terminal target、原子 loss 组件和 objective registry |
| `physical_criteria.py` | 保留旧 ADE/FDE smooth-max 模式 |
| `control_regularization.py` | absolute 或 trim-residual control regularization |
| `train.py` | 只负责选择 rollout、汇总命名组件和 airport-macro reduction |
| `fixed_anchor_validation.py` | 部署口径 common-grid dense/FDE/terminal-velocity 指标 |

新增其他 loss 配方时，应在 `control_loss_components.py` 增加组合器并注册，而不是在训练循环中
添加多层 `if/else`。每个 component 必须可以单独测试和记录。

## 配置与派生产物契约

新权重和尺度全部写入：

- `TSConfig`；
- checkpoint config；
- `control_recipe`；
- pipeline 命令；
- 输出目录 identity；
- 发布时的用户可见 label。

旧 control checkpoint 缺少任一新字段时要求重新训练，不添加默认升级或兼容 fallback。

## 开发实验命令骨架

```bash
conda run -n aeroviz python run_ts_pipeline.py \
  --training-mode pooled \
  --models itransformer \
  --prediction-output control \
  --control-dynamics-backend transport-chart-velocity \
  --control-state-clock observed \
  --control-state-loss-grid fixed-dt \
  --control-state-objective terminal-state \
  --checkpoint-selection-metric fixed-anchor-terminal-state \
  --control-dense-state-weight 0.25 \
  --control-terminal-position-weight 1 \
  --control-terminal-velocity-weight 1 \
  --control-terminal-position-scale-m 100 \
  --control-terminal-velocity-scale-mps 10 \
  --split development
```

任何权重选择和调试只允许使用 train/validation。设计冻结并得到用户明确授权前，不运行或查看
outer-test prediction。

## 2026-08-01 单航迹过拟合记录

该诊断只打开 outer-train 航迹：

```text
KSJC:ASA956_30L_a1f7fb_20260719T014743Z
```

模型、N=64、factorized duration、transport-chart-velocity dynamics、2秒 fixed-dt
监督、0.5秒积分步长、学习率和随机种子均与上一轮 normalized-MSE 过拟合保持一致；唯一主要
变量是 tracking objective。terminal-state 运行使用默认初始配方，control effort 和
smoothness 均为零，并允许 state loss 对 duration fractions 反向传播。

| 指标 | normalized-MSE | terminal-state |
|---|---:|---:|
| epochs / best epoch | 617 / 317 | 1000 / 999 |
| deployable common-grid ADE | 1808.46 m | 2993.18 m |
| deployable common-grid FDE | 6714.12 m | 59.86 m |
| native endpoint ADE | 1664.24 m | 2709.49 m |
| native endpoint FDE | 6714.12 m | 59.84 m |
| terminal velocity error | 未独立监督/记录 | 1.58 m/s |
| final-time absolute error | 0.00 s | 0.001 s |
| duration entropy | 4.1463 | 4.0704 |

结果表明新 objective 的反向传播链路有效：同一模型可以把终点位置从约6.7 km压到约60 m，
同时把三维终端速度误差压到约1.58 m/s。它也暴露了明确的权衡：`dense weight = 0.25` 时，
模型会优先修正终点而容忍中段绕行，common-grid ADE比旧 objective 增加约1.18 km。

因此该结果证明的是“终端约束可学”，不是“整条进近已经拟合”。下一步五机场 train/validation
实验必须同时报告 ADE、FDE、终端速度和时间误差；不能只按总 loss 或 FDE 宣称整体改善。

产物：

```text
4dTrajectory/outputs/KSJC/
  ts_control_fixed_dt_ASA956_transport_chart_velocity_terminal_state_single_flight_overfit_1000/
    overfit_result.json
```

## 2026-08-01 五机场 development 训练记录

本轮使用与既有 transport-chart-velocity baseline 相同的五机场 development cohort 和训练
配方：iTransformer、N=64、batch 512、学习率 `3e-5`、global clip 20、
`60/120/240 s` horizon curriculum、seed/split-seed 1337。除 objective 和与其绑定的
checkpoint selection 外不改变模型容量或训练超参数。

由于代码处于用户评审前的未提交状态，带 formal experiment identity 的首次启动被仓库的
clean-worktree 保护正确拒绝。实际训练以明确的 `review_pending` development run 执行，未
绕过保护，也没有登记为正式实验。

数据审计：

```text
train selected flights = 10239
validation selected flights = 2167
outer_test_tracks_loaded = false
outer-test selected_flights / identity hash = null
```

训练在 epoch 152 早停，best epoch 为132。下表只比较同一固定 anchor 的 deployable
common physical-time grid；native target grid 另列，避免混用口径。

| validation 指标 | physical-criteria baseline | terminal-state | 变化 |
|---|---:|---:|---:|
| common-grid ADE | 2038.52 m | 2717.21 m | +678.68 m / +33.29% |
| common-grid FDE | 2419.56 m | 2354.60 m | -64.96 m / -2.68% |
| final-time MAE | 75.71 s | 76.57 s | +0.85 s / +1.13% |
| native ADE | 2057.27 m | 2791.37 m | +734.10 m |
| native FDE | 2760.95 m | 2793.55 m | +32.61 m |
| terminal velocity error | 旧产物未独立记录 | 43.50 m/s | 不作跨实验结论 |

结论不是整体提升。新 objective 只在其选模使用的 common-grid FDE 上得到约65 m的小幅收益，
代价是 ADE 增加约679 m；native FDE 反而略差。时间预测也没有改善。

### observed-clock 与 deployable clock 的失配

best epoch 的 observed-clock validation loss 组件为：

```text
weighted dense state       = 0.3981
terminal position / 100 m  = 13.7918   -> 约1379 m
terminal velocity / 10 m/s = 2.1386    -> 约21.4 m/s
final time                 = 0.0291
```

但 best checkpoint 在 deployable predicted clock 上得到：

```text
common-grid FDE             = 2354.60 m
terminal velocity error     = 43.50 m/s
final-time MAE              = 76.57 s
```

首轮训练的 dense、terminal position 和 terminal velocity 都使用已知 observed total time；
inference 和 checkpoint selection 则只能使用预测总时长。单航迹实验把时间误差拟合到
0.001秒，因此终端监督可以转化成约60 m FDE；多航迹的时间 MAE 仍约77秒，导致控制序列在
训练时钟下得到的终点改善不能完整迁移到部署时钟。

### 梯度诊断

新 objective 的 best epoch 仍是20/20 batch 全部触发 global clip：

```text
pre-clip total gradient mean = 712.39
pre-clip control-head mean    = 699.73
pre-clip final-time-head mean = 0.116
mean clip coefficient         = 0.0446
```

旧 baseline 同样全部触发 clip，但 total mean 为592.26、mean coefficient 为0.0639。新终端项
进一步增大了 control-head 梯度，而很小的 final-time-head 梯度仍被同一个 global 系数缩放。
这与时间误差没有改善相符。控制饱和率接近零，因此当前主要问题不是输出撞限。

### 下一步优先级

1. 设计 dual-clock objective：dense fixed-dt tracking 保留 observed clock；terminal position
   和 terminal velocity 用 deployable predicted-clock rollout，直接消除训练/选模时钟错位。
2. 明确 predicted-clock terminal 对总时长的梯度策略，并同步处理 final-time 监督与 clip；优先
   评估让 final-time head 不受 control 大梯度的 global clip 缩放。
3. 完成时钟对齐后，再消融 dense weight；直接把0.25调大只能缓解 ADE，不能解决部署终点与
   observed-clock 终点不是同一个时刻的问题。

本记录不授权或启动上述后续实验。代码评审前不提交；模型冻结前不运行 outer-test。

产物：

```text
4dTrajectory/outputs/POOLED/experiments/
  openap_direct_20260801_terminal_state_transport_chart_velocity/
    itransformer_n64_b512_lr3e5_clip20_seed1337_review_pending/
      checkpoint.pt
      checkpoint_metadata.json
      data_selection.json
      fit_evaluation.json
      history.json
```
