# 双时钟终端监督消融

## 问题与边界

本实验只使用 KSJC outer-train / outer-validation，未读取、生成或发布 test 预测。目标是检验：
保持 dense/arc 几何损失使用真实观测时钟，同时让终端位置与终端速度使用部署时的模型预测时钟，
是否能缩小训练时钟与部署时钟之间的目标错配。

实现保持解耦：`control_terminal_clock.py` 独立拥有终端时钟策略，loss component 只读取统一的
`terminal_end_states`，不包含 rollout 或模式判断。三种公开模式为：

- `state-supervision`：原基线，终端与 dense loss 共用观测时钟 rollout；
- `predicted`：部署时钟终端，终端梯度同时更新 controls、duration fractions 和总时长头；
- `predicted-detached-time`：前向仍使用模型预测总时长，但终端梯度只更新 controls 和 duration
  fractions；总时长头只由独立 final-time loss 训练。

数值设置未改变：连续动力学 RK4 最大子步仍为 0.5 s，reference 监督仍为固定 2 s 网格，N=64。
数值 curriculum 的 60/120/240 s 阶段仍使用真实物理前缀终端；第二部署时钟 rollout 只在 full
horizon 启用。

## 实现验证与运行时修复

新增回归验证了隔离模式中 duration partition logits 接收非零终端梯度，而总时长张量的终端梯度
严格为零。完整相关测试为 280 passed、6 skipped。

联合模式的完整单航迹运行还发现 endpoint adjoint 的单 batch state view 保留了随 horizon 改变的
singleton stride，导致 `torch.compile` 达到重编译上限。修复仅把 adjoint 输入复制为规范连续布局，
不改变方程、积分步长或数值。CPU 回归覆盖不同 horizon 的相同 stride，12 epoch GPU 回归越过原
失败位置。

## 单航迹容量门槛

航迹：outer-train `KSJC:DAL1260_30L_a05f31_20260716T224959Z`。两种双时钟模式均使用
iTransformer、N=64、transport-chart-velocity、arc 2+4 objective、1000 epoch、LR=1e-4，除
终端时钟梯度策略外配方一致。

| 模式 | 部署 common ADE | 部署 FDE | 终端速度误差 | 总时长误差 |
|---|---:|---:|---:|---:|
| 原单时钟基线 | 313.11 m | 1.42 m | 7.67 m/s | 0.0013 s |
| predicted，联合梯度 | 941.58 m | 48.89 m | 49.55 m/s | 33.16 s |
| predicted-detached-time | 397.03 m | 1.16 m | 15.68 m/s | 0.0046 s |

联合梯度模式发生明确的时间捷径：33 s 误差在以 600 s 归一化的 final-time loss 中代价很小，
终端损失却可通过改变总时长迅速降低。隔离模式修复了该坍缩并显著优于联合模式，证明机制诊断
成立；但它仍未超过原单时钟基线。

输出：

- `outputs/KSJC/experiments/dual_clock_terminal_20260802/DAL1260_predicted_terminal_1000_retry1/`
- `outputs/KSJC/experiments/dual_clock_terminal_20260802/DAL1260_predicted_detached_time_1000/`

## 正式 KSJC train/validation

隔离变体通过“显著优于联合模式”的最低容量门槛后，进行一次固定配方正式消融。共同配方为
KSJC all-QFU OpenAP-direct、固定 anchor、iTransformer、N=64、batch=512、LR=3e-5、
60/120/240 s curriculum、global clip20、transport-chart-velocity、arc 2+4、seed/split seed=1337；
训练 2207 条，validation 505 条。

| 模式 | validation selection | ADE | FDE | cross-track p95 | altitude p95 | time MAE |
|---|---:|---:|---:|---:|---:|---:|
| state-supervision 基线 | 31.4126 | **1130.67 m** | **1202.94 m** | **3313.22 m** | 692.65 m | 20.84 s |
| predicted-detached-time | **29.2073** | 1297.70 m | 1394.54 m | 3441.31 m | **626.89 m** | **19.80 s** |

隔离模式把 selection objective 改善约 7.0%，并改善高度和时间；但 ADE 退化约 14.8%，FDE 退化
约 15.9%，cross-track p95 也变差。它优化了当前 arc objective 的某些组成部分，却没有改善完整
轨迹几何。full-horizon epoch 约 9.75 s，基线约 5.53 s，计算成本接近 1.76 倍。

正式输出：

- `outputs/KSJC/experiments/dual_clock_terminal_20260802/formal_predicted_detached_time/`

## 结论

不把双时钟终端监督升级为新基线。继续使用 `state-supervision`：它更简单、更快，并在主要
validation 轨迹指标上更好。`predicted` 保留为可复现的失败机制，`predicted-detached-time` 保留
为研究消融；二者都不用于后续基线实验，也没有启动 outer-test。
