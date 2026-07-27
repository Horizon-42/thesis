# 归一化时间轨迹与可切换控制输出设计

## 1. 当前实现结论

`ts_transformer` 现在保留三种可以独立切换的预测时域：

1. 输入仍为最近 `L` 个等间隔观测点；
2. 每个样本都预测从 anchor 到终点的完整剩余轨迹；
3. 剩余轨迹统一映射到 `tau ∈ (0, 1]` 的 `N` 个端点；
4. 模型同时预测物理量 `final_time_s`；
5. 推理时用 `final_time_s` 把归一化进度还原为真实时间；
6. 当前训练输出仍为 state，但输出层已经和预测契约分离；
7. controls 输出与非均匀 `segment_durations` 的 head 已实现，等有控制监督和可微动力学后再接入训练。

默认采用 `--horizon-mode normalized`：一次预测完整剩余航迹。`--horizon-mode full`
恢复单次 `H_full=300`、`dt_s=2 s` 的物理时间输出；短航迹的最后一个 segment 缩短到
真实终点，后续 padding 完全 mask。`--horizon-mode window` 恢复 `H_window=30` 的短时预测，
推理时把输出追加到历史并递归到 `H_full`。full/window 都按到跑道入口的空间最近点截断，
没有到达入口则显式标记 horizon-capped。

三种模式共用 state/final-time 输出层、loss、anchor 策略和模型结构，但目标时间网格与
推理策略分别在 `time_grids.py` 和 `forecast.py` 中独立分派。`final_time_s` 在 normalized
模式下决定节点时间，在 full/window 下作为辅助剩余时间预测，不改变固定 `dt_s` 时钟。

## 2. 归一化时间监督

设 anchor 的物理时间为 `t_a`，监督轨迹终点时间为 `t_f`：

```text
true_final_time_s = t_f - t_a
tau_i = i / N,                 i = 1, ..., N
t_i = t_a + tau_i * true_final_time_s
```

数据集在 `t_i` 上对剩余 state 和监督权重做线性插值，返回：

```text
x                  [L, C]
target_states      [N, C]
state_weights      [N, C]
true_final_time_s  scalar
```

其中 `C=6`，通道顺序仍为：

```text
(e, n, u, edot, ndot, udot)
```

`N` 只决定归一化路径的离散分辨率，不代表预测多少秒。因此短进近和长进近都产生相同形状，不再需要 padding，也不会因为固定时域不足而被截断。

## 3. state 输出与联合损失

当前模型返回结构化结果：

```text
StatePrediction
├── states        [B, N, C]  归一化后的状态通道
└── final_time_s  [B]        anchor 到终点的物理秒数
```

`final_time_s` 经过 `softplus`，所以为正。`final_time_scale_s` 只用于把时间误差无量纲化，不是时间上限。

训练目标为：

```text
L = L_state
    + lambda_t * L_time
    + lambda_kin * L_kinematic
    + lambda_terminal * L_terminal

L_state = weighted_MSE(predicted_states, target_states)
L_time  = mean(((predicted_final_time_s - true_final_time_s)
                / final_time_scale_s)^2)
delta_p_error = (p[i+1] - p[i])
                - ((v[i+1] + v[i]) / 2) * Delta_t
L_kinematic = MSE(delta_p_error / position_std)
L_terminal = MSE(predicted_position[N], target_position[N])
```

`L_kinematic` 在反归一化后的物理位置/速度上计算，把相邻位置位移与梯形积分得到的
速度位移进行比较，再按训练集位置标准差无量纲化。不能先除以 `Delta_t` 再按速度
标准差归一化；后一种写法会让位置梯度随 N 近似按 `N²` 放大。训练时
`Delta_t=true_final_time_s/N`，避免时间预测头通过缩短时长来降低一致性损失。

`L_terminal` 独立强调跑道入口节点，但默认权重为 `0.02`。在 iTransformer 当前
`N=64, C=6` 时，该值使终点位置约为普通单个位置节点的 2.6 倍权重，而不是
`1.0` 所造成的约 128 倍。held-out 消融后的模型级 N 默认值为 iTransformer 64、
PatchTST 256；共享 `lambda_kin=3.0`、`lambda_terminal=0.02`。训练历史会
分别保存 state、final-time、kinematic 和 terminal 四项加权贡献；训练、validation
early stopping 和 CV 候选选择使用四项之和。评估另外输出 `final_time_s` 的 MAE、
RMSE 和有符号均值误差。

评估还直接在未经样条、滤波或 CZML 插值的输出节点上计算五项 raw kinematic
指标：位置差分速度/状态速度 RMSE、航向一致性 p95、转弯率 p95、加速度 p95 和
jerk p95。逐航班结果写入 prediction rows，批量结果写入 fleet
`median/mean/p95/max`；消融使用 fleet p95 与同批 observed baseline 的比值，避免
单个接近零的 `final_time_s` 将均值放大后支配选模。

## 4. N 如何参与交叉验证

默认 CV 参数包含 `n_segments`。它的固定候选网格为：

```text
N ∈ {64, 128, 256}
```

CV 会完整遍历 `n_segments`、`learning_rate` 和 `d_model` 的 27 种默认组合，不再随机抽取候选。N 的选择会同时影响：

- 输出空间分辨率；
- state head 参数量；
- 显存和训练吞吐；
- 对转弯、下滑和终端状态细节的表达能力。

它不影响一个样本是否可用：只要有完整的 `L` 点历史并且 anchor 后还有监督轨迹，该样本就能映射到任意 N。

## 5. 推理时如何恢复真实时间

模型只前向一次。设模型预测时长为 `T_hat`：

```text
t_hat_i = t_anchor + (i / N) * T_hat
```

最后一个 state 的相对时间严格为 `T_hat`，因此导出记录满足：

```text
record.final_time_s == record.states[-1].t
```

这里不再进行几何截断。模型必须同时学习“终点在哪里”和“多久到达”，而不是依赖后处理替它决定停止位置。

## 6. 为什么要抽象预测输出层

当前层次是：

```text
vendored state forecaster
        │
        ├── state sequence
observed history ── FinalTimeHead
        │
        └── StateOutputLayer ── StatePrediction
```

训练和推理代码读取的是命名字段，而不是假设 `model(x)` 必须是一个 tensor。以后切换到 controls 时，可以使用另一种结构化结果，不必把 controls 伪装成 state，也不必把时段信息塞进额外状态通道。

这次没有强行重写 iTransformer/PatchTST 的内部 encoder 来构造统一 latent API。两套 vendored backbone 的内部表示差异较大，在还没有控制标签和 rollout loss 时提前统一会增加无用耦合。

## 7. controls 输出契约

控制量顺序与当前动力学/优化代码一致：

```text
controls[..., 0] = thrust_N
controls[..., 1] = bank_rad
controls[..., 2] = load_factor
```

未来 controls 模型返回：

```text
ControlPrediction
├── controls             [B, N, 3]
├── segment_durations    [B, N]
└── final_time_s         [B]
```

`ControlOutputHead` 接收通用特征、飞机对应的 `ControlBounds` 和已预测的 `final_time_s`。原始控制 logits 经 sigmoid 映射到物理上下界：

```text
u_i = lower + sigmoid(raw_u_i) * (upper - lower)
```

控制边界由调用者按飞机性能提供，head 本身不绑定 A320 或某一套固定性能参数。

## 8. 非均匀 segment_durations

每个控制段不再默认等时长。head 为 N 段预测 duration logits：

```text
alpha_i = softmax(duration_logits)_i
Delta_t_i = alpha_i * final_time_s
```

由构造可得：

```text
Delta_t_i > 0
sum_i Delta_t_i = final_time_s
```

这样模型可以给转弯、截获下滑道或近地阶段分配更短的控制段，给稳定直飞阶段分配更长的控制段，同时不引入独立的“总时长一致性”修补逻辑。

## 9. 后续接入 controls 训练的最小路径

后续只需要补齐三件事：

1. 从轨迹反演或仿真生成 controls 监督，或定义端到端 rollout loss；
2. 让选定 backbone 暴露供 `ControlOutputHead` 使用的特征；
3. 用 `controls + segment_durations` 调用现有的 piecewise-constant dynamics rollout，并对 rollout states 计算轨迹与终端损失。

state 数据集、`final_time_s` 监督、CV 的 N、导出所需的物理时间语义都不需要重新设计。

## 10. 主要代码位置

| 文件 | 职责 |
|---|---|
| `config.py` | `n_segments`、时间 loss 权重/尺度及 vendored `pred_len` 属性映射 |
| `dataset.py` | 剩余轨迹到固定 `tau` 网格的插值和 `final_time_s` 标签 |
| `prediction_outputs.py` | state/control 数据契约、时长 head、控制边界和非均匀时段 head |
| `models.py` | vendored state forecaster 与 `StateOutputLayer` 组装 |
| `train.py` | state/time 联合 loss、时间指标和 checkpoint |
| `cross_validation.py` | 将 N 纳入候选选择 |
| `forecast.py` | 单次预测及归一化进度到真实时间的恢复 |
| `export.py` | 输出 N、预测时长和 state evaluation records |
