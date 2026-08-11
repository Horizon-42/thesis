# 归一化时间轨迹与可切换控制输出设计

## 1. 当前实现结论

`ts_transformer` 现在保留三种可以独立切换的预测时域：

1. 输入仍为最近 `L` 个等间隔观测点；
2. 每个样本都预测从 anchor 到终点的完整剩余轨迹；
3. 剩余轨迹统一映射到 `tau ∈ (0, 1]` 的 `N` 个端点；
4. 模型同时预测物理量 `final_time_s`；
5. 推理时用 `final_time_s` 把归一化进度还原为真实时间；
6. `prediction_output=state|control` 显式选择输出架构，并完整写入 checkpoint；
7. control 模式输出有界 controls 与非均匀 `segment_durations`，经可微 Torch 动力学
   rollout 得到 state，再直接接受轨迹监督，不要求反演 control 标签。

默认采用 `--horizon-mode normalized`：一次预测完整剩余航迹。`--horizon-mode full`
恢复单次 `H_full=300`、`dt_s=2 s` 的物理时间输出；短航迹的最后一个 segment 缩短到
真实终点，后续 padding 完全 mask。`--horizon-mode window` 恢复 `H_window=30` 的短时预测，
推理时把输出追加到历史并递归到 `H_full`。full/window 都按到跑道入口的空间最近点截断，
没有到达入口则显式标记 horizon-capped。

三种时域都保留在 state 输出路径中，共用 state/final-time 输出层、loss、anchor 策略和
模型结构，但目标时间网格与推理策略分别在 `time_grids.py` 和 `forecast.py` 中独立分派。
`final_time_s` 在 normalized 模式下决定节点时间，在 full/window 下作为辅助剩余时间
预测，不改变固定 `dt_s` 时钟。control 输出目前只支持 normalized 时域，因为 N 在该
路径中表示非均匀控制段，而不是固定物理时钟的 state 节点。

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

control 模式另外返回按航班解析的初始 geodetic state、气动参数、控制上下界、坐标系参数
和 8 维条件向量；state 模式仍返回原来的五元组，原有数据 API 与 checkpoint 不变。

其中 `C=6`，通道顺序仍为：

```text
(e, n, u, edot, ndot, udot)
```

`N` 只决定归一化路径的离散分辨率，不代表预测多少秒。因此短进近和长进近都产生相同形状，不再需要 padding，也不会因为固定时域不足而被截断。

## 3. state 输出与联合损失

state 模式返回结构化结果：

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

state 路径保持原层次：

```text
vendored state forecaster
        │
        ├── state sequence
observed history ── FinalTimeHead
        │
        └── StateOutputLayer ── StatePrediction
```

训练和推理代码读取命名字段，而不是假设 `model(x)` 必须是一个 tensor。control 路径
因此返回另一种结构化结果，不把 controls 伪装成 state，也不把时段信息塞进额外状态通道。

两套 vendored 源码仍不修改。薄 adapter 分别复用它们在 state projector/head 之前的
表示。时间/patch 维可以池化，但通道维严格按
`(e,n,u,edot,ndot,udot)` 保留并展开为 `[B,C*d_model]`；两套 backbone 都没有通道索引
embedding，若在通道维做无标签 mean，会让水平位置/速度互换后的 control 几乎完全相同。
control 模型会移除不用的 state head，再把有序轨迹特征与按航班提供的动力学条件编码融合：

```text
history -> encoder -> per-channel temporal pool -> ordered flatten ┐
dynamics condition -> MLP                                         ├-> fused feature
history -> FinalTimeHead                                          ┘
fused feature + per-flight bounds -> ControlOutputHead
```

## 7. controls 输出契约

控制量顺序与当前动力学/优化代码一致：

```text
controls[..., 0] = thrust_N
controls[..., 1] = bank_rad
controls[..., 2] = load_factor
```

control 模型返回：

```text
ControlPrediction
├── controls             [B, N, 3]
├── segment_durations    [B, N]
└── final_time_s         [B]
```

`ControlOutputHead` 接收通用特征、逐样本控制上下界和已预测的 `final_time_s`。原始控制
logits 经 sigmoid 映射到物理上下界：

```text
u_i = lower + sigmoid(raw_u_i) * (upper - lower)
```

当前边界为推力 `[0, max_thrust]`、坡度角 `[-pi/4, pi/4]`、载荷因子 `[0.5, 2.0]`。
`max_thrust` 和气动参数来自该航班解析出的 aircraft/scenario，head 本身不绑定 A320
或某一套固定性能参数。模型条件向量包括 mass、max thrust、翼面积、`Cl_max`、`Cd0`、
诱导阻力系数、失速阈值和失速附加阻力系数。

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

## 9. control rollout 与训练目标

Torch rollout 的外部状态、控制量顺序与 `CasadiSimulator` 完全一致：

```text
state   = (lat_deg, lon_deg, alt_m, V, psi, gamma, mass_kg)
control = (thrust_N, bank_rad, load_factor)
```

每个非均匀控制段按 `control_rollout_integrator_dt_s`（默认 0.5 s）细分；最后一步截到
segment 边界。每个 substep 都执行与 CasADi 相同的过程：当前 geodetic state 建立局部
ENU、同一 ISA/阻力/失速与载荷限制、同一显式 RK4，然后经 WGS84 ECEF 把位置和速度
转换到新的局部 ENU。rollout 使用 float64，避免约 `6e6 m` ECEF 坐标在 float32 中做
局部位移差时产生约半米量化；转成 float64 不切断回到 FP32 网络参数的梯度。

rollout endpoints 转成同一套六通道并归一化。由于 duration partition 是非均匀的，第 i 个
rollout endpoint 位于 `cumsum(Delta_t_hat)[i]`，不能直接与均匀真值节点 `i*T_true/N`
按索引相减。训练和 fit evaluation 都先把 anchor 与均匀真值拼成 `t=0..T_true` 的折线，
再把 states/weights 插值到预测累计物理时钟；超过真实终点的查询 clamp 到 terminal，
总时长误差仍由独立 `L_time` 约束。control loss 为：

```text
L_control = L_rollout_state
          + lambda_t * L_time
          + lambda_terminal * L_terminal
          + lambda_effort * L_control_effort
          + lambda_smooth * L_control_smoothness
```

控制 effort/smoothness 都按每架飞机的控制范围无量纲化。state/velocity kinematic loss
在 control 路径记为 0，因为二者来自同一次动力学 rollout，一致性由构造保证。训练不需要
control 标签，状态轨迹与终端监督的梯度直接穿过 WGS84 变换、RK4、controls 和 durations。

明确的等价性测试位于 `aerodynamic_model/tests/test_torch_dynamics.py`：

1. 正常气动分支的单步结果与 `CasadiSimulator` 对比；
2. 超过 `Cl_max` 的失速/附加阻力分支单步对比；
3. 非均匀多段 rollout 的每个 segment endpoint 与 CasADi 数值 replay 对比；
4. controls 和每个 segment duration 的梯度均为有限且非零。

单步容差为 `rtol=2e-12, atol=2e-9`，多段端点容差为
`rtol=3e-12, atol=3e-8`。这些测试锁定的是离散行为等价，不只是连续方程“看起来相同”。

## 10. 主要代码位置

| 文件 | 职责 |
|---|---|
| `config.py` | output strategy、`n_segments`、control loss/rollout 设置与 checkpoint 配置 |
| `dataset.py` | 固定 `tau` 监督及逐航班 initial state/aero/bounds/conditioning |
| `prediction_outputs.py` | state/control 数据契约、时长 head、控制边界和非均匀时段 head |
| `models.py` | state head 或 encoder feature + dynamics conditioning + control head |
| `aerodynamic_model/torch_dynamics.py` | CasADi 离散行为的可微 Torch twin |
| `train.py` | state loss 或 control rollout loss、时间指标和 checkpoint |
| `cross_validation.py` | 将 N 纳入候选选择 |
| `forecast.py` | state 时钟恢复或 control rollout 推理 |
| `export.py` | state reference-shaped 或 control optimizer-shaped evaluation records |
