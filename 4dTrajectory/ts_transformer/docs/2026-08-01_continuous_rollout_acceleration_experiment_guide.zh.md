# Continuous dynamics rollout 加速实验指导（2026-08-01）

## 1. 目的与边界

本文用于评估 `transport-chart-velocity` continuous dynamics 的 rollout 加速方案。实验只回答
两个问题：

1. 在不改变动力学语义的前提下，能否减少 forward/backward wall time；
2. 若训练采用数值近似，使用当前精确配置复算时，模型质量是否仍然成立。

本文不修改 backbone、control head、duration parameterization、loss 权重或数据划分。所有开发
决策只使用 outer-train 和 validation；不得打开、生成或查看 outer-test 航迹与预测。

当前数值基线是：

- dynamics backend：`transport-chart-velocity`；
- event-aligned、float64 RK4；
- 最大内部积分步长：`0.5 s`；
- fixed-dt state supervision：`2 s`；
- 非均匀 control 边界作为积分事件，不能因加速而移除或近似错位。

RK4 每个积分步调用 4 次 dynamics。对 600 s 航迹，`0.5 s` 上限约对应 1200 个积分步和
4800 次 dynamics 调用；时间递推具有前后依赖，不能简单把所有时间步并行化。

## 2. 首轮候选及优先级

| 优先级 | 候选 | 是否改变数值方法 | 预期收益 | 主要风险 |
|---:|---|---|---|---|
| 1 | RK4 `dt=1.0 s` | 只改变步长 | 约 2 倍理论上限 | 长航迹累计误差 |
| 2 | 编译完整 rollout | 否 | 减少 Python/kernel launch 开销 | 动态 shape 导致重编译或 graph break |
| 3 | RK4 `dt=2.0 s` | 只改变步长 | 约 4 倍理论上限 | control 边界附近误差可能过大 |
| 4 | 训练用 RK2 `1.0 s`，验证用 RK4 `0.5 s` | 是 | 约 4 倍 dynamics 调用降幅 | train/eval dynamics gap |

首轮不做 adaptive ODE solver、adjoint 或 multiple shooting。adaptive solver 遇到
piecewise-constant control 边界后不保证更快；adjoint 主要节省显存；multiple shooting 会改变
优化问题，不属于执行层加速。

## 3. 实验前置接口与解耦要求

### 3.1 不通过修改默认值切换步长

正式训练 CLI 已有：

```text
--control-rollout-dt <seconds>
```

但 `run_ts_control_fixed_dt_overfit.py` 当前没有暴露这一参数。做单航迹步长消融前，应只增加
同名 CLI 参数并传入 `TSConfig.control_rollout_integrator_dt_s`。禁止临时修改 `TSConfig` 默认值，
否则旧实验目录和新实验配置无法可靠区分。

### 3.2 将“积分语义”和“执行方式”分开

推荐保持以下职责边界：

```text
control_dynamics_backends.py
  -> 定义 transport dynamics 和 RK4 数值语义

fixed_dt_control_loss.py
  -> 发起一次 dense rollout，并让 dense/terminal loss 共享结果

独立 rollout execution strategy
  -> eager 或 compiled；只改变执行方式，不改变返回契约
```

不要在 `train.py` 中用多层 `if/else` 混合 eager、compiled、RK4 和 RK2。若实现编译模式，使用
独立 strategy/adapter，并让它继续返回相同的 query states 和 segment-end states。

### 3.3 训练积分器和评估积分器必须显式区分

在研究“近似训练”前，需要分别记录：

- training rollout integrator 与 step；
- checkpoint selection/validation rollout integrator 与 step；
- artifact replay rollout integrator 与 step。

如果训练使用 `1.0 s` 而 validation 也自动沿用 `1.0 s`，只能说明模型适合近似积分器，不能
证明它在当前 `0.5 s` 数值契约下仍然有效。第一阶段步长审计可以直接比较 rollout；正式模型
选择必须在 RK4 `0.5 s` validation replay 上复核。

新增配置字段属于 checkpoint recipe。旧 checkpoint 若缺少字段，应要求重新生成，不添加默认
升级、双读或兼容 fallback。

## 4. 数据隔离与复现规则

- 数值审计使用锁定的 outer-train/validation 航迹身份；禁止抽取 outer-test。
- 单航迹过拟合只使用既有 outer-train 航迹
  `KSJC:ASA956_30L_a1f7fb_20260719T014743Z`。
- pooled 实验保持五机场 KMSY、KRDU、KSJC、KSMF、KSTL 和 `split_seed=1337` 不变。
- screening 固定 `seed=1337`；胜出方案才做多种子确认。
- 每个候选使用新的 output directory，禁止覆盖、续训或复用不同 recipe 的 checkpoint。
- 保存 manifest hashes、train/validation identity hashes、代码状态和完整 config。
- 性能实验必须在同一 GPU、同一 batch size、同一 dtype、同一软件环境中完成。

GPU 性能只能在确认设备映射成功的环境中测量。先运行 `nvidia-smi` 并确认 Python 进程实际占用
GPU；若沙箱没有映射 GPU，不得退回 CPU 后把结果与 GPU baseline 比较。

## 5. 阶段 A：纯 rollout 数值与性能审计

### 5.1 固定输入

锁定一组 train/validation 航迹及同一份：

- initial states；
- controls；
- segment durations；
- aero/frame parameters；
- `2 s` query offsets。

候选必须复用完全相同的 tensor。先比较 `dt in {0.5, 1.0, 2.0}`；`0.5 s` 是当前 reference。
如果需要进一步确认 reference 自身已收敛，可在少量样本上额外跑 `0.25 s`，但不能把它混入
首轮模型训练。

### 5.2 正确计时

forward 和 forward+backward 分开计时。CUDA 是异步执行，计时区间前后必须调用
`torch.cuda.synchronize()`。每个候选至少：

1. warm-up 10 次，不计入结果；
2. 正式计时 50 次；
3. 报告 median、p90，而不是只报告最快一次；
4. 记录 peak VRAM；
5. compiled 模式单独报告首次编译时间，steady-state 结果不能包含编译时间。

不要只测 dynamics forward。训练瓶颈必须用真实 loss、完整 rollout 和 backward 测量。

### 5.3 数值指标

以同一物理 `2 s` query grid 比较 candidate 与 RK4 `0.5 s` reference：

| 类别 | 指标 |
|---|---|
| 位置 | rollout-to-reference ADE、FDE、最大 3D 距离 |
| 速度 | dense velocity RMSE、terminal velocity 3D error |
| 其他状态 | final mass absolute error、任何非有限值 |
| 梯度 | loss 差、gradient relative L2、gradient max absolute difference |
| 性能 | forward median、forward+backward median、p90、peak VRAM、speedup |

建议在运行前冻结首轮门槛：

```text
rollout-to-reference ADE <= 10 m
rollout-to-reference FDE <= 25 m
terminal velocity difference <= 0.5 m/s
non-finite/failure count = 0
forward+backward speedup >= 1.5x
```

这些是工程 screening 门槛，不是物理定律。如果论文需要更严标准，应在看结果前修改并记录，
不能在结果出来后移动门槛。

## 6. 阶段 B：单航迹过拟合复核

只有通过阶段 A 的候选进入该阶段。保持现有单航迹实验的模型、N=64、seed、loss、LR、epoch、
patience、control regularization 完全不变，只改变 rollout execution recipe。

步长候选的目录必须明确编码，例如：

```text
..._transport_chart_velocity_terminal_state_rk4_dt05_...
..._transport_chart_velocity_terminal_state_rk4_dt10_...
..._transport_chart_velocity_terminal_state_rk4_dt20_...
```

在 runner 暴露 `--control-rollout-dt` 后，命令结构为：

```bash
conda run -n aeroviz python run_ts_control_fixed_dt_overfit.py \
  --airport KSJC \
  --flight-id KSJC:ASA956_30L_a1f7fb_20260719T014743Z \
  --epochs 1000 --patience 300 --lr-plateau-patience 150 \
  --learning-rate 1e-4 --n-segments 64 \
  --control-dynamics-backend transport-chart-velocity \
  --control-state-objective terminal-state \
  --control-rollout-dt <0.5|1.0|2.0> \
  --device cuda \
  --output-dir <unique-output-dir>
```

必须同时报告：

- 训练 recipe 下的 best loss、ADE/FDE 和 terminal velocity error；
- 将 best controls 用 RK4 `0.5 s` 重新 replay 后的同一组指标；
- epochs、optimizer updates、总 wall time 和每 epoch 中位时间；
- duration entropy、control saturation，确认加速没有改变优化退化方式。

单航迹通过条件不是只看 loss。candidate 在精确 replay 上不能出现 FDE 或终端速度明显退化；
若训练指标改善但精确 replay 退化，应判定为对近似积分器过拟合。

## 7. 阶段 C：pooled train/validation 实验

单航迹没有暴露明显 train/eval gap 后，才进入五机场 pooled development training。冻结当前正式
配方，包括：

```text
backbone = iTransformer
N = 64
batch = 512
learning rate = 3e-5
gradient clip = global 20
horizon curriculum = 60,120,240 s；每阶段 10 epochs
split seed / training seed = 1337 / 1337
```

比较顺序：

1. RK4 `0.5 s` eager baseline；
2. RK4 `1.0 s` eager；
3. RK4 `0.5 s` compiled；
4. 若前两项分别通过，再测 RK4 `1.0 s` compiled。

所有 candidate 的 checkpoint selection 和最终 fit evaluation 都必须补充 RK4 `0.5 s` 的固定
anchor validation replay。Primary quality metrics：

- validation common-grid ADE；
- validation common-grid FDE；
- validation terminal velocity 3D error；
- final-time MAE。

性能指标：

- optimizer updates/s；
- epoch median/p90 wall time；
- 端到端训练 wall time；
- peak VRAM；
- 首次编译时间与重编译次数。

单 seed validation 改善或退化小于 2% 先视为噪声。加速候选的目标是质量非劣且时间显著降低，
不是利用近似 dynamics 获得更好看的近似 validation 数字。

## 8. 阶段 D：compiled rollout 专项检查

编译模式必须满足：

- eager 与 compiled 使用相同 RK4 和 step；
- 固定 batch shape 或按少量 bucket 固定 shape；
- padding/mask 为 tensor 操作；
- 时间循环内没有 `.item()`、Python tensor 分支或动态 list append；
- 不因不同航迹时长每个 batch 重编译；
- dense、terminal position、terminal velocity 共享一次 rollout。

验收时先检查 eager/compiled 输出和梯度一致性，再测速度。如果 graph break 或重编译使 p90
变差，即使单次最快值很好也不能采用。

## 9. 阶段 E：RK2 近似训练（仅作为后续候选）

只有 RK4 步长和 compiled rollout 收益不足时才测试：

```text
training = midpoint/RK2, dt=1.0 s
checkpoint selection and final replay = RK4, dt=0.5 s
```

RK2 必须作为新的 integrator implementation，不允许在名为 RK4 的函数里加分支。模型 artifact
标签和 recipe 必须明确写出 train/eval integrator。任何 exact-validation 退化都应直接报告，
不要通过改变 loss 权重补偿，因为那会引入第二个实验变量。

## 10. 结果记录模板

### 10.1 数值与 microbenchmark

| candidate | F+B median ms | p90 ms | speedup | peak VRAM | ADE vs ref m | FDE vs ref m | terminal vel diff m/s | grad rel L2 | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RK4 0.5 eager | | | 1.00 | | 0 | 0 | 0 | 0 | baseline |
| RK4 1.0 eager | | | | | | | | | |
| RK4 2.0 eager | | | | | | | | | |
| RK4 0.5 compiled | | | | | | | | | |

### 10.2 单航迹与 pooled training

| candidate | train recipe | exact replay | best epoch | wall time | val ADE m | val FDE m | terminal vel m/s | time MAE s | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | RK4 0.5 eager | RK4 0.5 | | | | | | | |
| candidate | | RK4 0.5 | | | | | | | |

每个表格下方记录：代码状态、GPU 型号、PyTorch/CUDA 版本、manifest hashes、identity hashes、完整
命令和 output directory。

## 11. 决策树

```text
RK4 1.0 s 数值审计通过？
  否 -> 保留 0.5 s，测试 compiled 0.5 s
  是 -> 做单航迹，再做 pooled train/validation exact replay

compiled 输出/梯度一致且 steady-state 加速？
  否 -> 不采用，不为编译模式重写 dynamics 语义
  是 -> 与通过的最佳 RK4 step 组合复核

仍未达到目标速度？
  否 -> 冻结最快的质量非劣 recipe
  是 -> 最后测试 RK2 1.0 s train / RK4 0.5 s validation
```

最终采用条件：没有 rollout failure；RK4 `0.5 s` validation replay 质量非劣；端到端训练时间有
稳定、可复现的下降。outer-test 在模型、loss、integrator 和所有超参数完全冻结前始终保持未打开。
