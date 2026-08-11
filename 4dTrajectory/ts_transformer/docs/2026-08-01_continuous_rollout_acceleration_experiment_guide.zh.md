# Continuous dynamics rollout 加速实验指导（2026-08-01）

## 1. 目的与边界

本文用于评估 `transport-chart-velocity` continuous dynamics 的 rollout 加速方案。实验只回答
两个问题：

1. 在不改变动力学语义的前提下，能否减少 forward/backward wall time；
2. 若训练采用数值近似，使用当前精确配置复算时，模型质量是否仍然成立。

本文不修改 backbone、control head、duration parameterization、loss 权重或数据划分。所有开发
决策只使用 outer-train 和 validation；不得打开、生成或查看 outer-test 航迹与预测。

### 1.1 当前阶段的硬约束：加速不得改变模型质量语义

当前获准实施的加速仅限于**计算复用、固定输入缓存、等价批处理和性能观测**。优化前后必须执行
完全相同的浮点运算语义与训练协议；“验证结果接近”不能替代这一要求。具体禁止通过以下方式
换取速度：

- 修改 RK4 最大内部步长（包括 `0.5 -> 1.0/2.0 s`）；
- 更换积分器、使用训练期近似积分器，或改变 event-aligned control 边界；
- 修改 dynamics/state dtype、loss、control segments、horizon curriculum 或模型结构；
- 降低 validation/checkpoint-selection 频率、抽样 validation 航迹；
- 改变 train batch 的组成、顺序或 SGD optimizer-update 序列；
- 使用会改变结果数值语义的 compiled/AMP 路径，除非以后单独获得明确授权。

因此，本轮只实施第 12.3 节和第 12.6 节的 P0--P4 等价优化。第 2--11 节记录的是早期候选与
审计方法，保留作历史研究备忘，**不属于当前实施计划，也不得据此启动实验**。若未来考虑其中
任何会改变数值方法或训练协议的方案，必须先由用户明确授权，并使用独立 recipe/artifact。

当前数值基线是：

- dynamics backend：`transport-chart-velocity`；
- event-aligned、float64 RK4；
- 最大内部积分步长：`0.5 s`；
- fixed-dt state supervision：`2 s`；
- 非均匀 control 边界作为积分事件，不能因加速而移除或近似错位。

RK4 每个积分步调用 4 次 dynamics。对 600 s 航迹，`0.5 s` 上限约对应 1200 个积分步和
4800 次 dynamics 调用；时间递推具有前后依赖，不能简单把所有时间步并行化。

## 2. 首轮候选及优先级

> 历史候选，当前禁用：本节及第 3--11 节不属于当前等价加速实施范围。

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

## 12. 完整 train/validation 流程优化

前面各阶段主要优化单次 continuous rollout。本节补充完整 development training 的执行层分析，
涵盖一个 epoch 内的训练、validation loss、checkpoint selection 和训练结束后的 fit evaluation。
这些优化必须先于新的积分器近似进行，否则无法判断收益来自流水线消重还是数值语义改变。

### 12.1 当前一个 epoch 实际执行什么

`fit_model()` 当前包含三条主要计算路径：

```text
train batch
  -> model forward
  -> observed-supervision-clock rollout
  -> loss backward + optimizer step

validation objective
  -> model forward
  -> observed-supervision-clock rollout
  -> validation loss components

checkpoint selection: arc/common-grid metrics
  -> model forward
  -> deployable predicted-clock rollout
  -> fixed-anchor common-grid metrics
```

因此，使用 common-grid、terminal-state 或 arc-length checkpoint selection 时，每条 validation
航迹每个 epoch 会经过两次 model forward 和两种 clock 的 dynamics rollout。两个 rollout 的
物理时钟不同，不能直接把其中一个删除：validation objective 监督已知真实时长，deployable
metric 必须使用模型实际预测时长。但同一个 batch 的数据准备、Host→Device 传输和 model forward
可以共享。

训练结束后的 `evaluate_fit_splits()` 还有一处确定性重复：`evaluate_split()` 与
`evaluate_fixed_anchor_common_grid()` 分别调用同一份 `_predict_split()`。这两个报告可以从一次
deployable prediction 缓存分别计算，不需要重新 forward 和 rollout。

### 12.2 已观测的单航迹成本

在 RTX 4060、iTransformer、N=64、transport continuous dynamics、RK4 `0.5 s` 下，既有五条
train-only 容量实验的稳定 epoch 中位时间为：

| 剩余时域 | epoch 中位时间 | 每 1000 epoch 纯 epoch 时间 |
|---:|---:|---:|
| 107.7 s | 0.266 s | 4.4 min |
| 154.5 s | 0.352 s | 5.9 min |
| 179.9 s | 0.397 s | 6.6 min |
| 428.0 s | 0.866 s | 14.4 min |
| 580.0 s | 1.168 s | 19.5 min |

首个 epoch 另有约 4--5 s 的 CUDA、数据与 kernel 初始化开销。单航迹结果表明成本随时域近似
线性增加，但不能用上表直接外推 pooled epoch：完整训练每 epoch 覆盖全部 train flights，且
batch 中最长航迹、batch size、validation 双 rollout 和机场拆分都会影响实际利用率。

### 12.3 第一优先级：不改变科学实验定义

#### A. 分段计时后再优化

当前 `EpochResult.seconds` 只记录整个 epoch。至少应独立记录：

- `train_data_s`：组 batch 与 Host→Device；
- `train_forward_s`、`train_rollout_loss_s`、`train_backward_step_s`；
- `val_objective_s`；
- `val_checkpoint_selection_s`；
- epoch 总时间、peak VRAM、optimizer updates/s；
- 每个 validation airport 的航迹数、总 query 点数和 wall time。

CUDA 计时必须在边界同步。先用完整配置运行 3 个 warm-up epoch 和至少 10 个正式 epoch，报告
median/p90；不能依据首个 epoch 或单个最快值决定架构。

#### B. 一次 validation model forward 服务两个 clock

建立独立的 validation batch evaluator：每个 batch 只执行一次 backbone/control head，然后把同一
份 `ControlPrediction` 分发给：

1. observed-clock state-loss evaluator；
2. deployable-clock common-grid/checkpoint evaluator。

两次 dynamics rollout 保持独立，不能把 observed durations 与 deployable endpoints 混配。该优化
不改变模型输出、loss、checkpoint selection 或积分器，只消除重复 model forward、数据解包和
设备传输。预期端到端 epoch 收益约 5%--15%，实际值以分段计时为准。

#### C. 复用最终 fit-evaluation prediction

将 `_predict_split()` 的物理预测、truth、mask、predicted/true time、anchor 和 segment durations
视为一次不可变 replay 结果。native metrics 与 common-grid metrics 从同一结果计算。该改动主要
把训练结束后的 train/validation 报告阶段缩短约一半，不改变 epoch 时间。

#### D. Validation 按剩余时域分桶

建议先只对无梯度 validation 使用少量固定 bucket，例如：

```text
(0, 180] s
(180, 360] s
(360, 600] s
```

每个 bucket 分别探测最大安全 batch size。这样短航迹不必受最长航迹的调度 shape 和显存上限
约束，compiled 模式也更容易保持少量固定 shape。validation 没有 SGD 顺序问题；仍需确认分桶
前后 airport-macro metric 在既定浮点容差内一致。预期 validation 收益约 15%--35%。

#### E. 缓存固定输入并减少同步传输

以下 train/validation 固定数据应在 dataset 构建阶段生成一次：

- fixed-dt offsets、reference states、weights、valid masks；
- aero/frame/control bounds；
- fixed-anchor validation indices；
- normalizer 与机场/航班采样权重。

若 profiling 表明数据时间不可忽略，再引入 pinned host memory、non-blocking transfer 和双缓冲
prefetch。不要在未测得 Host→Device 瓶颈前增加多进程 DataLoader；continuous rollout 很可能仍是
主耗时。

#### F. 保存可独立 replay 的 checkpoint

单航迹和 pooled runner 都应先保存完整 checkpoint，再从 checkpoint 生成诊断、HTML 和派生
metrics。报告 schema、展示或 metric 实现变化时，只重新 replay，不重新训练。checkpoint 必须
绑定完整 config、normalizer、split identities 和 manifest hashes；旧 schema 派生产物要求重建，
不添加兼容 fallback。

### 12.4 第二优先级：改变执行协议，必须成为显式 recipe

> 当前禁用：这些候选会改变 validation 或 SGD 协议，本轮不得实施。

| 候选 | 预期收益 | 会改变什么 | 必须记录 |
|---|---:|---|---|
| common-grid checkpoint metric 每 5 epoch 一次 | 总时间约 10%--25% | checkpoint/LR scheduler 观察频率 | selection interval、patience 单位 |
| 完整 validation 每 2--5 epoch 一次 | 总时间约 15%--35% | early stop 与 scheduler 更新点 | validation interval、实际 validation 次数 |
| train 按时域分桶 | 约 20%--50%，需实测 | batch 组成与 SGD 更新顺序 | bucket 边界、bucket sampler 版本 |
| 训练阶段仅抽样 validation 航迹 | 依抽样率 | checkpoint 目标总体 | 抽样身份、seed；只可用于 screening |

降低 validation 频率时，`patience` 必须按“validation 次数”而非 epoch 解释。例如每 5 epoch 验证
一次、patience=60，等价于最多等待约 300 个训练 epoch。LR scheduler 也只能在获得新 selection
metric 时更新。禁止沿用旧字段含义却静默改变实际等待预算。

若 2+4 配方、backbone、N、LR 和其他超参数已经冻结，正式 baseline 可以跳过新的 CV 搜索，只做
development train/validation。若仍需调参，CV 是独立实验，其 candidate/fold 成本不能混入单次
最终训练 wall time；outer-test 始终不参与筛选。

### 12.5 第三优先级：改变数值或优化问题

> 当前禁用：这些候选可能改变数值结果或优化路径，本轮不得实施；尤其不得修改 RK4 `0.5 s`
> 步长、积分器或 dynamics dtype。

只有完成 12.3 的无损消重并取得端到端 profile 后，才按前文阶段 A--E 测试：

- RK4 step `0.5 -> 1.0 s`；
- compiled rollout；
- train-only 近似积分器、exact validation；
- dynamics float32；
- 更少 control segments 或 horizon curriculum。

这些候选不能称为纯提速：它们会改变积分误差、Jacobian、控制表达分辨率或优化路径。必须使用
独立 artifact identity，并在 RK4 `0.5 s`、fixed-anchor validation replay 上比较质量。

只对 Transformer 使用 AMP、dynamics 保持 float64 的风险较低，但 control 训练的主要成本在
dynamics，预期收益有限。减少 `validation_common_grid_points` 同样不是高优先级：它只减少 rollout
后的插值/metric 计算，不能减少 RK4 步数。

### 12.6 推荐实施顺序与验收

```text
P0  分解 epoch 计时，建立 eager 端到端 baseline
P1  合并 validation batch/model forward
P2  复用最终 fit-evaluation prediction
P3  validation 时域分桶 + 每 bucket batch-size probe
P4  checkpoint-first、报告独立 replay
```

P1--P4 的验收条件：相同输入、相同 model state 下，loss components、checkpoint selection、native
metrics 和 common-grid metrics 在冻结容差内一致；split identities 与 manifest hashes 完全相同；
outer-test 未打开；端到端 median/p90 和 peak VRAM 有完整记录。若加速只改善 microbenchmark、却
没有降低完整 development wall time，则不进入正式训练默认路径。

P0--P4 完成后即停止本轮速度优化。即使速度仍不理想，也不得自动转向 validation 降频、compiled、
AMP、RK4 `1.0 s`、RK2 或其他近似；这些方向需要新的明确授权。

### 12.7 P0--P4 实施与筛选结果

本轮实现保持 RK4 `0.5 s`、float64 dynamics、loss、validation 频率、train batch 顺序和 optimizer
update 序列不变：

- fixed-anchor 数据集缓存固定 target、dynamics 参数和 ragged fixed-dt supervision；组 batch 时只做
  padding/stack，不再重复插值和 reference gather；缓存与原 builder 已做逐位一致测试；
- validation 每个 batch 只做一次 backbone/control-head forward，同一 `ControlPrediction` 分别送入
  observed-supervision-clock loss rollout 和 deployable predicted-clock selection rollout；两个时钟的
  states、durations 和 timestamps 仍由独立 rollout 生成，不混配；
- final fit evaluation 每个 split 只生成一次 deployable prediction replay，native/common-grid
  metrics 共享该不可变结果；
- checkpoint 在派生 fit report 之前原子写入；报告失败后可用 checkpoint 独立 replay，无需重训；
- `EpochResult` 新增 train data/forward/rollout-loss/backward-step、validation objective/selection、
  epoch total、peak VRAM、updates/s 和逐机场 validation 航迹/query/wall-time 记录。

在 RTX 4060 上，以当前 2+4、N=64、batch=512、五机场完整 development validation（2167 条）
执行 3 次 warm-up 和 10 次正式交替顺序计时：

| backbone | 路径 | median | p90 | speedup | 240 个标量最大差 | peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| iTransformer | 旧双 forward | 3.888 s | 3.950 s | 1.000x | -- | 346.5 MiB |
| iTransformer | 共享 forward（同一缓存） | 3.820 s | 3.832 s | **1.018x** | **0** | 346.5 MiB |
| PatchTST | 旧双 forward | 4.049 s | 4.118 s | 1.000x | -- | 346.5 MiB |
| PatchTST | 共享 forward（同一缓存） | 3.891 s | 3.948 s | **1.040x** | **0** | 346.5 MiB |

收益低于原 5%--15% 预估，说明当前 validation 的绝对瓶颈是两次不可省略的 float64 RK4 rollout，
不是 Transformer forward。尽管收益有限，两个 backbone 的 median/p90 都稳定下降且指标逐位一致，
因此默认采用不分桶的缓存 + shared-forward 路径。上表让新旧 validation 都使用相同固定输入缓存，
只隔离测量 shared-forward；缓存本身由 epoch 分段计时记录，不把它的收益混入该 speedup。

时域分桶在完整 validation screening 中失败：shared-forward 分桶路径 median 6.468 s，而同轮旧路径
为 3.840 s，即 `0.594x`；虽然 peak VRAM 从 346.5 MiB 降到 122.0 MiB，且最大标量差仅
`8.55e-9`，但更多小 batch/动态 shape 的调度开销压倒了 rollout padding 收益。因此 P3 不进入
默认训练；实现只作为 benchmark 的显式 `--duration-bucketed` 候选保留，不能自动启用。

完整 development 的一 epoch GPU smoke（iTransformer、2+4、无 curriculum，因而第一 epoch 即
执行完整 checkpoint selection）记录到：train forward 0.39 s、train rollout/loss 11.80 s、
backward/optimizer 44.77 s、validation objective 1.89 s、validation selection 2.00 s，epoch total
61.06 s。该单次结果只用于定位，不作为稳定性能结论；它说明约 92.7% 的 epoch 时间位于保持不变
的精确 RK4 train rollout/backward 中，输入准备仅 0.17 s。P0--P4 已穷尽当前获准的明显执行重复，
不能为了继续提速而绕过本节硬约束。

完整结果：

- `4dTrajectory/outputs/POOLED/experiments/openap_direct_20260801_arc24_execution_optimization/validation_unbucketed_3warmup_10runs.json`
- `4dTrajectory/outputs/POOLED/experiments/openap_direct_20260801_arc24_execution_optimization/validation_patchtst_unbucketed_3warmup_10runs.json`
- `4dTrajectory/outputs/POOLED/experiments/openap_direct_20260801_arc24_execution_optimization/validation_bucketed_screening.json`
- `4dTrajectory/outputs/POOLED/experiments/openap_direct_20260801_arc24_execution_optimization/smoke_itransformer_epoch1/history.json`
