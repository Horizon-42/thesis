# Control 参数预测模式实验指导

日期：2026-07-28  
状态：开发集实验设计；control 模式尚无正式准确率结果；outer-test 保持封存

## 1. 目标与当前结论

新模式通过 `prediction_output=control` 预测分段控制量和每段持续时间，再经可微飞机
动力学 rollout 得到完整状态轨迹。第一轮实验要回答的不是“control 一定优于 state”，而是：

> 在相同航迹、相同 train/validation 身份和相同观测历史下，结构化 control rollout
> 能否在不牺牲 ADE/FDE 的前提下，提高状态一致性和可飞性？

目前只能确认实现闭环已经覆盖 control head、逐航班动力学条件、Torch rollout、训练 loss、
checkpoint、预测和优化器形状的导出；还没有真实 control 训练结果，因此不能把现有 state
指标写成 control 模式的预期结果。

现有 state 模式的五机场 pooled 参考如下：

| 项目 | 当前冻结值/结果 |
| --- | --- |
| 模型 | iTransformer / state / normalized / ENU / fixed `L-1` |
| 历史与输出 | `L=60`、`dt=2 s`、`N=64` |
| 结构 | `d_model=256`、`d_ff=512`、8 heads、3 layers、dropout 0.1 |
| 优化 | batch 512、LR `5e-4`、plateau factor/patience `0.5/8`、180 epochs、early stop 20 |
| 三种子 Validation ADE/FDE | `1523.0 ± 86.8 m / 731.3 ± 110.5 m` |
| 三种子 Validation time MAE | `69.57 ± 1.05 s` |
| seed 1337 common-grid ADE/FDE | `1585.3 m / 1526.4 m` |

完整依据和旧实验索引见[优化实验总结](2026-07-28_optimization_experiment_summary.zh.md)。

## 2. 正式实验前必须通过的三个阶段门

当前代码可以做合成测试和初步 smoke run，但在以下问题明确以前，不应发布 control 与
state 的正式准确率优劣结论。

### 2.1 非均匀时间段与监督 target 的时间语义尚未对齐

control head 预测：

```text
Delta_t_i = softmax(duration_logits)_i * predicted_final_time_s
```

rollout 第 `i` 个状态因此位于预测累计时间 `sum(Delta_t_1 ... Delta_t_i)`。但是当前
dataset 的 `target_states[i]` 仍位于真实剩余时间的均匀位置
`(i + 1) / N * true_final_time_s`，control loss 和 native ADE/FDE 直接按数组索引比较两者。

如果设计目标确实是“非均匀物理时间控制段”，正式实验前必须统一这两个时钟，例如把
rollout 重采样到固定监督时钟，或按预测累计时间取得对应 target；具体实现方案应单独确认，
本指南不替项目作这个设计决定。否则 native ADE/FDE 同时混入轨迹误差和时间索引错位，
而 duration head 也很难真正把更多控制段分配给转弯等局部阶段。

### 2.2 现有统一比较报告尚不支持 control 模型

[`run_ts_predictability_report.py`](../../../run_ts_predictability_report.py) 当前直接调用
`model(history)` 并读取 `output.states`，没有传入 control 模型所需的 per-flight
`dynamics`，也没有读取非均匀 `segment_durations`。因此现在不能把 control checkpoint
交给该脚本生成 common-grid 报告。

正式比较前，报告路径至少要能够：

1. 为 control 模型构建与训练/预测相同的 dynamics tensors；
2. 执行 control rollout；
3. 按累计 `segment_durations` 把 control 和 state 输出都重采样到同一个物理时间网格；
4. 保持 `evaluated_split=val`、`retrieval_reference_split=train`、
   `outer_test_loaded=false`；
5. 在报告中记录 `prediction_output`，避免把两种输出契约误标成同一种 state forecaster。

在这项支持完成前，`fit_evaluation.json` 可用于诊断训练是否发散，但不能作为 control 与
state 的最终公平比较依据。

### 2.3 顶层 development pipeline 尚未透传 control 配置

[`run_ts_pipeline.py`](../../../run_ts_pipeline.py) 当前没有暴露或传递：

- `--prediction-output control`；
- control effort/smoothness 权重；
- control rollout integration step；
- control 专用输出目录身份。

直接运行它会继续构造默认 state 模型，并可能落入既有 state 输出命名。正式实验前应先让
pipeline 显式携带 control 身份，并确认 checkpoint reuse 校验包含 output strategy 和
control recipe。之后开发运行统一使用 `--split development`，只发布 train 和 validation。

## 3. 新模式的实际训练契约

### 3.1 输出与动力学

control 模型输出：

```text
controls[..., 0] = thrust_N       in [0, aircraft max thrust]
controls[..., 1] = bank_rad       in [-pi/4, pi/4]
controls[..., 2] = load_factor    in [0.5, 2.0]
segment_durations > 0
sum(segment_durations) = final_time_s
```

每个控制段经 float64 Torch RK4 rollout，默认最大积分步长为 `0.5 s`。当前位置和速度都
来自同一次动力学 rollout，因此 state 模式使用的 kinematic consistency loss 在 control
模式中恒为 0；`--kinematic-consistency-weight` 对 control 结果不起选择作用，不应继续对它
做消融或把日志中的 0 误判为 loss 缺失。

control loss 为：

```text
L = L_rollout_state
  + lambda_time * L_final_time
  + lambda_terminal * L_terminal
  + lambda_effort * L_control_effort
  + lambda_smooth * L_control_smoothness
```

当前默认 `lambda_effort=1e-3`、`lambda_smooth=1e-2`。effort 是相对于零控制的绝对平方，
并不是相对于配平控制；它会同时偏好较低 thrust、较小 bank 和较低 load factor。因此
effort 权重必须单独消融，不能假设默认值一定有利于进近轨迹。

### 3.2 飞机类型假设

当前 harvested arrivals 的飞机类型通常不可解析，CLI 会使用 fallback aircraft。若本批数据
仍全部为 `UNK`，即使 condition 向量按航班传入，所有样本实际仍采用同一架 fallback
飞机的质量、推力和气动参数。

第一轮应显式固定 `--aircraft-type A320`，并把实验结论表述为“A320 fallback 假设下的
control prediction”，不能写成“混合机型物理条件已经验证”。训练开头的 build report 必须
归档；后续只有在来源中存在可信机型字段时，才单独开展 mixed-aircraft 实验。

## 4. 数据隔离与航迹安全

### 4.1 固定 development 身份

沿用现有五机场 pooled 数据与 `split_seed=1337`：

| Split | 航班数 | 身份 SHA-256 |
| --- | ---: | --- |
| Train | 13,807 | `32dd550dca616a1f85e8cde4012b3b262170c9df1f427f24ead5de8e687d2ea0` |
| Validation | 2,951 | `4df9d067f78e576c604cf08656e654fdfebc85d65b99a3ff736beea3fbdfb4ce` |

受保护输入：

| Airport | Manifest | SHA-256 |
| --- | --- | --- |
| KMSY | [manifest](../../../trajectory_data_process/outputs/harvest/KMSY/arrivals/manifest.json) | `831276322d9ab2306fa777e78398b12f1640c0d6f4479e4effbb65776b6bebde` |
| KRDU | [manifest](../../../trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json) | `80bf1a236eb755b88e35ef039494e50083a3d2ae3680ff53996233aa5a83b2ce` |
| KSJC | [manifest](../../../trajectory_data_process/outputs/harvest/KSJC/arrivals/manifest.json) | `687f5c6c1d94bb540d36c74ba443b57c7eddae5109441e2bef008b2498376d49` |
| KSMF | [manifest](../../../trajectory_data_process/outputs/harvest/KSMF/arrivals/manifest.json) | `90c9ecedbd63aa6ea6ef281d97ca4493f0a19d20bd323f252d127f191c3621a5` |
| KSTL | [manifest](../../../trajectory_data_process/outputs/harvest/KSTL/arrivals/manifest.json) | `ae90194bd4994710bf3e53384f51fd945df5af9bc4e2e3538275d8f93fe87f83` |

每轮实验前后用 `sha256sum` 复核这五个 manifest。实验只能读取它们及其引用的航迹；不要
运行 harvest、`--evaluate-only`、清理、覆盖、重命名或重新生成命令，也不要对
`trajectory_data_process/outputs/harvest` 做任何写操作。

### 4.2 test 禁令

整个本指南只允许：

- train split 用于梯度、normalizer 和诊断；
- validation split 用于 early stopping、候选比较和报告；
- outer-train 内部 CV（如果后期确有必要）。

禁止运行 `freeze-test`、`--split test`、`--test-release` 或 pipeline 的
`--release-test`。禁止读取 test prediction、test metric 或把 test 身份用于检索。只有所有
架构、loss、N、batch、学习率和 seed 决策冻结后，用户另行明确要求，才能执行一次性 final
test release。

### 4.3 输出目录隔离

所有新产物放到：

```text
4dTrajectory/outputs/POOLED/experiments/control_output/
├── capacity_probe/
├── baseline/
├── batch_size/
├── learning_rate/
├── effort_weight/
├── smoothness_weight/
├── n_segments/
├── dropout/
└── comparisons/
```

每个 candidate/seed 使用全新目录，并在目录名中写明 `control`、N、batch、关键权重和 seed。
绝不能把 control run 指向既有 state checkpoint 或 prediction 目录。`predict` 会清理它自己
输出目录中的旧 record，所以 prediction 目录也必须一候选一目录。

## 5. 哪些旧参数可以继承

| 项目 | 第一轮处理 | 原因 |
| --- | --- | --- |
| 五机场 manifests、split seed | 冻结 | 保持 development 身份完全一致 |
| iTransformer、ENU、fixed `L-1` | 冻结 | 当前 state 实验中最可靠的主配置 |
| `L=60`、`dt=2 s` | 冻结 | 避免同时改变输入信息量 |
| 256/512、8 heads、3 layers | 冻结 | 先只验证新输出结构 |
| 180 epochs、early stop 20 | 作为正式 baseline 起点 | 旧实验表明继续单纯增加 epoch 无效 |
| LR `5e-4`、plateau `0.5/8` | 只作为起点 | control rollout 的梯度尺度与 state head 不同 |
| `N=64` | 只作为起点 | state 中 N 是状态节点；control 中 N 是控制段，语义已改变 |
| batch 512 | 不继承 | float64 完整 rollout 反向图改变了显存、吞吐和更新噪声 |
| kinematic weight 3 | 不适用 | control 中该项由构造置 0 |
| terminal 0.02、time 1 | baseline 暂时冻结 | 先隔离 control 输出带来的影响 |

## 6. 阶段化实验流程

### 6.1 阶段 0：代码身份、契约测试和容量 smoke

正式 run 前先提交并记录包含 control 实现的 commit；不能用未提交工作树产生论文结果。
先运行：

```bash
conda run -n aeroviz python -m pytest \
  aerodynamic_model/tests/test_torch_dynamics.py \
  4dTrajectory/ts_transformer/tests/test_ts_transformer.py -q
```

必须通过的最小契约包括：

- Torch 单步、失速分支和非均匀 rollout 与 CasADi 等价；
- controls 与 durations 梯度 finite 且非零；
- controls 始终位于逐航班 bounds 内；
- durations 全正且总和等于 `final_time_s`；
- control checkpoint round trip 后仍是 control；
- control 预测能导出对齐的 states 与 controls。

pipeline 和 common-grid report 的阶段门处理完成后，可先做一个 `epochs=1`、独立目录的
capacity smoke。`--batch-size auto` 会用真实 control loss 和完整 rollout 做 CUDA probe；
它的结果只用于确定可运行上限 `B_max`，不能作为准确率实验。

### 6.2 阶段 A：建立 control baseline 与 batch

先在 seed 1337 比较：

```text
B in {B_max, B_max / 2}
若 B_max >= 64，再加入 B_max / 4
```

候选取不小于 8 的 2 的幂，其他配置完全相同。旧 state 结果显示更小 batch 可通过增加
optimizer updates 改善拟合，但 control 模式必须重新验证。记录每 epoch 秒数、peak VRAM、
updates/epoch 和 total updates。

基线训练参数模板如下；`<B>` 和输出目录必须替换为当前候选：

```bash
conda run -n aeroviz python \
  4dTrajectory/ts_transformer/__main__.py train \
  --data trajectory_data_process/outputs/harvest/KMSY/arrivals/manifest.json \
  --data trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
  --data trajectory_data_process/outputs/harvest/KSJC/arrivals/manifest.json \
  --data trajectory_data_process/outputs/harvest/KSMF/arrivals/manifest.json \
  --data trajectory_data_process/outputs/harvest/KSTL/arrivals/manifest.json \
  --output-dir 4dTrajectory/outputs/POOLED/experiments/control_output/batch_size/itr_control_n64_b<B>_seed1337 \
  --model itransformer \
  --prediction-output control \
  --horizon-mode normalized \
  --coordinate-frame enu \
  --seq-len 60 \
  --n-segments 64 \
  --d-model 256 \
  --n-heads 8 \
  --e-layers 3 \
  --batch-size <B> \
  --epochs 180 \
  --patience 20 \
  --learning-rate 0.0005 \
  --lr-plateau-factor 0.5 \
  --lr-plateau-patience 8 \
  --terminal-loss-weight 0.02 \
  --control-effort-weight 0.001 \
  --control-smoothness-weight 0.01 \
  --control-rollout-dt 0.5 \
  --aircraft-type A320 \
  --seed 1337 \
  --split-seed 1337
```

训练会自动写 best-checkpoint 的固定-anchor train/validation `fit_evaluation.json`。在阶段
2.1 的时间对齐问题处理前，这里的 native ADE/FDE 只用于发现明显发散，不能宣布胜负。

### 6.3 阶段 B：学习率复核

冻结阶段 A 的 batch 后，先筛：

```text
learning_rate in {3e-4, 5e-4}
plateau factor/patience = 0.5/8
```

只有 `5e-4` 出现 loss spike、NaN 或明显控制饱和时才加入 `1e-4`。旧 state 实验已经表明
`1e-4` 在 180 epochs 内欠拟合；新模式需要复核梯度尺度，但不应无证据重复大网格搜索。

### 6.4 阶段 C：control regularizer 单变量消融

先固定 smoothness 默认值，只改 effort：

```text
control_effort_loss_weight in {0, 1e-4, 1e-3}
control_smoothness_loss_weight = 1e-2
```

选定 effort 后，再固定它并改 smoothness：

```text
control_smoothness_loss_weight in {0, 1e-3, 1e-2, 1e-1}
```

不能同时改变两个权重。每个 candidate 必须报告未加权的控制分布，而不只报告加权后的
loss：thrust/bank/load-factor 的 p5、median、p95，贴近上下界的比例，相邻控制变化，以及
duration 的 min/p1/median/p99/max。若权重更大只让 controls 靠近下界而 ADE/FDE 没有改善，
这是 regularizer bias，不是物理性提升。

### 6.5 阶段 D：控制段数量 N

在 batch、LR 和 control weights 冻结后比较：

```text
n_segments in {16, 32, 64}
```

第一轮不做 128：完整 RK4 总步数主要由真实时长和 `0.5 s` 上限决定，但更多 segment 会
增加 head 参数、Python 段循环和 duration 自由度。需要同时检查：

- common-grid ADE/FDE 和 final-time MAE；
- duration 是否退化为大量接近 0 的段；
- 有效时间分配是否与转弯、下降阶段对应；
- controls 的饱和率与相邻跳变；
- wall time 与显存。

如果 N 增加只改善 native index-wise ADE、却不改善 common-grid ADE，应判定为时间网格或
采样密度效应，而不是模型能力提升。

### 6.6 阶段 E：dropout 与泛化

只有 control 的 train ADE 已明显优于 state，而 validation 没有同步改善时，才做：

```text
dropout in {0, 0.05, 0.1}
```

当前 CLI 没有单独的 `--dropout`，必须用记录在 run 目录中的 `config-overrides.json`，并
确认 checkpoint metadata 中只有 dropout 与预期字段发生变化。不要同时扩大 `d_model`；
旧实验已表明现阶段单纯加宽模型不是首要方向。

### 6.7 阶段 F：三种子确认

screening 固定 seed 1337。只有相对 control baseline 达到预注册改进门槛的前一至两个候选，
再用：

```text
training seeds = {1337, 2027, 4242}
split_seed = 1337
```

报告 mean ± sample SD。三种子中 train/validation 身份 SHA 必须完全相同；训练 seed 不能
改变 split。

`control_rollout_integrator_dt_s=0.5` 是与高保真 replay 对齐的数值契约，不是普通调参项。
如需做 `0.25 s` 敏感性检查，只能作为数值审计；不能因为 `1.0 s` 更快就直接选择它。

## 7. 指标与预注册选择规则

### 7.1 必须记录的指标

| 类别 | 指标 |
| --- | --- |
| 准确率 | common-grid Train/Validation ADE、FDE；按 remaining time 的误差 |
| 终点 | lateral、vertical、speed、heading、flight-path angle、final-time MAE/bias |
| 泛化 | val/train ADE 与 FDE ratio；按机场、跑道、轨迹类型分组 |
| 优化 | 各 loss component、best/run epoch、LR、optimizer updates、gradient/NaN 状态 |
| 控制 | 三个 controls 的分位数、上下界饱和率、相邻变化量 |
| 时段 | min/p1/median/p99/max duration、duration entropy、有效 segment 数 |
| 物理 | position/velocity、heading、turn rate、acceleration、jerk；直接 rollout 成功率 |
| 成本 | epoch 秒数、总 wall time、peak VRAM |

control 输出的 bounds 由 sigmoid 保证，但“没有越界”不等于控制合理。必须额外观察长期
贴边、极短 duration 和 controls 高频跳变。当前 `fit_evaluation.json` 尚未持久化所有这些
control 统计；缺失项补齐前，不应只凭一个 ADE 数字定 winner。

### 7.2 候选选择顺序

预注册顺序：

1. 任何 NaN、rollout failure、时间非正、checkpoint 身份不符的候选直接淘汰；
2. 以同一物理时间网格上的 Validation ADE 为 primary metric；
3. Train ADE 必须同方向改善，排除仅靠 validation 随机波动；
4. 单 seed 改善小于 2% 视为噪声，不宣布胜出；
5. 再看 FDE、final-time、机场/跑道 worst group 和 control/duration 诊断；
6. ADE 接近时，选择 FDE 更低、控制不饱和、duration 不退化且计算更简单的候选；
7. 最后用三种子 mean ± SD 决策，绝不查看 test。

### 7.3 state 与 control 的公平比较

两种模式必须使用：

- 同一代码 commit 和 manifest hashes；
- 相同 split 与 training seed；
- 相同 history、anchor、backbone、坐标系和机场集合；
- 相同 validation flight 顺序；
- 相同物理时间 query grid；
- 未经 spline、滤波或 CZML 插值的 rollout/state 原始结果。

不能直接把 control 的非均匀 native endpoint ADE 与 state 的均匀 native endpoint ADE
并列。现有 seed-1337 state checkpoint 可作为只读参考：

[`itr_norm_b512_plateau_p8_seed1337`](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p8_seed1337/fit_evaluation.json)。

state 模式的 flyability 是从预测状态反演“所需控制”，control 模式则直接输出并 rollout
控制；两者含义不同，应分别报告，不能把两个百分比当成完全相同的监督指标。

## 8. 结果诊断与停止规则

| 观察 | 解释 | 下一步 |
| --- | --- | --- |
| Train 和 Validation ADE 都优于 state | 结构约束可能有效 | 做三种子确认，再检查 FDE/control 分布 |
| Train ADE 明显差于 state | 优化或控制表达能力不足 | 先查 batch、LR、N、饱和率和 duration，不先加 dropout |
| Train 好、Validation 差 | 泛化不足 | 做 dropout/weight decay 单变量实验或补 intent 输入 |
| final-time MAE 仍约 70 s | duration head 仍是瓶颈 | 单独研究 time head；不要用 control smoothness 掩盖 |
| duration 大量接近 0 | 非均匀时间退化 | 停止 N/weight 搜索，先明确时间监督和 duration 约束 |
| controls 长期贴上下界 | 动力学假设、aircraft fallback 或 regularizer 有偏 | 先审计条件和 loss 定义，不宣布“物理可飞” |
| common-grid 不改善但 native 改善 | 采样/时间索引效应 | 否决候选 |
| 三种子排序不稳定 | 单 seed 信号不足 | 保留 baseline 或收集更多开发数据，不查看 test |

若 batch、LR、control weights 和 `N={16,32,64}` 后，control 的 Train ADE 仍系统性差于
state，应停止继续微调 scalar loss。下一步应检查 control 表达是否足以重建观测进近、
fallback A320 是否造成模型失配，以及是否需要显式 approach intent；不要通过更多 epoch
或查看 test 寻找偶然优势。

## 9. 每个 run 的数据索引模板

每个候选目录至少保留：

```text
checkpoint.pt
checkpoint_metadata.json
history.json
fit_evaluation.json
config-overrides.json        # 如果使用
val_prediction/summary.json  # common-grid 支持完成后
val_prediction/flyability_report.json
experiment_notes.md
```

`experiment_notes.md` 应在训练前写入：问题、唯一 changed variable、冻结参数、commit、manifest
hash、split hash、selection rule 和输出路径；训练后再追加结果与决策。最终总表至少包含：

| Candidate | Commit | Seed | Split SHA | Checkpoint SHA | History | Fit replay | Val report | Decision |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |

## 10. 代码与设计索引

| 资源 | 用途 |
| --- | --- |
| [normalized/control 输出设计](normalized_time_and_control_output.zh.md) | 模式契约与动力学说明 |
| [config.py](../config.py) | `prediction_output`、control weights、rollout step |
| [prediction_outputs.py](../prediction_outputs.py) | controls、bounds、durations head |
| [dataset.py](../dataset.py) | target grid 与逐航班 dynamics tensors |
| [models.py](../models.py) | encoder feature、condition fusion、control head |
| [train.py](../train.py) | differentiable rollout loss 与 fit replay |
| [forecast.py](../forecast.py) | control inference 与累计时间恢复 |
| [torch_dynamics.py](../../../aerodynamic_model/torch_dynamics.py) | CasADi-equivalent Torch dynamics |
| [动力学契约测试](../../../aerodynamic_model/tests/test_torch_dynamics.py) | 数值等价与梯度检查 |
| [现有实验总结](2026-07-28_optimization_experiment_summary.zh.md) | state baseline、实验结果与数据索引 |

## 11. 推荐的实际执行顺序

```text
确认时间监督语义
  -> 给 common-grid report 和 development pipeline 补齐 control 支持
  -> 提交并记录 control 实现 commit
  -> unit/contract tests
  -> auto batch capacity smoke（不参与选模）
  -> batch screening
  -> LR screening
  -> effort weight
  -> smoothness weight
  -> N segments
  -> 必要时 dropout
  -> seeds 1337/2027/4242 confirmation
  -> 冻结全部决策
  -> 等待用户另行批准一次性 outer-test release
```

这个顺序把最可能导致无效结论的“时钟和评价口径”放在最前面，也保留现有 state 模式及
全部旧实验产物不变。
