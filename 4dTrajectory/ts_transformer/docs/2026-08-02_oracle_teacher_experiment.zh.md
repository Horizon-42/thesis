# Oracle-teacher 初始化实验（2026-08-02）

## 1. 结论

Oracle-teacher 初始化通过正式 train/validation 对照，当前应作为 control 模型的新训练基线。

在完全相同的正式配方下，仅增加 32 条 outer-train 航迹产生的 teacher 初始化，KSJC validation 指标变化为：

| 指标 | 原基线 | teacher-only | 相对变化 |
|---|---:|---:|---:|
| selection，越低越好 | 31.413 | 27.628 | -12.0% |
| ADE | 1130.7 m | 771.0 m | -31.8% |
| FDE | 1202.9 m | 1040.4 m | -13.5% |
| cross-track p95 | 3313.2 m | 2039.6 m | -38.4% |
| altitude p95 | 692.6 m | 364.7 m | -47.3% |
| final-time MAE | 20.84 s | 14.19 s | -31.9% |

这说明当前模型的主要困难不只是 backbone 容量不足，还包括从随机参数出发寻找可飞控制序列的优化难度。Teacher 没有改变推理接口；正式训练结束后，模型仍只根据历史输入预测 control 和总时长。

## 2. 数据隔离

- 所有 teacher 构造、直接优化、模仿训练和 rollout gate 只打开 outer-train 航迹值。
- Teacher cohort 固定为 32 条航迹。
- 审计结果：32/32 teacher ID 属于 outer-train，和 validation 的交集为 0。
- 正式训练仍使用完整的 2207 条 train 航迹，validation 固定为 505 条；teacher 只负责初始化，不替代正式训练数据。
- 本组实验没有运行或查看 outer-test 预测及指标。
- Teacher schedule SHA-256：`a67ea14e0599dd5d2ecae55b854c3adf573043091e08cd4ce661bbf32cec4938`。

## 3. 方法链

### 3.1 Inverse-dynamics 初值

从 train-only 参考位置差分并平滑得到一致速度，再由 transport-chart-velocity dynamics 反推控制初值。拟合尾部的零权重速度占位值不会作为实测速度使用。

与中性控制相比，32 条航迹的固定 2 s rollout 中：

| 指标中位数 | 中性控制 | inverse-dynamics teacher |
|---|---:|---:|
| ADE | 3983.2 m | 2310.3 m |
| terminal distance | 10912.8 m | 6991.9 m |

32/32 航迹的 teacher ADE 均优于中性控制。

### 3.2 单航迹直接优化 teacher

每条 train-only 航迹的 64 段 control logits 独立优化，duration 固定为真实总时长均分，不训练 duration。目标函数使用生产训练相同的 arc-length geometry 2+4，阶段为 60 s、120 s、240 s、full，分别 30、30、30、150 步；学习率 `1e-4`，梯度裁剪 20。

| 指标中位数 | 优化前 | 优化后 |
|---|---:|---:|
| ADE | 2310.3 m | 1568.9 m |
| FDE at last complete 2 s | 6937.1 m | 3454.5 m |
| terminal distance | 6991.9 m | 3476.7 m |

### 3.3 Teacher imitation gate

在 32 条 teacher schedule 上训练同一 iTransformer 输出头 1000 步。Loss 为 unit-box control MSE、按 N 缩放的 duration-fraction MSE、以及 `time/600` MSE。由于此 teacher 使用均匀 duration，duration-fraction 项为 0，模型仍学习总时长。

Observed-clock 诊断中位数：

| 模型 | ADE | terminal distance | time error |
|---|---:|---:|---:|
| 未训练 | 3983.2 m | 10912.8 m | 248.89 s |
| imitation | 1909.2 m | 5374.9 m | 3.62 s |
| teacher target | 1568.9 m | 3476.7 m | 0 s |

模仿网络能吸收 teacher 信息，但纯参数模仿尚不能完整再现 rollout 质量。

### 3.4 Rollout gate

从 imitation 权重继续用生产 arc-length geometry 2+4 做小 cohort rollout fine-tuning。阶段为 60 s、120 s、240 s、full，分别 10、10、10、170 步；学习率 `3e-5`，梯度裁剪 20。

| 指标中位数 | rollout fine-tune 前 | 后 |
|---|---:|---:|
| ADE | 1909.2 m | 383.4 m |
| terminal distance | 5374.9 m | 558.1 m |
| time error | 3.62 s | 2.91 s |

该 gate 证明 teacher 初始化不是只能拟合参数标签；进入真实 rollout 目标后仍有很大的可优化空间。

## 4. 正式对照及一次混杂实验

正式配方为 iTransformer、64 control segments、transport-chart-velocity、RK4 0.5 s、固定 2 s 监督、arc-length geometry 2+4、horizon curriculum 60/120/240/full、learning rate `3e-5`、clip 20、180 epochs。

第一次正式 teacher run 意外保留了 `control_effort=0.001` 和 `control_smoothness=0.01`，而原基线两项均为 0。它只能记为“teacher + regularization”组合实验，不能用于 teacher 的单因素归因：

| 实验 | selection | ADE | FDE | cross p95 | altitude p95 | time MAE |
|---|---:|---:|---:|---:|---:|---:|
| teacher + regularization | 29.696 | 785.8 m | 1093.7 m | 2074.3 m | 433.2 m | 14.77 s |
| teacher-only，正则为 0 | 27.628 | 771.0 m | 1040.4 m | 2039.6 m | 364.7 m | 14.19 s |

修正后的 teacher-only 更好，因此当前基线明确采用 `control_effort=0`、`control_smoothness=0`。

## 5. 产物

- 原基线：`outputs/KSJC/experiments/reference_velocity_20260802/a_track_fit`
- Inverse-dynamics audit：`outputs/KSJC/experiments/oracle_teacher_20260802/inverse_dynamics_quality_32`
- Optimized schedules：`outputs/KSJC/experiments/oracle_teacher_20260802/optimized_arc24_32`
- Imitation gate：`outputs/KSJC/experiments/oracle_teacher_20260802/imitation_gate_32`
- Rollout gate：`outputs/KSJC/experiments/oracle_teacher_20260802/rollout_gate_32`
- 混杂的 teacher + regularization 正式实验：`outputs/KSJC/experiments/oracle_teacher_20260802/formal_teacher_pretrained`
- 正确的 teacher-only 正式实验：`outputs/KSJC/experiments/oracle_teacher_20260802/formal_teacher_pretrained_no_regularization`

## 6. 当前判断

Teacher 初始化是目前这组实验里最强且证据最完整的提升。它同时改善几何、终端和时间指标，没有以明显牺牲 FDE 换取 ADE。后续消融都应以 teacher-only 配方为比较基准，并继续保持固定 RK4 0.5 s、固定 2 s supervision 和同一 train/validation split。
