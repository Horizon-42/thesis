# 轨迹模型收敛诊断与后续优化实验指南

> 日期：2026-07-27  
> 适用代码：`4dTrajectory/ts_transformer` 当前 pooled、fixed-anchor 训练流程  
> 目标读者：首次接手该实验、不了解此前讨论和历史实验的人  
> 当前结论：模型的数值优化已经基本进入平台期，但训练集物理误差仍然很大；这属于“当前训练配方和输出表示下的欠拟合”，不能简单解释为 epoch 不够。

## 1. 本文解决什么问题

项目使用最近一段 ADS-B 历史航迹预测飞机到跑道入口的完整剩余四维航迹。当前正式模型可以正常训练、保存 checkpoint 并生成预测，但存在以下现象：

- 训练 loss 和 validation loss 后期基本不再下降；
- 最佳 checkpoint 在训练集上的 ADE 仍为千米量级；
- 训练集与 validation 的 ADE/FDE 很接近，而且二者都很差；
- 直接输出的轨迹节点不够平滑，位置通道与速度通道的一致性不足；
- 极少数 `final_time_s` 会接近零，导致运动学指标出现灾难性离群值；
- 继续增加 N、kinematic loss 权重或模型宽度，已经被现有实验否定为通用解决办法。

本文给出一条从低成本诊断到结构性改造的推进路线。重点不是一次性尝试很多参数，而是保证每轮实验能够回答一个明确问题，并且不读取 outer-test。

## 2. 当前任务与数据契约

### 2.1 输入、anchor 和预测目标

当前默认输入为：

```text
dt = 2 s
L = 60 observations
history duration = 120 s
anchor = L - 1
```

固定 anchor 表示每条航迹都从第一个完整 120 s 历史窗口结束处开始预测。每条 outer-train 航迹每个 epoch 使用一次，epoch 之间重新 shuffle；机场宏平均权重使五个机场对 loss 的总贡献相同。

默认 normalized-time state 模型输出：

```text
StatePrediction
├── states        [B, N, 6]
│   └── (e, n, u, edot, ndot, udot)
└── final_time_s  [B]
```

`N` 是从 anchor 到跑道入口的归一化进度分段数，不是预测秒数。当前模型级默认值为：

| 模型 | normalized-time N | 说明 |
|---|---:|---|
| iTransformer | 64 | 当前推荐基线；N=128/256 在现有直接节点输出中增加了高频自由度 |
| PatchTST | 256 | 只是在 PatchTST 候选中 ADE 最低，不代表轨迹物理质量合格 |

`full` 和 `window` 是另外两种保留且解耦的时域模式：

- `full`：一次输出最多 300 个固定 2 s 物理时间节点；
- `window`：每次输出 30 个固定 2 s 节点，推理时递归到完整时域上限；
- `normalized`：一次输出完整剩余路径，并用 `final_time_s` 恢复物理时间。

本文首先优化 iTransformer normalized-time，因为它是当前最稳定、最接近后续控制输出设计的基线。不要在第一阶段同时比较三种时域模式。

### 2.2 数据划分

当前五机场 pooled 数据为：

| Split | 航迹数 | 用途 |
|---|---:|---|
| outer-train | 13,807 | 梯度更新、内部 CV 的训练折 |
| outer-validation | 2,951 | 最终训练 early stopping、开发阶段选模 |
| outer-test | 2,983 | 所有结构和超参数冻结后的单次最终报告 |

机场为 KMSY、KRDU、KSJC、KSMF、KSTL。split 按航班身份划分，不按 window 划分。

后续所有实验必须遵守：

1. batch、学习率、loss、dropout、decoder 和输入特征的选择只能使用 train/validation；
2. 不运行 `--split test`，不读取 test 预测结果来决定参数；
3. 多次看过的历史 test 只能视为开发参考，不能继续作为最终盲测声明；
4. 最终论文结果需要新的、从未用于任何决策的时间外 holdout。

### 2.3 当前正式基线

```text
model = itransformer
horizon_mode = normalized
coordinate_frame = enu
N = 64
d_model = 256
d_ff = 512
heads = 8
layers = 3
dropout = 0.1
learning_rate = 5e-4
batch_size = 2048
epochs = 180
early_stopping_patience = 20 epochs
kinematic_consistency_loss_weight = 3.0
terminal_loss_weight = 0.02
final_time_loss_weight = 1.0
anchor = fixed L-1
seed = 1337
```

需要特别区分两个 patience：

- `TSConfig.patience=20` 是 early stopping patience；
- 当前 `ReduceLROnPlateau(..., patience=3)` 在 `train.py` 内单独定义，是学习率衰减 patience。

这两个参数不是同一件事。

## 3. 什么叫“收敛”

### 3.1 三个不同的问题

判断模型时必须分别回答：

| 问题 | 观察内容 | 当前判断 |
|---|---|---|
| 优化是否停止进步 | loss 曲线、梯度更新、学习率 | 基本进入平台期 |
| 训练集是否拟合充分 | best checkpoint 的 train ADE/FDE | 没有，仍是千米量级 |
| 是否能够泛化 | train 与 validation 的绝对误差及 gap | gap 很小，但二者都差；属于高偏差而非典型过拟合 |

所以“train/validation 很接近”不能自动解释为模型很好。它也可能表示模型对训练数据都学不下来。

### 3.2 当前 best-checkpoint 回放证据

这些结果来自 best checkpoint 的确定性回放：

```text
model.eval()
dropout disabled
fixed anchor L-1
sequential batches
shuffle disabled
```

| 模型/时域 | Train ADE | Val ADE | Val/Train | Train FDE | Val FDE | Val/Train |
|---|---:|---:|---:|---:|---:|---:|
| iTransformer normalized | 1854.3 m | 1857.4 m | 1.002 | 1318.8 m | 1346.3 m | 1.021 |
| iTransformer full | 2105.3 m | 2138.4 m | 1.016 | 2121.5 m | 2144.7 m | 1.011 |
| iTransformer window | 825.2 m | 847.5 m | 1.027 | 1343.3 m | 1366.6 m | 1.017 |
| PatchTST normalized | 4160.0 m | 4212.8 m | 1.013 | 3210.0 m | 3251.6 m | 1.013 |
| PatchTST full | 4398.3 m | 4398.7 m | 1.000 | 3646.7 m | 3660.5 m | 1.004 |
| PatchTST window | 1709.2 m | 1748.8 m | 1.023 | 2537.0 m | 2577.0 m | 1.016 |

上述是各模式原生 target grid 上的 measured-data 指标。`window` 只评价单次 60 s 原生输出，因此它的 ADE 不能直接与 normalized/full 的完整未来 ADE横向比较。跨时域模式比较应使用统一物理时间网格的 predictability report。

当前训练曲线也已基本变平。例如 iTransformer normalized 在 epoch 161 左右得到最佳 validation loss，此后到 epoch 180 的变化接近零。继续把上限扩展到 240 epoch 时，最佳点仍位于同一区域。因此，当前首要动作不应是继续堆 epoch。

### 3.3 模型并非完全没有容量

小样本记忆实验提供了必要的反证：

- 单航迹、`dropout=0`、`kinematic=0` 时，train ADE 可达到约 0.15 m；
- 单航迹、`dropout=0`、`kinematic=10` 时，train ADE 可达到约 7.3 m；
- 160 条航迹时，同一结构的 train ADE 又上升到数百至上千米。

因此，当前网络和 state head 具备基本函数拟合能力。问题更可能出在多航迹优化、loss 梯度竞争、输入条件不足以及直接多节点输出表示，而不是“Transformer 完全不能做这个任务”。

## 4. 当前根因假设及优先级

### Rank 1：大 batch 导致更新次数过少

当前 train 有约 13,807 条航迹，`batch=2048` 时：

```text
ceil(13807 / 2048) = 7 optimizer updates / epoch
180 epochs ≈ 1260 optimizer updates
```

对比：

| Batch | 约 updates/epoch | 180 epoch 约总 updates |
|---:|---:|---:|
| 2048 | 7 | 1,260 |
| 1024 | 14 | 2,520 |
| 512 | 27 | 4,860 |

2048 是吞吐量选择，不是经过拟合质量验证的选择。学习率调度又按 epoch 而不是 optimizer update 计数，会进一步放大 batch 对优化轨迹的影响。

### Rank 2：学习率过早衰减到近似冻结

当前调度器为：

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=3,
)
```

按照当前 validation loss 序列回放，部分训练最终学习率已进入 `1e-7` 到 `1e-6` 量级。此时即使 epoch 继续增加，参数也几乎不再移动。

当前 `history.json` 没有逐 epoch 保存 learning rate 和累计 optimizer step。这两项应在正式进行 scheduler 消融前补齐，否则只能间接推断。

### Rank 3：state、kinematic、terminal 和 time loss 梯度竞争

当前总损失为：

```text
L = L_state
  + lambda_time * L_time
  + lambda_kin * L_kinematic
  + lambda_terminal * L_terminal
```

已有实验表明：提高 `lambda_kin` 会改善部分 raw kinematic 指标，却显著恶化 ADE；把它降到零会改善小样本 ADE，却不能保证轨迹平滑。这说明当前 kinematic loss 更像在事后协调相互独立的位置/速度节点，而不是从结构上保证运动学一致。

### Rank 4：确定性单路径目标面对多模态未来

相似的 120 s 历史可能对应不同跑道、程序、转弯方向、ATC 向量引导和排序。当前确定性 MSE 模型倾向输出多个可能未来的条件均值。

validation-only 的候选覆盖实验显示：

- 单条确定性 iTransformer normalized ADE 约 1858 m；
- MC-dropout K=20 oracle minADE 约 1400 m；
- 相似历史检索 K=20 oracle minADE 约 453 m。

oracle 不能直接作为在线结果，但它证明数据中存在明显多模态性，也说明单一均值轨迹不是完整答案。

### Rank 5：直接回归独立状态节点不利于物理连续性

当前 decoder 直接预测 `N × 6` 个状态值，再通过 loss 要求位置和速度一致。增加 N 会同时增加高频自由度，并不会自动产生光滑曲线。

长期上更合适的结构是：预测控制量或低维曲线参数，再通过可微动力学/积分器生成状态轨迹。项目已经有 `ControlPrediction`、`ControlOutputHead` 和非均匀 `segment_durations` 契约，但还没有接入控制监督和 rollout loss。

## 5. 实验总原则

### 5.1 每轮只改变一个因素

推荐顺序：

```text
Batch size
  -> LR scheduler
  -> learning rate
  -> kinematic weight
  -> dropout
  -> duration head robustness
  -> structured control/state rollout
  -> intent/context and multimodal output
```

例如测试 batch 时，不要同时改变学习率、dropout 和 kinematic weight。否则即使结果改善，也无法知道原因。

### 5.2 所有候选必须使用相同数据身份

每次实验至少记录：

- Git commit 和 worktree 是否 dirty；
- 五个 arrivals manifest 的 SHA-256；
- outer train/validation split SHA-256；
- seed；
- 完整 `TSConfig`；
- checkpoint SHA-256；
- 最佳 epoch 和实际运行 epoch；
- 总 optimizer updates；
- wall-clock time。

如果数据重建后 manifest digest 发生变化，旧 checkpoint 不应与新候选直接比较，应在同一数据版本下重跑基线。

### 5.3 每个候选使用独立输出目录

不要让实验覆盖正式基线：

```text
4dTrajectory/outputs/POOLED/experiments/
  batch_size/
    itr_norm_b2048_seed1337/
    itr_norm_b1024_seed1337/
    itr_norm_b512_seed1337/
  lr_schedule/
  loss_weights/
  duration_head/
  structured_output/
```

候选目录至少应包含：

```text
checkpoint.pt
checkpoint_metadata.json
history.json
fit_evaluation.json
```

`fit_evaluation.json` 必须与 checkpoint SHA 绑定；不能复制其他候选的回放结果。

### 5.4 先单 seed screening，再做重复实验

第一轮用固定 `seed=1337` 快速筛掉明显差的候选。确定前两名后，再至少使用三个 seed 复验，报告均值和标准差。不能因为单次 seed 的小幅优势就宣布参数更好。

### 5.5 不用 test 选模型

开发阶段只生成 train 和 validation 结果。候选优劣由以下顺序决定：

1. train replay 判断是否真正提高了拟合能力；
2. validation 判断改进能否泛化；
3. raw kinematic 和 terminal 指标排除“更准但不可飞”的候选；
4. test 只在所有决策冻结后使用一次。

## 6. 阶段 A：Batch size 消融

### 6.1 要回答的问题

> `batch=2048` 是否因为 optimizer updates 太少而限制了多航迹拟合？

### 6.2 候选

```text
batch_size in {2048, 1024, 512}
```

其余配置严格保持第 2.3 节基线不变。即使已经有一个历史 `batch=2048` checkpoint，也建议在同一代码 commit 下重跑 2048；否则代码、数据或 metric contract 的变化会混入比较。

### 6.3 单个候选训练命令模板

下面以 `batch=512` 为例。1024 和 2048 只替换 `--batch-size` 与输出目录：

```bash
conda run -n aeroviz python \
  4dTrajectory/ts_transformer/__main__.py train \
  --data trajectory_data_process/outputs/harvest/KMSY/arrivals/manifest.json \
  --data trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
  --data trajectory_data_process/outputs/harvest/KSJC/arrivals/manifest.json \
  --data trajectory_data_process/outputs/harvest/KSMF/arrivals/manifest.json \
  --data trajectory_data_process/outputs/harvest/KSTL/arrivals/manifest.json \
  --output-dir 4dTrajectory/outputs/POOLED/experiments/batch_size/itr_norm_b512_seed1337 \
  --model itransformer \
  --horizon-mode normalized \
  --coordinate-frame enu \
  --n-segments 64 \
  --batch-size 512 \
  --epochs 180 \
  --patience 20 \
  --learning-rate 0.0005 \
  --kinematic-consistency-weight 3 \
  --terminal-loss-weight 0.02 \
  --seed 1337
```

`train` 完成后会自动对 retained best checkpoint 进行固定-anchor train/validation 回放，并生成 `fit_evaluation.json`。不需要用训练过程中的随机 shuffle loss 代替正式 train ADE/FDE。

### 6.4 第一轮观察指标

| 类别 | 指标 | 用途 |
|---|---|---|
| 优化 | best epoch、epochs run、train/val loss 曲线 | 判断是否更快进入较优区域 |
| 更新预算 | updates/epoch、total optimizer updates | 解释 batch 差异 |
| 训练拟合 | train ADE/FDE、train time MAE | 直接回答是否改善欠拟合 |
| 泛化 | validation ADE/FDE、time MAE | 排除只记忆训练集 |
| gap | val/train ADE、FDE ratio | 判断过拟合趋势 |
| 尾端 | terminal lateral/vertical error | 防止整段更准但终点更差 |
| 物理性 | position/velocity RMSE、turn rate、acceleration、jerk p95 | 防止通过抖动降低 ADE |
| 效率 | epoch seconds、total wall time、peak VRAM | 评估训练成本 |

### 6.5 生成统一 validation-only HTML 报告

三个候选全部训练完成后，可以把 checkpoint 一次性交给现有 predictability report。该脚本只使用 outer-validation 做模型比较，并使用 outer-train 作为检索参考，不读取 outer-test：

```bash
conda run -n aeroviz python run_ts_predictability_report.py \
  --checkpoint b2048=4dTrajectory/outputs/POOLED/experiments/batch_size/itr_norm_b2048_seed1337/checkpoint.pt \
  --checkpoint b1024=4dTrajectory/outputs/POOLED/experiments/batch_size/itr_norm_b1024_seed1337/checkpoint.pt \
  --checkpoint b512=4dTrajectory/outputs/POOLED/experiments/batch_size/itr_norm_b512_seed1337/checkpoint.pt \
  --multi-candidate-checkpoint b512 \
  --output-dir 4dTrajectory/outputs/POOLED/experiments/batch_size/comparison_seed1337
```

输出入口为：

```text
4dTrajectory/outputs/POOLED/experiments/batch_size/comparison_seed1337/report.html
```

这里的 `--multi-candidate-checkpoint` 只指定报告中额外 MC-dropout 覆盖实验使用哪个候选，不参与 batch 胜负判定。如果尚未选出候选，可以暂时指定任意一个，或者先只读取各目录的 `fit_evaluation.json`。报告脚本的 evaluation batch 与被比较模型的训练 batch 是不同概念，不能据此覆盖 checkpoint 中的训练配置。

### 6.6 决策规则

先看 train ADE，再看 validation：

- 若 512/1024 的 train 和 validation ADE 都明显下降，说明大 batch 确实限制了优化；
- 若 train ADE 下降而 validation 不降，开始出现过拟合，应再评估 dropout/weight decay，而不是继续减 batch；
- 若 train ADE 和 validation 都几乎不变，batch 不是主要瓶颈，进入 scheduler/loss 实验；
- 若 ADE 改善但 raw kinematic p95 大幅恶化，不能直接选中，应进入 Pareto 权衡；
- 若不同候选差异小于约 2%，先视为单 seed 噪声，对前两名做多 seed 复验。

本阶段不要把“每秒 samples 更多”当作胜出条件。吞吐量只是成本指标，train/validation 拟合质量才是实验目标。

## 7. 阶段 B：学习率调度实验

### 7.1 前置条件

先完成阶段 A，固定一个 batch。不要同时搜索 batch 和 scheduler。

正式运行前应先让训练产物记录：

```text
learning_rate per epoch
cumulative optimizer updates
```

并将 scheduler 参数纳入 `TSConfig`、checkpoint metadata 和复用身份。不要通过只改 `train.py` 中的硬编码常数产生无法追溯的 checkpoint。

### 7.2 最小实验

第一轮只比较 plateau patience：

```text
lr_plateau_patience in {3, 8, 12}
factor = 0.5
initial lr = 5e-4
```

这是比立刻引入复杂 scheduler 更小的改动。要回答的问题是：当前学习率是否被过早降到近似冻结。

如果更长 patience 明显改善 train/validation ADE，再考虑采用基于 optimizer update 的调度，使行为不依赖 batch，例如：

```text
warmup_updates = 100--200
total_updates = 3000--5000
cosine decay
```

### 7.3 第二轮学习率候选

固定 scheduler 后再比较：

```text
learning_rate in {1e-4, 3e-4, 5e-4}
```

这与当前 CV 网格一致。不要在 batch 变化时机械使用线性 LR scaling；本项目的 loss 由多项不同尺度目标组成，线性规则必须通过实验证明。

### 7.4 失败判据

以下情况说明继续调 scheduler 的收益有限：

- 学习率保持有效，但 train ADE 仍停在相同平台；
- 增加 optimizer updates 只降低 normalized loss，却不降低物理单位 ADE/FDE；
- 改进完全来自 terminal loss，而中段误差不降；
- raw kinematic 指标随训练显著恶化。

此时应进入 loss 或输出结构实验。

## 8. 阶段 C：Loss 与 dropout 消融

### 8.1 Kinematic weight

固定阶段 A/B 选出的 batch、scheduler 和 learning rate，比较：

```text
kinematic_consistency_loss_weight in {0.3, 1.0, 3.0}
terminal_loss_weight = 0.02
dropout = 0.1
```

`k=0` 可作为诊断对照，回答 state loss 单独能够拟合到什么程度，但不应在没有 raw kinematic 检查的情况下直接作为生产配置。

候选选择沿用已定义的 accuracy-first knee rule：

1. 找到 validation ADE 最低值；
2. 只保留 ADE 不超过最低值 10% 的候选；
3. 在保留候选中最小化五项 prediction/observed fleet-p95 比值的几何平均；
4. raw score 在 2% 内视为实际等效，再用较低 ADE 和更简单配置决胜。

五项 raw 指标为：

```text
position_velocity_rmse_mps
heading_consistency_p95_deg
turn_rate_p95_deg_s
acceleration_p95_mps2
jerk_p95_mps3
```

### 8.2 Dropout

固定 kinematic weight 后再比较：

```text
dropout in {0.0, 0.05, 0.1}
```

解释方式：

- train ADE 大幅改善、validation 恶化：正则化确实有用，不能简单关闭 dropout；
- train/validation 同时改善：原来的 dropout 对当前容量和更新预算过强；
- 二者都不变：dropout 不是主要瓶颈；
- 只有 loss 改善而物理 ADE 不变：loss 标度或目标间权衡需要重新检查。

当前通用 `train` CLI 尚未显式暴露 `--dropout`。继续实验时应正式增加该参数，或者使用受版本控制的 `--config-overrides` JSON；不要靠临时修改 `config.py` 默认值运行不同候选。

### 8.3 不优先做的容量实验

已有 `d_model=512` screening 没有优于 256，因此不要在 loss 和优化问题解决前继续增大 backbone。更大网络会增加显存和实验成本，但不解决独立节点输出的结构矛盾。

## 9. 阶段 D：`final_time_s` 鲁棒性

当前 `FinalTimeHead` 使用 `softplus` 保证时长大于零，但“正数”不等于“合理”。现有 validation 中出现过约 0.027 s 的极端预测。normalized-time 会把整条空间路径压进这段极短时间，从而使速度、加速度和 jerk 爆炸。

建议按以下顺序实验：

1. 统计 train/validation 的真实剩余时间分布和极端分位数；
2. 对比当前 scaled MSE 与 `log1p(duration)` 空间的 Huber/MSE；
3. 报告 time MAE、p95、bias、低时长异常比例，不只报告均值；
4. 检查 duration 改动是否影响路径 ADE 和 terminal error；
5. 不使用推理后硬裁剪来掩盖训练头失效。

如果未来使用非均匀 `segment_durations`，必须保证：

```text
delta_t_i > 0
sum(delta_t_i) = final_time_s
```

项目现有 `ControlOutputHead` 已通过 softmax fractions 实现这一契约。

## 10. 阶段 E：结构化输出

如果阶段 A--D 仍不能把 train ADE 从千米量级显著降低，同时保持合理 raw kinematics，应停止继续堆 loss，转向结构化 decoder。

### 10.1 推荐结构

```text
120 s history
    ↓
Transformer encoder
    ↓
bounded controls + positive nonuniform segment durations
    ↓
differentiable dynamics / kinematic rollout
    ↓
state trajectory
    ↓
path + terminal + duration + control regularization losses
```

控制量与当前输出契约一致：

```text
thrust_N
bank_rad
load_factor
```

### 10.2 为什么比继续增加 kinematic loss 更合适

- 位置和速度由同一个 rollout 产生，一致性成为结构性质而不是惩罚项；
- 控制边界可以直接施加在物理量上；
- 非均匀时段可在转弯、截获和拉平阶段分配更高时间分辨率；
- 后续多机智能交互本来就需要输出可执行控制或意图；
- N 表示控制复杂度，不再是大量相互独立的状态节点。

### 10.3 最小接入路径

当前已经存在：

- `StatePrediction` / `ControlPrediction` 类型；
- `ControlBounds`；
- `ControlOutputHead`；
- 非均匀 `segment_durations`；
- raw kinematic metrics；
- state/final-time 数据与固定 anchor 训练协议。

还需要补齐：

1. 从轨迹反演控制监督，或者直接定义 rollout state loss；
2. 让 backbone 提供稳定 latent feature 给 control head；
3. 接入可微分 piecewise-constant dynamics rollout；
4. 对 rollout state 计算 path、terminal 和 duration loss；
5. 将 state-output 与 control-output 作为平行策略，不能删除现有 state/full/window 功能；
6. train、predict、evaluation、checkpoint identity 和前端发布都显式记录 output kind。

不应在第一版同时实现复杂飞机性能模型、多模态 decoder 和多机策略。先完成单机、单候选、可微 rollout 的最小闭环。

### 10.4 可选中间方案：Spline/control-point decoder

若控制监督和动力学 rollout 尚未准备好，可先测试低维 spline/control-point decoder：

```text
encoder -> K control points -> differentiable spline -> N state samples
```

它能减少独立节点自由度并提高连续性，但不能替代最终控制模型，也不能自然保证完整飞行动力学约束。

## 11. 阶段 F：意图信息与多模态预测

当单机结构化输出能够稳定训练后，再加入影响未来分支的信息：

- airport/runway identity；
- runway-aligned coordinate frame；
- approach procedure、IAF/FAF 或 route intent；
- aircraft type/performance；
- 风场和天气；
- 邻机历史、相对位置、排序和冲突关系。

多机智能交互建议采用“每架飞机历史 encoder + 邻机集合/图交互模块 + 每架飞机控制 decoder”。邻机输入必须是 permutation-invariant，不能依赖航班在 batch 中的排列顺序。

如果同一条件仍对应多个合法未来，再引入：

- mixture trajectory/control heads；
- CVAE；
- diffusion/flow matching；
- best-of-K 或概率似然训练。

多候选模型必须同时报告：

1. 单候选可部署选择结果；
2. oracle minADE/minFDE，用于描述候选覆盖上限；
3. 候选多样性；
4. 概率校准或候选排序质量。

不能只报告 oracle，因为在线系统并不知道哪条候选是真值。

## 12. 统一评价协议

### 12.1 判断拟合使用什么

判断“模型能否拟合”优先使用：

```text
best checkpoint
model.eval()
fixed anchor L-1
dropout off
train split replay
```

训练循环中每个 epoch 的 `train_loss` 会受到 shuffle、dropout 和不同 loss 标度影响，不能替代 train ADE/FDE。

### 12.2 判断泛化使用什么

使用同一 best checkpoint、同一 anchor 和同一 metric contract 的 validation replay。先看绝对误差，再看 train-validation gap。

### 12.3 指标定义

- ADE：Average Displacement Error，所有有效未来节点的平均三维位置误差；
- FDE：Final Displacement Error，最后一个有效未来节点的三维位置误差；
- TTA：Time To Arrival，从 anchor 到跑道入口的剩余时间；
- terminal lateral/vertical：最终状态相对跑道入口目标门的横向/垂向误差；
- raw kinematic metrics：直接对未经滤波、样条或 CZML 插值的模型输出节点计算的物理一致性和平滑性指标。

normalized、full、window 的原生时间网格不同。模式内 train/validation 可用原生指标比较；模式间必须使用统一物理网格报告。

### 12.4 每个实验必须生成的表

| 字段 | Candidate A | Candidate B | Candidate C |
|---|---:|---:|---:|
| changed variable |  |  |  |
| seed |  |  |  |
| best epoch |  |  |  |
| epochs run |  |  |  |
| optimizer updates |  |  |  |
| final learning rate |  |  |  |
| train loss |  |  |  |
| validation loss |  |  |  |
| train ADE/FDE |  |  |  |
| validation ADE/FDE |  |  |  |
| ADE/FDE gap ratio |  |  |  |
| train/validation time MAE |  |  |  |
| terminal lateral/vertical p95 |  |  |  |
| raw kinematic score and five components |  |  |  |
| flyability |  |  |  |
| wall time / peak VRAM |  |  |  |
| checkpoint SHA-256 |  |  |  |

## 13. 阶段门与决策树

```text
Start
  |
  |-- Batch 512/1024 lowers train and val ADE?
  |       |-- yes -> fix batch, test scheduler
  |       `-- no  -> scheduler test directly
  |
  |-- Effective LR/update budget lowers train ADE?
  |       |-- yes -> fix optimizer recipe, test kinematic weight
  |       `-- no  -> loss/output representation is dominant
  |
  |-- k/dropout gives acceptable ADE + raw physics Pareto point?
  |       |-- yes -> duration robustness, then multi-seed confirmation
  |       `-- no  -> structured control or spline decoder
  |
  |-- Structured rollout fits train but validation remains poor?
  |       |-- yes -> add intent/context, then multimodal output
  |       `-- no  -> freeze design and prepare untouched temporal holdout
  |
  `-- Only after every decision is frozen: one-shot final test
```

“acceptable”必须在看结果前定义。当前第一阶段的现实目标不是立即达到跑道严格门限，而是证明 train ADE 能稳定下降至少一个有实际意义的幅度，同时 validation 同向改善且 raw kinematic 指标不崩溃。

## 14. 明确不建议的做法

- 不要只增加 epoch；当前主要曲线已经平台化；
- 不要仅因为显存有余就继续增加 batch；
- 不要认为 N 越大轨迹越平滑；当前直接节点 head 的实验证据相反；
- 不要继续提高 kinematic weight 来强压抖动；这会显著损害 ADE；
- 不要只看总 normalized loss，不看物理单位 ADE/FDE；
- 不要用滤波后的 CZML 视觉效果代替 raw-output 评价；
- 不要用 test 反复挑参数；
- 不要覆盖正式 checkpoint 或复用不同配置的 `fit_evaluation.json`；
- 不要为了新 decoder 删除 normalized/full/window 或 state-output 功能；新旧策略应并存并解耦；
- 不要在没有控制监督/rollout loss 时提前重构所有 backbone 内部接口。

## 15. 推荐的下一次实际执行

下一次实验只做阶段 A：

1. 固定当前代码和 arrivals manifests；
2. 在独立目录重跑 `batch=2048, 1024, 512`；
3. 每个候选保持 `N=64, lr=5e-4, k=3, terminal=0.02, dropout=0.1, epochs=180, seed=1337`；
4. 读取每个候选自动生成的 `fit_evaluation.json`；
5. 比较 train/validation ADE/FDE、gap、time MAE、最佳 epoch、运行时间；
6. 对候选运行统一 validation-only predictability/raw-kinematic 报告；
7. 选前两名做多 seed 复验；
8. 在 batch 结论冻结前，不开始 scheduler、loss 或结构改造。

如果 `batch=512` 明显降低 train/validation ADE，下一步是保留该 batch 并记录学习率/optimizer step，再比较 scheduler patience。如果没有明显改善，则应尽快进入 loss/decoder 方向，不再围绕 batch 消耗实验预算。

## 16. 相关代码与历史证据

| 资源 | 内容 |
|---|---|
| [`config.py`](../config.py) | 当前模型、训练和 loss 默认配置 |
| [`train.py`](../train.py) | 训练循环、loss、scheduler、best-checkpoint 回放 |
| [`metrics.py`](../metrics.py) | ADE/FDE 和 raw kinematic 指标 |
| [`prediction_outputs.py`](../prediction_outputs.py) | state/control 输出契约与非均匀时段 head |
| [`normalized_time_and_control_output.zh.md`](normalized_time_and_control_output.zh.md) | normalized-time、控制输出和 segment durations 设计 |
| [`2026-07-27_small_sample_overfit_diagnostic.zh.md`](2026-07-27_small_sample_overfit_diagnostic.zh.md) | 单航迹与 160 航迹拟合能力实验 |
| [`2026-07-27_kinematic_weight_epoch_ablation.zh.md`](2026-07-27_kinematic_weight_epoch_ablation.zh.md) | N、k、模型容量和 epoch 消融 |
| [`report.html`](../../outputs/POOLED/ts_time_parameterization_predictability_report/report.html) | validation-only 预测能力、训练回放和多候选覆盖报告 |

## 17. 实验记录模板

每轮实验复制以下内容到该轮输出目录的 `experiment_notes.md`：

```markdown
# Experiment: <name>

## Question

本实验只回答：<one question>

## Frozen controls

- data manifests + SHA-256:
- split SHA-256:
- git commit / dirty state:
- model / horizon / coordinate frame:
- L / N / anchor:
- loss weights:
- optimizer / scheduler:
- seed:

## Changed variable

- parameter:
- candidates:

## Selection rule declared before training

- primary:
- admissibility threshold:
- secondary:
- tie break:

## Results

| Candidate | Train ADE/FDE | Val ADE/FDE | Gap | Time MAE | Raw score | Terminal p95 | Best epoch | Updates | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Decision

- selected:
- rejected:
- evidence:
- remaining uncertainty:
- next single-variable experiment:
- outer-test accessed: no
```

该模板的目的不是增加形式工作，而是确保几周后仍能回答“为什么选择这个参数”，并阻止不同数据版本、不同 anchor 或不同 metric grid 的结果被误放在同一表格中比较。
