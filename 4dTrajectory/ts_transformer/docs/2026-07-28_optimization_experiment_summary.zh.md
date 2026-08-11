# iTransformer 轨迹预测优化实验总结与数据索引

日期：2026-07-28
状态：开发集实验阶段，尚未发布 outer-test 结果

## 1. 文档范围

本文汇总五机场 pooled iTransformer normalized-time 模型的以下实验：

1. batch size；
2. ReduceLROnPlateau patience；
3. ENU 与 runway-aligned 坐标系；
4. initial learning rate；
5. kinematic consistency loss weight。

实验设计依据见[收敛诊断与优化实验指南](2026-07-27_convergence_and_optimization_experiment_guide.zh.md)。本文只汇总 train、validation 和 validation-only 统一报告；outer-test 从未用于训练、回放、预测、指标计算或候选选择。

## 2. 数据与评价协议

### 2.1 数据范围

五个机场为 KMSY、KRDU、KSJC、KSMF、KSTL，共构建 19,741 条可用 arrival series；另有 1 条航迹短于 120 s 窗口而被跳过。

| Split | 航班数 | 身份 SHA-256 | 用途 |
| --- | ---: | --- | --- |
| Train | 13,807 | `32dd550dca616a1f85e8cde4012b3b262170c9df1f427f24ead5de8e687d2ea0` | 参数优化与 best-checkpoint train 回放 |
| Validation | 2,951 | `4df9d067f78e576c604cf08656e654fdfebc85d65b99a3ff736beea3fbdfb4ce` | 所有开发决策 |

`split_seed=1337` 固定数据身份；训练随机种子为 1337、2027、4242。三种子结果中的 `±` 表示样本标准差。只运行 seed=1337 的实验均明确标记为 screening，不代表稳定的重复实验结果。

### 2.2 指标口径

- Native ADE/FDE：normalized-time 模型原生 64 个 progress segment 上的 measured-data 三维位置误差；
- Common-grid ADE/FDE：validation-only 报告在统一物理时间网格上的误差；
- Final-time MAE：从 120 s history anchor 到航迹终点的剩余时间绝对误差；
- Raw physical diagnostics：position/velocity consistency、heading consistency、turn rate、acceleration 和 jerk；
- 所有正式 Train/Validation ADE/FDE 都来自 retained best checkpoint 的固定 `L-1` anchor 回放，不以训练循环内带 dropout 的 loss 代替。

已知极短 `final_time_s` 预测会支配 position/velocity RMSE，因此该项持续记录，但在本轮 batch、scheduler、坐标系、初始学习率和 kinematic-weight 选择中不作为单独胜负依据。

## 3. 当前冻结配置

```text
model                              itransformer
horizon_mode                       normalized
coordinate_frame                   enu
history                            60 × 2 s = 120 s
normalized segments                64
d_model / d_ff                     256 / 512
heads / encoder layers             8 / 3
dropout                            0.1
batch_size                         512
epochs / early-stop patience       180 / 20
initial learning rate              5e-4
ReduceLROnPlateau factor/patience  0.5 / 8
kinematic weight                   3.0
terminal weight                    0.02
final-time weight                  1.0
split_seed                         1337
```

该配置的三种子 native fixed-anchor 结果为：

| Seed | Best/run epoch | Train ADE/FDE (m) | Validation ADE/FDE (m) | Validation time MAE (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1337 | 124/144 | 1370.9 / 583.1 | 1451.5 / 631.2 | 69.42 |
| 2027 | 147/167 | 1428.1 / 668.7 | 1497.8 / 712.8 | 68.61 |
| 4242 | 88/108 | 1580.3 / 817.0 | 1619.6 / 849.8 | 70.69 |
| Mean ± SD | — | 1459.8 ± 108.3 / 689.6 ± 118.4 | 1523.0 ± 86.8 / 731.3 ± 110.5 | 69.57 ± 1.05 |

从最初 `batch=2048, patience=3, seed=1337` 的 Validation ADE/FDE 1857.4/1346.3 m，到当前同种子配置的 1451.5/631.2 m，ADE 降低 21.85%，FDE 降低 53.11%。这是阶段性开发集改进，不是 outer-test 结果。

## 4. 实验结论总览

| 阶段 | 改变变量 | 重复级别 | 结果 | 决策 |
| --- | --- | --- | --- | --- |
| A | batch `2048/1024/512` | 512、1024 三种子；2048 单种子 | 512 相对 1024 的平均 Validation ADE/FDE 改善 7.7%/19.8% | 冻结 batch 512 |
| B | plateau patience `3/8/12` | 8、12 三种子；3 有既有三种子基线 | patience 8 相对 patience 3 的平均 Validation ADE/FDE 改善 7.0%/21.4% | 冻结 patience 8 |
| C | ENU / runway-aligned | 两者三种子 | runway 的 Validation ADE 改善 0.81%，但 Train ADE、Validation FDE 和 time MAE 分别恶化 3.13%、12.75%、5.15% | 保留 ENU |
| D | initial LR `1e-4/3e-4/5e-4` | seed=1337 screening | `5e-4` 的 native Train/Validation ADE/FDE 最好 | 保留 `5e-4` |
| E | kinematic weight `0.3/1/3` | seed=1337 screening | `0.3` 仅改善 Validation ADE 0.65%，但 FDE 与四项物理指标显著恶化 | 保留 3.0 |

明确被当前证据否定的通用方向包括：继续增加 epoch、继续增大 batch、单纯提高模型宽度、将 ENU 整体替换为 runway-aligned、降低初始学习率，以及仅通过降低 kinematic weight 换取很小的 ADE 变化。

## 5. 分阶段结果

### 5.1 Batch size

三种子确认结果：

| Batch | Train ADE (m) | Validation ADE (m) | Train FDE (m) | Validation FDE (m) | Validation time MAE (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1772.7 ± 10.9 | 1774.6 ± 11.0 | 1140.2 ± 48.5 | 1161.3 ± 47.8 | 69.82 ± 1.28 |
| 512 | 1612.3 ± 34.2 | 1638.3 ± 33.3 | 901.1 ± 77.3 | 930.9 ± 74.9 | 68.73 ± 0.40 |

结论：batch 512 提供更多 optimizer updates，并且 Train/Validation 与物理指标同向改善。batch 2048 在 seed=1337 上只有 1,260 次更新，Validation ADE/FDE 为 1857.4/1346.3 m。

### 5.2 Learning-rate scheduler patience

三种子结果：

| Patience | Train ADE/FDE (m) | Validation ADE/FDE (m) | Validation time MAE (s) |
| ---: | ---: | ---: | ---: |
| 8 | 1459.8 ± 108.3 / 689.6 ± 118.4 | 1523.0 ± 86.8 / 731.3 ± 110.5 | 69.57 ± 1.05 |
| 12 | 1494.2 ± 79.2 / 725.5 ± 79.2 | 1546.9 ± 66.8 / 765.6 ± 73.1 | 70.20 ± 0.56 |

patience 3 的三种子既有基线为 Validation ADE/FDE 1638.3/930.9 m。patience 8 避免学习率过早衰减到近似冻结，并在 ADE、FDE 和典型 raw physical diagnostics 上均优于 12。

### 5.3 Coordinate frame

三种子结果：

| Frame | Train ADE/FDE (m) | Validation ADE/FDE (m) | Validation time MAE (s) |
| --- | ---: | ---: | ---: |
| ENU | 1459.8 ± 108.3 / 689.6 ± 118.4 | 1523.0 ± 86.8 / 731.3 ± 110.5 | 69.57 ± 1.05 |
| Runway-aligned | 1505.5 ± 48.1 / 747.4 ± 48.3 | 1510.6 ± 43.2 / 824.5 ± 48.5 | 73.16 ± 0.73 |

Runway-aligned 将 validation cross-track mean 从 676.0 m 降到 524.0 m，但 along-track mean 从 981.6 m 增至 1254.9 m，altitude mean 从 69.1 m 增至 80.8 m。它提高横向约束和平滑性，却没有形成整体 ADE/FDE 优势，因此不替换 ENU。

### 5.4 Initial learning rate

Seed=1337 screening：

| Initial LR | Best/run epoch | Train ADE/FDE (m) | Validation ADE/FDE (m) | Common-grid ADE/FDE (m) |
| ---: | ---: | ---: | ---: | ---: |
| `1e-4` | 177/180 | 1638.2 / 955.3 | 1653.8 / 970.4 | 1712.4 / 1728.4 |
| `3e-4` | 166/180 | 1414.2 / 641.8 | 1464.2 / 681.1 | 1557.5 / 1532.8 |
| `5e-4` | 124/144 | 1370.9 / 583.1 | 1451.5 / 631.2 | 1585.3 / 1526.4 |

`3e-4` 的 common-grid ADE 比 `5e-4` 好 1.76%，但低于预注册的 2% 确认门槛，native Train/Validation ADE 与两种 FDE 均未同时改善。`1e-4` 在 180 epoch 内明显欠拟合。因此没有触发额外种子确认。

### 5.5 Kinematic consistency weight

Seed=1337 screening：

| Weight | Train ADE/FDE (m) | Validation ADE/FDE (m) | Common-grid ADE/FDE (m) |
| ---: | ---: | ---: | ---: |
| `0.3` | 1363.0 / 607.8 | 1442.1 / 661.2 | 1567.0 / 1577.7 |
| `1.0` | 1422.0 / 609.1 | 1484.9 / 658.0 | 1657.4 / 1499.2 |
| `3.0` | 1370.9 / 583.1 | 1451.5 / 631.2 | 1585.3 / 1526.4 |

| Weight | Heading p95 (deg) | Turn p95 (deg/s) | Acceleration p95 (m/s²) | Jerk p95 (m/s³) |
| ---: | ---: | ---: | ---: | ---: |
| `0.3` | 65.06 | 27.16 | 38.78 | 30.08 |
| `1.0` | 49.92 | 20.22 | 34.50 | 28.25 |
| `3.0` | 37.37 | 12.00 | 19.62 | 15.00 |

`k=0.3` 的 native Validation ADE 只改善 0.65%，却使 heading、turn、acceleration、jerk 分别恶化约 74%、126%、98%、101%；`k=1` 的准确率和物理性也没有形成优势。两者都没有触发额外种子确认。

## 6. 实验数据索引

### 6.1 受保护的输入 manifests

| Airport | Manifest | SHA-256 |
| --- | --- | --- |
| KMSY | [manifest.json](../../../trajectory_data_process/outputs/harvest/KMSY/arrivals/manifest.json) | `831276322d9ab2306fa777e78398b12f1640c0d6f4479e4effbb65776b6bebde` |
| KRDU | [manifest.json](../../../trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json) | `80bf1a236eb755b88e35ef039494e50083a3d2ae3680ff53996233aa5a83b2ce` |
| KSJC | [manifest.json](../../../trajectory_data_process/outputs/harvest/KSJC/arrivals/manifest.json) | `687f5c6c1d94bb540d36c74ba443b57c7eddae5109441e2bef008b2498376d49` |
| KSMF | [manifest.json](../../../trajectory_data_process/outputs/harvest/KSMF/arrivals/manifest.json) | `90c9ecedbd63aa6ea6ef281d97ca4493f0a19d20bd323f252d127f191c3621a5` |
| KSTL | [manifest.json](../../../trajectory_data_process/outputs/harvest/KSTL/arrivals/manifest.json) | `ae90194bd4994710bf3e53384f51fd945df5af9bc4e2e3538275d8f93fe87f83` |

所有实验前后哈希一致。实验只读取这些 manifests 与其引用的航迹，没有执行 harvest、evaluate-only、清理、覆盖或源数据改写。

### 6.2 单次训练目录的标准文件

每个 run 目录包含：

- `checkpoint.pt`：retained best model；
- `checkpoint_metadata.json`：配置、模型身份和数据身份；
- `history.json`：逐 epoch loss、learning rate、optimizer updates 和训练摘要；
- `fit_evaluation.json`：best checkpoint 的固定-anchor Train/Validation 回放指标。

下面的索引链接到后三种可直接检查的 JSON；对应 `checkpoint.pt` 与它们位于同一目录。

### 6.3 Batch-size runs

实验说明：[experiment_notes.md](../../outputs/POOLED/experiments/batch_size/experiment_notes.md)

| Candidate | Fit evaluation | History | Metadata |
| --- | --- | --- | --- |
| b2048 / seed1337 | [fit](../../outputs/POOLED/experiments/batch_size/itr_norm_b2048_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/batch_size/itr_norm_b2048_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/batch_size/itr_norm_b2048_seed1337/checkpoint_metadata.json) |
| b1024 / seed1337 | [fit](../../outputs/POOLED/experiments/batch_size/itr_norm_b1024_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/batch_size/itr_norm_b1024_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/batch_size/itr_norm_b1024_seed1337/checkpoint_metadata.json) |
| b1024 / seed2027 | [fit](../../outputs/POOLED/experiments/batch_size/itr_norm_b1024_seed2027/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/batch_size/itr_norm_b1024_seed2027/history.json) | [metadata](../../outputs/POOLED/experiments/batch_size/itr_norm_b1024_seed2027/checkpoint_metadata.json) |
| b1024 / seed4242 | [fit](../../outputs/POOLED/experiments/batch_size/itr_norm_b1024_seed4242/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/batch_size/itr_norm_b1024_seed4242/history.json) | [metadata](../../outputs/POOLED/experiments/batch_size/itr_norm_b1024_seed4242/checkpoint_metadata.json) |
| b512 / seed1337 | [fit](../../outputs/POOLED/experiments/batch_size/itr_norm_b512_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/batch_size/itr_norm_b512_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/batch_size/itr_norm_b512_seed1337/checkpoint_metadata.json) |
| b512 / seed2027 | [fit](../../outputs/POOLED/experiments/batch_size/itr_norm_b512_seed2027/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/batch_size/itr_norm_b512_seed2027/history.json) | [metadata](../../outputs/POOLED/experiments/batch_size/itr_norm_b512_seed2027/checkpoint_metadata.json) |
| b512 / seed4242 | [fit](../../outputs/POOLED/experiments/batch_size/itr_norm_b512_seed4242/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/batch_size/itr_norm_b512_seed4242/history.json) | [metadata](../../outputs/POOLED/experiments/batch_size/itr_norm_b512_seed4242/checkpoint_metadata.json) |

### 6.4 Scheduler runs

实验说明：[experiment_notes.md](../../outputs/POOLED/experiments/lr_schedule/experiment_notes.md)

| Candidate | Fit evaluation | History | Metadata |
| --- | --- | --- | --- |
| p3 / seed1337 | [fit](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p3_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p3_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p3_seed1337/checkpoint_metadata.json) |
| p8 / seed1337 | [fit](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p8_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p8_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p8_seed1337/checkpoint_metadata.json) |
| p8 / seed2027 | [fit](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p8_seed2027/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p8_seed2027/history.json) | [metadata](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p8_seed2027/checkpoint_metadata.json) |
| p8 / seed4242 | [fit](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p8_seed4242/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p8_seed4242/history.json) | [metadata](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p8_seed4242/checkpoint_metadata.json) |
| p12 / seed1337 | [fit](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p12_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p12_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p12_seed1337/checkpoint_metadata.json) |
| p12 / seed2027 | [fit](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p12_seed2027/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p12_seed2027/history.json) | [metadata](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p12_seed2027/checkpoint_metadata.json) |
| p12 / seed4242 | [fit](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p12_seed4242/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p12_seed4242/history.json) | [metadata](../../outputs/POOLED/experiments/lr_schedule/itr_norm_b512_plateau_p12_seed4242/checkpoint_metadata.json) |

### 6.5 Coordinate-frame runs

实验说明：[experiment_notes.md](../../outputs/POOLED/experiments/coordinate_frame/experiment_notes.md)

ENU 参考即上一节的 p8 三种子。Runway-aligned 新 runs：

| Candidate | Fit evaluation | History | Metadata |
| --- | --- | --- | --- |
| runway / seed1337 | [fit](../../outputs/POOLED/experiments/coordinate_frame/itr_norm_b512_plateau_p8_runway_aligned_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/coordinate_frame/itr_norm_b512_plateau_p8_runway_aligned_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/coordinate_frame/itr_norm_b512_plateau_p8_runway_aligned_seed1337/checkpoint_metadata.json) |
| runway / seed2027 | [fit](../../outputs/POOLED/experiments/coordinate_frame/itr_norm_b512_plateau_p8_runway_aligned_seed2027/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/coordinate_frame/itr_norm_b512_plateau_p8_runway_aligned_seed2027/history.json) | [metadata](../../outputs/POOLED/experiments/coordinate_frame/itr_norm_b512_plateau_p8_runway_aligned_seed2027/checkpoint_metadata.json) |
| runway / seed4242 | [fit](../../outputs/POOLED/experiments/coordinate_frame/itr_norm_b512_plateau_p8_runway_aligned_seed4242/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/coordinate_frame/itr_norm_b512_plateau_p8_runway_aligned_seed4242/history.json) | [metadata](../../outputs/POOLED/experiments/coordinate_frame/itr_norm_b512_plateau_p8_runway_aligned_seed4242/checkpoint_metadata.json) |

### 6.6 Initial-learning-rate runs

实验说明：[experiment_notes.md](../../outputs/POOLED/experiments/initial_learning_rate/experiment_notes.md)

`5e-4` 参考即 scheduler p8 / seed1337。新 screening runs：

| Candidate | Fit evaluation | History | Metadata |
| --- | --- | --- | --- |
| `1e-4` / seed1337 | [fit](../../outputs/POOLED/experiments/initial_learning_rate/itr_norm_b512_plateau_p8_lr1e4_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/initial_learning_rate/itr_norm_b512_plateau_p8_lr1e4_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/initial_learning_rate/itr_norm_b512_plateau_p8_lr1e4_seed1337/checkpoint_metadata.json) |
| `3e-4` / seed1337 | [fit](../../outputs/POOLED/experiments/initial_learning_rate/itr_norm_b512_plateau_p8_lr3e4_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/initial_learning_rate/itr_norm_b512_plateau_p8_lr3e4_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/initial_learning_rate/itr_norm_b512_plateau_p8_lr3e4_seed1337/checkpoint_metadata.json) |

### 6.7 Kinematic-weight runs

实验说明：[experiment_notes.md](../../outputs/POOLED/experiments/kinematic_weight/experiment_notes.md)

`k=3` 参考即 scheduler p8 / seed1337。新 screening runs：

| Candidate | Fit evaluation | History | Metadata |
| --- | --- | --- | --- |
| `k=0.3` / seed1337 | [fit](../../outputs/POOLED/experiments/kinematic_weight/itr_norm_b512_plateau_p8_k03_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/kinematic_weight/itr_norm_b512_plateau_p8_k03_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/kinematic_weight/itr_norm_b512_plateau_p8_k03_seed1337/checkpoint_metadata.json) |
| `k=1.0` / seed1337 | [fit](../../outputs/POOLED/experiments/kinematic_weight/itr_norm_b512_plateau_p8_k1_seed1337/fit_evaluation.json) | [history](../../outputs/POOLED/experiments/kinematic_weight/itr_norm_b512_plateau_p8_k1_seed1337/history.json) | [metadata](../../outputs/POOLED/experiments/kinematic_weight/itr_norm_b512_plateau_p8_k1_seed1337/checkpoint_metadata.json) |

### 6.8 Validation-only comparison reports

每个 report 的 `report.json.data_policy` 记录 `evaluated_split=val`、`retrieval_reference_split=train` 和 `outer_test_loaded=false`。

| Experiment | HTML | JSON | Per-flight | By airport | By runway | By trajectory type | By remaining time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Batch size | [report](../../outputs/POOLED/experiments/batch_size/comparison_seed1337/report.html) | [json](../../outputs/POOLED/experiments/batch_size/comparison_seed1337/report.json) | [CSV](../../outputs/POOLED/experiments/batch_size/comparison_seed1337/flight_metrics.csv) | [CSV](../../outputs/POOLED/experiments/batch_size/comparison_seed1337/metrics_by_airport.csv) | [CSV](../../outputs/POOLED/experiments/batch_size/comparison_seed1337/metrics_by_runway.csv) | [CSV](../../outputs/POOLED/experiments/batch_size/comparison_seed1337/metrics_by_trajectory_type.csv) | [CSV](../../outputs/POOLED/experiments/batch_size/comparison_seed1337/error_by_remaining_time.csv) |
| Scheduler | [report](../../outputs/POOLED/experiments/lr_schedule/comparison_seed1337/report.html) | [json](../../outputs/POOLED/experiments/lr_schedule/comparison_seed1337/report.json) | [CSV](../../outputs/POOLED/experiments/lr_schedule/comparison_seed1337/flight_metrics.csv) | [CSV](../../outputs/POOLED/experiments/lr_schedule/comparison_seed1337/metrics_by_airport.csv) | [CSV](../../outputs/POOLED/experiments/lr_schedule/comparison_seed1337/metrics_by_runway.csv) | [CSV](../../outputs/POOLED/experiments/lr_schedule/comparison_seed1337/metrics_by_trajectory_type.csv) | [CSV](../../outputs/POOLED/experiments/lr_schedule/comparison_seed1337/error_by_remaining_time.csv) |
| Initial LR | [report](../../outputs/POOLED/experiments/initial_learning_rate/comparison_seed1337/report.html) | [json](../../outputs/POOLED/experiments/initial_learning_rate/comparison_seed1337/report.json) | [CSV](../../outputs/POOLED/experiments/initial_learning_rate/comparison_seed1337/flight_metrics.csv) | [CSV](../../outputs/POOLED/experiments/initial_learning_rate/comparison_seed1337/metrics_by_airport.csv) | [CSV](../../outputs/POOLED/experiments/initial_learning_rate/comparison_seed1337/metrics_by_runway.csv) | [CSV](../../outputs/POOLED/experiments/initial_learning_rate/comparison_seed1337/metrics_by_trajectory_type.csv) | [CSV](../../outputs/POOLED/experiments/initial_learning_rate/comparison_seed1337/error_by_remaining_time.csv) |
| Kinematic weight | [report](../../outputs/POOLED/experiments/kinematic_weight/comparison_seed1337/report.html) | [json](../../outputs/POOLED/experiments/kinematic_weight/comparison_seed1337/report.json) | [CSV](../../outputs/POOLED/experiments/kinematic_weight/comparison_seed1337/flight_metrics.csv) | [CSV](../../outputs/POOLED/experiments/kinematic_weight/comparison_seed1337/metrics_by_airport.csv) | [CSV](../../outputs/POOLED/experiments/kinematic_weight/comparison_seed1337/metrics_by_runway.csv) | [CSV](../../outputs/POOLED/experiments/kinematic_weight/comparison_seed1337/metrics_by_trajectory_type.csv) | [CSV](../../outputs/POOLED/experiments/kinematic_weight/comparison_seed1337/error_by_remaining_time.csv) |

Coordinate-frame 实验当前以三种子 native replay 和误差分解为主，详见其 `experiment_notes.md`，没有单独生成 common-grid comparison report。

## 7. 当前结论与后续阶段门

当前最可靠的优化来自 batch 512 和 plateau patience 8。坐标旋转、较低初始学习率和较低 kinematic weight 都没有形成同时满足 Train/Validation ADE、FDE 与物理一致性的稳定改进。

下一步应先做 dropout `0/0.05/0.1` 单变量消融。如果仍不能稳定改善 Train 与 Validation ADE，应停止围绕现有 scalar loss 和 epoch 继续细调，转向低维 spline/control-point decoder、显式 approach intent 输入以及后续多模态输出。所有开发决策仍只能使用 train/validation；只有设计完全冻结后才能执行一次性 final test release。
