# Kinematic 权重、输出分段数、容量与 epoch 消融（2026-07-27）

## 结论

当前 iTransformer normalized-time state baseline 的默认配置定为：

```text
N = 64
d_model = 256, d_ff = 512, e_layers = 3
kinematic_consistency_loss_weight = 3.0
terminal_loss_weight = 0.02
epochs = 180, patience = 20
batch_size = 2048（五机场 pooled 全量训练）
```

PatchTST 使用同一 loss 权重和训练契约，但其独立 screening 选择 `N=256`，不能套用
iTransformer 的 `N=64`；该模型级默认值由 `config.py` 的单一映射解析。它在完整
validation/test 上仍明显失败，因此保留它是为了继续作为对照模型，不代表推荐用于当前
state-output 生产预测。

这里的 `epochs=180` 是训练上限，不是强制使用最后一个 epoch。完整 CUDA 训练在
epoch 161 得到最低 validation joint loss，并在 epoch 181 达到 patience 后停止，最终
checkpoint 回载 epoch 161。

`k=3` 是当前独立状态节点输出架构中的精度/物理一致性折中，不表示轨迹已经达到
实测可飞水平。最终 test 的转弯率、加速度和 jerk fleet p95 仍分别为实测基线的约
15.2、1.8 和 1.6 倍；严格跑道终端门限成功率仍为 0%。

## 实验协议

- 数据：KMSY、KRDU、KSJC、KSMF、KSTL arrivals；共构建 19,741 条可用航迹。
- 固定 outer split：train 13,807 / validation 2,951 / test 2,983，按航班身份切分。
- anchor：固定 `L-1`；每条训练航迹每 epoch 使用一次，重新 shuffle，并做机场 macro
  loss 加权。
- 超参数选择只使用 outer-validation。test 在 `N=64、k=3` 冻结后才首次读取；随后
  epoch 上限由 160 调整到 180 的依据仅为 validation 曲线，但此后 test 已不再是严格
  意义上的首次盲测，报告保留这一实验顺序。
- 选择规则：先保留 validation ADE 不超过最优值 10% 的候选，再最小化五项
  prediction/observed fleet-p95 比值的几何平均；raw score 相差 2% 内视为实际等效，
  再按 ADE 和较小模型容量决胜。
- 所有 raw metrics 使用未经样条、滤波或 CZML 插值的模型节点和显式 segment
  durations。批量结果同时保存 median、mean、p95、max；选择使用 p95，避免单个接近
  零的 final time 支配均值。

主要产物：

- `4dTrajectory/outputs/POOLED/ts_itransformer_kinematic_joint_confirmation/`
- `4dTrajectory/outputs/POOLED/ts_itransformer_kinematic_capacity_confirmation/`
- `4dTrajectory/outputs/POOLED/ts_itransformer_kinematic_full_confirmation/`
- `4dTrajectory/outputs/POOLED/ts_itransformer_kinematic_k3_epoch_confirmation/`

## N 与 k 的筛选

机场均衡 screening（每机场 128 train / 32 validation，`batch=64`）显示，增加 N 会显著
增加独立输出自由度和 raw-node 抖动。表中的 raw score 是五项 fleet-p95 ratio 的几何
平均，1.0 表示与同批 observed baseline 相同。

| N | k | validation ADE (m) | raw score |
|---:|---:|---:|---:|
| 64 | 0.3 | 2646.6 | 16.02 |
| 64 | 1 | 2645.2 | 13.22 |
| 64 | 3 | 2713.6 | 9.50 |
| 128 | 0.3 | 2639.1 | 40.52 |
| 128 | 1 | 2664.3 | 31.92 |
| 128 | 3 | 2886.1 | 25.32 |
| 256 | 0.3 | 2643.0 | 114.06 |
| 256 | 1 | 2796.1 | 93.61 |
| 256 | 3 | 2832.9 | 74.55 |

因此选择 `N=64`。提高 N 只能增加输出采样密度，不能自动带来曲线平滑；在当前“一次
直接回归所有节点”的 head 中，它反而增加高频自由度。

## 模型容量

在相同约 950–1200 optimizer updates 的筛选预算下，`d_model=512` 没有解除精度与
一致性的冲突：

| d_model | k | validation ADE (m) | raw score |
|---:|---:|---:|---:|
| 256 | 3 | 2701.6 | 8.86 |
| 512 | 3 | 2789.7 | 9.49 |
| 256 | 10 | 2736.5 | 6.99 |
| 512 | 10 | 2790.8 | 6.97 |

512 在 `k=3` 下精度和物理指标都更差；在 `k=10` 下 raw score 仅改善约 0.3%，但 ADE
更差。按 practical-equivalence 规则保留 `d_model=256`。

## PatchTST 的 N/k screening

相同 flight IDs、`batch=64` 和最多 120 epoch 的 PatchTST screening 得到：

| N | k | validation ADE (m) | raw score |
|---:|---:|---:|---:|
| 64 | 3 | 5252.2 | 21.59 |
| 64 | 10 | 5454.3 | 17.39 |
| 128 | 3 | 5295.5 | 51.92 |
| 128 | 10 | 5157.9 | 43.23 |
| 256 | 3 | 4387.1 | 72.49 |
| 256 | 10 | 5258.6 | 142.30 |

PatchTST 的最佳 ADE 来自 `N=256, k=3`；N=64 虽更平滑，但 ADE 比最优值高 19.7%，
不在 10% 精度准入带内。因此共享 k=3，但保留 PatchTST `N=256`。这同时说明 N 是
decoder/backbone 相关参数，不能在两个模型间机械共享。

完整 pooled 训练确认 batch 2048 可用，best epoch=114、epoch 134 early-stop；
`N=256, k=10` 在 screening 中同时劣于 k=3 的 ADE 和 raw score，因此没有进入 full
finalist。

| PatchTST 指标 | outer-validation | outer-test |
|:---|---:|---:|
| ADE mean / p95 (m) | 4185.5 / 9178.0 | 4057.8 / 8991.0 |
| FDE mean / p95 (m) | 3826.6 / 8086.8 | 3750.1 / 7789.0 |
| final-time MAE / p95 (s) | 70.4 / 204.9 | 69.6 / 200.0 |
| flyability | 0.1% | 0.0% |
| turn-rate fleet p95 pred/obs (deg/s) | 319.6 / 2.9 | 321.8 / 2.9 |
| acceleration fleet p95 pred/obs (m/s²) | 672.7 / 35.1 | 671.2 / 35.3 |
| jerk fleet p95 pred/obs (m/s³) | 2220.1 / 31.9 | 2204.1 / 32.5 |
| strict terminal-gate success | 0/2951 | 0/2983 |

结论：PatchTST 当前的直接多节点 state head 不可用。N=256 是其候选中的“精度最不差”
而不是物理合格；后续应改变 decoder/输出结构，不应继续靠增加 k 或 N。

## 完整 outer-validation 的 k 确认

小样本只用于 screening。最终在完整 13,807 train / 2,951 validation、`batch=2048`、
相同 160 epoch cap 下重新训练：

| k | ADE (m) | FDE (m) | time MAE (s) | raw score | 结果 |
|---:|---:|---:|---:|---:|:---|
| 3 | 1884.5 | 2442.3 | 68.4 | 5.82 | 选中 |
| 10 | 2234.0 | 2769.3 | 68.4 | 4.72 | ADE 比最优差 18.5%，淘汰 |

`k=10` 确实把 acceleration/jerk p95 拉近 observed baseline，但代价不是轻微波动，而是
约 350 m 的平均 ADE 增量。因此默认从原来的 10 降为 3。

## epoch 确认与最终 val/test

对选定 `N=64、d_model=256、k=3` 放宽到 240 epoch：epoch 161 最优，epoch 181 早停。
默认 cap 取 180，能够覆盖最优区域，同时避免把一次硬件运行的 epoch 161 当作固定训练
长度。该结论只适用于当前五机场 pooled 数据量和 `batch=2048`；改变 batch 后应按
optimizer updates 重新换算，不能机械复用 epoch 数。

| 指标 | outer-validation | outer-test |
|:---|---:|---:|
| flights | 2,951 | 2,983 |
| ADE mean / p95 (m) | 1883.8 / 5056.8 | 1879.4 / 5332.1 |
| FDE mean / p95 (m) | 2441.7 / 6485.9 | 2483.3 / 6812.1 |
| final-time MAE / p95 (s) | 68.4 / 202.3 | 69.0 / 203.8 |
| flyability | 48.8% | 48.6% |
| pos/velocity RMSE fleet p95 (pred/obs, m/s) | 44.6 / 13.3 | 44.6 / 13.0 |
| heading consistency fleet p95 (pred/obs, deg) | 121.0 / 2.2 | 121.0 / 2.2 |
| turn-rate fleet p95 (pred/obs, deg/s) | 42.1 / 2.9 | 43.7 / 2.9 |
| acceleration fleet p95 (pred/obs, m/s²) | 60.7 / 35.1 | 64.1 / 35.3 |
| jerk fleet p95 (pred/obs, m/s³) | 47.4 / 31.9 | 51.2 / 32.5 |
| strict terminal-gate success | 2/2951 | 0/2983 |

validation 与 test 的 ADE、时间误差、flyability 和 raw p95 接近，没有出现明显 split
崩溃；但绝对误差和严格成功率说明模型仍是欠约束的 kinematic baseline。

## 剩余问题

1. validation/test 各仍有 2 条预测的 `final_time_s < 30 s`；极端最小值分别约 0.027 s
   和 0.117 s。这些样本会让 raw mean/max 爆炸，因此正式比较使用 fleet p95，但时间头
   的灾难性短时长仍应通过独立的 duration 建模（例如 log-duration loss）解决，而不是
   用 metric 隐藏。
2. heading consistency 和 turn-rate 的中位数也明显偏离 observed，不只是少数 outlier。
   单纯继续提高 consistency weight 已被 full 实验证明会严重损害 ADE。
3. 若要进一步改善物理性，应优先改为结构化输出：预测速度/控制并积分得到位置，或使用
   低维 spline/control-point decoder；不要继续用更大的 N 或更强的同一 loss 强行压制
   彼此独立的状态节点。
