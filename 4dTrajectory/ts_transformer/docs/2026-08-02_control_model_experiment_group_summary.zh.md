# Control 模型实验组总结（2026-08-02）

## 1. 最终结论

本组实验完成后，当前 KSJC development 最佳配方为：

```text
backbone: iTransformer
d_model / d_ff / layers / heads: 512 / 1024 / 4 / 8
output: N=64 absolute controls + factorized final time
initialization: 32-flight outer-train direct oracle teacher, 1000 imitation steps
dynamics: scaled-transport-chart-velocity
loss: arc-length geometry 2+4
training: 60/120/240/full curriculum, LR=3e-5, global clip20
clock: observed state supervision, detached state-duration gradient
control effort / smoothness: 0 / 0
RK4 / supervision: 0.5 s / fixed 2 s
```

该配置仍是 development 模型，尚未冻结为 final release，也没有打开 outer-test。

## 2. 累计结果

| 阶段 | selection | ADE | FDE | cross p95 | altitude p95 | time MAE |
|---|---:|---:|---:|---:|---:|---:|
| 原正式基线，small iT | 31.4126 | 1130.7 m | 1202.9 m | 3313.2 m | 692.6 m | 20.84 s |
| + direct train-only teacher | 27.6277 | 771.0 m | 1040.4 m | 2039.6 m | 364.7 m | 14.19 s |
| + scaled transport | 27.5097 | 751.7 m | 1039.2 m | 1930.2 m | 396.3 m | 14.35 s |
| + large iTransformer | **23.2743** | **694.7 m** | **915.4 m** | **1539.8 m** | **355.3 m** | **13.72 s** |

最终相对原基线：selection -25.9%，ADE -38.6%，FDE -23.9%，cross-track p95 -53.5%，
altitude p95 -48.7%，final-time MAE -34.1%。

## 3. 每个方向的判断

### 3.1 Reference velocity

因果平滑位置差分在单航迹和 validation 中改善 ADE/横向误差，但 validation selection 从 31.4126
退化到 35.2837，FDE 从 1202.9 m 退化到 1367.1 m。保留 `track-fit`。

### 3.2 Dual-clock terminal

联合 predicted-clock 发生 33 s 时间捷径坍缩。Detached-time 修复机制后 selection 为 29.2073，
但 ADE/FDE 退化到 1297.7/1394.5 m，且训练约慢 1.76 倍。保留单 state-supervision clock。

### 3.3 Oracle teacher

这是本组最大单项提升。32 条 outer-train optimized schedules 的初始化使 ADE 从 1130.7 m 降到
771.0 m，并同时改善 FDE、横向、高度和时间。隔离审计为 teacher 32/32 属于 train、与 validation
交集为 0。推理时不读取 future。

### 3.4 State scaling

Scaled transport 将 selection 从 27.6277 小幅改善到 27.5097，ADE/cross 也改善，但高度和时间略
退化。Clip rate 仍为 100%，最大裁剪前梯度反而更高，因此它是数值路径 Pareto 点，不是 Jacobian
根治方案。

### 3.5 Progressive N

16→32→64 的同预算 imitation MSE 更低，但正式 selection 退化到 33.7088，ADE/FDE 退化到
863.1/1239.2 m，推力饱和明显上升。拒绝该模式。

### 3.6 Backbone 与容量

PatchTST selection 28.5444、ADE 821.4 m，未超过 small iTransformer；只在 cross-track p95 上
更好。Large iTransformer 则六项全部改善，证明 teacher 改善初始化后，模型容量重新成为真实瓶颈。

## 4. 数据与复现审计

- Arrival manifest SHA-256：`687f5c6c1d94bb540d36c74ba443b57c7eddae5109441e2bef008b2498376d49`。
- Train selected identity SHA-256：`77fb674e93b7f39863fc2409e3169f56de463d60a9cb5aef0cbd9098d0925690`。
- Validation selected identity SHA-256：`eacbf823144ea2af0b1241ee92c98f3df0c54365149b51a9d9aeea7734a10d4b`。
- Teacher schedule SHA-256：`a67ea14e0599dd5d2ecae55b854c3adf573043091e08cd4ce661bbf32cec4938`。
- 所有正式后续 run 均为 2207 train / 505 validation，split 和 teacher schedule 完全相同。
- Outer-test 轨迹值、预测和指标均未运行或查看。
- 相关全量测试：348 passed，6 skipped。

## 5. 下一步建议

按当前证据排序：

1. 在不看 test 的前提下，用第二个固定 seed 复现 small/large iTransformer 容量差异；large 的优势
   足够大，但一次 seed 尚不能估计方差。
2. 预注册扩大 train-only teacher cohort（例如 32→128），保持 validation、正式 epoch 和 large
   backbone 不变。Teacher 是本组最大收益来源，且当前只覆盖 32/2207 train 航迹。
3. 单独研究推力边界：large 最佳 epoch 的推力饱和约 38.2% 控制点。先诊断这些点是否对应真实
   idle/max-thrust 需要，再决定是否改参数化或正则；不要直接惩罚所有饱和。
4. 如果 large 在第二 seed 仍稳定，预注册更长 epoch 上限。当前最佳位于 epoch 180，但本组不事后
   延长预算，以免破坏容量对照。
5. 在模型、teacher cohort、训练预算和 selection contract 全部冻结后，再由用户明确授权一次
   `--split test --release-test` 最终发布。

## 6. 详细报告

- [`2026-08-02_reference_velocity_consistency_ablation.zh.md`](2026-08-02_reference_velocity_consistency_ablation.zh.md)
- [`2026-08-02_dual_clock_terminal_ablation.zh.md`](2026-08-02_dual_clock_terminal_ablation.zh.md)
- [`2026-08-02_oracle_teacher_experiment.zh.md`](2026-08-02_oracle_teacher_experiment.zh.md)
- [`2026-08-02_nondimensional_transport_experiment.zh.md`](2026-08-02_nondimensional_transport_experiment.zh.md)
- [`2026-08-02_progressive_n_experiment.zh.md`](2026-08-02_progressive_n_experiment.zh.md)
- [`2026-08-02_backbone_capacity_experiment.zh.md`](2026-08-02_backbone_capacity_experiment.zh.md)
