# Backbone 与容量实验（2026-08-02）

## 1. 实验问题

在 train-only oracle teacher 已显著改善优化起点后，重新回答两个过去被优化失败掩盖的问题：

1. 相同参数规模下，PatchTST 是否优于 iTransformer；
2. 保持 iTransformer 时，纯粹扩大 latent/backbone 容量是否继续改善模型。

共同配方为：32 条 outer-train optimized teacher schedules、direct N=64 imitation 1000 steps、
scaled-transport、arc 2+4、60/120/240/full curriculum、LR=`3e-5`、clip20、batch=512、N=64、
2207 train / 505 validation、seed/split seed=1337。RK4 仍为 0.5 s，监督仍为固定 2 s。
Outer-test 没有运行或查看。

## 2. 模型

| 模型 | d_model / d_ff / layers | 参数量 |
|---|---:|---:|
| iTransformer small | 256 / 512 / 3 | 2,217,729 |
| PatchTST | 256 / 512 / 3 | 2,207,745 |
| iTransformer large | 512 / 1024 / 4 | 10,601,217 |

PatchTST 与 small iTransformer 只改变 `model`。Large 与 small iTransformer 的 config 精确差异
只有 `d_model`、`d_ff`、`e_layers`；batch、LR、epoch、teacher、dynamics 和 loss 均不变。

## 3. 正式 train/validation 结果

| 指标 | iTransformer small | PatchTST | iTransformer large |
|---|---:|---:|---:|
| best / run epoch | 163 / 180 | 107 / 127 | 180 / 180 |
| selection | 27.5097 | 28.5444 | **23.2743** |
| ADE | 751.7 m | 821.4 m | **694.7 m** |
| FDE | 1039.2 m | 1056.2 m | **915.4 m** |
| cross-track p95 | 1930.2 m | 1755.0 m | **1539.8 m** |
| altitude p95 | 396.3 m | 434.7 m | **355.3 m** |
| final-time MAE | 14.35 s | 14.88 s | **13.72 s** |

### 3.1 Backbone 替换

PatchTST 的 cross-track p95 比 small iTransformer 好 9.1%，但 selection、ADE、FDE、高度和时间
均更差，并在 epoch 127 早停。因此相同规模下，PatchTST 不是当前部署基线；它说明 patch 归纳
偏置更偏向横向走廊，但没有提供更好的综合进近预测。

### 3.2 纯容量扩大

Large 相对 small iTransformer：

- selection 改善 15.40%；
- ADE 改善 7.58%；
- FDE 改善 11.91%；
- cross-track p95 改善 20.23%；
- altitude p95 改善 10.35%；
- final-time MAE 改善 4.39%。

六项全部改善，且最佳 checkpoint 位于预算最后一个 epoch，说明在 teacher 已解决一部分初始化
困难后，backbone 容量确实成为可见瓶颈。过去没有 teacher 时 `d_model=512` screening 的负结果
不能外推到当前训练配方。

## 4. 计算与稳定性

| 模型 | full epoch median | 训练 epoch wall time | 最大裁剪前梯度 | clip rate |
|---|---:|---:|---:|---:|
| iTransformer small | 5.50 s | 909.6 s / 180 epochs | 1.55e5 | 100% |
| PatchTST | 5.71 s | 619.2 s / 127 epochs | 9.33e4 | 100% |
| iTransformer large | 5.55 s | 894.8 s / 180 epochs | 1.01e5 | 100% |

Large 参数量约为 small 的 4.78 倍，但 epoch wall time没有相应增加，因为主成本仍是相同的
dynamics rollout/adjoint。它也没有解决梯度裁剪：clip 仍为 100%。最佳 epoch 推力控制点饱和率
由 small 的 23.1% 升到 large 的 38.2%（overall 7.70%→12.74%），后续继续扩大容量时应监控边界
依赖，不能无限增大模型。

## 5. 与最初基线的累计改善

相对于 reference-velocity 实验中尚未使用 teacher、尚未 state scaling、`d_model=256` 的正式基线：

| 指标 | 最初基线 | 当前最佳 large | 累计变化 |
|---|---:|---:|---:|
| selection | 31.4126 | 23.2743 | -25.9% |
| ADE | 1130.7 m | 694.7 m | -38.6% |
| FDE | 1202.9 m | 915.4 m | -23.9% |
| cross-track p95 | 3313.2 m | 1539.8 m | -53.5% |
| altitude p95 | 692.6 m | 355.3 m | -48.7% |
| final-time MAE | 20.84 s | 13.72 s | -34.1% |

## 6. 结论

- 当前开发基线更新为 large iTransformer + direct train-only teacher + scaled transport。
- PatchTST 已完成同配方实验，但不替代 iTransformer。
- Progressive-N 不叠加；它已经单独证明会退化。
- Large 的最佳点位于 epoch 180，未来可预注册更长预算或单独扩大 teacher cohort；本组实验不根据
  当前 validation 结果临时追加 epoch，以保持本轮比较预算一致。
- Outer-test 继续封存，尚未进入冻结后的最终 release。

输出目录：

- small iTransformer：`outputs/KSJC/experiments/nondimensional_transport_20260802/formal_teacher_scaled`
- PatchTST：`outputs/KSJC/experiments/backbone_capacity_20260802/patchtst_d256_l3_direct_teacher_scaled`
- large iTransformer：`outputs/KSJC/experiments/backbone_capacity_20260802/itransformer_d512_l4_direct_teacher_scaled`
