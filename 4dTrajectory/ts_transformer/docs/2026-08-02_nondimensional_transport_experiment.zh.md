# Transport state 无量纲化实验（2026-08-02）

## 1. 目的与边界

本实验检验 `transport-chart-velocity` 长时 single-shooting 的内部 state 尺度是否导致较差的
Jacobian/adjoint 数值条件。新 backend 为 `scaled-transport-chart-velocity`，内部状态采用固定、
与数据集统计无关的参考单位：

```text
E,N : 10000 m
U   : 1000 m
VE,VN: 100 m/s
VU  : 10 m/s
mass: 100000 kg
```

它只做对角线性坐标变换。物理 RHS、WGS84 transport term、控制和气动参数、物理时间、RK4 0.5 s
以及 fixed-2s supervision 均不改变。训练和推理边界恢复为原来的物理 chart state，因此模型、loss、
数据集和公开输出 contract 不感知内部缩放。

所有实验只使用 outer-train 和 validation；outer-test 预测及指标未运行、未查看。

## 2. 数值 contract

- Float64 state scale/unscale 往返成立。
- 原 backend 与 scaled backend 的 20 s RK4 rollout 恢复到物理 state 后，在约 `1e-9` 绝对误差
  范围内一致。
- 对相同物理 channel loss，control 和 duration 梯度在 float64 下逐值一致。
- 两个 backbone 的一批完整可微 control loss smoke test 均通过。

这证明新模式没有借修改步长、监督网格或物理模型获得优势；float32 长链中若出现差异，只来自内部
数值路径和由此进入的不同优化 basin。

## 3. 单航迹过拟合

同一条 outer-train 航迹 `KSJC:DAL1260_30L_a05f31_20260716T224959Z`，iTransformer、N=64、
arc 2+4、300 epochs、LR=`1e-4`、seed=1337：

| backend | fixed-2s ADE | last-complete-2s FDE | terminal distance | common-grid ADE |
|---|---:|---:|---:|---:|
| physical transport | 309.4 m | 104.9 m | 150.1 m | 307.4 m |
| scaled transport | 260.0 m | 40.3 m | 13.9 m | 256.9 m |

Scaled 单航迹进入了更好的终端 basin，但垂直 arc 误差没有同步全面改善。因此这里只作为进行正式
validation 对照的理由，不把单航迹结果解释为泛化提升。

## 4. 正式 train/validation 对照

对照基础为 teacher-only 新基线：32 条 train-only optimized schedules 做 1000 步 direct N=64
imitation，然后使用完整 2207 train / 505 validation 航迹训练 180 epochs。两组 config 精确比较后，
唯一不同字段是 `control_dynamics_backend`。

| 指标 | physical transport | scaled transport | 相对变化 |
|---|---:|---:|---:|
| selection | 27.6277 | 27.5097 | -0.43% |
| ADE | 771.0 m | 751.7 m | -2.51% |
| FDE | 1040.4 m | 1039.2 m | -0.11% |
| cross-track p95 | 2039.6 m | 1930.2 m | -5.36% |
| altitude p95 | 364.7 m | 396.3 m | +8.68% |
| final-time MAE | 14.19 s | 14.35 s | +1.15% |

Scaled 在预注册 selection、ADE、FDE 和横向长尾上小幅改善，但高度和时间退化，属于 Pareto 点。
提升幅度远小于 oracle-teacher 初始化本身，不能称为全面的新能力跃升。

## 5. 梯度诊断

| 诊断 | physical transport | scaled transport |
|---|---:|---:|
| epoch 平均 clip 触发率 | 100% | 100% |
| 全程最大裁剪前总梯度 | 4.46e4 | 1.55e5 |
| 最佳 epoch 裁剪前总梯度均值 | 1012 | 2251 |
| 最佳 epoch control-head 梯度均值 | 1003 | 2237 |
| 最佳 epoch control saturation | 8.13% | 7.70% |

核心机制假设没有成立：无量纲化没有缓解持续 clip，也没有缩小参数梯度；最大值和最佳 epoch 均值
反而更大。原因是线性 state 变换在精确链式法则下会被输出到物理 loss 的反向缩放抵消，它能改变
中间表示和浮点误差，却不会从根本上改变 control 参数到物理 loss 的真实敏感度。

## 6. 结论

- 保留 scaled backend 作为解耦的实验模式和一个小幅 validation Pareto 点。
- 不把它描述为 Jacobian/梯度爆炸的解决方案。
- 后续 progressive-N 实验以 scaled/direct-teacher 结果作为严格直接对照，因为其预注册 selection
  略好；最终推荐仍需同时报告高度和时间退化。
- 不再围绕参考单位做 validation 调参，避免把 7 个 scale 变成新的过拟合超参数。

输出目录：

- `outputs/KSJC/experiments/nondimensional_transport_20260802/single_physical`
- `outputs/KSJC/experiments/nondimensional_transport_20260802/single_scaled`
- `outputs/KSJC/experiments/nondimensional_transport_20260802/formal_teacher_scaled`
