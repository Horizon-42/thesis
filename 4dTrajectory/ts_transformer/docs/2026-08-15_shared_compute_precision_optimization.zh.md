# TS 模型通用计算精度优化档案

日期：2026-08-15

适用范围：iTransformer、PatchTST；state、control、control-mixture 输出

实验边界：仅使用 outer-train/validation；未读取或发布 outer-test 预测

## 1. 问题与根因

正式 PatchTST CV 慢的主因是 GPU 网络计算量，不是文件读取。当前归一化时域配置为

\[
L=60,\quad C=6,\quad P=16,\quad S=8.
\]

末端 padding 后，每个状态通道产生

\[
N_p=1+\left\lfloor\frac{L+S-P}{S}\right\rfloor=7
\]

个 patch token。PatchTST 把 batch 重排为 \([B C,N_p,d]\)，因此每条航迹实际处理

\[
C N_p=6\times 7=42
\]

个隐藏 token；iTransformer 只处理 6 个变量 token。两者参数量接近，但 PatchTST 的激活和矩阵乘计算明显更大。

## 2. 精度边界

优化只改变 backbone 的矩阵计算精度，不改变数据、模型结构、loss、控制动力学或评估定义。

设网络为 \(f_\theta\)，网络输出为 \(\hat y\)，state loss 为 \(\mathcal L_s\)，control 动力学为 \(g\)：

\[
\hat y_{bf16}=f_\theta^{bf16}(x),
\qquad
\hat y_{32}=\operatorname{cast}_{fp32}(\hat y_{bf16}),
\]

\[
\mathcal L_s=\mathcal L_s^{fp32}(\hat y_{32},y),
\]

\[
\dot x=g^{fp64}(x,u),
\qquad
x_{k+1}=\operatorname{RK4}^{fp64}(x_k,u_k,\Delta t_k).
\]

严格边界如下：

- 模型参数和优化器状态保持 FP32。
- CUDA 上仅网络 forward 使用 BF16 autocast；CPU 自动保持 FP32。
- 网络输出在离开 autocast 后立即转换为 FP32；state objective 保持 FP32，control objective 中经过物理 rollout 的项按既有规则提升为 FP64。
- control/control-mixture 的状态、控制量、时长和物理 rollout 继续显式使用 FP64。
- BF16 不需要 FP16 的 loss scaling；不引入 `GradScaler`。
- `float32_matmul_precision=high` 允许 CUDA 对剩余 FP32 矩阵乘使用高吞吐实现；序列化配方明确记录该设置。

这是一条共享执行路径，不为不同 backbone 或输出类型维护不同实现。

## 3. 单步吞吐实验

硬件：NVIDIA GeForce RTX 4060 8 GB；PyTorch 2.11.0+cu128。所有数据来自 KMSY outer-train，未使用 validation/test 作吞吐调优。

| Backbone / 输出 | batch | FP32 highest | BF16 + high | 变化 |
|---|---:|---:|---:|---:|
| PatchTST / state | 2048 | 5,403 sample/s | 9,541 sample/s | +76.6% |
| iTransformer / state | 2048 | 25,688 sample/s | 33,709 sample/s | +31.2% |
| PatchTST / control | 512 | 662 sample/s | 676 sample/s | +2.1% |
| iTransformer / control | 512 | 691 sample/s | 706 sample/s | +2.2% |

PatchTST/state 的 CUDA reserved memory 从约 4,278 MiB 降至 2,920 MiB。control 提速小不是优化失效，而是总耗时主要来自必须保留 FP64 的物理 rollout。

## 4. 开发集质量检查

比较使用相同 KMSY outer-train/validation、相同 seed、split、batch、模型和优化器。36 epoch 检查覆盖正式 CV 的单折训练长度。

| Backbone | 精度 | val ADE (m) | val FDE (m) | endpoint (m) | ETA MAE (s) |
|---|---|---:|---:|---:|---:|
| PatchTST | FP32 | 2,832.7 | 2,793.8 | 2,613.2 | 83.2 |
| PatchTST | BF16 | 2,837.1 | 2,837.1 | 2,675.4 | 84.8 |
| iTransformer | FP32 | 2,354.9 | 2,605.2 | 1,969.4 | 63.7 |
| iTransformer | BF16 | 2,422.4 | 2,587.2 | 2,005.0 | 64.6 |

相对差异均在预先采用的 3% 容差内，所有 loss、梯度和运动学指标有限。另做 6 epoch iTransformer/control 检查，BF16 全程有限且 validation 指标未退化。该检查只用于排除数值不稳定，不替代正式 CV。

## 5. 正式决策

`run_ts_pipeline.py` 的统一正式默认值为：

```text
training_precision=bfloat16
float32_matmul_precision=high
```

底层 `TSConfig` 和直接调用的 TS CLI 仍保留 FP32/highest 默认值；这使独立工具和 CPU 使用者不会被隐式改变。正式 pipeline 总是显式传入优化配方。

精度不进入目录或前端 category 名称，因为它不是新的模型/实验模式。但它进入：

- 完整 `TSConfig`；
- CV `base_config` 和运行契约；
- checkpoint metadata；
- throughput benchmark JSON。

checkpoint metadata schema 升级，旧派生产物必须重训，不增加双读兼容路径。

## 6. 未采用的优化

- 增大 batch size：会改变每 epoch 的优化器更新次数和有效训练配方，不能当作纯执行优化直接替换。
- 修改 PatchTST patch 长度/stride：属于模型超参数和结构变化，必须通过 CV 决定，不是两个 backbone/control 通用的后端优化。
- `torch.compile`：正式网格为 45 个候选、3 folds，即 135 次短生命周期拟合；逐模型编译成本难以摊销，也增加 checkpoint/诊断复杂度。
- 降低 control rollout 精度：可能改变物理积分和梯度，是数值方法变更，不属于本次安全优化范围。

## 7. 验收要求

1. 两个 backbone、state/control 的生产 forward 和 loss 均由同一 benchmark/test 路径覆盖。
2. prediction 对象离开 autocast 后所有 tensor 均为 FP32。
3. control 动力学继续 FP64。
4. 精度配方变化必须使 CV/checkpoint 复用检查失败。
5. outer-test 在用户冻结实验并明确要求 release 前保持关闭。
