# iTransformer 小样本拟合能力诊断（2026-07-27）

## 结论

当前 iTransformer **具备基本拟合能力**，但当前正式训练配方对多机场、多航迹数据存在明显欠拟合。

- 单航迹、关闭 dropout 后，即使保留 `kinematic_consistency_loss_weight=10`，训练集回放 ADE 也能降至 **7.3 m**，终点水平误差降至 **0.8 m**。
- 同一单航迹去掉 kinematic loss 后，ADE 可降至 **0.15 m**，说明 `N=256` 的状态输出头本身可以逼近目标航迹。
- 但五机场各 32 条、共 160 条固定训练航迹，在当前 `dropout=0.1` 配方下：
  - `kinematic=10` 的训练集回放 ADE 仍为 **1725.3 m**；
  - `kinematic=0` 的训练集回放 ADE 为 **730.1 m**。

因此，当前问题不是“模型完全不会拟合”，也不能靠增加更多数据直接解决。更准确的判断是：

1. 模型能够记住单航迹；
2. 当前损失权重和正则化会显著降低多航迹拟合速度与拟合精度；
3. 在继续扩大数据或正式训练前，应先降低 kinematic 权重，并联合检查 dropout、模型容量和多机场条件表示。

## 数据恢复

实验前，派生的 arrivals 和 scenario inputs 已被清理。使用以下命令从保留的 harvested tracks 重建：

```bash
conda run -n aeroviz python prepare_scenario_inputs.py
```

这一步没有重新下载原始数据。重建后五机场 arrivals manifest 共登记 19742 条航迹，构造出 19741 条可用 series，1 条因不足 120 s 被跳过。

## 实验协议

实验脚本：[`run_ts_overfit_diagnostic.py`](../../../run_ts_overfit_diagnostic.py)

共同设置：

| 项目 | 设置 |
|---|---|
| 模型 | iTransformer |
| 坐标 | ENU |
| 历史输入 | `L=60`，即 120 s |
| 输出 | `N=256` normalized-progress segments + `final_time_s` |
| anchor | 固定 `L-1` |
| terminal loss weight | 0.02 |
| 数据来源 | 只从固定 outer-train split 抽样 |
| checkpoint 选择 | 在同一训练样本上关闭 dropout 回放 |
| 评价性质 | 训练集记忆能力，不是泛化能力 |

同一批样本同时用于梯度更新和回放评价是有意设计的。该实验只回答“模型能否把已见样本拟合下来”，不能用于报告 validation/test 性能。

### 160 航迹生产配方检查

- 五个机场各 32 条，共 160 条；
- `dropout=0.1`；
- 500 epoch，patience 100；
- 对比 kinematic weight 10 和 0；
- 样本 SHA-256：`b6a650c42ac913b7ad5b0a8c590f1da4d518991b6540f172ea4e213c3dfdb985`。

```bash
conda run -n aeroviz python run_ts_overfit_diagnostic.py \
  --epochs 500 \
  --patience 100 \
  --output-dir 4dTrajectory/outputs/POOLED/ts_itransformer_small_sample_overfit_500
```

### 单航迹结构容量检查

- 仅使用 KSJC outer-train 中一条固定航迹；
- `dropout=0`，排除训练正则化的干扰；
- 1000 epoch，patience 200；
- 对比 kinematic weight 10 和 0；
- 航迹：`KSJC:ASA956_30L_a1f7fb_20260719T014743Z`；
- 样本 SHA-256：`177a9e8649b4d6356c11c938f45ebd86df888e17b4f6f2cc8fec3329b208e283`。

```bash
conda run -n aeroviz python run_ts_overfit_diagnostic.py \
  --airports KSJC \
  --samples-per-airport 1 \
  --epochs 1000 \
  --patience 200 \
  --batch-size 1 \
  --dropout 0 \
  --output-dir 4dTrajectory/outputs/KSJC/ts_itransformer_single_flight_overfit_1000
```

## 结果

### 160 条训练航迹

| Kinematic weight | Best epoch | Replay loss | ADE (m) | FDE (m) | Time MAE (s) | 终点水平误差 (m) | 终点垂直绝对误差 (m) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 497 | 0.171860 | 1725.3 | 1877.0 | 16.8 | 1636.6 | 76.5 |
| 0 | 289 | 0.017375 | 730.1 | 703.4 | 16.1 | 466.2 | 29.0 |

相同训练样本上，`kinematic=10` 相比 `kinematic=0`：

- 总 replay loss 高约 **9.9 倍**；
- ADE 高约 **2.36 倍**；
- FDE 高约 **2.67 倍**；
- 终点水平误差高约 **3.51 倍**；
- final-time MAE 几乎没有改善。

把训练从 200 epoch 延长到 500 epoch 后：

- `kinematic=10` 的 ADE 从 1993.9 m 降至 1725.3 m，仅再改善约 13.5%；
- `kinematic=0` 的 ADE 从 859.3 m 降至 730.1 m，并在 epoch 289 达到最好值，epoch 389 early-stop。

所以单纯继续增加 epoch 收益已经很低，尤其不能解释两种 loss 设置之间的大幅差距。

### 单航迹、无 dropout

| Kinematic weight | Best epoch | Replay loss | ADE (m) | FDE (m) | Time MAE (s) | 终点水平误差 (m) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1000 | 0.00000409 | 7.316 | 0.819 | 0.000 | 0.798 |
| 0 | 522 | 4.78e-14 | 0.146 | 0.018 | 0.000 | 0.018 |

该结果排除了以下结构性怀疑：

- `N=256` 输出维度并没有让输出头天然无法拟合；
- normalized-progress target 并非完全不可学习；
- `final_time_s` head 可以拟合训练目标；
- kinematic target 与状态 target 不是严格数学矛盾，因为权重 10 最终也能把单航迹拟合到米级。

但是，权重 10 达到的误差比权重 0 大得多，而且收敛明显更慢。这说明它在多航迹训练中很可能形成了强烈的梯度竞争或过度约束。

## 结果边界

- 160 条实验使用 `dropout=0.1`，测的是当前生产训练配方的有效拟合能力。
- 单航迹实验使用 `dropout=0`，测的是网络和输出头的结构容量。
- 不同实验会分别重新拟合 normalizer，因此不能只横向比较 normalized loss；物理单位 ADE/FDE 更有意义。
- `kinematic=0` 的位置误差更低，不代表其航迹更平滑或更可飞。该实验尚未计算 raw-output smoothness、速度一致性和 flyability 指标。
- 所有指标都是已见训练样本上的回放，不代表 validation/test 泛化。

## 下一步建议

优先做一个受控的中间权重实验，而不是直接把正式配置设为 0：

```text
kinematic_consistency_loss_weight = 0.1, 0.3, 1.0, 3.0
```

每个候选同时记录：

1. outer-validation ADE/FDE 和 terminal error；
2. raw-output position/velocity consistency residual；
3. 相邻段速度、加速度、曲率或转弯率的 p95；
4. flyability；
5. train-validation gap。

如果中间权重仍无法降低训练集 ADE，再比较 `dropout=0`、`0.05`、`0.1`，并测试更大的 `d_model`/`d_ff`。在训练误差仍为数百米时，增加更多数据通常只会改善泛化覆盖，不会修复当前拟合瓶颈。

## 产物

- 160 条实验汇总：[`plots/index.md`](../../outputs/POOLED/ts_itransformer_small_sample_overfit_500/plots/index.md)
- 160 条完整结果：[`overfit_diagnostic.json`](../../outputs/POOLED/ts_itransformer_small_sample_overfit_500/overfit_diagnostic.json)
- 单航迹实验汇总：[`plots/index.md`](../../outputs/KSJC/ts_itransformer_single_flight_overfit_1000/plots/index.md)
- 单航迹完整结果：[`overfit_diagnostic.json`](../../outputs/KSJC/ts_itransformer_single_flight_overfit_1000/overfit_diagnostic.json)
