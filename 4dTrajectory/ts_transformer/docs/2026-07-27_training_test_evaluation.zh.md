# 2026-07-27 归一化时间模型训练集/测试集评估

> 历史结果说明：本文记录的是 2026-07-27 当时的实验与旧 evaluation gate。
> 当前 runtime 已使用 `terminal-approach-evaluation-v4`：横向为跑道/程序特定界限，
> 垂直为 published-TCH path 的 `[-22,+22] m`。本文的旧 terminal pass 数量必须在
> 重新发布 prediction 后更新，不能与 v4 observed pass rate 直接比较。

## 1. 评估对象

本报告评估当前 pooled ENU checkpoint：

- 机场：KMSY、KRDU、KSJC、KSMF、KSTL
- 模型：iTransformer、PatchTST
- 输入：`L=60`，`dt=2 s`，即 120 s 历史航迹
- 输出：`N=256` 个归一化进度段和 `final_time_s`
- anchor：固定为 `L-1`
- loss：`final_time=1`、`kinematic=10`、`terminal=0.02`
- batch size：2048
- 最多训练 50 epoch

数据量为 13,807 条 train、2,951 条 validation 和 2,983 条 held-out test 航迹。

## 2. 指标口径

以下两套指标不能直接混用：

1. `history.json` 中的 loss 和 validation ADE/FDE 在归一化进度监督网格上计算，适合判断优化过程和 checkpoint 选择。
2. 发布目录 `summary.json` 中的 train/test ADE、FDE 和到达时间误差，均按预测结果与实际观测航迹的物理时间重叠段计算。本文只用这套同口径指标比较 train 与 test。

终点 lateral/vertical 指标来自独立的 `evaluation_report.json`，表示最终状态相对跑道目标门的偏差。本节以下原始数字使用当时的旧 gate；当前标准见文首说明和 `evaluation/FINAL_APPROACH_VERDICT_STANDARD.md`。

## 3. 训练过程

| 模型 | 参数量 | 最优 epoch | 最优 validation loss | 对应 train loss | validation ADE | validation FDE | validation time MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| iTransformer | 1,755,905 | 50 | 0.4142 | 0.3595 | 3510.0 m | 3324.7 m | 71.2 s |
| PatchTST | 2,139,137 | 48 | 0.5540 | 0.4807 | 4763.6 m | 3133.8 m | 72.2 s |

iTransformer 在 epoch 41--50 的 validation loss 从 0.4425 连续下降到 0.4142，最优点正好位于训练上限，尚未出现验证集恶化。PatchTST 在 epoch 48 达到 0.5540，随后两个 epoch 维持在 0.5548--0.5568，已接近平台期。

因此，当前 iTransformer 可能仍受 epoch 上限约束；PatchTST 单纯增加 epoch 的预期收益较小。

## 4. 同口径 train/test 对比

### 4.1 航迹与时间误差

| 模型 | split | 航班数 | ADE | FDE | `final_time_s` MAE | `final_time_s` bias |
|---|---|---:|---:|---:|---:|---:|
| iTransformer | train（in-sample） | 13,807 | 3473.1 m | 3784.2 m | 71.4 s | -8.9 s |
| iTransformer | test（held-out） | 2,983 | 3399.0 m | 3794.1 m | 69.9 s | -8.0 s |
| PatchTST | train（in-sample） | 13,807 | 4553.5 m | 2993.9 m | 72.6 s | -3.2 s |
| PatchTST | test（held-out） | 2,983 | 4477.1 m | 3019.7 m | 71.2 s | -2.4 s |

test 相对 train 的变化很小：

- iTransformer：ADE -2.1%，FDE +0.3%，时间 MAE -2.1%。
- PatchTST：ADE -1.7%，FDE +0.9%，时间 MAE -1.8%。

test 并未系统性差于 train。这说明当前误差主要不是由 train-to-test 泛化落差造成，而是训练分布内也没有拟合出足够准确的完整航迹。更符合“高偏差、目标权衡或模型/表征能力不足”，而不是典型过拟合。

### 4.2 跑道终点误差

| 模型 | split | terminal lateral mean / p95 | terminal vertical absolute mean / p95 | 严格成功率 |
|---|---|---:|---:|---:|
| iTransformer | train | 3058.1 / 6438.8 m | 186.9 / 438.5 m | 0 / 13,807 = 0% |
| iTransformer | test | 3084.7 / 6575.2 m | 186.4 / 439.5 m | 0 / 2,983 = 0% |
| PatchTST | train | 3845.8 / 5753.5 m | 435.8 / 515.8 m | 0 / 13,807 = 0% |
| PatchTST | test | 3849.9 / 5734.7 m | 436.6 / 513.0 m | 0 / 2,983 = 0% |

终点误差在 train/test 上同样几乎一致，但远大于严格跑道门限。当前模型可以对所有样本生成解，却不能可靠落到目标门内；因此 `solve rate=100%` 不能解释为轨迹预测成功。

### 4.3 机场差异（ADE）

| 模型 | 机场 | train ADE | test ADE | test 相对 train |
|---|---|---:|---:|---:|
| iTransformer | KMSY | 3959.1 m | 4154.5 m | +4.9% |
| iTransformer | KRDU | 4290.2 m | 4072.3 m | -5.1% |
| iTransformer | KSJC | 2144.2 m | 2134.4 m | -0.5% |
| iTransformer | KSMF | 3141.4 m | 3227.3 m | +2.7% |
| iTransformer | KSTL | 3637.2 m | 3666.6 m | +0.8% |
| PatchTST | KMSY | 5249.5 m | 5411.7 m | +3.1% |
| PatchTST | KRDU | 4454.1 m | 4169.3 m | -6.4% |
| PatchTST | KSJC | 3634.4 m | 3648.2 m | +0.4% |
| PatchTST | KSMF | 4571.8 m | 4582.1 m | +0.2% |
| PatchTST | KSTL | 5389.0 m | 5485.9 m | +1.8% |

机场内的 train/test 差异也都较小。KSJC 明显最容易；KRDU 的到达时间最难，两个模型的 test time MAE 都约为 139 s。这更像机场/程序分布差异，而不是测试集泄漏或单一 split 异常。

## 5. 模型比较与结论

held-out test 上：

- iTransformer 的 ADE 比 PatchTST 低 1078.1 m（24.1%），跑道终点 lateral mean 低 19.9%，vertical mean absolute error 低 57.3%。
- PatchTST 的 FDE 比 iTransformer 低 774.4 m（20.4%），但它的整段 ADE 和终点门偏差更差。这表示 PatchTST 更偏向末段收束，不能据此认为完整航迹更好。
- 两个模型的到达时间误差接近，iTransformer 略好约 1.4 s。
- 两个模型严格成功率均为 0%，当前结果还不能满足跑道终端约束。

综合来看，iTransformer 是当前更好的基线，但现有模型仍处于明显欠拟合或 loss 权衡不当状态。优先级应是：

1. 保留独立 held-out test，不用 test 参与超参数选择。
2. 对 iTransformer 增加 epoch/patience 做一次受控实验，因为 validation loss 到 epoch 50 仍在下降。
3. 对 `kinematic=10` 做权重消融，并同时看 ADE、终点误差和轨迹平滑性；高平滑约束不能以数公里位置偏差为代价。
4. 按机场/跑道报告结果，重点检查 KRDU 的到达时间和 KMSY/KSTL 的空间误差。

## 6. 前端发布隔离

train 和 test 已分别发布，且不会共用目录或分类：

- 训练集目录/category：`..._normalized_time_train`，显式元数据 `datasetSplit: "train"`
- 测试集目录/category：`..._normalized_time_test`，显式元数据 `datasetSplit: "test"`
- 前端选择框分别显示 `Held-out test results` 和 `Training results (in-sample)` 两个分组。
- 显示标签分别以 `Test split (held-out)` 和 `Training split (in-sample)` 开头。
- CZML 发布命令会校验 `--dataset-split` 与预测 `summary.json` 的 `split` 一致，不一致时直接拒绝发布。

默认运行下面命令会复用 checkpoint，并分别生成/发布 train 和 test：

```bash
conda run -n aeroviz python run_ts_pipeline.py \
  --models itransformer,patchtst \
  --skip-train
```

若只想刷新某一个 split，可显式使用 `--split train` 或 `--split test`。
