# 单 QFU 与进近几何聚类实验记录

日期：2026-08-02

## 1. 实验问题

本轮实验依次回答三个问题：

1. A：使用 KSJC 所有 QFU 的训练数据，作为当前基线。
2. B：只使用 RWY 30L 航迹，观察单 QFU 是否降低意图混杂。
3. C：只使用 RWY 30L 的主进近几何簇，观察进一步消除进近路径多模态是否有帮助。

三个实验使用完全相同的 iTransformer、control 输出、2+4 arc-length loss、180 epochs、2 s 监督网格和 0.5 s RK4 rollout 步长。只改变训练/验证 cohort，不修改模型质量相关设置。

所有模型选择均使用 train/validation；未加载或评估 outer-test 航迹。最终比较固定为 C cohort 中相同的 322 条 validation 航迹，不能用各实验难度不同的原生 validation 集直接排名。

Development cohort 不定义 test roster。使用该 roster 训练的 checkpoint 被明确标记为仅开发用途，不能执行 `freeze-test` 或消耗一次性 final-test release；最终发布必须在实验冻结后使用具有明确 outer-test population 的配方重新训练。

## 2. 解耦设计

聚类实现位于独立子包 `ts_transformer/approach_clustering`：

- `features.py`：把 anchor 后的水平路径按弧长重采样，构造与飞行速度和总时长无关的几何特征。
- `model.py`：只包含标准化、PCA、K-means 和 silhouette 选择。
- `artifacts.py`：写出冻结聚类模型、QFU cohort、主簇 cohort 和摘要。
- `cli.py`：负责 train-only 拟合和 validation 冻结分配。
- `evaluation.py`：在同一 validation cohort 上比较 checkpoint。
- `development_cohorts.py`：训练入口通用的显式 train/validation roster，不把聚类逻辑写进训练器。

聚类标签由真实未来路径构造，因此 C 是“同质意图下的模型容量消融”，不是可直接部署的路由器。部署时若要使用簇，需要另行用历史观测预测意图，不能查看未来路径。

## 3. 聚类结果

RWY 30L cohort 包含 1810 条 train 和 411 条 validation 航迹。PCA 和 K-means 只在 1810 条 train 航迹上拟合，validation 使用冻结模型分配。

| K | Train 簇大小 | Silhouette |
|---:|---|---:|
| 2 | 1487 / 323 | 0.8057 |
| 3 | 1477 / 216 / 117 | 0.7844 |
| 4 | 1475 / 93 / 27 / 215 | 0.7649 |

按最高 silhouette 选择 K=2。主簇为 cluster 0，包含 1487 条 train 和 322 条 validation 航迹。

## 4. 三个训练实验

| 实验 | 训练范围 | Train | Validation | 最佳固定-anchor弧长选择值 |
|---|---|---:|---:|---:|
| A | KSJC 全部 QFU | 2207 | 505 | 31.41 |
| B | RWY 30L 全部进近 | 1810 | 411 | 26.29 |
| C | RWY 30L 主进近簇 | 1487 | 322 | 18.99 |

选择值随 cohort 收窄显著下降，但不同训练集拥有不同 normalizer 和 validation 分布，不能据此宣称 B/C 优于 A。

## 5. 同一 validation cohort 的公平比较

以下三个 checkpoint 全部重新评估于完全相同的 322 条 C-validation 航迹。ADE/FDE 使用 64 点共同物理时间网格。

| 模型 | ADE (m) | FDE (m) | 终端速度误差 (m/s) | 最终时间 MAE (s) |
|---|---:|---:|---:|---:|
| A：全部 QFU | **491.6** | **713.8** | 16.06 | 15.63 |
| B：单 QFU 30L | 516.0 | 924.0 | 15.41 | **12.88** |
| C：30L 主簇 | 521.2 | 977.7 | **15.06** | 13.49 |

相对 A：

- B 的 ADE 增加 5.0%，FDE 增加 29.5%；终端速度误差降低 4.1%，时间 MAE 降低 17.6%。
- C 的 ADE 增加 6.0%，FDE 增加 37.0%；终端速度误差降低 6.2%，时间 MAE 降低 13.7%。

弧长几何分项进一步说明了取舍：

| 模型 | 弧长距离均值 (m) | 水平速度 MAE (m/s) | 垂直速度 MAE (m/s) | 弧长终端位置 (m) | 终端横向位置 (m) |
|---|---:|---:|---:|---:|---:|
| A | 480.9 | 12.30 | 5.99 | 814.5 | 109.1 |
| B | **466.8** | 10.22 | 6.28 | **808.1** | **90.8** |
| C | 499.7 | **9.67** | **5.70** | 898.6 | 117.9 |

## 6. 结论

单 QFU 和主簇训练让任务表面上更容易，并改善速度、时间或局部跑道对齐分项，但没有提高相同目标进近族上的整体 4D 预测。当前公平结果中 A 最好。

主要原因不是“意图完全不重要”，而是硬筛选同时删除了 18% 到 33% 的训练样本和有用的动力学变化。A 从其他 QFU 学到的共享减速、下滑和控制规律带来的正迁移，大于意图混杂的负面影响。B/C 的主要退化集中在共同物理时钟的 FDE 和沿航向终端位置，而不是局部横向路径形状。

因此，当前不应把单 QFU 或单簇硬筛选替换为新基线。若继续研究意图，优先保留全部训练数据，再把 QFU/进近簇作为条件输入、辅助分类目标或解耦路由信号；先验证“共享 backbone + 条件化”的收益，再考虑独立 expert。聚类仍可作为 train-only 标签生成器复用。

## 7. 当前 CLI

聚类构建与同 cohort 比较现已统一到 TS CLI：

```bash
TS=4dTrajectory/ts_transformer/__main__.py
DATA=trajectory_data_process/outputs/harvest/KSJC/arrivals/manifest.json
EXP=4dTrajectory/outputs/KSJC/experiments/approach_intent_20260802

conda run -n aeroviz python "$TS" approach-cohorts build \
  --data "$DATA" \
  --recipe "$EXP/a_all_qfu/history.json" \
  --runway 30L \
  --output-dir "$EXP/clustering"

conda run -n aeroviz python "$TS" approach-cohorts compare \
  --data "$DATA" \
  --cohort "$EXP/clustering/dominant_cluster_cohort.json" \
  --checkpoint "$EXP/a_all_qfu/checkpoint.pt" --label A-all-QFU \
  --checkpoint "$EXP/b_qfu_30l/checkpoint.pt" --label B-QFU-30L \
  --checkpoint "$EXP/c_dominant_cluster/checkpoint.pt" --label C-dominant-cluster \
  --output "$EXP/comparison.json"
```

开发 cohort 仍通过训练命令的 `--development-cohort` 参数使用。若只处理聚类子包，亦可在
`4dTrajectory/ts_transformer/` 下运行 `python -m approach_clustering build|compare`。

## 8. 可复现实验产物

- 聚类摘要：`4dTrajectory/outputs/KSJC/experiments/approach_intent_20260802/clustering/summary.json`
- 冻结聚类模型与分配：`4dTrajectory/outputs/KSJC/experiments/approach_intent_20260802/clustering/approach_clusters.json`
- QFU cohort：`4dTrajectory/outputs/KSJC/experiments/approach_intent_20260802/clustering/qfu_cohort.json`
- 主簇 cohort：`4dTrajectory/outputs/KSJC/experiments/approach_intent_20260802/clustering/dominant_cluster_cohort.json`
- A/B/C checkpoint：`4dTrajectory/outputs/KSJC/experiments/approach_intent_20260802/{a_all_qfu,b_qfu_30l,c_dominant_cluster}/checkpoint.pt`
- 同 cohort 完整分项：`4dTrajectory/outputs/KSJC/experiments/approach_intent_20260802/comparison.json`

所有聚类和比较产物均记录 `outer_test_tracks_opened: false`，cohort 与 checkpoint 同时记录 SHA-256，便于复核数据谱系。
