# Plan

为 `ts_transformer` 增加一个可复现的多机场联合训练模式，并在最终训练前加入严格隔离测试集的交叉验证阶段。实现沿用现有按航班切分和 checkpoint 自包含原则，同时加入机场/航班均衡采样、跑道对齐坐标消融入口、GPU 显存探测式 batch-size 自动选择，以及逐机场评估产物。

## Scope

- In: 多 manifest 数据集、`per-airport`/`pooled` 两种训练模式、机场感知身份与 provenance、outer split、outer-train 内 K-fold 超参数搜索、GPU batch 自动探测、联合训练均衡采样、固定评估 anchor、逐机场预测与指标、测试和文档。
- Out: 使用 outer-test 选择模型或超参数、自动发布/提交训练好的大体积 checkpoint、修改 vendored iTransformer/PatchTST、把稀有跑道强制重采样到等频、声称未见机场泛化而不运行 leave-one-airport-out。

## Action items

[x] 将一个或多个 `arrivals/manifest.json` 规范化为按机场排序的数据集描述，聚合加载航班，并把 split/checkpoint 身份扩展为 `airport + flight_key`，同时保持逐机场导出文件继续使用原始 `flight_key`。

[x] 升级 arrival-data provenance 和轻量 checkpoint metadata，使其记录全部机场 manifest 与源记录摘要；训练时要求完全匹配，逐机场预测时只允许验证过的 checkpoint 数据子集。

[x] 保留“按航班、绝不按窗口”切分，先锁定机场感知的 outer train/val/test；让 K-fold 只消费 outer-train，最终训练使用 outer-val 早停，并在最终 checkpoint 固定后才允许 outer-test 预测与评分。

[x] 增加确定性的机场分层 K-fold 与超参数候选生成，默认搜索学习率、模型宽度、编码层数、注意力头数、dropout 和 weight decay；持久化每折分数、均值/方差、数据 split 摘要和 `best_config.json`。

[x] 为 pooled 训练增加“机场 → 航班 → anchor”分层采样和固定每 epoch 样本预算，避免大机场、长轨迹及高度重叠窗口主导 loss；验证和测试使用与实际整段预测一致的每航班最早有效 anchor，并以机场 macro validation loss 选模。

[x] 增加 GPU batch-size 自动探测：在目标模型、序列长度和 horizon 下执行隔离的前向/反向探测，按可用显存留安全余量选择 2 的幂 batch，OOM 时自动减半；显式 `--batch-size` 始终覆盖自动值，CPU 使用保守默认值。

[x] 扩展 `ts_transformer` CLI：支持重复 `--data`、`cross-validate` 子命令、读取最优配置、采样策略、epoch 样本预算、评估 anchor 策略及 auto batch，并把最终解析后的训练配方写入 checkpoint/history。

[x] 重构 `run_ts_pipeline.py`：`per-airport` 保留每机场独立模型；`pooled` 对选中机场只执行一次 `CV → final train`，然后使用同一 checkpoint 逐机场运行 outer-test prediction、evaluation 和 CZML，且 `--skip-cv`/`--skip-train` 的复用检查覆盖全部 manifest。

[x] 为跑道对齐坐标保留显式配置和转换边界，使 pooled ENU baseline 与 runway-aligned ablation 都能独立训练且 checkpoint 不可混用；不修改 vendored 模型。

[x] 添加数据身份、分层 split、CV 无泄漏、均衡 sampler、provenance 子集校验、auto-batch 回退、命令规划与 pooled 产物路径测试；运行 `conda run -n aeroviz python -m pytest 4dTrajectory/ts_transformer/tests trajectory_data_process/tests/test_ts_pipeline.py -q`，并对两种模式执行 `run_ts_pipeline.py --dry-run`。

[x] 增加 raw-output smoothness/kinematic metrics：持久化位置差分速度与预测速度的 RMSE、航向一致性 p95、转弯率 p95、加速度 p95 和 jerk p95；指标基于未经样条、滤波或 CZML 插值的模型原始输出，支持显式非均匀 segment durations，并已用于比较 N=64/128/256。逐航班指标及 fleet median/mean/p95/max 贯穿 validation history、CV、prediction summary 和正式消融报告；后续 controls rollout 复用同一接口。

## Open questions

- 无阻塞问题；默认采用 3-fold、三个显式 CV 参数的完整固定网格和较短 CV epoch，全部通过 CLI 可覆盖。
- leave-one-airport-out 作为独立泛化实验保留，不与默认“已知机场上的新航班”outer-test 混合。
- 默认 pooled 训练启用机场/航班均衡采样；`per-airport` 保留原有 all-window 采样，便于与历史结果对照。
