# Plan

日期：2026-07-30  
状态：**control-mixture 方向暂停；以下事项全部延期到明确恢复该方向之后执行**

本轮 `K=3` 验证集筛选没有通过阶段门。当前决定是停止追加 seed、expert 数量和标量超参数实验，保留现有产物作为不可覆盖的失败证据；outer-test 继续封存。在重新训练之前，先把 WTA 责任分配、selector 路由、物理时间目标和 duration 约束变成可观测、可验证的独立模块。

现有结论以共同时钟 validation 报告为准：

| 模型/选择方式 | ADE (m) | FDE (m) | final-time MAE (s) | 结论 |
| --- | ---: | ---: | ---: | --- |
| single-control baseline | 2634.8 | 4399.0 | 110.7 | 当前保留基线 |
| K=3 deployable selector | 6728.4 | 9587.2 | 78.8 | ADE `+155.4%`，不通过 |
| K=3 oracle minADE | 4991.3 | 7894.7 | 不适用 | 仅表示候选集覆盖，ADE 仍 `+89.4%` |

同时，deployable selector 对 2167 个 validation 航班全部选择 expert 0；oracle minADE 则分别使用 expert 0/1/2 的 45.18%/0.88%/53.95%。这证明当前可部署路由已经坍缩，但现有日志没有记录逐 epoch 的训练 WTA winner 分布，因此“训练责任分配已经坍缩”只能作为待验证机制，不能当作已证事实。候选轨迹平均两两差异约 15.6 km，说明存在多样性，但这种多样性没有转化为正确候选。

K=3 的 duration 分布也出现明显异常：`min=0.0011 s`、`p1=0.0122 s`、`median=0.0381 s`、`p99=164.6 s`、`max=480.6 s`，jerk p95 为 `404.1 m/s³`。训练历史中的 native、按预测时钟对齐的 validation ADE `212.8 m` 因而不能用于宣布胜负；模型可以通过扭曲时间分配得到很低的 native ADE，正式比较仍必须使用统一物理时间网格。

审计入口：

- [K=3 共同时钟比较报告](../../outputs/POOLED/experiments/openap_direct_20260730_control_mixture/comparisons/stage_a_k3_seed1337_retry1/report.html)
- [K=3 mixture 专项诊断](../../outputs/POOLED/experiments/openap_direct_20260730_control_mixture/diagnostics/stage_a_k3_seed1337_retry1/report.json)
- [成功 run 的训练历史](../../outputs/POOLED/experiments/openap_direct_20260730_control_mixture/stage_a_k3/itr_control_mixture_k3_n64_b512_lr3e-4_seed1337_retry1/history.json)
- [成功 run 的实验清单](../../outputs/POOLED/experiments/openap_direct_20260730_control_mixture/stage_a_k3/itr_control_mixture_k3_n64_b512_lr3e-4_seed1337_retry1/experiment_manifest.json)
- [首次失败 run 的审计清单](../../outputs/POOLED/experiments/openap_direct_20260730_control_mixture/stage_a_k3/itr_control_mixture_k3_n64_b512_lr3e-4_seed1337/experiment_manifest.json)

## Scope

- In:
  - 仅使用既有 train/validation split 的离线诊断、损失设计、路由设计和恢复后的最小筛选。
  - 保持 single-control baseline、`split_seed=1337`、OpenAP-direct 数据筛选和 common-grid 比较口径不变。
  - 在再次训练前补齐 WTA、selector、各 expert 和 duration 的可观测性与单元测试。
  - 将 control-mixture 保持为可开关、与 single-control 解耦的模式。
- Out:
  - 暂停期间不启动训练，不追加 `K`、seed、模型宽度或 loss-weight sweep。
  - 不读取、预测或发布 outer-test；不删除、重置或绕过 `test_release.json` 审计机制。
  - 不把 native clock-aligned ADE、validation oracle minADE 或候选多样性当作可部署收益。
  - 不直接重复已经失败的 observed-clock loss 方案；该方案曾改善时间误差但显著恶化 common-grid ADE，重新设计必须解释与它的实质差异。

## Action items

[ ] 恢复实验时，先冻结并复核本页列出的 baseline checkpoint、K=3 checkpoint、validation split SHA-256 和两个报告；所有新 run 使用新目录，不覆盖当前失败证据。

[ ] 在任何新训练之前补齐逐 epoch、train/validation 分开的路由日志：WTA winner 占比、selector 预测占比、selector-vs-WTA confusion matrix、selector entropy、每个 expert 的几何/终点/时间/正则损失及梯度有效性。先用现有 checkpoint 验证统计链路，不把诊断运行计作候选实验。

[ ] 用既有 train/validation 数据做 objective-attribution 诊断：对同一批候选同时计算当前 WTA composite loss、统一物理时间网格 ADE/FDE 和 final-time loss，量化“训练 winner”与“common-grid oracle winner”的一致率，并单独检查极短/极长 duration 对 winner 的影响。

[ ] 将 expert 责任目标改为与部署指标一致的物理时间目标，同时保留独立的 final-time 监督。优先比较可微 soft responsibility 与带容量约束的 balanced assignment；在小规模、固定样本诊断中确认目标对航迹几何和时间都敏感后，才进入完整训练。不得把简单替换为 observed-clock loss 当作新方案。

[ ] 为 duration head 增加独立、可测试的物理约束。约束应基于 train 数据和 single-control baseline 的 duration 分布预先确定，至少覆盖过小分段、过大分段和过度集中的分配，同时允许合理的非均匀时间；阈值必须在查看新 validation 结果前写入实验清单。

[ ] 将 expert 学习与 selector 学习分阶段：先让多个 expert 获得非退化责任和有效梯度，再训练 selector 匹配冻结的责任标签，最后才允许短程联合微调。selector 的监督标签必须与统一物理时间验证目标一致，并记录 load-balance/entropy 项对责任分布的实际影响。

[ ] 增加最小合成测试和回归测试：所有 expert 都能获得有限且非零的梯度；责任分配不会因 batch 顺序改变；selector 能学习已知分群；duration 约束能阻止毫秒级坍缩和单段吞掉绝大部分时长；single-control 行为与 checkpoint 契约保持不变；development 流程不会加载 test。

[ ] 只有上述诊断与测试全部通过后，才允许一次固定 `K=3, seed=1337, split_seed=1337` 的 train/validation 筛选。运行前预注册阶段门：oracle minADE 至少比 2634.8 m 改善 5%（不高于 2503.1 m）；deployable selector ADE 至少改善 2%（不高于 2582.1 m）；最大 expert 使用率低于 80%，且至少两个 expert 各承担不低于 10%；final-time MAE 不高于 110.7 s；duration 与运动学诊断通过上一项预先冻结的阈值。

[ ] 只有单次筛选通过全部阶段门，才追加 `seed={2027,4242}`，并保持同一个 `split_seed=1337`。若任一主门失败，停止本方向并记录失败，不以增加 `K`、换 seed 或继续调参补救。

[ ] 三个开发种子完成且模型、损失、路由和超参数全部冻结后，仍不得自动运行 outer-test；只有用户明确宣布最终模型冻结并要求一次性 test release，才能使用 `--split test --release-test`。

## Open questions

- 该方向恢复时的首要研究目标是“可部署 selector 的单条轨迹精度”，还是“供下游规划器使用的候选集覆盖率”？前者应以 selector ADE 为主门，后者需要另行定义不偷看真值的候选消费方式，不能用 oracle 指标代替部署结果。
- duration 的可接受上下限应采用机理约束、train 数据分位数，还是两者交集？该决定需要在下一次 validation 训练之前冻结。
- 若 balanced assignment 提高使用均衡却降低 oracle 覆盖，应优先保留自然不均衡的 expert 专门化，还是把均衡视为防坍缩硬门？需要先由 objective-attribution 结果回答，而不是直接选择正则权重。
