# 随机 Anchor 训练实验计划

日期：2026-07-30  
状态：设计已冻结；仅使用 train/validation；outer-test 封存

## 1. 问题与唯一变量

本实验检验：在最终部署仍从固定 `anchor=L-1` 预测完整剩余进近的条件下，每个训练
epoch 为每个航班随机选择一个较晚 anchor，能否通过增加历史/未来组合而改善固定-anchor
预测。

首轮使用已经保留的 single-control 配方，不使用 control-mixture。两个实验臂唯一不同变量
是训练 anchor 策略：

| 实验臂 | Train anchor | Validation/预测 anchor |
| --- | --- | --- |
| A0 fixed | 每航班固定 `L-1` | 固定 `L-1` |
| A1 random-60s | 每航班每 epoch 均匀随机一个有效 anchor；至少剩余 60 s | 固定 `L-1` |

两臂均保持每航班每 epoch 一个样本，不能通过增加随机窗口数获得额外 optimizer updates。
随机 anchor 的输入严格为 `values[anchor-seq_len+1 : anchor+1]`；拟合尾迹只能用于 target，
不能进入历史输入。

single-control 的随机候选还必须满足 anchor 处观测速度
`V >= 1.10 × Vstall`（与现有轨迹优化器相同的海平面失速裕度），因为机载点质量 ODE
含 `1/V`，不定义于已停止的地面报告。train-only 审计发现 1279634 个时间合法候选中
202367 个速度为零、208548 个低于该裕度；10236 架 train 航班均仍至少有一个可飞候选，
且所有固定 `L-1` anchor 均满足此条件。该规则是 control 随机-anchor 的输入域约束，
不筛 validation；state-output 随机模式仍只使用时间条件。

为保证唯一变量确实只有 anchor 策略，两臂在 split 完成后、训练开始前共同应用
`training_cohort_min_future_s=60`。该规则只检查 train 航班在固定 `L-1` 后是否仍有至少
60 s 监督未来；不筛 validation。当前 seed 1337 的原始 train 10239 架中有 3 架不满足，
因此两臂冻结为同一 train roster 10236 架，validation 仍为 2167 架。

## 2. 冻结配置

- model：iTransformer
- prediction output：single `control`
- coordinate frame：ENU
- aircraft filter：OpenAP-direct
- `seq_len=60`，`N=64`，`dt=2 s`
- batch size：512
- learning rate：`3e-4`
- LR plateau：factor `0.5`、patience `8`
- epoch cap：180；early-stop patience：20
- control effort/smoothness：`1e-3` / `1e-2`
- seed：1337；split seed：1337
- random minimum future：60 s
- common train-cohort minimum future：60 s（两臂相同，仅 train）
- validation common grid：64 points
- checkpoint selection：固定-anchor、机场宏平均 common-grid ADE

由于 checkpoint 选择协议从旧 native objective 改为 common-grid ADE，A0 也必须重新训练；
不能把新 A1 直接与旧 checkpoint 判胜负。

## 3. Validation 契约

每个 validation 航班只有一个固定 `L-1` 窗口，`model.eval()`、dropout 关闭、顺序固定且
不使用随机 sampler。每 epoch 同时记录：

1. native composite loss，仅用于解释优化过程；
2. 固定-anchor common-grid ADE，作为 scheduler、early stopping 和 checkpoint 保存指标；
3. 分机场 common-grid ADE，先对每个机场求值，再等权平均，避免大机场独占模型选择。

best checkpoint 另行输出固定-anchor train/validation native replay 与 common-grid replay。
正式横向结论使用 validation 的统一物理时间网格 ADE/FDE、final-time MAE、分机场指标及
control/duration/运动学诊断，不使用随机 validation 或 native clock-aligned ADE 判胜负。

## 4. 预注册阶段门

先只运行 seed 1337。相对于同协议重新训练的 A0，A1 必须同时满足：

- validation common-grid ADE 至少改善 2%；
- FDE 与 final-time MAE 各自恶化不超过 5%；
- 任一机场 ADE 恶化不超过 10%；
- 没有非有限 loss、duration 非正、控制越界或严重运动学异常；
- checkpoint 与报告确认 `outer_test_loaded=false`。

只有全部通过，才追加 `seed={2027,4242}`，并保持 `split_seed=1337`。任一主门失败即停止
该方向，不通过增加窗口数、调整 60 s 下限或修改其他超参数补救。

## 5. 产物位置

```text
4dTrajectory/outputs/POOLED/experiments/openap_direct_20260730_random_anchor/
  stage_a/
    itr_control_fixed_anchor_cohort60_n64_b512_lr3e-4_seed1337_retry2/
    itr_control_random_anchor_min60_n64_b512_lr3e-4_seed1337_retry2/
  comparisons/
    stage_a_seed1337_cohort60/
```

每个 run 必须使用正式 `campaign_id` / `experiment_id` 写入不可覆盖
`experiment_manifest.json`，并保存 config、split SHA-256、checkpoint SHA-256、每 epoch
anchor 分布摘要和 60 s 有效性审计。首次训练前代码必须已提交且工作树干净。

首个未应用共同队列的 A0、因 3 架短航班触发 60 s 合约而失败的首个 A1，以及共同队列
修复后的 `retry1` 产物均保留审计但不进入横向比较。`retry1` A1 暴露了停止地面 anchor
导致的 `1/V` 动力学奇点；对应 A0 虽完成，也因早于可飞候选契约提交而只作 preliminary。
正式横向比较只使用同一新源码提交下的 `retry2` 两臂，任何旧目录均不得覆盖或删除。

## 6. 数据隔离

- 训练命令只加载 outer-train 和 outer-validation 身份对应的航迹。
- outer-test 身份可进入 checkpoint 审计 roster，但不得加载其轨迹值。
- 本阶段禁止 `--split test`、`--release-test`，禁止创建或删除 `test_release.json`。
- 若后续根据 validation 结果做任何选择，outer-test 仍保持未暴露状态。
