# Control 固定 2 秒监督与单航迹过拟合（2026-07-30）

## 实现结论

新增 `control_state_loss_grid=fixed-dt` 模式，保留原有
`native-segment-endpoints` 模式。新模式的契约是：

- reference 直接读取预处理阶段已经生成的 2 秒规则网格，不做第二次插值；
- 每条航迹先对全部有效 2 秒状态点求平均，batch 再按 flight weight 求平均；
- 预测 duration 仍决定 piecewise-constant control 的切换时刻；
- state loss rollout 使用 observed total time，final-time head 仍由独立 loss 训练；
- RK4 的事件表是“全局 0.5 秒积分网格 ∪ 非均匀 control 边界”，因此 2 秒查询值是
  一次 rollout 中的真实积分状态，不是 segment endpoint 之间的线性插值；
- 没有按 supervision 点数截断长航迹。batch 内按最长航迹 padding + mask，积分器仅保留
  一个 65536 event 的异常保护上限，它不是数据语义上的预测长度上限。

启用参数：

```text
--prediction-output control
--control-state-clock observed
--control-state-loss-grid fixed-dt
--dt 2
--control-rollout-dt 0.5
```

旧 control checkpoint 若缺少 `control_state_loss_grid` 会要求重新生成，不添加兼容 fallback。

## 数据隔离

全部诊断只打开 outer-train 航迹。程序读取 manifest 中的 split 身份以确认成员关系，但没有
打开 validation 或 outer-test 航迹值，也没有生成或查看 test prediction。

正式同航迹对照使用：

```text
KSJC:ASA956_30L_a1f7fb_20260719T014743Z
```

它与 2026-07-27 state-output 单航迹容量实验相同；后者在 `N=256`、无 kinematic loss
时达到 0.146 m ADE，因此可用于区分自由 state head 与受动力学约束的 control head。

## 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | iTransformer / deterministic control |
| duration | factorized，observed-clock state rollout |
| control segments | N=64 |
| dense state grid | 2 s，共 193 点 |
| RK4 最大步长 | 0.5 s |
| dropout | 0 |
| control effort / smoothness | 0 / 0（容量诊断） |
| optimizer | Adam，learning rate `1e-4` |
| scheduler patience | 150 |
| epoch 上限 / early-stop patience | 1000 / 300 |

命令：

```bash
conda run -n aeroviz python run_ts_control_fixed_dt_overfit.py \
  --airport KSJC \
  --flight-id KSJC:ASA956_30L_a1f7fb_20260719T014743Z \
  --epochs 1000 --patience 300 --lr-plateau-patience 150 \
  --learning-rate 1e-4 --n-segments 64 --device cuda \
  --output-dir \
    4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_single_flight_overfit_1000
```

## 结果

训练在 epoch 497 early-stop，最佳 checkpoint 位于 epoch 197。

| 指标 | 结果 |
|---|---:|
| replay loss | 5.30136 → 0.18313（下降 96.55%） |
| fixed-2s normalized state loss | 0.17366 |
| fixed-2s dense ADE | 1849.4 m |
| fixed-2s 最后完整 dt 误差 | 7087.5 m |
| 真实终点 3D 误差 | 7131.0 m |
| predicted-clock common-grid ADE | 1896.3 m |
| predicted-clock common-grid FDE | 7130.8 m |
| final-time absolute error | 0.00076 s |
| duration fraction min / max | 0.01174 / 0.01781 |
| duration entropy | 4.15495（均匀 64 段上限约 4.15888） |

完整 JSON：
`4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_single_flight_overfit_1000/overfit_result.json`。

## 诊断

这次单航迹实验**没有成功过拟合空间轨迹**。它成功拟合了总时长，duration 分配也没有发生
lock collapse，但仍保留公里级空间误差。因此当前结果排除了以下解释：

1. loss 只监督 64 个 segment endpoints；新 loss 已监督全部 193 个固定 2 秒点；
2. total-duration head 学不会；时间误差已小于 1 ms；
3. dropout 或 control regularizer 阻止记忆；二者均已关闭；
4. duration routing collapse；分配接近均匀而非少数超长段。

reference 自身的 2 秒位置/速度一致性诊断为：

| reference raw metric | 数值 |
|---|---:|
| position/velocity RMSE | 5.60 m/s |
| heading consistency p95 | 1.76° |
| turn rate p95 | 2.52°/s |
| acceleration p95 | 19.18 m/s² |
| jerk p95 | 18.03 m/s³ |

这说明 reference 含有很强的 2 秒高频加速度/jerk，而 control 模式必须同时满足简化飞机动力学、
control bounds、piecewise-constant N=64 表示和六通道监督。自由 state head 可以逐点记忆，不能
证明同一轨迹在该动力学可行域内。因此当前最重要的下一项诊断不是立即做多航迹正式训练，而是
直接优化单航迹 control/duration tensor（绕过 Transformer）得到动力学 oracle 下界；若 oracle
仍为公里级，应先处理 reference 动力学一致性或提高 control 时间分辨率。

## Transport chart-velocity 同航迹对照（2026-08-01）

为单独检验新动力学是否改变模型容量上限，复用完全相同的 outer-train 航迹、模型、损失、
`N=64`、随机种子和优化超参数，仅把动力学后端从 `reanchored-rk4` 切换为
`transport-chart-velocity`。本次没有加入无量纲化，也没有打开 validation 或 outer-test 航迹。

命令：

```bash
conda run -n aeroviz python run_ts_control_fixed_dt_overfit.py \
  --airport KSJC \
  --flight-id KSJC:ASA956_30L_a1f7fb_20260719T014743Z \
  --epochs 1000 --patience 300 --lr-plateau-patience 150 \
  --learning-rate 1e-4 --n-segments 64 \
  --control-dynamics-backend transport-chart-velocity \
  --device cuda \
  --output-dir \
    4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_single_flight_overfit_1000
```

训练在 epoch 617 early-stop，最佳 checkpoint 位于 epoch 317。与上述同航迹 RK4 baseline
相比：

| 指标 | reanchored RK4 | transport chart-velocity | 相对变化 |
|---|---:|---:|---:|
| best replay loss | 0.18313 | 0.16738 | -8.60% |
| fixed-2s normalized state loss | 0.17366 | 0.15860 | -8.67% |
| fixed-2s dense ADE | 1849.4 m | 1764.6 m | -4.59% |
| fixed-2s 最后完整 dt 误差 | 7087.5 m | 6679.5 m | -5.76% |
| 真实终点 3D 误差 | 7131.0 m | 6714.1 m | -5.85% |
| predicted-clock common-grid ADE | 1896.3 m | 1808.5 m | -4.63% |
| predicted-clock common-grid FDE | 7130.8 m | 6714.1 m | -5.84% |
| predicted-clock native endpoint ADE | 1833.2 m | 1664.2 m | -9.22% |

duration 仍接近均匀分配：64 段 fraction 范围为 `0.01315–0.01996`，entropy 为
`4.14633`（均匀上限约 `4.15888`）；总时长误差为 0。因此改善不是由 duration lock 或
总时长偏差造成的。

结论：新动力学把这次同航迹实验的**经验拟合上限小幅提高**，说明它不是完全等价的数值替换，
但没有带来数量级变化。ADE 仍为 1.76 km、终点误差仍为 6.71 km，远未达到“正确降落”；
所以它没有消除当前主要容量/可优化性瓶颈。该结论仅针对同一 seed 的一次受 early stopping
约束的训练结果，不把它表述为数学上的全局最优上限。

完整 JSON：
`4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_single_flight_overfit_1000/overfit_result.json`。
