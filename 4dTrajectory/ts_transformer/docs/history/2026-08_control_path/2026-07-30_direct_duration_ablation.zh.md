# Control 直接分段时长参数化消融（2026-07-30）

## 1. 问题与假设

当前 `factorized` control head 将时长写成：

\[
\hat T=S\,\operatorname{softplus}(g),\qquad
\Delta t_i=\hat T\,\operatorname{softmax}(z)_i.
\]

本实验只改变时长参数化。`direct` 使用相同的全局历史时长网络、局部 duration
projection、参数量和初始时钟，但直接生成每段正时长：

\[
\Delta t_i={S\over N}\operatorname{softplus}(g+z_i),\qquad
\hat T=\sum_i\Delta t_i.
\]

假设是：去掉 softmax 的段间竞争后，局部控制阶段可以独立调整时长，从而改善固定-anchor
validation 的轨迹和时钟误差。两者能表达相同的正时长集合，因此这是优化参数化消融，不是
输出能力不同的新任务。

## 2. 隔离边界

- 唯一自变量：`control_duration_parameterization=factorized|direct`；
- 冻结：iTransformer、五机场 OpenAP-direct roster、fixed `L-1` anchor、`L=60`、
  `N=64`、batch 512、LR `3e-4`、plateau `0.5/8`、seed/split-seed 1337、控制损失权重和
  float64 RK4 `0.5 s` 上限；
- 两种模式使用相同 `ControlPrediction`、dynamics rollout、clock-aligned state loss 和
  common-grid validation；
- development 只加载 train/validation；outer-test 保持关闭；
- 当前改动在 review 前不提交，因此第一轮结果标记为 provisional；代码确认后应在干净提交上
  重跑正式 run，而不是把 provisional 指标冒充论文最终结果。

## 3. 对照与候选

- Factorized baseline：
  `openap_direct_20260729_control_prediction/stage_b_learning_rate/`
  `itr_control_n64_b512_lr3e-4_seed1337`；
- Direct candidate：
  `openap_direct_20260730_direct_duration/stage_a/`
  `itr_control_direct_duration_n64_b512_lr3e-4_seed1337`。

旧 baseline 的 checkpoint 配置没有该字段。按派生制品策略，该 checkpoint 现在会被明确拒绝，
不能再通过默认值升级为 `factorized`；正式对照必须用当前完整 recipe 重新训练 baseline。

## 4. 决策指标

主指标使用2167架固定 validation 航班、64点统一物理时间网格：

- ADE、FDE、final-time MAE；
- position/velocity consistency、turn rate、acceleration、jerk；
- duration min/p1/median/p99/max、接近零的比例；
- control bounds 饱和率和相邻控制变化。

只有 direct 的 common-grid ADE 改善至少5%，并且 FDE、时间 MAE及物理诊断没有超过5%的
实质退化，才进入配对三种子确认。单 seed 不作为最终选择依据。

## 5. 初步结果（旧 baseline 已失效）

以下数值保留为历史诊断，但因 factorized 一侧复用了缺少
`control_duration_parameterization` 的旧派生 checkpoint，不能作为正式消融结论。重新训练
factorized baseline 后，必须在相同 validation 协议下重新生成比较报告。

数据策略审计：2167架 validation 航班，split SHA-256 为
`ea0d70fa25c4f21bde024f361216500e65193a27cda00ea864ab8d1445e0af33`；报告明确记录
`outer_test_loaded=false`。

### 5.1 统一物理时间网格

| 参数化 | ADE (m) | FDE (m) | 时间 MAE (s) | epoch |
| --- | ---: | ---: | ---: | ---: |
| factorized | 2634.8 | 4399.0 | 110.7 | 31 |
| direct | 3188.5 | 6008.5 | 102.1 | 46 |

Direct 相对 factorized：

- ADE 变差21.0%；
- FDE 变差36.6%；
- 时间 MAE 改善7.8%。

五个机场的 direct ADE 均更差；FDE 只在 KSMF 有约56 m的小幅改善，不能抵消整体退化。

### 5.2 Native 指标为何误导

Direct best checkpoint 的 native validation ADE 为337.2 m，远低于 factorized 的2555.3 m，
但它没有转化为同一物理时间上的提升。Duration 分布显示模型通过移动采样时钟降低逐段平均误差：

| 参数化 | duration p1 (s) | median (s) | p99 (s) | max (s) |
| --- | ---: | ---: | ---: | ---: |
| factorized | 0.342 | 1.130 | 61.2 | 170.6 |
| direct | 0.064 | 0.507 | 123.2 | 188.9 |

Direct 的 p1 缩短81.1%，p99 增长101.2%。大量短段让 native endpoint loss 对轨迹早期重复计权，
少数长段承担其余时间；64点统一物理时钟移除这一采样密度效应后，真实误差反而更大。

### 5.3 优化和物理诊断

- 最佳 objective 出现在 epoch 26；epoch 27 起 state loss 突然上升，之后未恢复，epoch 46早停；
- position/velocity consistency RMSE 从1.20增至2.15 m/s，变差79.2%；
- jerk p95 从0.92增至3.05 m/s³，变差231.9%；
- 虽然 turn-rate 和 acceleration p95 降低，但这是控制趋于平缓、load factor 靠近1且时间分布极化共同
  产生的结果，不能据此宣称整体物理性提升。

### 5.4 决策

本次初步比较中，`direct N=64 seed=1337` 未达到预注册门槛；但旧 factorized checkpoint
已按新制品契约失效，因此暂不把该结果作为正式淘汰结论。下一步应先用当前完整 recipe
重新训练 factorized baseline，再决定是否进入三种子确认。outer-test 继续封存。

机器可读报告：
`outputs/POOLED/experiments/openap_direct_20260730_direct_duration/comparisons/`
`stage_a_seed1337/report.json`。
