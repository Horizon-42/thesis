# 可部署 Control 模型训练优化清单

日期：2026-07-31

## 适用边界

本文只保留能够用于模型训练、且部署推理时不依赖未来真实航迹的优化。所有开发选择只使用
outer-train 与 validation；outer-test 继续封存。Oracle 可以使用完整 outer-train future 作为
诊断或离线 teacher，但任何部署前向都不能读取真实未来状态、控制或总时长。

当前共同基线为：固定 anchor `L-1`、N=64、factorized final-time head、状态 rollout 使用
observed training clock、均匀 duration/state-duration 梯度解耦、fixed-2s physical criteria，
checkpoint 由固定-anchor validation common-grid criteria 选择。

## 优先级

| 优先级 | 优化 | 可部署方式 | 当前状态 |
|---:|---|---|---|
| 1 | Horizon curriculum | 同一模型逐步延长 single-shooting 训练链 | 在 `3e-5 + clip20` 下已通过 clip-only 对照；空间指标改善、时间 MAE 退化 |
| 2 | 低初始 LR + 梯度裁剪 | 从训练开始限制不稳定更新 | iTransformer 组合消融完成；`3e-5` 的预注册准则最好 |
| 3 | Final-time clip 解耦 | final-time head 不随巨大 control 梯度一起缩放 | 已完成；形成时间/ADE Pareto 点，未替代预注册 selection 胜者 |
| 4 | Trim baseline + residual control | 只用 anchor 状态和飞机参数计算基准控制 | iTransformer 首轮完成；ADE 略好但 selection/FDE/时间退化，未迁移 PatchTST |
| 5 | Train-only oracle teacher | control imitation 预训练后 rollout 微调 | 已通过正式 KSJC train/validation，成为新基线 |
| 6 | Progressive N refinement | N=16→32→64，复制控制并平分时长 | 已完成；正式 validation 明显退化，不采用 |

已冻结为后续基础配置、无需作为新方向重复声明的两项是：

- `physical-criteria = smooth-max(ADE/100m, terminal_error/100m)`；
- factorized duration 的 state-gradient 解耦。final-time head 仍独立训练，推理仍用预测时间。

## 1. Horizon curriculum（当前实验）

### 训练语义

模型始终输出完整 N=64 controls；历史输入、anchor 和网络结构不变。训练阶段依次为：

```text
60 s × 10 epochs
  → 120 s × 10 epochs
  → 240 s × 10 epochs
  → full horizon（剩余 epoch，正常 scheduler/early stop）
```

每个短阶段真正裁掉 horizon 之后的 control duration、dense supervision 与动力学事件，而不是
只 mask full-rollout loss。因此正向积分链和反向 adjoint 链都随阶段缩短。对于真实剩余时长短于
当前阶段的航迹，该航迹自然使用自己的 full horizon，不外推或改 roster。

短阶段只使用固定-anchor prefix validation objective 观察本阶段训练，不允许覆盖最终 checkpoint；
进入 full 阶段后重置 scheduler/early-stop 计数，最终 checkpoint 只由完整固定-anchor validation
common-grid criteria 选择。这保证开发任务仍对齐真正部署的 full prediction。

### 首轮单变量配方

为测量 curriculum 本身，首轮保持刚完成的 iTransformer 配方不变：

- 五机场 pooled OpenAP-direct train/validation；
- iTransformer、N=64、batch=512、LR=`3e-4`；
- physical criteria、固定均匀 duration、control effort=`1e-3`、smoothness=`1e-2`；
- seed/split seed=`1337`；
- 唯一新增变量：`60,120,240 s × 10 epochs` curriculum。

比较基线为无 curriculum 的 iTransformer development run：最佳 epoch 15，validation
common-grid ADE/FDE=`2424.69/2688.55 m`，等权机场 criteria=`27.6909`。首轮不同时降低 LR，
避免无法判断改善来自 curriculum 还是优化器。

### 首轮结果：未形成可用 checkpoint

五机场 roster、split、模型、LR 和损失均与基线一致，且全程只使用 train/validation。短阶段
本身都能下降：

| 阶段末 | train objective | validation prefix objective |
|---:|---:|---:|
| 60 s / epoch 10 | 4.2409 | 4.1346 |
| 120 s / epoch 20 | 8.8534 | 8.9917 |
| 240 s / epoch 30 | 28.5255 | 27.3842 |

但进入 full horizon 后发生剧烈不连续：

| epoch | train full objective | validation full objective | validation common-grid criteria |
|---:|---:|---:|---:|
| 31 | 506.5842 | 603.5438 | 561.4 |
| 32 | 323.9715 | 320.6297 | 317.9 |

epoch 33 的某个 optimizer update 后，共享 backbone 输出非有限值，继而
`segment_durations` 变为 NaN，训练终止。由于 `fit_model` 只在完整训练结束后写最佳权重，本次
没有生成 checkpoint；没有进行 prediction，更没有读取 outer-test。

结论是：`60→120→240→full` 的 prefix-cropping curriculum 在当前 `LR=3e-4` 下没有提高
可部署 full-horizon 模型。短阶段只约束给定物理时间内的控制；对长航迹而言，240 s 之后的控制
没有获得相同条件下的状态监督，进入 full 后同时暴露长尾控制和更长的 single-shooting 梯度链，
objective 放大约 22 倍并最终梯度爆炸。这是当前 curriculum 配方的负结果，不能用短阶段较低的
objective 宣称改善。

实现过程中还发现并修复了一项独立数值问题：短阶段裁剪原先在 float32 中求和，转换为 float64
积分时可能让精确的 60 s query 超出总时长数微秒。现在裁剪统一使用 fixed-dt supervision 的
float64 时钟，并有 64 段分数时长回归测试。该修复只保证边界语义正确，不改变上述优化结论。

按本清单的预注册顺序，下一项应单独测试“低 LR＋梯度裁剪”。如果保留 curriculum 再加入它，
实验名称必须明确写成组合消融，不能再称为 curriculum-only。

## 2. 低初始 LR 与梯度裁剪

两个 backbone 都在约 epoch 20 从较好 basin 突然发散，事后 ReduceLROnPlateau 无法恢复。
curriculum-only 发生同类失稳后，固定 curriculum 比较：

```text
LR=1e-4, clip_norm=20
LR=3e-5, clip_norm=20
```

必须记录裁剪前总梯度、backbone/control head 分模块梯度、control 饱和率及分机场 validation。

### 实现与实验约束

梯度稳定功能是独立训练模块，不嵌入 backbone 或预测头：

- `control_training_diagnostics.py` 负责 float64 全局/分组梯度范数、全局 L2 clip 和控制饱和率；
- `control_gradient_clip_norm=0` 保持原训练路径，正值仅允许 deterministic control 输出；
- checkpoint recipe 必须显式带该字段，旧派生 checkpoint 不做兼容升级，必须重训；
- 历史记录保存每 epoch 的裁剪前 backbone/control/final-time/total 梯度、clip 触发率和三类控制饱和率。

两组组合消融使用完全相同的五机场 OpenAP-direct roster、split、N=64、batch=512、
`60→120→240→full` curriculum、clip norm=20、seed/split seed=1337。train/validation
分别为 10239/2167 条固定身份航迹；outer-test source tracks 始终关闭。比较表中的 ADE/FDE
均来自最佳 checkpoint 的固定-anchor、64 点 common physical-time grid，不是 native grid：

| 配方 | best / run epoch | updates | 机场宏平均 criteria | common ADE (m) | common FDE (m) | time MAE (s) |
|---|---:|---:|---:|---:|---:|---:|
| 无 curriculum 基线，LR=`3e-4`，无 clip | 15 / 35 | 700 | 27.6909 | 2424.69 | 2688.55 | 63.64 |
| 无 curriculum，LR=`3e-5` + clip20 | 146 / 166 | 3320 | 23.9121 | 2127.77 | 2323.65 | **63.39** |
| curriculum + LR=`1e-4` + clip20 | 144 / 164 | 3280 | 21.6476 | **1706.44** | 2128.25 | **63.67** |
| curriculum + LR=`3e-5` + clip20 | 165 / 180 | 3600 | **20.8005** | 1852.05 | **2055.70** | 69.51 |

相对无 curriculum、无 clip 的旧基线，两组组合都显著改善：

- `1e-4`：criteria -21.82%，ADE -29.62%，FDE -20.84%，时间 MAE 基本不变；
- `3e-5`：criteria -24.88%，ADE -23.62%，FDE -23.54%，时间 MAE +9.22%。

但两个组合不是简单的“低 LR 全面更好”。与 `1e-4` 组合相比，`3e-5` 的预注册机场宏平均
criteria 低 3.91%，common FDE 低 3.41%，而 pooled common ADE 高 8.53%、时间 MAE 高
9.17%。这是指标定义导致的真实 Pareto 权衡：checkpoint criteria 对每个机场计算
`smooth-max(ADE/100m, FDE/100m)` 后再机场等权平均；当前 FDE 通常是较差的一项，因而
`3e-5` 的终点改善压过了 ADE 退化。`3e-5` 在五个机场上的 criteria 均低于 `1e-4`：

| 机场 | LR=`1e-4` | LR=`3e-5` |
|---|---:|---:|
| KMSY | 30.3679 | 27.9611 |
| KRDU | 27.5793 | 27.1583 |
| KSJC | 10.6518 | 9.9965 |
| KSMF | 17.5680 | 17.2834 |
| KSTL | 22.0711 | 21.6031 |

因此按实验前已声明的 selection contract，`3e-5` 是组合消融胜者；若论文另行声明 ADE 为
唯一主指标，则 `1e-4` 是 ADE Pareto 点，不能事后把二者混写成一个“全面最优”模型。

clip20 成功把 curriculum-only 的 NaN 失败变为可完成训练，但没有消除病态梯度：

- `1e-4` 全程最大裁剪前总梯度约 `4.17e17`，epoch 平均 clip 触发率 99.94%；
- `3e-5` 最大裁剪前总梯度仍约 `1.37e13`，epoch 平均 clip 触发率 98.83%；
- 两个最佳 epoch 都是 20/20 batch 触发 clip；`1e-4` 最佳 epoch 的总体控制饱和率
  约 0.336%，`3e-5` 为 0。

所以降低 LR 不是梯度爆炸的根治手段；长时 single-shooting rollout 的局部敏感度仍是根因。
同时训练成本由旧基线 700 updates 增至 3280/3600 updates，组合收益伴随明显计算开销。

### Clip-only 因果对照：curriculum 的空间收益成立

保持 `LR=3e-5 + clip20`、模型、roster、split、epoch budget 与所有 loss 不变，只去掉
curriculum。clip-only 在 epoch 146 达到最佳 criteria=`23.9121`，epoch 166 早停；curriculum
组合在 epoch 165 达到 `20.8005` 并跑满 180 epoch。相对 clip-only，curriculum：

- 机场宏平均 criteria 降低 13.01%；
- pooled common-grid ADE/FDE 分别降低 12.96%/11.53%；
- 时间 MAE 从 63.39 s 增至 69.51 s，退化 9.65%；
- optimizer updates 多 8.43%（3600 vs 3320），但训练墙钟少 6.30%（5879 s vs 6274 s），
  因为 30 个 prefix epoch 远比 full rollout 便宜；
- 五机场中 KMSY/KRDU/KSJC/KSMF criteria 更好，KSTL 更差，收益不是每机场一致。

因此在数值已由 clip20 稳定的 `LR=3e-5` 配方下，可以独立确认 horizon curriculum 改善了
固定-anchor 空间轨迹和预注册 selection criteria；它也更快到达较好的 full-horizon basin。
这个结论不能外推成“任何 LR 下 curriculum 都有效”：`LR=3e-4` 的 curriculum-only 仍然是
NaN 失败。当前部署候选按预注册 selection contract 选 `LR=3e-5 + curriculum + clip20`，但若
4D 时间误差与空间误差同等重要，该候选尚未结束优化，因为时间 MAE 明显退化。

检查实现后确认，短阶段的 final-time loss 始终比较原始 `prediction.final_time_s` 与完整真实总时长，
没有把 clock head 错教成 60/120/240 s；时间退化不是 crop label bug。当前最可能的下一项最小
消融是 final-time clip 解耦：control/backbone 路径继续使用 clip20，final-time head 独立裁剪或
不参与该全局缩放。其动机来自诊断而非既成结论：最佳 epoch 的 final-time-head 梯度最大值仅约
0.15--0.16，而 control-head 为约 990--1380；全局 clip 会把小的 clock 梯度一起缩小。该消融
必须继续用同一 train/validation roster 验证，不能用 test 决策。

输出目录：

- `outputs/POOLED/experiments/openap_direct_20260731_horizon_curriculum_lr_clip/itransformer_n64_b512_lr1e4_clip20_seed1337/`
- `outputs/POOLED/experiments/openap_direct_20260731_horizon_curriculum_lr_clip/itransformer_n64_b512_lr3e5_clip20_seed1337/`
- `outputs/POOLED/experiments/openap_direct_20260731_horizon_curriculum_lr_clip/itransformer_n64_b512_lr3e5_clip20_no_curriculum_seed1337/`

## 3. Final-time clip 解耦

当前 `clip_norm=20` 是所有参数共享的全局 L2 cap。控制 rollout 梯度比 final-time head 梯度高
数千到数万倍时，后者也被同一缩放系数压低。最小实现应放在独立 gradient policy 模块中，而非
向 backbone 或预测头加入模式分支；先只比较“全局 clip20”与“control/backbone clip20、
final-time head 独立”的 validation 空间/时间 Pareto，不改变推理结构。

### 实现边界

新增的 `final-time-decoupled` 策略只把模型参数分成两个裁剪 scope：

- `backbone + control_head` 作为一个向量继续做 global L2 clip20；
- `final_time_head` 参数不参与上述缩放。

共享 backbone 中来自空间 loss 和时间 loss 的梯度在一次反向传播后已经相加，本消融不尝试拆分
它们；否则需要分别 backward 或改变优化目标，便不再是最小单变量实验。模型 forward、loss、
optimizer、推理结构和数据完全不变。

### Train/validation 结果

实验保持上一轮预注册胜者的五机场 roster、iTransformer、N=64、batch=512、LR=`3e-5`、
`60→120→240→full` curriculum、clip20、seed/split seed=1337，只把 clip policy 从
`global` 改为 `final-time-decoupled`。outer-test source tracks 始终关闭。

| clip policy | best / run epoch | criteria | common ADE (m) | common FDE (m) | time MAE (s) |
|---|---:|---:|---:|---:|---:|
| global | 165 / 180 | **20.8005** | 1852.05 | **2055.70** | 69.51 |
| final-time-decoupled | 144 / 164 | 21.0431 | **1813.11** | 2067.91 | **63.63** |

相对 global clip，解耦策略的 common ADE 改善 2.10%、时间 MAE 改善 8.46%，但预注册机场
宏平均 criteria 退化 1.17%、FDE 退化 0.59%。分机场 criteria 也不一致：KSTL 改善 9.29%，
KMSY/KRDU/KSJC 分别退化 5.66%/3.13%/7.67%，KSMF 基本持平（+0.10%）。因此这是一个
真实 Pareto 点，不能称为全面优于 global clip；按预注册 selection contract，global clip20
仍是下一项空间模型消融的基础配置。

机制假设得到部分支持：解耦模型最佳 epoch 的 final-time validation loss 为 0.02132，而 global
clip 为 0.02489；时间 MAE也恢复到与旧基线相近的 63.63 s。但它没有解决长时 rollout 梯度：
control/backbone 的 epoch 平均 clip 触发率仍为 98.75%，全程最大裁剪前范数约 `8.71e18`。
最佳 epoch 的 control/backbone clip 系数均值仅 0.0689，而 final-time head 始终保持 1.0。

输出目录：

- `outputs/POOLED/experiments/openap_direct_20260731_final_time_clip_decoupling/itransformer_n64_b512_lr3e5_clip20_final_time_decoupled_seed1337/`

## 4. Trim baseline + residual control

用历史末状态和飞机气动参数计算保持当前 `V/psi/gamma` 的基准控制：

```text
bank_trim = 0
n_trim = cos(gamma)
T_trim = D + m*g*sin(gamma)
```

模型在有界 logit 空间只预测 residual；零 residual 返回每架飞机自己的 trim control。该方法
不读取未来，可同时服务 iTransformer/PatchTST。Residual 模式的 effort/smoothness 应作用于
residual，而不是惩罚维持飞行所必需的绝对推力和载荷因子。

### 已实现的独立模式与首轮协议

`trim-residual` 是 control value parameterization，不是新 backbone，也不改变 duration head。
独立模块使用共享 Torch 动力学中的 ISA 密度、升阻模型和飞机参数计算 anchor baseline；预测头
通过 `sigmoid(logit(trim_fraction) + residual_logit)` 保持每架飞机自己的物理上下界；为避免
边界 logit 无穷，trim fraction 仅在数值上内缩 `1e-6`，所有
residual 权重/偏置初始化为零。effort 惩罚 `(control-trim)/(upper-lower)`，smoothness 对同一
residual 做相邻段差分。训练和推理都只需要历史最后状态与随航班提供的飞机参数。

对固定 development roster 的 12,406 条航迹做了不读取 outer-test 的 anchor 审计：没有 baseline
失速或 load-factor 裁剪，平均 `n_trim=0.9987`；但 16.65% 的下降状态按上述公式会要求负推力，
因此被物理约束裁到 0。对这部分样本，零 residual 是 idle-thrust constrained baseline，不能精确
维持速度。该限制预先保留，不使用真实 future 修正 trim。

首轮是相对当前预注册 selection 胜者的单变量消融：五机场 iTransformer、N=64、batch=512、
LR=`3e-5`、global clip20、`60→120→240→full` curriculum 及全部 split/seed/loss 保持不变，
只把 absolute control head/absolute effort 改为 trim-residual head/residual effort。先按相同的
固定-anchor validation criteria 判断；只有通过后再迁移到 PatchTST，outer-test 始终封存。

### 首轮 train/validation 结果：未通过迁移门槛

实验使用与 absolute-control 胜者完全相同的 10,239/2,167 条 train/validation 航迹、固定 anchor、
split seed 和训练预算；outer-test source tracks 始终关闭。trim-residual 跑满 180 epoch，最佳
checkpoint 位于 epoch 175，selection criteria 精确值为 `21.9754557`：

| control value 参数化 | best / run epoch | criteria | common ADE (m) | common FDE (m) | time MAE (s) |
|---|---:|---:|---:|---:|---:|
| absolute | 165 / 180 | **20.8005** | 1852.05 | **2055.70** | **69.51** |
| trim-residual | 175 / 180 | 21.9755 | **1818.94** | 2242.58 | 72.29 |

相对 absolute，trim-residual 的 pooled common-grid ADE 改善 1.79%，但 FDE 退化 9.09%、时间
MAE 退化 4.00%，预注册机场宏平均 criteria 最终退化 5.65%。分机场结果揭示了宏平均退化的主要
来源：四个机场有小幅改善，KRDU 则明显恶化。

| 机场 | absolute criteria | trim-residual criteria | 相对变化 |
|---|---:|---:|---:|
| KMSY | 27.9611 | 27.3158 | -2.31% |
| KRDU | **27.1583** | 34.4850 | **+26.98%** |
| KSJC | 9.9965 | 9.7272 | -2.69% |
| KSMF | 17.2834 | 16.8847 | -2.31% |
| KSTL | 21.6031 | 21.4646 | -0.64% |

优化难度也没有被 trim 消除。全程 epoch 平均 clip 触发率为 99.86%，最大裁剪前总梯度约
`3.87e11`；虽然低于 absolute 对照的 `1.37e13`，但最佳 epoch 的平均裁剪前总梯度为 948.0，
高于 absolute 最佳 epoch 的 331.5。trim 最佳 checkpoint 的总体饱和率为 11.45%，全部来自
推力，约 34.36% 的推力控制点落在上下界 1% 内；这与 16.65% anchor 需要 idle-thrust constrained
baseline 的先验审计一致，不应解释为所有控制维度的路由或输出坍缩。

结论：当前 trim baseline 改善了平均路径贴合，但牺牲终点和 KRDU 泛化，未通过预先声明的
`criteria < 20.8005` 迁移门槛。依照实验协议不启动 PatchTST，不因看到结果而修改门槛，也不查看
outer-test。后续若重新研究 trim，应作为新的独立实验定位 KRDU/下降状态的 constrained-trim
失配，而不能把本轮结果称为 backbone 无关的有效优化。

输出目录：

- `outputs/POOLED/experiments/openap_direct_20260731_trim_residual/itransformer_n64_b512_lr3e5_clip20_seed1337/`

## 5. Train-only oracle teacher

仅在 outer-train 航迹上离线生成确定性 oracle schedule，先做 control imitation，再切换到可微
rollout physical criteria。Inverse dynamics 在这里是 teacher 构建工具，不进入部署前向。由于
control 解不唯一且生成成本高，先限制为每机场 20--50 条 train 航迹。

### 2026-08-02 结果

32 条 outer-train 航迹的 inverse-dynamics schedule 经单航迹直接优化后，用于 1000 步 N=64
imitation，再进入普通全量 rollout 训练。Teacher-only 正式 KSJC validation 相对此前同配方基线：

- selection `31.4126 -> 27.6277`；
- ADE `1130.7 -> 771.0 m`；
- FDE `1202.9 -> 1040.4 m`；
- cross-track p95 `3313.2 -> 2039.6 m`；
- altitude p95 `692.6 -> 364.7 m`；
- time MAE `20.84 -> 14.19 s`。

Teacher ID 审计为 32/32 outer-train、与 validation 交集为 0；部署前向不读取 future。完整链路、一次
正则混杂 run 和修正后的单因素结果见
[`2026-08-02_oracle_teacher_experiment.zh.md`](2026-08-02_oracle_teacher_experiment.zh.md)。

## 6. Progressive N refinement

从较低 N 开始训练，升级时复制每段 control 并平分 duration，使升级瞬间的 rollout 等价。
此前 absolute/factorized 模型中 N=4/8 表达不足、N=32 出现时钟坍缩；固定均匀 duration 消除了
后一机制，但 oracle 分辨率链只改善终点、没有改善最佳平均 ADE，因此本方向排在最后。

### 2026-08-02 结果

已完成 `N=16 × 300 -> N=32 × 300 -> N=64 × 400` 的同预算 teacher pretraining。Head 升级
通过复制 control logits 和等分 softmax duration 保证瞬间 rollout 等价。尽管最终 imitation loss
优于 direct N=64，正式 validation selection 从 `27.5097` 退化到 `33.7088`，ADE/FDE 从
`751.7/1039.2 m` 退化到 `863.1/1239.2 m`，并出现更高推力饱和。因此不进入基线。详见
[`2026-08-02_progressive_n_experiment.zh.md`](2026-08-02_progressive_n_experiment.zh.md)。

## 明确排除

以下 oracle 特权不能直接迁移为部署模型能力：推理时 inverse dynamics、用真实 future schedule
初始化单航班、逐航班直接优化 control tensor、推理时使用真实总时长。随机 anchor 已在固定部署
任务上显著失败；在单专家尚未学好前也不恢复无监督 WTA mixture。

## 7. Transport chart + ENU velocity dynamics

### 动机与状态定义

旧 `reanchored-rk4` backend 持久化
`(lat, lon, alt, V, psi, gamma, mass)`：每个 RK4 子步先在当前局部 ENU 中积分，再把结果经
ECEF 转回大地坐标并重建下一步局部坐标系。新 `transport-chart-velocity` backend 持久化

```text
(E, N, U, V_E, V_N, V_U, mass)
```

其中 `E/N/U` 始终是 runway threshold 对应的固定 chart；速度仍表示在随飞机位置移动的局部
地理 ENU 基底中。因此速度微分显式包含 WGS84 的完整 transport rate：

```text
v_dot_ENU = a_force_ENU - omega_transport × v_ENU
omega_transport = (-V_N/(R_M+h), V_E/(R_N+h),
                   V_E*tan(lat)/(R_N+h))
```

位置微分同时带 `R_M/R_N`、高度和纬度比例。这里没有加入地球自转/Coriolis，因为当前 CasADi
和 re-anchored baseline 都不包含它；本项是同一个三自由度点质量模型的连续坐标表达，不是增加
第二套飞机物理。模型输入/监督输出的 `coordinate_frame=enu|runway-aligned` 仍只是边界旋转，
没有把内部 transport state 与 Transformer 的输入 frame 混为一谈。

### 解耦实现

- endpoint 和 fixed-dt dense rollout 共用 backend-neutral 的分段调度/离散 adjoint；
- 两个 backend 各自拥有初始状态转换、RK4 step、CUDA compile cache 和输出转换；
- 训练 loss、iTransformer、PatchTST、control head 和数据集只依赖统一 rollout contract，不含
  dynamics 模式分支；
- 配置、checkpoint recipe、target contract、输出目录与前端标签显式记录 backend；旧 control
  checkpoint 缺少该字段时必须重训，不做兼容升级。

### 数值一致性门槛

正式训练前固定了三层 contract：

1. chart/geodetic 往返与既有 ENU、runway-aligned channel adapter 在 float64 下逐值一致；
2. 对 300 s、`dt=0.5 s` 的 full-transport CasADi 连续模型，新 backend 的终点位置差约
   `1.9e-4 m`，速度差约 `2.7e-6 m/s`；heading 仅有等价的 `2π` wrap；
3. 对 600 s 变控制 rollout，新 backend 与生产 `reanchored-rk4` baseline 在 `dt=0.5 s` 的终点
   位置差为 `2.62 m`，缩小到 `dt=0.25 s` 后降为 `1.23 m`，说明两个有限步实现向同一连续
   动力学收敛，而不是靠放宽单点 tolerance 掩盖结构差异。

CPU 的 endpoint/dense 梯度测试、两个 backend × 两个 backbone 的完整可微训练步，以及宿主
GPU 的 endpoint forward + dense adjoint smoke test均通过。

### Train/validation 单变量对照

正式对照保持当前预注册胜者的五机场 OpenAP-direct roster、固定 anchor、iTransformer、N=64、
batch=512、LR=`3e-5`、global clip20、`60→120→240→full` curriculum、factorized/absolute
control、fixed-2s physical criteria、seed/split seed=1337 和 180 epoch 预算不变。唯一变量是：

```text
reanchored-rk4 -> transport-chart-velocity
```

outer-test source tracks 始终关闭。实验输出目录：

- `outputs/POOLED/experiments/openap_direct_20260731_transport_chart_velocity/itransformer_n64_b512_lr3e5_clip20_seed1337/`

### Train/validation 结果：数值更平滑，但预测全面退化

新 backend 在 epoch 148 达到最佳 validation criteria=`24.6380163`，epoch 168 因连续 20 epoch
没有改善而早停；baseline 在 epoch 165 达到 `20.8004671` 并跑满 180 epoch。两者的 checkpoint
使用同一 10,239/2,167 train/validation 航迹和完全相同的 split digest。

| dynamics backend | best / run epoch | criteria | common ADE (m) | common FDE (m) | time MAE (s) |
|---|---:|---:|---:|---:|---:|
| reanchored-rk4 | **165 / 180** | **20.8005** | **1852.05** | **2055.70** | **69.51** |
| transport-chart-velocity | 148 / 168 | 24.6380 | 2038.52 | 2419.56 | 75.71 |

相对 baseline，新 backend 的预注册 criteria 退化 18.45%，common ADE 退化 10.07%，FDE 退化
17.70%，时间 MAE 退化 8.93%。这不是单一机场造成的；五机场 criteria 全部变差：

| 机场 | reanchored-rk4 | transport-chart-velocity | 相对变化 |
|---|---:|---:|---:|
| KMSY | 27.9611 | 35.4158 | +26.66% |
| KRDU | 27.1583 | 33.5870 | +23.67% |
| KSJC | 9.9965 | 11.2845 | +12.88% |
| KSMF | 17.2834 | 18.3869 | +6.38% |
| KSTL | 21.6031 | 24.5159 | +13.48% |

新表示确实削弱了最极端的数值爆炸：全程最大裁剪前总梯度由 baseline 的约 `1.37e13` 降到
`1.10e8`。但 epoch 平均 clip 触发率几乎没变（98.75% vs 98.83%）；新 backend 最佳 epoch
的裁剪前总梯度均值/最大值反而为 `592/4322`，高于 baseline 最佳 epoch 的 `331/1008`。
两个模型的控制饱和率均为 0。full 阶段第 31 epoch 的 validation criteria 也很接近
（67.72 vs 66.95），说明退化不是 curriculum 切换时发生的简单坐标 bug，而是之后在同一
clip/LR 配方下进入了更差的优化路径。

新 backend 运行 168 epoch、3360 updates、训练 epoch 墙钟合计约 5445 s；baseline 为 180 epoch、
3600 updates、约 5879 s。总时间少 7.39%主要来自早停，单个 full epoch 都约 37--38 s，不能
宣称 dynamics 本身显著加速。

结论：`transport-chart-velocity` 的物理与数值 contract 成立，但在当前训练配方下没有提高模型
能力，不能替代 `reanchored-rk4` 默认 backend，也不因实现完成而启动 outer-test。由于五机场均
退化，本轮不迁移 PatchTST；如继续研究，应作为新的优化实验针对 transport state 的尺度/Jacobian
或 LR/clip 配方，而不能复用本轮结果声称 backbone 无关收益。
