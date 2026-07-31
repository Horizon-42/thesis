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
| 1 | Horizon curriculum | 同一模型逐步延长 single-shooting 训练链 | 本轮实现与实验 |
| 2 | 低初始 LR + 梯度裁剪 | 从训练开始限制不稳定更新 | 待 curriculum 单变量结论后实验 |
| 3 | Trim baseline + residual control | 只用 anchor 状态和飞机参数计算基准控制 | 待设计实现 |
| 4 | Train-only oracle teacher | control imitation 预训练后 rollout 微调 | 待小样本可行性验证 |
| 5 | Progressive N refinement | N=16→32→64，复制控制并平分时长 | 低优先级 |

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

## 2. 低初始 LR 与梯度裁剪

两个 backbone 都在约 epoch 20 从较好 basin 突然发散，事后 ReduceLROnPlateau 无法恢复。
若 curriculum-only 仍发生同类失稳，再固定 curriculum 比较：

```text
LR=1e-4, clip_norm=20
LR=3e-5, clip_norm=20
```

必须记录裁剪前总梯度、backbone/control head 分模块梯度、control 饱和率及分机场 validation。

## 3. Trim baseline + residual control

用历史末状态和飞机气动参数计算保持当前 `V/psi/gamma` 的基准控制：

```text
bank_trim = 0
n_trim = cos(gamma)
T_trim = D + m*g*sin(gamma)
```

模型在有界 logit 空间只预测 residual；零 residual 精确返回每架飞机自己的 trim control。该方法
不读取未来，可同时服务 iTransformer/PatchTST。Residual 模式的 effort/smoothness 应作用于
residual，而不是惩罚维持飞行所必需的绝对推力和载荷因子。

## 4. Train-only oracle teacher

仅在 outer-train 航迹上离线生成确定性 oracle schedule，先做 control imitation，再切换到可微
rollout physical criteria。Inverse dynamics 在这里是 teacher 构建工具，不进入部署前向。由于
control 解不唯一且生成成本高，先限制为每机场 20--50 条 train 航迹。

## 5. Progressive N refinement

从较低 N 开始训练，升级时复制每段 control 并平分 duration，使升级瞬间的 rollout 等价。
此前 absolute/factorized 模型中 N=4/8 表达不足、N=32 出现时钟坍缩；固定均匀 duration 消除了
后一机制，但 oracle 分辨率链只改善终点、没有改善最佳平均 ADE，因此本方向排在最后。

## 明确排除

以下 oracle 特权不能直接迁移为部署模型能力：推理时 inverse dynamics、用真实 future schedule
初始化单航班、逐航班直接优化 control tensor、推理时使用真实总时长。随机 anchor 已在固定部署
任务上显著失败；在单专家尚未学好前也不恢复无监督 WTA mixture。
