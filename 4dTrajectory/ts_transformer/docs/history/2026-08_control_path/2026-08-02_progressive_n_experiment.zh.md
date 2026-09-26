# Progressive-N teacher 实验（2026-08-02）

## 1. 设计

目标是检验先学习粗控制序列、再提高分辨率，能否降低 N=64 control head 的优化难度。实验只改变
teacher imitation 初始化：

```text
N=16 × 300 steps
  -> N=32 × 300 steps
  -> N=64 × 400 steps
```

总计仍为 1000 imitation steps，与 direct-N64 baseline 的 1000 steps 相同。每次升级时：

- 每个 coarse control logit 复制到相邻 child segments；
- 每个 duration logit 同样复制，softmax 自动把 coarse fraction 等分；
- backbone、condition encoder、feature fusion 和 final-time head 原样复制；
- 升级瞬间的 control 值、总时间和完整物理 rollout 严格不变。

64 段 teacher 降采样到 16/32 段时，按真实 segment duration 对相邻 control 加权平均，duration 求和。
实现位于独立的 `oracle_teacher/progressive_pretraining.py`，普通训练和 direct teacher 路径不含
progressive 分支。

所有 teacher 航迹仍为相同 32 条 outer-train；正式训练使用相同 2207 train / 505 validation，
outer-test 未运行、未查看。

## 2. Imitation 结果

| 初始化 | 最终总 loss | control loss | time loss |
|---|---:|---:|---:|
| direct N=64，1000 steps | 0.000340 | 0.000208 | 0.000132 |
| 16→32→64，1000 steps | **0.000251** | **0.000169** | **0.000082** |

Progressive 在 teacher 参数模仿上更好，但参数 MSE 不是最终部署目标，必须继续由 rollout validation
判断。

## 3. 正式 train/validation 结果

两组都使用 iTransformer、scaled-transport、arc 2+4、60/120/240/full curriculum、LR=`3e-5`、
clip20、N=64 正式 rollout。唯一差异是 direct 或 progressive teacher pretraining。

| 指标 | direct N=64 | progressive 16→32→64 | 相对变化 |
|---|---:|---:|---:|
| best / run epoch | 163 / 180 | 52 / 72 | — |
| selection | **27.5097** | 33.7088 | +22.53% |
| ADE | **751.7 m** | 863.1 m | +14.82% |
| FDE | **1039.2 m** | 1239.2 m | +19.25% |
| cross-track p95 | **1930.2 m** | 1996.7 m | +3.44% |
| altitude p95 | **396.3 m** | 406.4 m | +2.53% |
| final-time MAE | **14.35 s** | 18.06 s | +25.85% |

Progressive 模型在 epoch 52 达到最佳，随后 20 epochs 无改善而早停。最佳 checkpoint 的推力控制点
饱和率为 38.45%，direct baseline 为 23.09%。这说明粗粒度阶段虽然更容易拟合平均 teacher
controls，却把模型带入更强的推力边界 basin；复制成细段只保证升级瞬间 rollout 等价，并不保证
之后 N=64 rollout objective 的优化几何更好。

## 4. 结论

- Progressive-N 未通过 validation 门槛，不进入新基线，也不与 backbone 容量实验叠加。
- 更低 imitation MSE 不能作为 control 模型更好的代理指标；最终必须用可部署 rollout 目标选择。
- 本实验已经完成原清单中的 16→32→64 最小方案。若未来重启，必须引入每个 coarse stage 的
  物理 rollout 训练或减少推力饱和，而不是继续调 imitation step 比例。

输出目录：

- direct 对照：`outputs/KSJC/experiments/nondimensional_transport_20260802/formal_teacher_scaled`
- progressive：`outputs/KSJC/experiments/progressive_n_20260802/formal_teacher_scaled_n16_n32_n64`
