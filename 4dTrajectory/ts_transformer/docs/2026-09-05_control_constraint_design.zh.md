# control 输出的约束方案（dev 文档，2026-09-05，第三版）

承接 `2026-09-04_constraint_methods_survey.zh.md`（方法综述）和 `2026-09-05_final_constraint_results.zh.md`（state 输出上的结果）。问题：**同一个五边约束（LPV 走廊 + 下滑道窗口）在 control 输出上，除了罚项，还能怎么加？**

## 结论

control 输出上值得做的是两条"进 rollout 回路"的路线，都作用在每一步的控制命令上，位置始终由动力学积分得到：

1. **屏障过滤器**：把走廊写成屏障，逐步算出坡度的允许区间，把网络的命令压进去。只在要越界时干预。
2. **名义跟踪律 + 有界残差**：门控生效时，命令 = 固定跟踪律（把飞机拉向中线和下滑道）+ 网络的有界修正。

两者接口相同、语义相反。关键差别在**已经在走廊外**的路径：过滤器无事可做，名义律会把它拉回来。state 路径上 A 臂有 55–77 % 的真值门控行在走廊外，说明"拉回来"才是需要的行为，所以两条都做、先比。参考路径 + 制导律、优化器求解、采样筛选降为低优先级；事后投影对 control 不适用。

## 出发点

- control 路径的 rollout（`control/dynamics/rollout.py`，点质量 + 一阶滞后，RK4 子步 0.5 s）整个在反向传播路径上，梯度经积分器传回控制量和分段时长。所以放进 rollout 的任何可微操作都自动进训练回路。
- 约束只能改**命令**，不能改位置。改位置（state 上的事后投影）会破坏动力学一致性，正好抵消 control 路径的意义。
- 已有的硬约束只有包线：控制头用 sigmoid 把推力比、坡度、载荷因子压进包线盒。走廊还没有任何硬约束。
- 罚项已经接到 rollout 端点状态上（`procedure_loss`，提交 9d9e66e）。KRDU 的结果：小剂量（1e-3）终点变好、路径中段变差、坡度略变平；大剂量（5e-3）全面变差，坡度调度坍缩成共享形状。罚项改变的是目标，不是可行集。
- 什么时候算"在五边上"（门控）是所有方案共同的难点。state 路径已有三种门：自门控（路径在成员锥内且方向对齐）、FAF 距离、真值。control 上复用同一套。

## 方案一：屏障过滤器

**是什么。** 安全强化学习里的"安全层"：策略照常输出动作，一个薄层在每一步把动作投影到不会越界的集合里。单个约束下这个投影有闭式解（Dalal 等人 2018），不需要 QP。

**怎么算。** 走廊写成屏障 h(x) = k·hw(d) − |xt| ≥ 0。点质量模型里横向位置由航向决定、航向变化率由坡度决定，h 相对坡度是二阶的，所以分两层一阶条件：

1. 位置层：给定到走廊边界的余量 (hw − |xt|)，允许的航向误差区间 ψ_err ∈ [−ψ_max, +ψ_max]。ψ_max 在中线附近大、贴近边界时趋于零，且只允许朝回中线的一侧。
2. 航向层：把 ψ_err 拉回区间所需的航向变化率区间，经 ψ̇ = g·n·sin μ /(V cos γ) 换成坡度区间 [μ_lo, μ_hi]（n、V、γ 取当前状态）。

区间由当前状态和跑道航向决定，都是解析式；梯度沿 rollout 传回前面的段。

**怎么压。** 训练回路里用 tanh 式软饱和：μ' = μ_c + w·(sat(μ_c; [μ_lo, μ_hi]) − μ_c)，w 是门控权重。软饱和是 C¹ 的，被夹住的步对命令仍有梯度；硬 clamp 在被夹住的步梯度为零（死区），只用于推理对照臂。

**屏障的收敛率 α。** 它决定过滤器多早介入：固定值要调；BarrierNet（Xiao 等人 2023）把 α 作为网络输出、随任务学出来，是这条路线的可微版本。第一版固定 α，"α 可学"作为一个开关，不另立臂。

**滞后。** 过滤器作用在命令上，实际坡度滞后 τ_bank = 2 s，屏障条件要留裕度（α 保守），并记录"实际坡度越出区间的步比例"。

**范围。** 第一版只做横向。下滑道窗口交给罚项或以后的载荷因子过滤器，避免两个过滤器在同一步抢命令。

**优点与短板。** 优点：最小干预，命令合规就原样通过，对已学好的调度几乎无影响；实现只在一处。短板：对门控生效时已在走廊外的路径不作为（屏障只定义在可行集内）；不管垂直；网络可能学会"懒"——命令偏得很远、指望过滤器夹回来。

**检验。**
- 单元测试：hook 恒等时 rollout 逐比特一致；硬版不变性（从走廊内出发、随机命令，每步都留在走廊内）；软版越出量有界；只开过滤器时控制量有非零梯度，且软版被夹步梯度非零、硬版为零；门控为零是恒等；诊断计数与手算一致。
- 读数：真值门控行上的走廊违反率下降，ADE/FDE 不退化（分层、两机场）；被夹步比例（> 20 % 且坡度技能低于基线减种子噪声，读作"懒"，否决）。
- 对照臂：训练不加、推理加，量"训练穿过约束"值多少。

## 方案二：名义跟踪律 + 有界残差

**是什么。** 残差策略学习（Silver 等人 2018、Johannink 等人 2019）和自动驾驶里"学习规划器 + 跟踪控制器 / 规则兜底"（SafetyNet，Vitelli 等人 2022）的分工，搬到 rollout 内部：一个已知稳定的控制律负责跟踪，网络负责有界的修正。也是真实自动驾驶仪的分工（LOC/GS 跟踪律 + 上层意图）。

**怎么算。** 门控生效的步上，网络命令被解释为对名义律的修正：

```
u_nominal(x_k)  横向：L1 制导（Park 等人 2004）把 xt 拉回中线，输出坡度
                纵向：下滑道跟踪，输出载荷因子
u_k'          = w_k · (u_nominal(x_k) + r(u_k)) + (1 − w_k) · u_k
r(u)          = tanh 有界残差；上限按观测航迹相对名义律的偏差标定（坡度约 ±5°）
```

门控关闭（w = 0）时与现在完全一样；门控打开时路径以名义律的收敛率趋向中线和下滑道，网络只能在残差上限内偏离。走廊保证不是逐点硬的，但由"名义律的收敛区间 + 残差上限"决定，可分析、可测。

**优点与短板。** 优点：从走廊外也能拉回；横向垂直一起覆盖；残差上限天然防"懒"；从锚点出发，没有起点问题。短板：名义律和数据打架的地方（真实航班在 10 km 处离中线 200 m 缓慢收敛）要靠残差上限留够；多两个要标定的量（L1 前视距离 / 增益、残差上限）；名义律本身要先对真实航迹验证。

**检验。**
- 单元测试：门控为零逐比特一致；残差为零、门控全开时，从走廊内任意偏移出发 |xt| 单调收敛、高度收敛到下滑道（含 τ_bank 滞后）；残差饱和 |u' − u_nominal| ≤ 上限；残差路径和门控路径对控制量都有非零梯度；名义律输出与逆动力学得到的观测坡度的相关性作为底线记录。
- 读数：同方案一；另加残差饱和步比例（> 30 % 读作"名义律与数据打架"，放宽上限或降增益）。
- 与方案一比较的关键维度：走廊外路径的回收率（门控生效时已在走廊外的行，之后进入走廊的比例）。

## 两个方案怎么选

| | 屏障过滤器 | 名义律 + 残差 |
|---|---|---|
| 对走廊内的路径 | 只在要越界时改 | 持续拉向中线 |
| 对走廊外的路径 | 不作为 | 拉回 |
| 垂直通道 | 第一版不管 | 覆盖 |
| "懒"的风险 | 有（要监控被夹步比例） | 残差上限限制 |
| 要标定的量 | α（或可学） | L1 增益、残差上限 |
| 与数据打架 | 少（最小干预） | 可能（名义律的收敛率 vs 真实汇入速率） |

预期：直线进近两者都能压违反率；雷达引导航班只有方案二可能改善（它们的问题是路径在走廊外）；方案一更不容易伤到 ADE。两者共用门控、几何、诊断和读数，各自一个模块，先并行跑一轮再定。

## 其他方案

- **罚项**：已完成，实验中（`control_procedure_20260905`），结果另有报告。
- **决策量 + 封闭末段**（P2）：网络只预测汇入距离和时长，汇入后按程序几何闭式给出，控制量由逆动力学得到（`control/dynamics/inverse.py` 已有）。走廊从约束变成定义。适合直线进近；雷达引导航班汇入前那段仍要网络出。若两个 P1 方案都在直线进近上过关但雷达引导层没有改善，再做它。
- **参考路径 + 制导律**（低）：网络输出有界参考路径，制导律跟踪它出控制量。是一条新的输出路径，改动大；它与方案二共用制导律模块。只有当两个 P1 都失败、且原因是"控制头本身学不出五边意图"时再考虑。
- **优化器跟踪求解、采样 + 筛选**（低）：见综述 §2.5、§2.6。
- **事后投影**：不适用于 control。

## 接入方式

三条规则，够用即可：

1. **rollout 加一个 `command_hook`**（`rollout_control_endpoints(..., command_hook=None)`），签名 `hook(state_k, command_k, context) -> command_k'`，在每个积分子步的命令处调用，默认恒等（恒等时逐比特一致是第一条测试）。两个方案各是一个 hook 实现，放在 `control/constraints/`；rollout 不 import 它们，方向只有约束模块 → dynamics。推理（`forecast` 的 control 批预测）用同一个 hook，训练和部署一致。记录里导出的控制量是过滤后的有效值。
2. **门控、几何、每航班常量复用 state 路径的**：`final_approach_geometry.py` 的公式和三种门；batch 上下文里的跑道航向和 tan GPA（`FINAL_APPROACH_KEYS`）。hook 不自己判断"在不在五边上"，它接收一个门控对象。
3. **诊断进 history，读数不加脚本**：每个 hook 报自己的计数（被夹步比例、残差饱和比例、门控权重 > 0.5 的步比例），走 `ControlLossTerms.diagnostics` 到 `EpochResult`；读数仍用 `compare_constraint_arms.py`（违反率、终点、分层配对）和 `score_control_arms.py`（坡度技能、共享分量），只加"走廊外回收率"一列。

config 字段照 `PROCEDURE_LOSS_FIELDS` 的做法对命名配方开放，run 名显示为 `simple-v3+(…)`。

## 实施顺序

| 顺序 | 内容 | 前置 |
|---|---|---|
| P0 | `command_hook` + 恒等测试；门控对象化；制导律模块（L1 + 下滑跟踪，方案二用） | 无 |
| P1 | 两个 hook 各带测试；臂：基线、`F_barrier_soft`、`F_barrier_infer`（训练不加、推理硬夹）、`R_nominal_residual`；两机场一个种子；读数加走廊外回收率 | P0；罚项报告决定是否叠 λ = 1e-3 |
| P2 | 决策量 + 封闭末段 | P1 结论 |
| 不做 | 硬 clamp 进训练回路；把过滤器写成损失；事后改位置；改动力学方程 | |

## 阅读

- Dalal et al., *Safe Exploration in Continuous Action Spaces*, arXiv 2018：单约束闭式安全层（方案一的原型）。
- Pham, De Magistris, Tachibana, *OptLayer*, ICRA 2018：多约束时的可微 QP 投影层。
- Xiao et al., *BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control*, IEEE T-RO 2023：可微屏障层，类-K 函数随网络学出。
- Taylor, Singletary, Yue, Ames, *Learning for Safety-Critical Control with Control Barrier Functions*, L4DC 2020；Cheng, Orosz, Murray, Burdick, *End-to-End Safe RL through Barrier Functions*, AAAI 2019。
- Fisac et al., *A General Safety Framework for Learning-Based Control in Uncertain Robotic Systems*, IEEE TAC 2019：可达性安全过滤器（屏障解析式不够用时的下一步）。
- Silver et al., *Residual Policy Learning*, arXiv 2018；Johannink et al., *Residual Reinforcement Learning for Robot Control*, ICRA 2019（方案二的原型）。
- Vitelli et al., *SafetyNet: Safe Planning for Real-World Self-Driving Vehicles Using Machine-Learned Policies*, ICRA 2022：学习规划器 + 规则安全检查 + 兜底轨迹。
- Park, Deyst, How, *A New Nonlinear Guidance Logic for Trajectory Tracking*, AIAA GNC 2004：L1 横向制导律（方案二的名义律）。
- Brunke et al., *Safe Learning in Robotics*, Annual Review of Control, Robotics, and Autonomous Systems 2022；Dawson, Gao, Fan, *Safe Control With Learned Certificates*, IEEE T-RO 2023：综述。
- Achiam et al., *Constrained Policy Optimization*, ICML 2017：罚项 / 对偶一脉，作对照。
