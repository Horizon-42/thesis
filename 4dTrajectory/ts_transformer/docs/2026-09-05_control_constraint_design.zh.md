# control 输出的约束设计（dev 文档，2026-09-05）

承接 `2026-09-04_constraint_methods_survey.zh.md`（方法综述）和 `2026-09-05_final_constraint_results.zh.md`（state 输出上的结果：有界输出采纳、罚项否决）。本文回答：**同一个五边约束（LPV 走廊 + 下滑道窗口）在 control 输出上除了罚项还能怎么加，模块怎么切，先做哪个。** 罚项臂（`control_procedure_20260905`）正在跑，它的结果另有报告；这里的设计不依赖那个结果，但 §6 的顺序会按它调整。2026-09-05 第二版：对照机器人 / 自动驾驶里"学习策略 + 安全层"的做法复核了一遍施加方式（§3、§5），把参考路径 + 制导律降为低优先级（§4）。

## 0. control 路径的现状

```
历史 → 主干 → 控制头（推力比、坡度、载荷因子 × N 段 + 分段时长）
     → 可微 RK4 rollout（control/dynamics/rollout.py，点质量 + 一阶滞后）
     → 分段端点状态 → 位置 / 速度 / 终端 / 模仿 / 罚项
```

- 已有的硬约束只有一层：控制头用 sigmoid 把三个控制量压进包线盒（`control/envelope.py`）。约束对象是包线，不是走廊。
- rollout 全程在反向传播路径上（`test_control_dataset_and_rollout_loss_form_one_differentiable_training_step`），梯度经积分器传回控制量和分段时长；模仿项是另一条不经过 rollout 的路径。
- 罚项（`procedure_loss`）已接到 rollout 的分段端点状态上（提交 9d9e66e）；它是软约束，改变的是目标，不是可行集。
- state 路径的经验：训练穿过约束（B 臂）优于事后投影（P0）；罚项改变违反率、改不了终点误差；门控（什么时候算"在五边上"）是所有方式共同的设计难点。

## 1. 模块划分

原则：把"约束是什么"、"在哪些行/步生效"、"怎么施加"、"施加在动力学的哪一点"、"每航班常量从哪来"、"怎么诊断"、"怎么读数"七件事分开，每件一个模块，只通过窄接口相连。新方式加进来时，只应新增一个"施加"模块和一个 config 开关，其余不动。

| 层 | 模块 | 现状 | 接口 |
|---|---|---|---|
| 几何 | `final_approach_geometry.py`（顶层，state/control 共用） | 已有 | 纯函数：`runway_axes`、`corridor_halfwidth`、`glidepath_height`、`corridor_violations`、`bound_to_final`；只能增加纯函数 |
| 门控 | 同上，`membership(gate, …)` / `truth_final_gate` | 已有，但和几何混在一个文件 | 抽成协议 `FinalGate`：`soft(rows) -> [B,N] ∈ [0,1]`、`hard(rows) -> bool`；实现 `OnFinalGate`（自门控，读预测位置的路径方向）、`FafGate`、`TruthGate`（训练用，读真值行）。使用者只依赖协议，不知道是哪种门 |
| 施加（enforcer） | 见 §2，每种一个模块 | 罚项、StateBounder 已有 | 各自独立，互不 import，各由一个 config 字段开关 |
| 动力学接入点 | `control/dynamics/rollout.py` | 无 | RK4 步前的 `command_hook: Callable[[state_k, command_k, context], command_k']`，默认恒等；施加模块以 hook 形式注入，rollout 不 import 约束模块（方向：约束模块 → dynamics，不反向；`test_architecture` 的规则保持） |
| 上下文 | `dataset.final_approach_arrays`（`FINAL_APPROACH_KEYS`） | 已有；control 的 dynamics 字典带 `runway_heading_rad`、`glidepath_tan` | 施加模块只能从 batch 的 context 取每航班常量；要新常量先加进这一处 |
| 诊断 | `LossComponents.diagnostics` / `ControlLossTerms.diagnostics` → `EpochResult.procedure` | 已有（违反率、λ） | 每个施加模块报告自己的计数；history 里按模块名分组，不进 `total` |
| 读数 | `docs/compare_constraint_arms.py`（真值门控行上的违反率、终点、分层配对）、`docs/score_control_arms.py`（坡度技能、共享分量、平滑度）、`flyability` | 已有 | 新方式不新增读数脚本，只加列 |

放置规则（`tests/test_architecture.py`）：只被 control 消费的模块放 `control/`（`control/constraints/bank_filter.py`），被两条路径共用的门控和几何留在顶层。

## 2. 施加方式与模块

| 方式 | 模块 | 作用点 | 保证 | 可飞性 | 改动量 | 优先级 |
|---|---|---|---|---|---|---|
| 罚项 | `train.procedure_loss`（已有） | 目标函数，rollout 端点状态 | 软 | 保持 | 已完成 | 实验中 |
| 每步屏障过滤器（安全层） | `control/constraints/barrier_filter.py`（新） | rollout 每步的命令（第一版只夹坡度） | 门控步上硬：**不让越出**，对已在走廊外的路径不作为 | 保持（只改命令，位置仍由积分得到） | 中 | P1 |
| 名义跟踪律 + 有界残差 | `control/constraints/nominal_residual.py`（新） | rollout 每步的命令（坡度 + 载荷因子） | 近硬：名义律**把路径拉向**中线 / 下滑道，残差有界 | 保持 | 中 | P1 |
| 决策量 + 封闭末段 | 网络预测汇入距离、时长；汇入后几何由程序固定 | 输出参数化 | 硬 | 需末段控制律 | 中 | P2 |
| 参考路径 + 制导律 | 新输出路径，网络只出有界参考 | 输出参数化 | 硬（参考）+ 可飞（制导律） | 保持 | 大 | 低（§4） |
| 优化器跟踪求解 | `collocation/optimizer.py` 新目标项 | 推理期 | 硬且可飞 | 保证 | 大（综述 §2.6） | 低 |
| 采样 + 筛选 | 需要随机控制头 | 推理期 | 视筛选 | 保持 | 阻塞于多模态 | 低 |

事后投影（state 上的 P0）对 control 不适用：改位置会破坏动力学一致性，正好抵消 control 路径的意义。

两个 P1 候选都以 `command_hook` 的形式进 rollout，接口相同，差别在语义：屏障过滤器是"最小干预"（命令合规就原样通过，只在要越界时改），来自安全强化学习的安全层一脉；名义律 + 残差是"分工"（一个已知稳定的控制律负责跟踪，网络负责有界的修正），来自残差策略学习和自动驾驶里"学习规划器 + 跟踪控制器"的分工。关键差别是对**已经在走廊外**的路径：过滤器无事可做（屏障只定义在可行集内），名义律会把它拉回来。state 路径的经验（A 臂 55–77 % 的真值门控行在走廊外）说明"拉回来"才是需要的行为，所以两个都做，先比。

## 3. 两个 P1 候选

### 3A. 每步屏障过滤器（`BarrierFilter`，安全层）

#### 3A.1 想法

走廊写成屏障 h(x) = k·hw(d) − |xt| ≥ 0。点质量模型里横向运动由航向决定、航向变化率由坡度决定，所以 h 相对坡度是二阶的。做法是分两层一阶条件：

1. 位置层：给定 |xt| 和到边界的余量，允许的航向误差区间 ψ_err ∈ [−ψ_max, +ψ_max]，ψ_max 随 (hw − |xt|) 缩小（在中线附近可以大，贴近边界时趋于 0，并且方向上只允许朝回中线的一侧）。
2. 航向层：把 ψ_err 拉回区间所需的 ψ̇ 区间，经 ψ̇ = g·n·sin μ /(V cos γ) 解析换成坡度区间 [μ_lo, μ_hi]（n、V、γ 取当前状态）。

两层都是解析式，不需要 QP：单个约束下"把动作投影到线性化的可行集"有闭式解（Dalal 等人的安全层就是这么做的），多约束才需要 OptLayer / cvxpylayers 那类 QP 层。区间由当前状态 x_k 和上下文（跑道航向）决定，梯度沿现有 rollout 传回前面的段。

屏障的"收敛率" α（类-K 函数）决定过滤器多早介入：固定 α 要调；BarrierNet 的做法是把 α 作为网络输出的一部分、随任务学出来（可微屏障层）。这里第一版用固定 α，把"α 可学"作为 P1 内部的一个开关，不另立臂。

#### 3A.2 接口

```python
class CommandHook(Protocol):                     # rollout 的接入点，所有施加模块共用
    def __call__(self, state_k, command_k, context) -> command_k_filtered  # 逐步，可微
    def diagnostics(self) -> dict[str, Tensor]   # 累计计数，进 history

class BarrierFilter(CommandHook): ...            # 被夹步比例、|Δμ| 均值、门控权重>0.5 的步比例
```

- 注入：`rollout_control_endpoints(..., command_hook=bank_filter)`；`command_hook=None` 时 rollout 与现在逐比特一致（这是第一条单元测试）。
- 门控：过滤器自己不判断是否在五边上，它接收一个 `FinalGate` 实例，`w = gate.soft(state_k)`，输出 μ' = μ_c + w·(sat(μ_c; [μ_lo, μ_hi]) − μ_c)。
- 饱和函数：训练回路里用 tanh 式软饱和（与 `bounded_cross_track` 同形，C¹，梯度不为零）；硬 clamp 只在推理臂里作为对照。不用硬 clamp 训练：被夹住的步对命令的梯度为零（死区）。
- 一阶滞后：过滤器作用在命令上，实际坡度滞后 τ_bank = 2 s；屏障条件要留裕度（α 取保守值，并在诊断里记"实际坡度越出区间的步比例"）。
- 垂直：第一版只做横向。下滑道窗口由罚项或以后的载荷因子过滤器处理，避免两个过滤器在同一步互相抢命令。

#### 3A.3 config

```
control_bank_filter: str = "off"        # off | soft | hard        （hard 只允许在预测时开）
control_bank_filter_gate: str = "on-final"   # 复用 CORRIDOR_GATES
control_bank_filter_alpha: float        # 屏障收敛率
control_bank_filter_heading_max_deg: float   # 中线附近允许的最大航向误差
```

命名规则沿用 `run_naming`：这些字段进 `CONTROL_LOSS_FIELDS` 旁边的新列表 `CONTROL_ENFORCER_FIELDS`，同样对命名配方开放（像 `PROCEDURE_LOSS_FIELDS`）。

#### 3A.4 单元测试（写在实现之前）

1. hook 恒等 → rollout 输出与现有实现逐比特相同。
2. 硬版不变性：从走廊内任意状态出发、随机命令序列，rollout 的每一步都留在走廊内（门控全开）。
3. 软版有界性：越出量有上界，且随饱和尺度趋零。
4. 梯度：只开过滤器时 `controls` 有非零梯度；被夹步的命令梯度在软版下非零、硬版下为零（把死区写成测试，防止以后误用）。
5. 门控为零时过滤器是恒等。
6. 诊断计数与手算一致。

#### 3A.5 预注册读法

- 主指标：真值门控行上的走廊违反率下降；ADE/FDE 不退化（分层，两机场）。
- 否决：被夹步比例超过阈值（建议 20 %）且坡度技能低于基线减种子噪声，读作"网络学会了懒"——命令偏得很远、指望过滤器夹回来。
- 对照臂：训练不加、推理加（安全过滤器的经典用法），量"训练穿过约束"值多少，与 state 上的 B 对 P0 同构。
- 已知短板：对门控生效时已在走廊外的路径不作为；第一版不管垂直通道。

### 3B. 名义跟踪律 + 有界残差（`NominalResidual`）

#### 3B.1 想法

在门控生效的步上，把网络的命令解释为对一个固定跟踪律的有界修正，而不是绝对命令：

```
u_nominal(x_k)  = 横向：L1 / 比例导引把 xt 拉回中线（输出坡度）；纵向：下滑道跟踪（输出载荷因子）
u_k'            = w_k · (u_nominal(x_k) + r(u_k))  +  (1 − w_k) · u_k
r(u)            = 有界残差（tanh，幅度上限按观测航迹相对名义律的偏差标定，坡度约 ±5°）
```

w_k 是门控权重（同一套 `FinalGate`）。门控关闭时 u_k' = u_k，与现在完全一样；门控打开时路径以名义律的收敛率趋向中线和下滑道，网络只能在残差上限内偏离。走廊保证不是逐点硬的，但由"名义律的收敛区间 + 残差上限"决定，可分析、可测。

这是残差策略学习（Silver 2018、Johannink 2019）和自动驾驶里"学习规划器 + 跟踪控制器 / 规则兜底"（SafetyNet 一类）的做法搬到 rollout 内部；也是真实自动驾驶仪的分工（LOC/GS 跟踪律 + 上层意图）。相对 3A 的优点：从走廊外也能拉回；同时覆盖横向和垂直；残差上限天然限制"懒"。缺点：名义律和数据打架的地方（真实航班在 10 km 处离中线 200 m 缓慢收敛）要靠残差上限留够；多两个要标定的量（L1 距离 / 增益、残差上限）。

#### 3B.2 接口与 config

同 `CommandHook`；`control/constraints/nominal_residual.py` 内含名义律（`control/guidance_laws.py` 单独放 L1 和下滑跟踪，将来参考路径方案（§4）也用它）。

```
control_nominal_residual: str = "off"     # off | on
control_nominal_residual_gate: str = "on-final"
control_nominal_l1_distance_m: float      # L1 前视距离
control_nominal_residual_bank_max_deg: float
control_nominal_residual_load_max: float
```

诊断：残差饱和步比例、名义律与真值坡度的差（名义律本身对数据的拟合度，作为底线）。

#### 3B.3 单元测试

1. 门控为零 → 与现有 rollout 逐比特一致。
2. 残差为零、门控全开：从走廊内任意偏移出发，名义律使 |xt| 单调收敛到中线、高度收敛到下滑道（在滞后 τ_bank 下仍收敛）。
3. 残差饱和：|u' − u_nominal| ≤ 上限。
4. 梯度：残差路径和门控路径对 `controls` 都有非零梯度。
5. 名义律对真实航迹的复现：用逆动力学得到的观测坡度与名义律输出的相关性作为底线（记录，不断言阈值）。

#### 3B.4 预注册读法

- 主指标同 3A；额外：残差饱和步比例（> 30 % 读作"名义律与数据打架"，上限该放宽或增益该降）。
- 与 3A 的比较维度：走廊外路径的回收率（门控生效时已在走廊外的行，之后进入走廊的比例）；这是两者的本质差别。

## 4. 低优先级：参考路径 + 制导律

网络输出有界参考路径（复用 `StateOutputLayer` 的 corridor-bounded 输出），rollout 内用固定制导律跟踪参考生成控制量。它和 3B 共用制导律模块，差别是网络输出的是路径而不是残差控制量，因此是一条新的 `prediction_output`，改动大。按用户决定降为低优先级：只有当 3A、3B 都失败、且失败原因是"控制头本身学不出五边意图"时再考虑。

## 5. 参考阅读（跨领域）

安全层 / 动作投影：
- Dalal et al., *Safe Exploration in Continuous Action Spaces*, arXiv 2018：单约束线性化投影的闭式安全层（3A 的直接原型）。
- Pham, De Magistris, Tachibana, *OptLayer*, ICRA 2018：动作空间的可微 QP 投影层（多约束时用）。
- Xiao et al., *BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control*, IEEE T-RO 2023：可微屏障层，类-K 函数随网络学出。
- Taylor, Singletary, Yue, Ames, *Learning for Safety-Critical Control with Control Barrier Functions*, L4DC 2020；Cheng, Orosz, Murray, Burdick, *End-to-End Safe RL through Barrier Functions*, AAAI 2019。
- Fisac et al., *A General Safety Framework for Learning-Based Control in Uncertain Robotic Systems*, IEEE TAC 2019：可达性安全过滤器（最不受限的干预，若屏障解析式不够用时的下一步）。

残差 / 分工：
- Silver et al., *Residual Policy Learning*, arXiv 2018；Johannink et al., *Residual Reinforcement Learning for Robot Control*, ICRA 2019（3B 的原型）。
- Vitelli et al., *SafetyNet: Safe Planning for Real-World Self-Driving Vehicles Using Machine-Learned Policies*, ICRA 2022：学习规划器 + 规则安全检查 + 兜底轨迹的工业分工。
- Park, Deyst, How, *A New Nonlinear Guidance Logic for Trajectory Tracking*, AIAA GNC 2004：L1 横向制导律（3B 的名义律）。

综述：
- Brunke et al., *Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning*, Annual Review of Control, Robotics, and Autonomous Systems 2022。
- Dawson, Gao, Fan, *Safe Control With Learned Certificates: A Survey of Neural Lyapunov, Barrier, and Contraction Methods*, IEEE T-RO 2023。
- Achiam et al., *Constrained Policy Optimization*, ICML 2017（罚项 / 对偶一脉，已在 state 上否决，列作对照）。

## 5b. 决策量 + 封闭末段

网络预测汇入距离 d_join 和总时长，汇入点之后的路径按程序几何（沿中线、沿下滑道）闭式给出，控制量由逆动力学得到（`control/dynamics/inverse.py` 已有）。走廊不再是约束而是定义。适合直线进近；雷达引导航班汇入前那段仍要网络出。P2。

## 6. 实施顺序

| 顺序 | 内容 | 前置 | 读数 |
|---|---|---|---|
| P0 | 门控抽成 `FinalGate` 协议；rollout 加 `command_hook`（恒等时逐比特一致的测试）；`control/guidance_laws.py`（L1 + 下滑跟踪，3B 用） | 无 | 单元测试 |
| P1 | 两个施加模块：`BarrierFilter`（3A，横向）与 `NominalResidual`（3B，横向 + 垂直），各带自己的测试；臂：`A_control_v3`（已有）、`F_barrier_soft`、`F_barrier_infer`（训练不加、推理硬夹）、`R_nominal_residual`；两机场一个种子 | P0；罚项臂的结果决定是否同时带 λ = 1e-3 | `compare_constraint_arms` + `score_control_arms` + flyability + 走廊外回收率 |
| P2 | 决策量 + 封闭末段（§5 原第 5 节的思路），若 P1 两者都在直线进近上过关但雷达引导层没有改善 | P1 | 同上 |
| 低 | 参考路径 + 制导律；优化器跟踪；采样 + 筛选 | 见综述 | |
| 不做 | 硬 clamp 进训练回路；把过滤器写成损失；事后改位置的投影；改动力学方程 | | |

## 7. 与 state 路径的对称性

| | state 输出 | control 输出 |
|---|---|---|
| 软约束 | 罚项（否决） | 罚项（实验中） |
| 构造上硬 / 近硬 | 有界位置输出 B（采纳） | 屏障过滤器、名义律 + 有界残差（P1） |
| 事后 | 投影（上限/兜底） | 不适用 |
| 门控 | on-final / faf / 真值 | 同一套 |
| 起点问题 | 未解决（state-v3） | 天然没有（rollout 从锚点出发） |

两条路径各有"构造上硬"的方式，一个夹位置、一个夹命令，共用几何、门控、上下文和读数。
