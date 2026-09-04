# control 输出的约束设计（dev 文档，2026-09-05）

承接 `2026-09-04_constraint_methods_survey.zh.md`（方法综述）和 `2026-09-05_final_constraint_results.zh.md`（state 输出上的结果：有界输出采纳、罚项否决）。本文回答：**同一个五边约束（LPV 走廊 + 下滑道窗口）在 control 输出上除了罚项还能怎么加，模块怎么切，先做哪个。** 罚项臂（`control_procedure_20260905`）正在跑，它的结果另有报告；这里的设计不依赖那个结果，但 §6 的顺序会按它调整。

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

| 方式 | 模块 | 作用点 | 保证 | 可飞性 | 改动量 |
|---|---|---|---|---|---|
| 罚项 | `train.procedure_loss`（已有） | 目标函数，rollout 端点状态 | 软 | 保持 | 已完成 |
| 每步坡度过滤器 | `control/constraints/bank_filter.py`（新） | rollout 每步的坡度命令 | 门控步上硬（软饱和形式为"近硬"） | 保持（只改命令，位置仍由积分得到） | 中 |
| 参考路径 + 制导律 | 新输出路径 `prediction_output="reference"`：复用 `StateOutputLayer` 的有界输出作参考，`control/guidance.py` 用固定制导律出控制量 | 输出参数化 | 硬（参考）+ 可飞（制导律） | 保持 | 大 |
| 决策量 + 封闭末段 | 网络预测汇入距离、时长；汇入后几何由程序固定 | 输出参数化 | 硬 | 需末段控制律 | 中 |
| 优化器跟踪求解 | `collocation/optimizer.py` 新目标项 | 推理期 | 硬且可飞 | 保证 | 大（见综述 §2.6） |
| 采样 + 筛选 | 需要随机控制头 | 推理期 | 视筛选 | 保持 | 阻塞于多模态 |

事后投影（state 上的 P0）对 control 不适用：改位置会破坏动力学一致性，正好抵消 control 路径的意义。

## 3. 核心新模块：每步坡度过滤器（`BankFilter`）

### 3.1 想法

走廊写成屏障 h(x) = k·hw(d) − |xt| ≥ 0。点质量模型里横向运动由航向决定、航向变化率由坡度决定，所以 h 相对坡度是二阶的。做法是分两层一阶条件：

1. 位置层：给定 |xt| 和到边界的余量，允许的航向误差区间 ψ_err ∈ [−ψ_max, +ψ_max]，ψ_max 随 (hw − |xt|) 缩小（在中线附近可以大，贴近边界时趋于 0，并且方向上只允许朝回中线的一侧）。
2. 航向层：把 ψ_err 拉回区间所需的 ψ̇ 区间，经 ψ̇ = g·n·sin μ /(V cos γ) 解析换成坡度区间 [μ_lo, μ_hi]（n、V、γ 取当前状态）。

两层都是解析式，不需要 QP。区间由当前状态 x_k 和上下文（跑道航向）决定，梯度沿现有 rollout 传回前面的段。

### 3.2 接口

```python
class BankFilter(Protocol):
    def __call__(self, state_k, command_k, context) -> command_k_filtered  # 逐步，可微
    def diagnostics(self) -> dict[str, Tensor]   # 累计：被夹步比例、|Δμ| 均值、门控权重>0.5 的步比例
```

- 注入：`rollout_control_endpoints(..., command_hook=bank_filter)`；`command_hook=None` 时 rollout 与现在逐比特一致（这是第一条单元测试）。
- 门控：过滤器自己不判断是否在五边上，它接收一个 `FinalGate` 实例，`w = gate.soft(state_k)`，输出 μ' = μ_c + w·(sat(μ_c; [μ_lo, μ_hi]) − μ_c)。
- 饱和函数：训练回路里用 tanh 式软饱和（与 `bounded_cross_track` 同形，C¹，梯度不为零）；硬 clamp 只在推理臂里作为对照。不用硬 clamp 训练：被夹住的步对命令的梯度为零（死区）。
- 一阶滞后：过滤器作用在命令上，实际坡度滞后 τ_bank = 2 s；屏障条件要留裕度（α 取保守值，并在诊断里记"实际坡度越出区间的步比例"）。
- 垂直：第一版只做横向。下滑道窗口由罚项或以后的载荷因子过滤器处理，避免两个过滤器在同一步互相抢命令。

### 3.3 config

```
control_bank_filter: str = "off"        # off | soft | hard        （hard 只允许在预测时开）
control_bank_filter_gate: str = "on-final"   # 复用 CORRIDOR_GATES
control_bank_filter_alpha: float        # 屏障收敛率
control_bank_filter_heading_max_deg: float   # 中线附近允许的最大航向误差
```

命名规则沿用 `run_naming`：这些字段进 `CONTROL_LOSS_FIELDS` 旁边的新列表 `CONTROL_ENFORCER_FIELDS`，同样对命名配方开放（像 `PROCEDURE_LOSS_FIELDS`）。

### 3.4 单元测试（写在实现之前）

1. hook 恒等 → rollout 输出与现有实现逐比特相同。
2. 硬版不变性：从走廊内任意状态出发、随机命令序列，rollout 的每一步都留在走廊内（门控全开）。
3. 软版有界性：越出量有上界，且随饱和尺度趋零。
4. 梯度：只开过滤器时 `controls` 有非零梯度；被夹步的命令梯度在软版下非零、硬版下为零（把死区写成测试，防止以后误用）。
5. 门控为零时过滤器是恒等。
6. 诊断计数与手算一致。

### 3.5 预注册读法

- 主指标：真值门控行上的走廊违反率下降；ADE/FDE 不退化（分层，两机场）。
- 否决：被夹步比例超过阈值（建议 20 %）且坡度技能低于基线减种子噪声，读作"网络学会了懒"——命令偏得很远、指望过滤器夹回来。
- 对照臂：训练不加、推理加（安全过滤器的经典用法），量"训练穿过约束"值多少，与 state 上的 B 对 P0 同构。

## 4. 第二候选：参考路径 + 制导律

网络输出有界参考路径（复用 `StateOutputLayer` 的 corridor-bounded 输出，走廊由构造保证），rollout 内用固定制导律生成控制量：横向用 L1 / 比例导引跟踪参考的横向偏差，纵向跟踪参考高度和速度。网络只学"意图"，可飞性由制导律保证，走廊由参考保证，"起点先验"也自然解决（制导律从锚点出发）。接口：`GuidanceLaw.controls(state_k, reference_window) -> control_k`，同样以 hook 形式进 rollout，控制头退化为参考头。这是一条新的 `prediction_output`，评估链不变。改动大，放在过滤器之后，若过滤器出现"懒"或垂直方向压不住再做。

## 5. 决策量 + 封闭末段

网络预测汇入距离 d_join 和总时长，汇入点之后的路径按程序几何（沿中线、沿下滑道）闭式给出，控制量由逆动力学得到（`control/dynamics/inverse.py` 已有）。走廊不再是约束而是定义。适合直线进近；雷达引导航班汇入前那段仍要网络出。作为过滤器的补充实验，不单独立项。

## 6. 实施顺序

| 顺序 | 内容 | 前置 | 读数 |
|---|---|---|---|
| P0 | 门控抽成 `FinalGate` 协议；rollout 加 `command_hook`；`BankFilter` 模块 + §3.4 六条测试 | 无 | 单元测试 |
| P1 | 臂：`A_control_v3`（已有）、`F_bank_soft`（训练进回路，软饱和）、`F_bank_infer`（训练不加，推理硬夹）；两机场一个种子 | P0；罚项臂的结果决定是否同时带 λ = 1e-3 | `compare_constraint_arms` + `score_control_arms` + flyability |
| P2 | 参考路径 + 制导律原型 | P1 出现"懒"或垂直问题 | 同上 |
| 不做 | 硬 clamp 进训练回路；把过滤器写成损失；事后改位置的投影；改动力学方程 | | |

## 7. 与 state 路径的对称性

| | state 输出 | control 输出 |
|---|---|---|
| 软约束 | 罚项（否决） | 罚项（实验中） |
| 构造上硬 | 有界位置输出 B（采纳） | 每步坡度过滤器（P0/P1） |
| 事后 | 投影（上限/兜底） | 不适用 |
| 门控 | on-final / faf / 真值 | 同一套 |
| 起点问题 | 未解决（state-v3） | 天然没有（rollout 从锚点出发） |

两条路径各有一个"构造上硬"的方式，一个夹位置、一个夹坡度，共用几何、门控、上下文和读数。
