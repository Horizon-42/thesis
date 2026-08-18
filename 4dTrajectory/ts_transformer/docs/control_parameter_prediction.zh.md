# control 参数预测：架构、消融轴与模块状态

> **2026-08-18 状态说明（先读这段）**
>
> 本文描述的是 **2026-08-16 的代码状态**。2026-08-18 的 control 设计梳理之后，下面
> 第 2 节的消融矩阵有相当一部分已经被删除，第 6 节的模块状态表随之过期。仍然准确的
> 是：第 1 节的两层配方机制、第 3 节的调用链、第 4 节的 oracle-teacher 生产链路结构
> （其中的两个 bug 已修，见新文）、第 5 节的诊断脚本用途、第 8 节对架构本身的评价。
>
> 已删除的轴与模块（各自附有当时的否定结论）：`control-mixture`、`direct` duration、
> `trim-residual`、`physical-criteria` 与 `terminal-state` objective 及其
> checkpoint-selection metric、`progressive_pretraining`、`rollout_finetuning`、
> `benchmark_validation_execution.py`、`control_oracle_curriculum` 的重复实现。
> 第 8 节 a/b/c/d/e 五条建议均已落实。
>
> 新增的轴：`control_dynamics_model`（`point-mass` / `first-order-lag`）与配方
> `simple-v1-lag`；控制量契约改为无量纲。
>
> 当前参考：**`docs/2026-08-18_control_dynamics_lag_model.zh.md`**。

本文档取代 `docs/normalized_time_and_control_output.zh.md` 作为 `prediction_output=control`
（学习控制量 + 可微动力学 rollout）的当前参考。旧文档描述的是最初版本（联合
`kinematic_consistency`/`terminal_loss` 目标、单一 factorized duration、单一 rollout
后端），此后经过 2026-07-28 至 2026-08-16 共约 20 篇实验记录（`docs/2026-07-28_*` 到
`docs/2026-08-16_*`）的持续消融，代码已演化出一套完整的、按 `TSConfig` 字段分派的策略
矩阵。本文只描述 **当前代码的真实行为**（截至 2026-08-16 的最后一批提交），逐条对照
源码函数/注册表，不复述已被取代的设计。

**读者假设**：已读过 README「[State baseline and dynamics-constrained control
output](../README.md#state-baseline-and-dynamics-constrained-control-output)」一节，
了解 `prediction_output=state|control` 的基本切换、controls 契约
`(thrust_N, bank_rad, load_factor)` 和非均匀 `segment_durations`。

**已知空白**：截至本文写作时，`control`（含 `simple-v1`）从未在 README 那种
"Historical results on real KRDU data" 表格里发布过一次完整训练后的
ADE/FDE/gate-pass 结果——本文和 `2026-08-16_control_simple_v1_development.zh.md`
记录的都是配方本身的设计推导和局部/单飞行/消融级别的数字，不是一次正式
train→predict→evaluate 全流程之后的机场级结果表。这是目前"架构已完成、
度量未完成"的最大缺口，`run_ts_simple_teacher_paired_cv.py` 是解决它的
在跑实验（见第 4.2 节），但截至本文写作没有已发布的结果。

---

## 1. 两层配方：`custom` 与 `simple-v1`

`TSConfig.control_recipe_name`（`config.py:282`）是新增的顶层开关：

- **`custom`**（默认）— 保留每一个历史实验模式；`TSConfig` 的其余 ~40 个
  `control_*` 字段可以任意组合（受 `__post_init__` 的交叉校验约束，见第 3 节）。
  已有 checkpoint 全部落在这一档。
- **`simple-v1`** — `control_simple_v1_overrides()`（`config.py:204-270`）冻结的
  **一套具体取值**。`__post_init__`（`config.py:518-532`）强制：只要
  `control_recipe_name="simple-v1"`，这些字段的实际值必须逐一等于冻结值，否则在
  构造 `TSConfig` 时直接抛 `ValueError`（列出每个字段的 实际值/期望值）。这不是
  "默认值恰好如此"，而是构造期强校验的**冻结契约**。

`simple-v1` 是 2026-08-16 那一批提交（`config.py`/`train.py`/`__main__.py` 单日各
30+ 次提交）落地的产物，是当前推荐的、被最新根级脚本
（`run_ts_simple_teacher_paired_cv.py` 等，见第 8 节）当作基线使用的配方。README
的 `train --control-recipe simple-v1 ...` 示例就是这一档。

### simple-v1 的具体取值（对照 `control_simple_v1_overrides()`）

| 轴 | simple-v1 取值 | 对照的历史/其他取值 |
|---|---|---|
| `control_duration_parameterization` | `uniform`（无 duration head，`final_time/N` 等分） | `factorized`（原始默认）、`direct` |
| `control_value_parameterization` | `absolute` | `trim-residual` |
| `control_dynamics_backend` | `scaled-transport-chart-velocity` | `reanchored-rk4`（原始默认）、`transport-chart-velocity` |
| `control_state_supervision_clock` | `observed` | `predicted` |
| `control_state_loss_grid` | `native-segment-endpoints` | `fixed-dt` |
| `control_state_objective` | `true-time-position`（最新注册的目标，见 2.2） | `normalized-mse`、`physical-criteria`、`terminal-state`、`arc-length-geometry` |
| `control_terminal_supervision_clock` | `state-supervision`（no-op） | `predicted`、`predicted-detached-time` |
| `control_horizon_curriculum_s` | `()`（关闭） | 非空 tuple 开启分阶段 single-shooting |
| 各 arc/terminal/dense 权重 | 全部为 `0.0`（`control_dense_state_loss_weight`、`control_geometry_loss_weight`、`control_arc_*_weight`、`control_terminal_position/velocity_loss_weight`） | `terminal-state`/`arc-length-geometry` 配方下非零 |
| `final_time_loss_weight` / `state_endpoint_loss_weight` | `1.0` / `0.25` | — |
| `n_segments` | `64` | 状态输出路径默认按模型 16/256 |
| `aircraft_filter` | `openap-direct` | `all` |

净效果：simple-v1 只保留「物理 3D 路径 MSE + 软终点位置 + final_time MSE」（见
2.2 的 `_true_time_position_objective`），删掉了 duration head、trim-residual
分支、effort/smoothness 正则、horizon curriculum、以及全部 arc/terminal 辅助项。
这是目前代码里**最简**的 control 配方，"simple-v1" 之名即由此而来。

**重要澄清：`uniform` 不是三种 duration 参数化正面对照后的赢家。**
`2026-08-16_control_simple_v1_development.zh.md` 的原话是"彻底移除未受有效
梯度训练的 duration projection"——是作为一次删复杂度的工程决策直接引入的，
没有做 factorized/direct/uniform 三者在同一 recipe 下的正面消融。`direct`
自己的对照实验（`2026-07-30_direct_duration_ablation.zh.md`）结果是负面的
（ADE +21.0%、FDE +36.6%，怀疑模型在利用极端时长偏斜钻采样密度空子），但
该文档自己标注对比用的基线 checkpoint 已过期，**从未重跑过**，所以是"初步
证据下判负"，不是被正式关闭的方向。`factorized` 仍是 `custom` 档的字段默认值
（2026-08-01/08-02 那一整条 arc-length-geometry 实验链都在用它）。

---

## 2. 完整消融矩阵（`custom` 档可用的全部轴）

每个轴都是一个「按 `config.<field>` 取值查表」的注册表，不是散落的 `if/elif`。
这是好消息：新增一个策略只需要在对应字典里加一项，不会牵动调用方。下面逐轴列出
注册表位置、可选值、以及 simple-v1 选了哪一个。

### 2.1 duration 参数化 — `control_duration_parameterization`

分派在 `models.py:144-155`（`_build_control_output` 的 `builders` 字典，键是
`(duration_param, value_param)` 二元组）：

| 取值 | 模型类 | 文件 | 机制 |
|---|---|---|---|
| `factorized`（原始默认） | `ControlOutputModel` | `control_models.py` | `ControlOutputHead`（`prediction_outputs.py`）内置：一个 softplus 总时长 + 一个 softmax 划分，`control_duration_uniform_floor` 控制划分与均匀分布的混合比例 |
| `direct` | `DirectDurationControlOutputModel` | `direct_duration_control.py` | 每段独立预测正时长（`softplus(global+local)`），求和得总时长；同样支持 uniform floor 混合 |
| `uniform` | `UniformDurationControlOutputModel` | `uniform_duration_control.py` | **无 duration head**：`UniformDurationControlHead` 把 `duration_projection` 直接设为 `None`，`final_time/N` 等分。`control_duration_uniform_floor` 在此模式下完全不被读取——不是 0 生效，而是这条代码路径根本不存在这个概念（见第 9 节「结构问题」a） |

`models.py` 的 builders 字典只注册了 4 个组合，不是 3×2=6 个：
`trim-residual` 只和 `factorized` 配对（→`TrimResidualControlOutputModel`）；
`direct`/`uniform` 只支持 `absolute`。字典本身对 `(direct, trim-residual)`/
`(uniform, trim-residual)` 会触发裸 `KeyError`——但这两个组合在实践中**不可能
到达这里**：`config.py:852-856`（`TSConfig.__post_init__`）已经在配置构造期
检查「`control_value_parameterization=="trim-residual"` 则
`control_duration_parameterization` 必须是 `factorized`」，不满足就直接
`raise ValueError`，早于任何 `_build_control_output` 调用。所以字典缺口是
纵深防御层面的代码整洁问题（见第 9 节 d），不是可以从 CLI 实际触发的 bug。

### 2.2 状态跟踪目标 — `control_state_objective`

分派在 `control_loss_components.py:344-350`（`_TRACKING_OBJECTIVES` 字典）：

| 取值 | 函数 | 需要 | 备注 |
|---|---|---|---|
| `normalized-mse`（原始默认） | `_normalized_mse_objective` | 无额外约束 | 六通道归一化 MSE + `terminal_loss_weight` 加权终点 MSE |
| `physical-criteria` | `_physical_criteria_objective` | `control_state_loss_grid=fixed-dt` | `physical_criteria_loss`：fixed-dt ADE(m)/100 与终点误差(m)/100 的 smooth-max |
| `terminal-state` | `_terminal_state_objective` | `fixed-dt`；`checkpoint_selection_metric=fixed-anchor-terminal-state` | dense state MSE + 独立的终点位置/速度项（`terminal_state_loss.py`） |
| `arc-length-geometry` | `_arc_length_geometry_objective` | `fixed-dt`；`n_segments>=2`；`checkpoint_selection_metric=fixed-anchor-arc-length-geometry` | 位置 SmoothL1（`arc_length_geometry.py`）+ 局部弧速度项（vector 或 tangent-speed 两种参数化，`control_arc_local_velocity_parameterization`）+ 终点（vector-norm 或 runway-components 两种参数化） |
| `true-time-position`（**simple-v1**） | `_true_time_position_objective` | `control_state_loss_grid=native`；`control_duration_parameterization=uniform`；`control_state_supervision_clock=observed` | 物理 3D 路径 MSE（原生 uniform-clock，不需要 fixed-dt 重采样）+ 软终点位置 MSE（`state_endpoint_loss_weight` 加权）。最新加入，唯一要求 `native` grid 而非 `fixed-dt` 的目标 |

`config.py:625-658` 的交叉校验把上表「需要」列强制成构造期异常，不是运行期
静默降级。

**这一列的演化是本包最长的一条否定结果链**（日期均取自对应实验文档）：
`physical-criteria`（07-30/07-31，从单飞行 oracle 迁移而来）迁移到共享模型
训练后"仍停留在公里级拟合能力"（`2026-07-30_direct_control_oracle.zh.md`）；
`terminal-state`（08-01 设计）自己的文档记录了一个未解决的观测/预测时钟
不匹配——在观测时钟上训练的稠密终端损失单飞行层面能到 ~60 m FDE，换到可部署
的预测时钟上只有 ~2354 m FDE；`arc-length-geometry`（08-01，"方法 2+4"）成为
08-02 那组实验采用的目标，但它自己的多飞行容量诊断发现 ADE 仍在
226–6000 m（取决于飞行长度），存在"终点对、中段轨迹错"的局部最优，且参考
速度（位置差分）与参考状态速度之间有 4.76–10.23 m/s RMSE 的不可约张力。
`true-time-position` 能在 08-16 一天内替换掉这一整条链，直接依据是
`2026-08-15_comprehensive_metrics_review.zh.md` §6.4 的度量结果：在 **state**
路径上，kinematic-consistency + terminal 两项加起来对总损失的贡献
**< 0.5%**，却带来了大量额外复杂度——如果 control-output 要"转正"，最低
目标应直接复用 state 路径同款的 `L_p + 0.25·L_E + L_T + λ_u L_effort +
λ_Δu L_smooth`。`normalized-mse`/`physical-criteria`/`terminal-state`/
`arc-length-geometry` 四个目标函数今天仍然是可达、可测试的 `custom` 消融轴，
但没有一个是被推荐的方向。

### 2.3 loss grid — `control_state_loss_grid`

分派在 `train.py:650-653`（`_CONTROL_STATE_LOSS_HANDLERS`）：

- `native-segment-endpoints` → `_native_endpoint_control_state_loss`
  （`train.py:561`）：只在模型自己学出的、非均匀的 segment 端点上算 loss，用
  `align_control_targets_to_prediction_clock` 把真值插值到预测的累积时钟上。
  **simple-v1 用这一档**，且此时 `dense_supervision` 恒为 `None`——因此
  `fixed_dt_control_loss.py`、horizon curriculum（2.6）、
  `terminal-state`/`physical-criteria`/`arc-length-geometry` 三个目标在
  simple-v1 下**永远不会被调用**（不是关闭，是这条分支物理上不会走到）。
- `fixed-dt` → `_fixed_dt_control_state_loss`（`train.py:617`），委托给
  `fixed_dt_control_loss.fixed_dt_control_state_loss`：在规则的 `dt_s` 网格上
  评估密集监督（`fixed_dt_supervision.FixedDTControlSupervision`），是
  `physical-criteria`/`terminal-state`/`arc-length-geometry` 三个目标以及
  horizon curriculum 的前提条件。

`fixed_dt_supervision.py` 有 16 个 import 方（本包内 import 数最高的
control 相关模块），但**只在 `control_state_loss_grid=fixed-dt` 时才真正参与
loss 计算**——它被 `dataset.py`、`arc_length_geometry.py`、
`terminal_state_loss.py`、`physical_criteria.py`、`control_oracle*.py` 等广泛
import 是因为它定义了共享的 `FixedDTControlSupervision` 数据结构和查询-网格
工具函数，而不是因为每个 import 方都在 simple-v1 下执行到它。

### 2.4 控制量参数化 — `control_value_parameterization`

分派在 `control_regularization.py:53-62`（`_REGULARIZATION_SIGNALS`，只用于
effort/smoothness 正则的量纲化信号）和 `models.py` 的 builders 字典（决定用
哪个模型类）：

- `absolute`（**simple-v1**）— 控制量直接 sigmoid 映射到物理上下界
  （`ControlOutputHead.bounded_controls`）。
- `trim-residual` — `trim_residual_control.py`：`trim_control_baseline` 用
  锚点观测状态 + 该机型气动参数，代数求解「近似维持 V/psi/gamma 不变」的配平
  控制（bank=0 时 `n=cos(gamma)`，`T=D+mg·sinγ`），越界则 clamp；模型学习的是
  **零中心的 logit 残差**，零残差 = 精确复现配平点。`TrimResidualControlHead`
  的 duration head 权重初始化为零（`control_oracle`-style 冷启动)。

  **已被拒绝**：`2026-07-31_deployable_control_training_optimizations.zh.md`
  §4 —— ADE 改善 1.79%，但 FDE 恶化 9.09%、time MAE 恶化 4.00%、机场宏平均
  criteria 恶化 5.65%（主要由 KRDU 单机场恶化 26.98% 拖累，其余 4 个机场小幅
  改善），未通过预注册的迁移门槛（`criteria < 20.8005`），PatchTST 上从未
  尝试迁移。`simple-v1` 用 `absolute`；`trim_residual_control.py` 保留是为了
  让该实验在 `custom` 档下可复现，不代表这是可选的等价方案。

### 2.5 动力学 rollout 后端 — `control_dynamics_backend`

`control_rollout.py` 本身**不含任何积分实现**——2026-08-16 之后它只是一个
瘦分派层（`rollout_control_endpoints`/`rollout_control_dense`，强制 float64
边界），真正的三个后端在 `control_dynamics_backends.py:302-308`
（`_BACKENDS` 字典，`ControlDynamicsBackend` ABC 的三个子类）：

| 取值 | 类 | 数值实现来源（本包外部） |
|---|---|---|
| `reanchored-rk4`（原始默认） | `ReanchoredRK4Backend` | `aerodynamic_model/torch_dynamics.py`：每个子步重新以当前 geodetic state 建立局部 ENU、RK4、再经 WGS84 ECEF 转换——与 `CasadiSimulator` 逐子步契约测试等价（`aerodynamic_model/tests/test_torch_dynamics.py`） |
| `transport-chart-velocity` | `TransportChartVelocityBackend` | `aerodynamic_model/torch_transport_chart_dynamics.py`：状态本身就是阈值锚定图表坐标 + 物理速度，避免每子步的 ENU 重锚定 |
| `scaled-transport-chart-velocity`（**simple-v1**） | `ScaledTransportChartVelocityBackend` | `aerodynamic_model/torch_scaled_transport_chart_dynamics.py`：内部状态整理为量级为 1 的无量纲量（"Order-one internal state"，类文档字符串原话），对外契约不变，目的是数值条件数 |

三个后端共享同一个 `EndpointControlRollout`/`DenseControlRolloutChannels`
返回契约，训练/评估/推理代码**不知道也不需要知道**当前用的是哪个后端——这是
本包里注册表模式用得最彻底的一处。

**`transport-chart-velocity`（未缩放版）单独引入时是退步，不是改进。**
`2026-07-31_deployable_control_training_optimizations.zh.md` §7：criteria
+18.45%、ADE +10.07%，5 个机场全部变差——尽管它确实压低了最严重的梯度尖峰
（裁剪前最大梯度范数 1.37e13 → 1.10e8），诊断为未解决的尺度/Jacobian 条件数
问题，不是实现 bug。`scaled-transport-chart-velocity`（08-02，
`2026-08-02_nondimensional_transport_experiment.zh.md`）用无量纲化状态修复了
这一点：配合 oracle-teacher 初始化，选择指标从 27.6277 降到 27.5097——文档
自己标注"不是 Jacobian 根因修复，是一个数值上的 Pareto 改进点"。`simple-v1`
冻结的是 scaled 版本；未缩放版本已知会退步，只作为 `custom` 消融保留，不要
在新实验里把它当作"reanchored-rk4 和 scaled 版之间的中间选项"来选用。

### 2.6 终端时钟 — `control_terminal_supervision_clock`

分派在 `control_terminal_clock.py:129-135`（`_TERMINAL_CLOCK_STRATEGIES`）：

- `state-supervision`（**simple-v1**，no-op）— 直接复用 dense-loss 算出的端点。
- `predicted` — 额外用模型自己预测的（非 detach）总时长再 rollout 一次终点，
  只在 horizon curriculum 的 full-horizon 阶段生效。
- `predicted-detached-time` — 同上，但总时长 `detach()`，只留划分梯度；要求
  `control_duration_parameterization=factorized`。

以及 horizon curriculum（`control_horizon_curriculum_s`，
`control_training_curriculum.py`）：非空时把训练切成「短物理horizon前缀 →
完整 horizon」的分阶段 single-shooting（`build_control_training_stage_view`
按 float64 参考时钟裁剪 duration 前缀并重新闭合总和）。simple-v1 关闭
（`()`），且在 `native` grid 下即使打开也不会被调用（2.3）。

**关闭课程是工程简化的连带产物，不是重新测量后判定无效。**
08-16 的文档没有论证课程本身有害——它是在"砍掉 arc-length-geometry 及其
配套复杂度"这一轮简化里被一起关掉的；此前 `2026-07-31_deployable_control_
training_optimizations.zh.md`（"Clip-only 因果对照"一节）记录过课程本身
"空间收益成立"。如果未来要在 `native`/`true-time-position` 路径上重新引入
分阶段训练，`control_training_curriculum.py`/`control_terminal_clock.py`
的裁剪逻辑本身没有被证明有问题，只是现在没被用到。

### 2.7 control-mixture — 独立的第三个 `prediction_output`

**不是** `control` 的一个子选项，而是 `PREDICTION_CONTROL_MIXTURE`
（`prediction_output=control-mixture`）独立枚举值，走独立的模型类
（`ControlMixtureOutputModel`，`control_models.py:106`）和独立的 loss 适配器
（`train.py:962` `_mixture_loss_adapter` → `control_mixture_loss.py`）。

结构（`control_mixture.py`）：`K`（默认 3，`control_expert_count`）个独立的
`ControlOutputHead` "专家" 各自输出完整的 controls/durations/final_time，外加
一个只看历史特征（不看 rollout 结果）的线性 `selector` head，输出
`selection_logits ∈ R^K`——这是可部署的部分：推理时不需要真的跑 K 次 rollout
比较,直接用 selector 的 argmax。

训练目标（`control_mixture_loss.mixture_prediction_loss_components`）是
best-of-K hindsight：K 个专家全部展平后**复用同一个
`control_prediction_loss_terms`**（和单一 `control` 路径完全相同的 rollout +
loss 代码，只是 batch 维度乘以 K）算出每个专家的 total loss,取
`argmin`（`detach()`，不回传选择本身的梯度）作为「winner」，只用 winner 的
reconstruction loss 更新对应专家的参数；`selector` 用交叉熵学习「不跑
rollout 也能提前猜中哪个专家会赢」；额外的 `_candidate_diversity` 惩罚项防止
K 个专家收敛到同一个解（用一个以 0.01 为尺度的指数相似度核，鼓励控制量/时长
划分/总时长三者中任一显著不同）。

这是本包里唯一一次直接尝试 README「Deliberate scope」一节承认的
"deterministic point prediction" 局限（多模态覆盖）的方案。

**但截至本文写作时，这条分支的研究结论是"暂停"，不是"活跃演进中"。**
`2026-07-30_control_mixture_pause_and_resume_plan.zh.md` 记录的首次正式训练
结果：K=3 部署态（selector 选出的那条）ADE 6728.4 m，对比单控制基线
2634.8 m，**+155.4%**，未过 gate；selector 在 2167 条验证飞行里 **100%**
都选了 0 号专家（彻底坍缩，没学到有意义的路由）。该文档列出的恢复前置条件
（可观测路由诊断、容量约束分配、专家/选择器分阶段训练、回归测试）截至
2026-08-16 没有任何文档报告过已满足。`run_ts_control_mixture_report.py`
（2026-08-16 修改）是报告/审计脚本，不是重新训练——它能跑不代表这条分支
已经被重新验证过。`simple-v1` 的设计文档原话是"不把 PatchTST、mixture
head、trim residual 或其他 dynamics backend 纳入新 recipe"。代码层面完全
接得通（结构上不是孤儿），但把它当作"当前活跃演进方向"会误导读者。

---

## 3. 一次 `prediction_output=control` 训练/推理的真实调用链

### 训练（`train.py`，simple-v1 路径加粗标出实际会走到的函数）

```
model_forward(model, history, dynamics)
  -> UniformDurationControlOutputModel.forward   [simple-v1; 见 2.1]
     -> ControlFeatureModel.fused_features        (backbone 特征 + 8 维气动条件编码融合)
     -> UniformDurationControlHead.forward        (bounded controls + final_time/N 等分)

prediction_loss_components()                      [train.py:1004, PREDICTION_LOSS_HANDLERS 按类型分派]
  -> _control_loss_adapter -> control_prediction_loss_components -> control_prediction_loss_terms
       state_prediction = control_state_supervision_prediction(...)   [控制训练时钟, 2.5 之外的 clock 轴]
       (dense_supervision is None under native grid -> 跳过 horizon curriculum 分支)
       rollout_loss = _CONTROL_STATE_LOSS_HANDLERS["native-segment-endpoints"](...)
           -> control_rollout.rollout_control_endpoints
                -> control_dynamics_backend(config).endpoint_rollout   [scaled-transport-chart-velocity]
       rollout_loss = apply_control_terminal_clock(..., "state-supervision")   [no-op]
       tracking = control_tracking_loss_terms(...)   [true-time-position objective]
       effort_signal, smoothness_signal = control_regularization_signals(...)  ["absolute"]
       (control_effort_loss_weight = control_smoothness_loss_weight = 0 in simple-v1
        -> 计算了但对总 loss 无贡献)
```

`control_gradient_clip_norm=20.0 > 0`（simple-v1）→ `batching.py`/`train.py`
中的 `ControlTrainingDiagnosticsAccumulator`（`control_training_diagnostics.py`）
被启用，记录裁剪前梯度范数、按子系统拆分、以及 bounded-control 饱和度审计，
不影响 loss 数值本身。`training_performance.EpochProfiler`
（`training_performance.py`）与 `prediction_output` 无关，对 state/control/
control-mixture 三种输出统一记录 CUDA-event 计时，无条件启用。

### 推理（`forecast.py`）

`_forecast_control_batch` → `_control_prediction_batch`（模型前向一次）→
`deployable_control_prediction`（`control_prediction_adapters.py`：单一
`control` 路径原样返回；`control-mixture` 路径调用
`ControlMixturePrediction.selected()` 取 selector argmax）→
**始终**再调用一次 `control_rollout.rollout_control_dense`（同一个后端分派）
产出稠密时间戳上的完整轨迹。也就是说：control 路径的推理输出从来不是「读出
模型的原始输出」，而是「模型输出的 controls+durations 经过与训练时相同的
可微动力学重新积分」——一致性由构造保证，不需要额外的 kinematic-consistency
loss（`config.py` 注释也如此说明，`kinematic_consistency_loss_weight` 在
control 路径被忽略）。

---

## 4. oracle-teacher 热启动：完整生产链路（代码验证，非推测）

README 的 `--control-teacher-schedules <path>/teacher_schedules.npz` 示例背后
是一条完整的、可复现的离线生产流水线，**全程不依赖 casadi**（与本包
"casadi-free by design" 的定位一致）。之前 `--control-teacher-schedules`
只在 `__main__.py:709-726` 消费一个已经存在的 `.npz`；生成它的入口在
**仓库根目录**，不在 `4dTrajectory/ts_transformer/__main__.py` 的任何子命令里
——找它必须知道去看 `run_ts_oracle_teacher_optimize.py`。

### 4.1 生产 `teacher_schedules.npz`

```
run_ts_oracle_teacher_optimize.py --output-dir <dir> [--airport ...] [--cohort-size 32]
  -> oracle_teacher.cohort.select_outer_train_cohort   (只开 outer-train 身份;
                                                         --airport 缺省时跨全部
                                                         已发现的 K 机场按航班数均衡分配)
  -> oracle_teacher.targets.build_inverse_dynamics_target   (每个航班的初值猜测)
       -> control_oracle_initialization.inverse_dynamics_controls
            (用「偷看未来」的真实观测轨迹做代数反演——psi_dot/gamma_dot/V_dot
             解出 load factor/bank/thrust，闭式,不是数值优化;这是仅教师可用的
             特权,模型自身训练/推理绝不会看到未来)
  -> oracle_teacher.optimization.BatchedOracleTeacher   (每个航班一组独立可训练
                                                          参数,不是神经网络)
       + optimize_teacher_controls: 分阶段 direct-shooting 梯度下降
         (60s -> 120s -> 240s -> full, torch.optim.Adam, 梯度裁剪)
       内部固定使用一套独立的 TSConfig：
         control_dynamics_backend=transport-chart-velocity   (不是 simple-v1 用的 scaled 变体)
         control_state_loss_grid=fixed-dt
         control_state_objective=arc-length-geometry
       (教师拟合本身的目标函数与它将要热启动的训练配方是两回事——教师要的是
        丰富的几何拟合信号,不需要和下游训练配方一致)
  -> 写出 teacher_schedules.npz (dataset_ids, controls, segment_durations_s)
     + teacher_optimization.json (每机场审计 + 每航班 ADE 从 initial 到 optimized 的改进)
```

### 4.2 消费 `teacher_schedules.npz`

```
__main__.py train --control-teacher-schedules <path> --control-recipe simple-v1
  -> oracle_teacher.pretraining.CachedSchedulePretrainer
       __post_init__: 若 recipe_name == "simple-v1",强校验
         steps==1000, learning_rate==1e-4, gradient_clip_norm==20.0,
         SHA-256(schedule) == 冻结值 "60e40e0...",
         schedule 形状 == (32 flights, 64 segments, 3 controls)
         (任何偏离直接 ValueError——不是警告)
       __call__: 用当前 TSConfig 的 normalizer 重建 outer-train 中这 32 个航班的
         FixedAnchorTrajectoryWindows,对模型做 1000 步 Adam 模仿学习
         (oracle_teacher.imitation.control_imitation_loss:
          unit-box 控制 MSE + N 缩放的 duration-fraction MSE + time/600 MSE)
       随后模型权重带着这个热启动状态,进入 train.py 正常的 rollout-loss 训练循环
```

`SIMPLE_V1_TEACHER_SCHEDULE_SHA256` 硬编码在
`oracle_teacher/pretraining.py:28-30`，把 simple-v1 的教师热启动钉死为**一个
具体文件**（32 条 KSJC 航班的 direct-shooting 结果），不是「用最新一批教师
数据」这种会随生产批次漂移的说法。README 例子里的路径
`.../oracle_teacher_20260816_current_manifest/optimized_arc24_32/teacher_schedules.npz`
就是这个被钉死的产物。

`run_ts_simple_teacher_paired_cv.py`（2026-08-16，与 `simple-v1` 同批提交）
是「教师热启动是否真的有用」的成对消融——在锁定的 outer-train fold 上分别跑
有/无教师热启动两支,保持其余 simple-v1 配方完全冻结。

### 4.3 两代 oracle-teacher 脚本

`git log` 按最后修改日期把根目录的 oracle/control 相关脚本分成两代：

| 代 | 日期 | 脚本 | 备注 |
|---|---|---|---|
| 第一代 | 2026-07-31 ~ 08-02 | `run_ts_control_oracle.py`（07-31）、`run_ts_oracle_teacher_train.py`、`run_ts_oracle_teacher_audit.py`、`run_ts_oracle_teacher_rollout_gate.py`、`run_ts_oracle_teacher_imitation.py`、`run_ts_control_fixed_dt_overfit.py`（均 08-02） | 早于 `control_recipe_name`/`simple-v1` 存在的时间点；`run_ts_oracle_teacher_rollout_gate.py`/`_imitation.py` 分别对应 `oracle_teacher/rollout_finetuning.py`、`oracle_teacher/progressive_pretraining.py`——这两个子模块目前**只被第一代脚本引用**,不在 `run_ts_oracle_teacher_optimize.py` + `CachedSchedulePretrainer` 这条当前链路里 |
| 第二代 | 2026-08-16 | `run_ts_oracle_teacher_optimize.py`、`run_ts_simple_teacher_paired_cv.py`、`run_ts_control_mixture_report.py`、`run_ts_control_capacity_ceiling.py`、`run_ts_clock_attribution.py` | 与 `simple-v1`/`config.py` 32 次提交同日；是本文第 4.1/4.2 节描述的当前链路 |

第一代脚本语法仍可解析（已用 `ast.parse` 核实），测试套件不会执行它们
（不是 `pytest` 收集范围），**没有证据证明它们已损坏**，但也没有证据证明它们
在 08-16 重构后仍被跑过。`oracle_teacher/rollout_finetuning.py`（82 行）和
`oracle_teacher/progressive_pretraining.py`（235 行）因此处于「代码活着、
调用它们的脚本疑似停用」的中间状态——建议下次触碰 oracle-teacher 链路时先确认
这两个脚本是否还在用，再决定归档还是继续维护（第 6 节的可执行建议之一）。

对 `progressive_pretraining.py` 这一份，"疑似"其实可以去掉：
`2026-08-02_progressive_n_experiment.zh.md` §4 明确写着"**Progressive-N 未通过
validation 门槛，不进入新基线**"——由粗到细（N=16→32→64）渐进 refine 教师
schedule 的方向已经被正式测过并否决，不只是没人碰。`rollout_finetuning.py`
没有对应的否决性结论：它是 08-02 那一轮的一次性"门禁"验证（确认
imitation-only 的结果在真实 rollout 下站得住，才敢把更便宜的 imitation-only
定为生产路径），没有文档说它错了，只是 08-16 的 `CachedSchedulePretrainer`
契约本来就只到 imitation 为止，没有再触发这道门禁。

---

## 5. 独立诊断：`control_oracle.py` 家族的第二种用途

除了给 oracle-teacher 当初值/精修引擎，`control_oracle.py`
（`DirectControlOracle`：单个已知航班的 direct-shooting 拟合，参数就是控制
schedule 本身，不含历史编码器——"isolates control/dynamics representability
from Transformer capacity"，类文档字符串原话）还被两个**诊断类**根级脚本
直接复用，目的与「造教师数据」完全不同：

- `run_ts_control_oracle.py`（806 行,07-31）—— **可表达性诊断**：给定
  一条真实观测轨迹，动力学模型本身（不经过任何 Transformer）能不能被直接
  拟合到位？回答的是「即使模型完美，rollout 契约本身的表达能力够不够」。
- `run_ts_control_capacity_ceiling.py`（674 行,08-16）—— **能力上限诊断**：
  以训练好的网络输出为初值，对选定的开发集航班做逐航班 direct-shooting 精修，
  测量「网络输出」与「同一动力学契约下能达到的最优解」之间的差距，即网络
  当前性能相对于该架构理论上限还差多少。

两者都不修改任何 checkpoint、不参与训练，是只读的验证工具（脚本文档字符串
明确写"validation-safe"/"representability diagnostic, not a deployable
model"）。`control_oracle_curriculum.py` 见第 6 节——它与这两个诊断脚本
**没有**调用关系。

---

## 6. 模块状态表：在用 / 仅消融可达 / 孤立

判定口径：`LIVE-default` = simple-v1 训练一次会执行到；`LIVE-ablation` =
需要显式偏离 simple-v1（即回到 `control_recipe_name="custom"`）才会执行到；
`LIVE-offline` = 通过某个根级 `run_ts_*.py` 脚本可达，但不在
`__main__.py train` 的直接路径上；`test-only` = 仅被自己的单元测试
import，无任何生产代码路径引用；`orphan` = 连测试也没有引用。全部 377 个
测试当前 100% 通过（`python -m pytest 4dTrajectory/ts_transformer/tests -q
--import-mode=importlib`），所以下表的 `test-only`/`orphan` 判定是「未被
接线」，不是「已损坏」。

| 模块 | 状态 | 触发条件 / 引用方 |
|---|---|---|
| `control_loss_components.py` | LIVE-default | 5 个 objective 注册表；simple-v1 用 `true-time-position` |
| `control_models.py` | LIVE-default | 单一/mixture 共享的 backbone 融合层 + factorized 头 |
| `control_prediction_adapters.py` | LIVE-default | 推理/诊断路径的类型自适配 |
| `control_rollout.py` / `control_dynamics_backends.py` | LIVE-default | dtype 边界 + 3 后端注册表；simple-v1 用 `scaled-transport-chart-velocity` |
| `uniform_duration_control.py` | LIVE-default | simple-v1 的 duration 策略 |
| `control_terminal_clock.py` | LIVE-default | 3 策略注册表；simple-v1 落到 no-op 分支 |
| `control_training_curriculum.py` | LIVE-default | 定义 `ControlTrainingStage`,`control_prediction_loss_terms` 无条件构造一个 full-horizon 哨兵 stage；分阶段裁剪逻辑本身仅 `dense_supervision is not None` 时执行 |
| `control_regularization.py` | LIVE-default | 2 策略注册表；即使权重为 0 也会被调用算出信号 |
| `control_training_diagnostics.py` | LIVE-default | `control_gradient_clip_norm>0`（simple-v1 是 20.0）时启用 |
| `training_performance.py` | LIVE-default | 与 prediction_output 无关,每 epoch 无条件计时 |
| `reference_velocity.py` / `coordinate_frames.py` / `anchor_eligibility.py` / `lateral_eligibility.py` / `evaluation_protocol.py` / `fixed_anchor_validation.py` | LIVE-default | 数据/评估基础设施,state 与 control 两条路径共用 |
| `fixed_dt_supervision.py` | LIVE-default（数据结构）/ LIVE-ablation（作为 loss 输入） | `FixedDTControlSupervision` 类型被广泛 import；仅在 `control_state_loss_grid=fixed-dt` 时真正驱动 loss |
| `control_mixture.py` / `control_mixture_loss.py` | LIVE-ablation | 需要 `prediction_output=control-mixture`；结构上不是孤儿（`run_ts_control_mixture_report.py`，08-16，仍能跑），但研究结论层面**已暂停**——见 2.7 的 pause-and-resume 结果，恢复前置条件截至 08-16 未见满足 |
| `direct_duration_control.py` | LIVE-ablation | `control_duration_parameterization=direct` |
| `trim_residual_control.py` | LIVE-ablation | `control_value_parameterization=trim-residual`;同时被 `control_regularization.py` 无条件 import(用于任意 value_parameterization 下的信号分派表,不代表恒执行) |
| `fixed_dt_control_loss.py` | LIVE-ablation | `control_state_loss_grid=fixed-dt` |
| `terminal_state_loss.py` | LIVE-ablation | `control_state_objective=terminal-state` 或 `arc-length-geometry` |
| `arc_length_geometry.py` | LIVE-ablation | `control_state_objective=arc-length-geometry` |
| `physical_criteria.py` | LIVE-ablation | `control_state_objective=physical-criteria`;同时被 `control_oracle.py` 复用为 oracle 目标之一 |
| `cross_validation.py` | LIVE-default(非 control 专属) | `__main__.py cross-validate` 子命令 |
| `development_cohorts.py` / `experiment_index.py` | LIVE-default(非 control 专属) | `__main__.py` 直接 import,`approach_clustering/` 也用 |
| `oracle_teacher/pretraining.py` / `oracle_teacher/optimization.py` / `oracle_teacher/targets.py` / `oracle_teacher/cohort.py` / `oracle_teacher/evaluation.py` | LIVE-offline | 第 4 节链路；生产入口是根级 `run_ts_oracle_teacher_optimize.py`,消费入口是 `__main__.py train --control-teacher-schedules` |
| `control_oracle.py` / `control_oracle_initialization.py` | LIVE-offline | 被 `oracle_teacher/targets.py`（初值）、`oracle_teacher/optimization.py`(通过 `BatchedOracleTeacher`,间接) 以及 `run_ts_control_oracle.py`/`run_ts_control_capacity_ceiling.py`(诊断脚本,第 5 节) 复用；**不被 `__main__.py train` 的默认路径直接 import** |
| `train_only_diagnostics.py` | LIVE-offline | 仅被 `oracle_teacher/cohort.py` 引用(第一代/第二代通用) |
| `oracle_teacher/rollout_finetuning.py` / `oracle_teacher/progressive_pretraining.py` | LIVE-offline,疑似停用 | 只被第一代根级脚本（`run_ts_oracle_teacher_rollout_gate.py`/`_imitation.py`,08-02)引用；08-16 重构未触碰,当前 `run_ts_oracle_teacher_optimize.py` 链路不经过它们 |
| `benchmark_validation_execution.py` | **orphan** | `grep` 全仓库零引用,零测试覆盖；是可独立运行的脚本（有 `argparse`/`if __name__`），但没有任何证据表明它仍被使用 |
| `build_multiflight_capacity_report.py` | test-only（脚本） | 仅 `tests/test_ts_transformer.py` import;本身是 `argparse` 独立脚本,设计上就该被直接运行而非 import,测试覆盖但未见文档提及入口用法 |
| `control_oracle_curriculum.py` | **test-only** | 仅 `tests/test_control_oracle.py` import；`oracle_teacher/optimization.py` 的分阶段逻辑（`TeacherOptimizationStage`,4.1 节的 60/120/240/full）是**另一套独立实现**,不复用这个模块的 `HorizonCurriculumStage`/`build_horizon_curriculum`。没有任何生产代码路径引用它——见第 9 节 a |

---

## 7. 根目录 `run_ts_*.py` 脚本总表

`README.md` 此前只记录了 4 个（`run_ts_pipeline.py`、`run_ts_cv.py`、
`run_ts_history_ablation.py`、`run_ts_coordinate_ablation.py`）。仓库根目录
实际有 17 个，按最后修改日期分三代：

| 脚本 | 日期 | 用途（首行 docstring） |
|---|---|---|
| `run_ts_cv.py` | 07-27 | 跑 pooled CV（README 已记录） |
| `run_ts_overfit_diagnostic.py` | 07-27 | 小规模固定子集记忆能力测试（state 路径通用诊断） |
| `run_ts_coordinate_ablation.py` | 07-27 | ENU vs runway-aligned 配对消融（README 已记录） |
| `run_ts_history_ablation.py` | 07-27 | 历史长度 L 消融（README 已记录） |
| `run_ts_kinematic_ablation.py` | 07-27 | state 路径 kinematic-loss 权重筛选 |
| `run_ts_control_oracle.py` | 07-31 | 单航班 direct-shooting 可表达性诊断（第 5 节） |
| `run_ts_oracle_teacher_train.py` | 08-02 | 第一代：教师热启动初始化的正式 KSJC 训练 |
| `run_ts_oracle_teacher_audit.py` | 08-02 | 第一代：审计逆动力学教师质量 |
| `run_ts_oracle_teacher_rollout_gate.py` | 08-02 | 第一代：在 train-only rollout 目标上微调「已模仿」的教师模型（`oracle_teacher/rollout_finetuning.py`） |
| `run_ts_oracle_teacher_imitation.py` | 08-02 | 第一代：train-only 优化后教师 cohort 的记忆化门控（`oracle_teacher/progressive_pretraining.py`） |
| `run_ts_control_fixed_dt_overfit.py` | 08-02 | fixed-dt control 状态监督的单航班容量诊断 |
| `run_ts_control_mixture_report.py` | 08-16 | control-mixture checkpoint 的 selector/oracle 验证诊断 |
| `run_ts_oracle_teacher_optimize.py` | 08-16 | 第二代：生产 `teacher_schedules.npz`（第 4.1 节，当前链路） |
| `run_ts_clock_attribution.py` | 08-16 | control checkpoint 的误差按「时长头/几何/…」来源归因 |
| `run_ts_control_capacity_ceiling.py` | 08-16 | 逐航班能力上限诊断（第 5 节） |
| `run_ts_simple_teacher_paired_cv.py` | 08-16 | simple-v1 下教师热启动 vs 无教师的成对消融（第 4.2 节） |
| `run_ts_pipeline.py` | 08-16 | 总入口：CV→训练→逐机场预测/评估/CZML（README 已记录） |
| `run_ts_predictability_report.py` | 08-16 | pooled 预测器的纯验证集诊断报告（README 已记录） |

建议：README 的「Running it」一节保留当前四个通用入口不变，新增一行指向本文
第 7 节，而不是把 17 个脚本的说明搬进 README——这类实验驱动脚本的数量还会继续
增长，单独维护一张表比反复膨胀 README 更可持续。

---

## 8. 结构优化潜力

整体评价：**架构本身是干净的**。每一条消融轴都是「字典/ABC 注册表 + 一个
`config.<field>` 键」，没有发现散落的 `if backend == "x": ... elif ...`
式分支纠缠；`control_prediction_loss_terms`/`control_dynamics_backend`/
`control_tracking_loss_terms`/`_TERMINAL_CLOCK_STRATEGIES`/
`_REGULARIZATION_SIGNALS` 全部是这种模式,新增一种策略理论上只需要新增一个
注册表项。`config.py` 的交叉校验（第 2 节表格里的「需要」列）把非法组合堵在
构造期而不是训练到一半才报错，符合仓库 `CLAUDE.md` "fail loudly" 的约定。
`control_recipe_name` 机制（`custom` 保留全部历史实验、`simple-v1` 冻结一套
最简配方并做 SHA-256 级别的可复现性校验）本身就是这份复杂度的正确解法：不
删除历史消融的可运行性,同时给出一个不必理解全部 40 个字段就能用的默认路径。

以下是具体可执行的改进点，不是「重写架构」级别的建议：

**a. `control_oracle_curriculum.py` 是死代码，应二选一。** 它实现的
"single-shooting 分阶段裁剪" 与 `oracle_teacher/optimization.py` 里
`TeacherOptimizationStage` 手写的 60/120/240/full 阶段是同一个问题的两份
独立实现，后者目前在跑（4.1 节），前者只有单元测试覆盖、零生产引用。要么把
`oracle_teacher/optimization.py` 迁移过去复用它（消除重复逻辑），要么确认它
是被放弃的早期尝试后删除，两者都好于现状的「两份实现并存,其中一份没人调用」。

**b. `benchmark_validation_execution.py` 零引用、零测试，是最强的孤立信号。**
本次调查里唯一一个「连测试都没有」的模块。若它仍是某个人工基准流程的一部分,
应该在文档里留一行说明如何调用；若已废弃,建议删除而不是留着让下一次搜索的人
误以为它是活的。

**c. `oracle_teacher/rollout_finetuning.py` / `progressive_pretraining.py`
疑似被第一代脚本废弃但代码仍在。** 08-16 的重构（`optimize.py` +
`CachedSchedulePretrainer` + `simple_teacher_paired_cv.py`）没有触碰这两个
子模块，也没有新脚本引用它们。建议下次碰 oracle-teacher 时先确认：如果第二代
链路已经完全替代了「模仿预训练 + rollout 微调」两阶段设计（现在只有一阶段
`CachedSchedulePretrainer`），就把这两个文件和对应的两个第一代脚本一起标记为
`docs/` 里常见的「保留但已知过时」并说明原因,而不是让新读者反复重新发现这个
问题。

**d. `models.py` 的 `_build_control_output` builders 字典只覆盖 4/6 种
`(duration, value)` 组合，纵深防御层面不够干净（非活跃 bug）。**
`(direct, trim-residual)` 和 `(uniform, trim-residual)` 会在 `builders[(...)]`
查找时抛裸 `KeyError: (...)`；但 `config.py:852-856` 已经在
`TSConfig.__post_init__` 里为这两个组合抛出可读的 `ValueError`（"trim-residual
controls currently require factorized durations"），所以 `_build_control_output`
在正常 CLI 路径下永远不会真的收到这两个组合——**这条 `KeyError` 目前不可达**。
仍然值得低成本修一下（在 `_build_control_output` 里对未命中做一次显式检查，
给出比裸 `KeyError` 更可读的信息），理由是维护纵深防御一致性、以及防止未来
有人绕过 `TSConfig.__post_init__`（例如直接构造 dataclass 字段而不走
`__init__` 校验）时收到难以定位的错误，而不是因为它现在会被触发。

**e. `control_duration_uniform_floor` 在 `control_duration_parameterization
=uniform` 下是静默死字段。** `UniformDurationControlHead` 根本不读这个
config 字段（第 2.1 节）；simple-v1 把它设成 `0.0` 恰好无害，但如果有人在
`custom` 档下把 duration 参数化设成 `uniform` 又调这个字段期望有效果,不会有
任何报错或警告。建议要么在该组合下让 `TSConfig.__post_init__` 拒绝
非默认值,要么在字段文档里显式点名"仅 factorized/direct 下生效"。

**f. 两代 oracle-teacher 根级脚本共存但无索引。** 本文第 4.3 节和第 7 节的
表格是目前唯一一处把 17 个 `run_ts_*.py` 集中列出的地方；这类脚本的数量在
2026-07-27→08-16 三周内翻了两番，如果没有本文这样的活文档,下一次新增脚本时
大概率会重复发明第一代已经写过的诊断逻辑（正如第二代已经不止一次和第一代
职责重叠：`run_ts_control_oracle.py` vs `run_ts_control_capacity_ceiling.py`,
`run_ts_oracle_teacher_train.py` vs `run_ts_oracle_teacher_optimize.py`）。
建议：新增 `run_ts_*.py` 时，在本文第 7 节表格追加一行是比事后靠 `git log`
重新梳理更便宜的习惯。

---

## 9. 相关文档索引

- `docs/normalized_time_and_control_output.zh.md` — **已被本文取代**，只作为
  最初设计（2026-07-20 前后）的历史记录保留，不再更新。
- 按时间顺序的实验记录（本文的结论均来自对当前代码的直接验证，未逐篇复述这些
  文档的实验过程，需要具体数值/中间失败尝试时查看对应日期）：
  `docs/2026-07-28_control_prediction_experiment_guide.zh.md`、
  `docs/2026-07-30_control_mixture_pause_and_resume_plan.zh.md`、
  `docs/2026-07-30_direct_control_oracle.zh.md`、
  `docs/2026-07-30_direct_duration_ablation.zh.md`、
  `docs/2026-07-30_fixed_dt_control_overfit.zh.md`、
  `docs/2026-07-31_deployable_control_training_optimizations.zh.md`、
  `docs/2026-08-01_arc_length_geometry_loss_experiments.zh.md`、
  `docs/2026-08-01_terminal_state_loss_design.zh.md`、
  `docs/2026-08-02_control_model_experiment_group_summary.zh.md`、
  `docs/2026-08-02_dual_clock_terminal_ablation.zh.md`、
  `docs/2026-08-02_oracle_teacher_experiment.zh.md`、
  `docs/2026-08-02_progressive_n_experiment.zh.md`、
  `docs/2026-08-02_train_only_oracle_teacher_explainer.zh.html`、
  `docs/2026-08-15_comprehensive_metrics_review.zh.md`、
  `docs/2026-08-16_control_simple_v1_development.zh.md`（simple-v1 的落地
  记录，日期上紧邻本文描述的代码状态）。
- README「[State baseline and dynamics-constrained control
  output](../README.md#state-baseline-and-dynamics-constrained-control-output)」
  — 面向读者的精简版本，本文是它的技术展开。
