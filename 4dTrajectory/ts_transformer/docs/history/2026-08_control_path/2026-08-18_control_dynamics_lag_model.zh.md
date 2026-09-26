# control 动力学：一阶滞后模型与反求 teacher 的一致性

日期：2026-08-18
状态：已实现并通过定向测试；τ_μ CV 扫描完成（§5b，未分辨）；
      point-mass vs lag 成对发布实验完成（§5d，滞后模型 jerk −28 %、ADE −3.4 %）
数据边界：只使用 outer-train / outer-validation；本文未读取、预测或评估 outer-test
来源：`docs/MeetingNotes/note_8_17.md`

---

## 1. 问题

会议记录第一条：

> change dynamic model to keep the bank angle parameter continue
> d/dt mu = (mu_cmd - mu) / tau_mu

现有 point-mass 模型把分段常值控制**瞬时**施加。一条 N=64 段的学习 schedule 因此在
64 个分段边界上让 bank 角发生阶跃：航迹曲率不连续 64 次，隐含滚转角速度无界。真实
飞机滚入转弯、推油门、加载机翼都需要有限时间。

同时，会议还要求：

- thrust 归一化到 1.0；
- load factor 连续，合理区间 0.2–2 g；
- thrust 与 load factor 的 τ 可以更接近瞬时；
- τ_μ 用 CV 测。

---

## 2. 模型

### 2.1 增广状态

状态从 transport-chart 的 7 维扩展到 10 维：

```
(e, n, u, ve, vn, vu, mass,   δT, μ, n_z)
 └────── 原 point-mass 状态 ──┘ └─ 执行器状态 ─┘
```

三个执行器状态各自被命令驱动：

\[
\dot{\delta T}=\frac{\delta T_{cmd}-\delta T}{\tau_T},\quad
\dot{\mu}=\frac{\mu_{cmd}-\mu}{\tau_\mu},\quad
\dot{n}=\frac{n_{cmd}-n}{\tau_n}
\]

**关键实现约束：`torch_lag_dynamics.lag_rhs` 调用的是原封不动的
`transport_chart_rhs`，只是把 `actual` 而不是 `commands` 喂进去。** 受力方程、失速
处理、WGS84 transport 项、chart 投影全部是同一份代码。因此"滞后模型 = point-mass
模型 + 三个一阶执行器"是结构事实，不是文档里的声明。

模型输出的 schedule 从"控制量"变成"控制指令"，形状和契约不变。

### 2.2 τ→0 的退化，以及 τ 真的改变了什么

这两条都必须实测，否则"新模型"可能只是个平滑器，或者根本是另一个模型：

| 检验 | 结果 |
|---|---|
| τ = 0.1 s 与 point-mass 的终点距离 | **< 路径长度的 0.5 %**，且随 τ 一阶收敛 |
| τ = 2.0 s（默认）与 point-mass 的终点距离 | **~3 km / 240 s rollout** |
| 25° 阶跃指令下实际 bank 的逐样本变化 | ≤ `25°·(1−e^{−Δt/τ_μ})`，而非 25° 阶跃 |

即：τ→0 时两个模型是同一个模型（可比较），τ=2 s 时是实质不同的模型（值得比较）。

### 2.3 数值稳定性（新增的硬约束）

执行器 ODE 与其余状态用同一个显式 RK4 积分。显式 RK4 对 `y' = −y/τ` 的稳定域是
`h/τ < 2.785`。**τ 小于积分步长时 rollout 不是变差，而是直接产生 NaN。**

所以 `TSConfig.__post_init__` 在构造期拒绝 `τ < control_rollout_integrator_dt_s`，
而不是让它在扫描 τ 时变成一次报废的训练。CV 网格最小值 0.5 s 恰好等于默认积分步长
（`h/τ = 1.0`，远在稳定域内）。

---

## 3. 控制量契约：无量纲化与负推力

### 3.1 无量纲化

本包内的控制契约改为：

```
(thrust_fraction ∈ [-0.2, 1.0],  bank_rad ∈ ±π/4,  load_factor ∈ [0.2, 2.0])
```

`thrust_fraction = T / T_max`。单一来源是 `control_envelope.py`。牛顿只出现在两个
函数里：进 dynamics 的 `physical_controls()`，和出到 evaluation record 的
`forecast.py`（record 契约仍是牛顿，与 CasADi optimizer 共用，未改动）。

收益不只是"量级一致"：**改之前同一个 sigmoid 输出在小型机上是 100 kN、在重型机上是
400 kN**，teacher schedule 和学习到的 bias 都不能跨机队迁移。另外，滞后模型的执行器
状态必须是 order-one 的，否则 1e5 量级的推力状态混在 chart 状态里会毁掉长 rollout 的
条件数。

### 3.2 负推力下界（实测驱动）

下界从 0 改为 **−0.2**。理由不是审美：真实进近需要净负推力——慢车推力加上减速板、
襟翼、起落架的阻力，而这套 clean-configuration 阻力极曲线不建模其中任何一项。
README 的 flyability 结论早就写着"真实进近的中位需求推力是 0.43 kN（慢车），负推力
只代表阻力增量"，`flyability.py` 也因此把 `thrust_negative` 列为 SOFT violation。

实测（KSJC outer-train，24 条航班 × 64 段）：

| | 0 N 下界 | −0.2·T_max 下界 |
|---|---:|---:|
| thrust 被上下界削掉的比例 | **39.7 %** | **0.33 %** |
| bank 被削比例 | 0 % | 0 % |
| load factor 被削比例 | 0 % | 0 % |

即改之前**教师在结构上无法复现飞机实际飞出的减速**。改之后 thrust_fraction 的分布是
p5 = −0.12（阻力增量）、p50 = 0.036（慢车）、p95 = 0.18。

顺带记录：反求出的 load factor 实测只落在 **[0.93, 1.16]**，`[0.2, 2.0]` 这个盒子在
观测数据上永远不 binding——它约束的是**学习头可以输出什么**，这才是它存在的理由。

### 3.3 与 optimizer 包络的关系

`optimization/casadi_optimizer.make_control_bounds` **没有改**，仍是非负推力下界和
0.5 的 load 下界。这是刻意的：optimizer 发布的是它声称可飞的轨迹，它的盒子是一个
适航声明；本包的盒子是一个学习头的搜索空间，事后再判。因此本次改动**不使任何
optimizer 产物过期**。

---

## 4. 反求 teacher 的一致性（本次审核的核心）

### 4.1 为什么这件事只能靠测试保证

用错误方程解出来的 teacher schedule 是有限的、在界内的、形状正确的，而且它自己的
optimizer 会报告 loss 在下降。它只是**复现不了任何东西**，而下游没有任何环节能发现。

因此设计上：`control_inverse_dynamics.py` 的每个反解注册在**与其前向模型相同的
config key** 下（`CONTROL_INVERSES` 与 `control_dynamics_backends._BACKENDS` 同键），并且
`tests/test_control_inverse_dynamics.py` 对**每一个注册的模型**做闭环：

```
已知 schedule → 前向 rollout → 稠密参考 → 反解 → 要求拿回原 schedule
```

实测复原精度：thrust_fraction 与 load factor 5e-3，bank 0.5°（这是 2 s 网格对
段内变化控制量做有限差分的误差，不是自由参数）。新增一个没有反解的动力学模型会在
注册表查找时直接失败。

### 4.2 排查出的具体问题

| 问题 | 影响 |
|---|---|
| `build_inverse_dynamics_target` 硬解包 7 元组 | `dataset.batch()` 只有在 `control_state_loss_grid='fixed-dt'` 时返回 7 个字段。**所有 native-grid 配方（含 simple-v1）下 teacher 构造器直接抛裸 `ValueError`**。它一直能跑只是因为 `run_ts_oracle_teacher_optimize.py` 自建了一套 `custom` + fixed-dt 配置 |
| 参考速度源硬编码 | 无视 `config.reference_velocity_source`，固定用 `smoothed-position-difference` 反求，即反解的速度定义与监督目标的速度定义不同 |
| 反解与后端无关联 | 手写的 flat-ENU RHS 代数反解，与 `config.control_dynamics_backend` 没有任何联系。它与 transport-chart 后端**碰巧**吻合（实测 0.5 N / 1e-6 load factor），不是构造上吻合；加入 lag 状态后会静默失配 |
| 缺 transport 项 | 反解现在加回 chart RHS 减掉的 `ω × v`（量级仅 ~1.6e-4 g，但这是"精确反解"与"接近的反解"的区别） |

### 4.3 滞后模型的反解

分两级，共用第一级：

1. **实际控制量**：`actual_controls()` —— 与 point-mass 反解同一份代码。因为滞后模型
   的受力方程吃的就是执行器状态，所以从轨迹反求出的就是执行器状态。
2. **指令**：`u_cmd = u + τ·du/dt`，连续系统下精确。

因为共用第一级，两个反解不可能漂移。`du/dt` 在一个 5 样本中心平滑后的副本上取，因为
`u` 本身已经是位置的二阶数值微分——这是一个**显式声明**的平滑（
`COMMAND_SMOOTHING_SAMPLES`），只作用于 teacher 的指令估计，不作用于任何被测量。

### 4.4 滞后状态的初值

`(δT, μ, n)` 在 anchor 处不可观测。取值来自**同一个反解**作用于观测 lookback
（`dataset.anchor_controls`，11 个样本 ≈ 20 s）：

- 只读 anchor 及之前的样本，因此与 history window 一样可部署；
- 让"反解必须与前向模型一致"成为**主路径上的结构约束**，而不是 teacher 专有的额外
  要求；
- 不能用"第一段指令"当初值：那等于把飞机放在它还没滚入的坡度上，滞后会被算两次。

---

## 5. 配方与实验

`control_dynamics_model` 是一条独立的轴，与状态表示（`control_dynamics_backend`）正交；
注册表按 **(model, backend) 二元组** 查表。滞后模型要求 transport-chart 家族的后端，
因为它需要一个连续 RHS 可以增广——`reanchored-rk4` 是离散映射（每个子步重建局部 ENU
系），没有可增广的 RHS。这条约束在 config 构造期检查。

新增配方 `simple-v1-lag`：**simple-v1 改一个字段**，其余全部冻结。这样成对比较测的是
飞行模型本身，而不是一整包选择。三个时间常数刻意**不冻结**——τ_μ 正是要扫的量。

CV 网格新增：

```python
"control_bank_time_constant_s": (0.5, 1.0, 2.0, 3.0, 4.0)
```

括住 2 s：1 s 是利落的滚转，4 s 是慵懒的，0.5 s 足够接近瞬时，兼作"滞后到底有没有
用"的对照臂。`applicable_cv_parameters` 在 `point-mass` 下把这条轴判为 inert 并剔除，
否则候选数会乘 5 而每一折结果完全相同——那会读成"在一个从未变化的参数上收敛了"。

**命名配方不能以自身身份做 CV**：`simple-v1`/`simple-v1-lag` 冻结了 `epochs`/`patience`，
而搜索刻意用更短的预算。所以 CV 候选携带 `control_recipe_name='custom'`，父配方保留在
run contract 的 `base_config` 里，正式训练仍从冻结配方构造。

---

## 5b. τ_μ CV 结果（2026-08-18 实测）

产物：`4dTrajectory/outputs/KSJC/experiments/cv_tau_bank_20260818/cv_results.json`
配置：KSJC，`simple-v1-lag`，3 folds × 36 epochs，selector = common true-time
airport-macro 3D ADE。

| τ_bank | mean ADE (m) | fold 0 | fold 1 | fold 2 |
|---|---:|---:|---:|---:|
| 0.5 s | 1029.5 | 1022.4 | 1101.8 | 964.3 |
| 1.0 s | 1019.3 | 1011.9 | 1076.1 | 970.0 |
| **2.0 s** | **1014.2** | 988.4 | 1083.0 | 971.3 |
| 3.0 s | 1066.1 | 1028.3 | **1198.8** | 971.3 |
| 4.0 s | 1072.3 | **1159.6** | 1088.9 | 968.3 |

**结论：本轮未分辨。** 最优到最差只差 **5.7 %**，而同一 τ 内部的 fold 间离散度是
**11–23 %**——噪声是信号的 2–4 倍。配对比较（同折同 seed）：τ=1.0 相对 τ=2.0 平均只差
5.1 m 且 3 折里赢 2 折；τ=3/4 的劣势各自完全由一个 fold 撑起（分别是 fold 1 的 +115.9
和 fold 0 的 +171.2），其余两折基本打平；**fold 2 上五个 τ 的跨度只有 0.7 %**。

按 `2026-08-16_control_simple_v1_development.zh.md` §5 的判据（平均 ADE 差异 < 2 % 即判
无可靠差异），τ ∈ {0.5, 1.0, 2.0} 相互差 ≤1.5 %，判定为**没有可靠差异**。取
**τ_bank = 2.0 s** 作为默认值（赢了均值、量级符合真实运输机滚入时间、没有候选可靠地
赢它），但必须记为"未分辨"而不是"CV 选定"。

### 一个必须记住的坑：`--cv-patience` 不能用默认的 6

第一轮扫描用 `--cv-patience 6`，全部作废。**两个飞行模型都会穿过一个早期 ADE 暂态凸包**
（未改动的 point-mass `simple-v1` 实测：ep2 最优 1534 m → ep6 凸包顶 2132 m → ep13
回到 1268 m 且仍在降），而训练 loss 的每一个分量在整个过程中单调下降。patience 6 正好
停在凸包里：同一个 τ=0.5，被早停抓住的 fold 选出 1674.6 m，没被抓住的 fold 走到
1234.4 m——**26 % 的差异纯粹是停止时机的产物**。凸包宽约 7 个 epoch，patience 15 可以
留足余量；冻结配方自己的 patience=20 本来就够。

---

## 5c. 实验产物保留

CV **不保存 fold checkpoint**（`cross_validate` 只写 `cv_results.json` /
`best_config.json`），所以超参搜索的产物无法回放、无法发布到前端。需要可发布产物的实验
必须走完整链路，`run_ts_flight_model_paired.py` 是这条链路的模板：

```
train    -> checkpoint.pt (+ history / fit_evaluation / checkpoint_metadata)
predict  -> 逐航班 prediction record：states、controls、observed lookback
evaluate -> evaluation_report.json / .html（与 optimizer 同一套 gate）
publish  -> aeroviz-4d/public/data/airports/<ICAO>/comparison/<category>/ 的对比 CZML
```

---

## 5d. point-mass vs first-order-lag 成对结果（2026-08-19 实测）

产物：`4dTrajectory/outputs/KSJC/experiments/flight_model_paired/`
（两臂各含 `checkpoint.pt` / prediction records / evaluation report），
前端：`aeroviz-4d/public/data/airports/KSJC/comparison/ts_ksjc_flight_model_{point_mass,first_order_lag}/`

配置：KSJC，两个冻结配方各训 180 epochs，同一 manifest / roster / `split_seed=1337`，
**两臂评分的是同一批 1083 条 validation 航班**，因此可以做逐航班配对检验（下面的 p 值
是双侧符号检验）。

### 精度：小幅改善，但极其一致

| 指标 | point-mass | lag τ=2 | 配对中位差 | lag 更优的航班 | p |
|---|---:|---:|---:|---:|---:|
| **ADE（路径）** | 732.6 m | **707.8 m** | **−20.3 m** | **730/1083 = 67.4 %** | **7.9e-31** |
| FDE | 891.8 | 856.2 | +2.5 | 49.2 % | 0.63 |
| arrival endpoint error | 688.4 | 681.3 | +3.5 | 49.1 % | 0.58 |
| \|final time error\| | 14.39 s | 14.29 s | −0.04 s | 53.9 % | 0.011 |

ADE 平均改善 **−3.4 %**（bootstrap 95 % CI **[−32.7, −17.0] m**）。**终点没有改善**
（FDE / endpoint 的胜率都是 49 %，即无差异）。这个组合是自洽的：滞后平滑的是沿途的
转弯几何，不改变模型认为跑道在哪里。

### 平滑性：这才是滞后模型要解决的问题

| 指标 | point-mass | lag τ=2 | 变化 | lag 更低的航班 | p |
|---|---:|---:|---:|---:|---:|
| **jerk p95** | 1.267 m/s³ | **0.912** | **−28.0 %** | **99.8 %** | ~1e-320 |
| turn rate p95 | 1.270 °/s | 1.193 | −6.0 % | 86.4 % | 5e-141 |
| acceleration p95 | 3.869 m/s² | 3.658 | −5.5 % | 77.4 % | 2e-76 |

**结论：滞后模型在不牺牲精度的前提下把 jerk 降低 28 %，而且精度还略有提升。**
这一点很关键——README 记录过的陷阱是"更差的预测器因为画得更平淡而在 flyability 上
得分更高"；这里没有掉进去，因为平滑与精度是同向改善的。

### 必须同时记录的反面读数

两个模型**本来就都比真实飞行轨迹更平滑**，滞后只是让它们更远离观测统计：

| 指标 | 观测 | point-mass | lag τ=2 |
|---|---:|---:|---:|
| turn rate p95 | 1.360 °/s | 0.93× | 0.88× |
| acceleration p95 | 4.806 m/s² | 0.81× | 0.76× |
| jerk p95 | 4.253 m/s³ | 0.30× | **0.21×** |

"更接近观测"的航班比例分别是 26.7 % / 43.0 % / **0.2 %**。所以**不能**把这个结果说成
"滞后让预测更像真实飞行"。正确的说法是：滞后去掉的是一个**建模伪影**（分段常值控制在
段边界上造成的曲率不连续），而观测 jerk 本身主要是 2 s ADS-B 位置采样三次微分后的
量化噪声，不是真实飞机的 jerk——它不是一个可追的目标。

### Gate

两臂的 terminal-approach verdict 都是 **0 pass / 1083 fail**。与 README 既有结论一致
（"forecast ≠ certifiable approach"）：换飞行模型不改变这一条。

---

## 6. 过期产物

**所有现存的 control-output checkpoint。** 控制契约换了单位（牛顿 → 比例），且
`TSConfig` 增加了必需的序列化字段，`load_checkpoint` 会拒绝它们——而不是把推力
错算五个数量级。`state`-output checkpoint 不受影响。optimizer、harvest、evaluation
和 comparison 产物均不受影响。

---

## 7. 未做的事

- 会议提到的 **speed gate at threshold（1.3 Vstall / 各机型进近速度）** 属于
  evaluation 的判据，不在本次范围；
- 会议提到的 **不同 dt 的优化器实验** 属于 `4dTrajectory/optimization`，不在本次范围；
- τ_T 与 τ_n 固定不扫（会议原话是它们"可以更接近瞬时"）；
- 滞后模型尚未产出正式的 train→predict→evaluate 机场级结果表——这与
  `control_parameter_prediction.zh.md` 记录的"架构已完成、度量未完成"是同一个缺口。
