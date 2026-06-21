# Direct Collocation 与 Hermite-Simpson 算法

本文配合 `4dTrajectory/optimization/casadi_direct_collocation_optimizer.py` 阅读。目标是把"为什么写成这样、跟原来的 multiple shooting 差在哪里"讲清楚，让你看代码时知道每一段对应论文里的哪一行。

读完本文你应该能回答：

1. 直接配点 (direct collocation) 是怎么把一个连续时间 OCP 离散成一个 NLP 的；
2. Hermite-Simpson 公式是怎么"凭空"冒出来的，3 阶精度是哪来的；
3. 为什么把 `T`（到达时间）作为决策变量在 direct collocation 里几乎是免费的，而在 multiple shooting 里需要外层 bisection；
4. 这两种方法相对于经典数值积分（RK4）的工程取舍。

---

## 1. 我们要解的问题

终端区 4D 轨迹优化的标准最优控制问题 (Optimal Control Problem, OCP)：

$$
\begin{aligned}
\min_{x(\cdot),\,u(\cdot),\,T} \quad & J = T + \lambda \int_0^T \|u(t)\|^2_W \, dt \\
\text{s.t.} \quad & \dot{x}(t) = f(x(t), u(t)) && t \in [0, T] \\
& x(0) = x_0,\quad x(T) = x_T \\
& x(t) \in \mathcal{X},\quad u(t) \in \mathcal{U}
\end{aligned}
$$

状态、控制和动力学函数在我们的项目里是：

$$
x = (e,\, n,\, h,\, V,\, \psi,\, \gamma)^\top, \qquad
u = (T_\text{thrust},\, \mu,\, n_\text{cmd})^\top
$$

$f$ 就是 `aerodynamic_model.casadi_simulator.make_dynamics_model()` 返回的那个连续 ENU 三自由度点质量模型：

$$
\dot{x} = \begin{bmatrix}
V \cos\gamma \cos\psi \\[2pt]
V \cos\gamma \sin\psi \\[2pt]
V \sin\gamma \\[2pt]
(T_\text{thrust} - D)/m - g\sin\gamma \\[2pt]
g\,n_\text{cmd}\sin\mu / (V\cos\gamma) \\[2pt]
g\,(n_\text{cmd}\cos\mu - \cos\gamma)/V
\end{bmatrix}
$$

OCP 是连续的、无限维的；NLP 求解器（如 IPOPT）只会做有限维优化。把 OCP 变成 NLP 的过程叫做 **离散化 (transcription)**。

---

## 2. 两种主流离散化方法

把一段 $[0,T]$ 切成 $N$ 段，每段长 $h = T/N$。每一段的左右端点叫**节点 (knot)**：$t_0=0, t_1, \ldots, t_N = T$。

### 2.1 Multiple Shooting（多重打靶）

> 在每段里**用积分器跑动力学**，然后用代数约束保证段与段相连。

每段 $k$ 内部用一个 RK4 之类的固定积分器把 $x_k$ 推到 $\tilde{x}_{k+1}$：

$$
\tilde{x}_{k+1} = \Phi_h(x_k,\, u_k) \quad\text{(RK4 几步以保精度)}
$$

下一节点 $x_{k+1}$ 作为决策变量，约束：

$$
\underbrace{x_{k+1} - \tilde{x}_{k+1}}_{\text{shooting defect}} = 0
$$

NLP 的决策变量是 $\{x_0, x_1, \ldots, x_N, u_0, \ldots, u_{N-1}\}$（控制在每段内段常数），约束是上面那串 defect。

特点：

- 段内动力学 **被精确积分**（RK4 是 4 阶），所以 defect 的"物理含义"清晰；
- 但 NLP 的雅可比里要塞进 RK4 的链式求导，结构稠密；
- 每次 NLP 探针都得跑一次 RK4，慢；
- 如果想把 $T$ 也作为变量，$h = T/N$ 就出现在 RK4 的内部递推里，雅可比对 $T$ 的导数非常脏。所以工程上一般固定 $T$，外面套一层 bisection 找最短可行时间——`casadi_optimizer.py` 走的就是这条路。

### 2.2 Direct Collocation（直接配点）

> **不**积分。在每一段里假设一个解析的多项式 $\tilde{x}(t)$，让它在若干"配点 (collocation points)"上同时满足 ODE。

代数化以后的样子是：在节点 $k, k+1$（以及可能的内部配点）上要求

$$
\dot{\tilde{x}}(t_\text{col}) = f(\tilde{x}(t_\text{col}),\, u(t_\text{col}))
$$

成立。这条等式叫 **collocation defect**。

direct collocation 的关键特征：

- 状态多项式是**预先选定的形式**（线性、Hermite 三次、Lagrange 高次…），系数由节点状态唯一决定；
- defect 只需要**计算 $f$ 在几个点的值**，不需要积分；
- 因此 NLP 的雅可比稀疏、结构整齐；
- $T$ 作为决策变量是天然的——$h$ 只是个标量，直接以符号形式出现在公式里。

我们的优化器选了三次 Hermite 多项式 + Simpson 求积，下面把这个组合推导一遍。

---

## 3. Hermite-Simpson 公式怎么来的

### 3.1 一段内的 Hermite 三次多项式

在 $[t_k, t_{k+1}]$，把
$x_k = x(t_k),\; x_{k+1} = x(t_{k+1}),\; f_k = f(x_k,u_k),\; f_{k+1} = f(x_{k+1},u_k)$
都看作"已知"。要求多项式 $\tilde{x}(t)$ 在两端**值和导数都匹配**：

$$
\tilde{x}(t_k) = x_k,\quad \tilde{x}(t_{k+1}) = x_{k+1},\quad
\dot{\tilde{x}}(t_k) = f_k,\quad \dot{\tilde{x}}(t_{k+1}) = f_{k+1}.
$$

4 个条件、唯一确定的 3 次多项式就叫 **Hermite 三次插值**。用归一化变量 $\tau = (t - t_k)/h \in [0,1]$：

$$
\tilde{x}(\tau) = (1 - 3\tau^2 + 2\tau^3)\,x_k + h(\tau - 2\tau^2 + \tau^3)\,f_k
+ (3\tau^2 - 2\tau^3)\,x_{k+1} + h(-\tau^2 + \tau^3)\,f_{k+1}.
$$

### 3.2 中点的值和导数

代 $\tau = 1/2$ 立得：

$$
\boxed{\;x_\text{mid} = \tfrac{1}{2}(x_k + x_{k+1}) + \tfrac{h}{8}\,(f_k - f_{k+1})\;}
$$

这就是代码里那一行：

```python
x_mid = 0.5 * (x_k + x_kp1) + (h / 8.0) * (f_k - f_kp1)
```

注意：$x_\text{mid}$ 是**从多项式读出来的**，不是新的决策变量。这是所谓的 **compressed（紧凑）Hermite-Simpson**：节点状态全是决策变量，中点状态用代数公式表达。还有一种叫 separated（分离）形式，把 $x_\text{mid}$ 也设成决策变量并加一条等式，规模大一倍但雅可比更稀疏，工程上各有用场。

### 3.3 用 Simpson 积分把 defect 写出来

对 ODE 两边在 $[t_k, t_{k+1}]$ 积分：

$$
x_{k+1} - x_k = \int_{t_k}^{t_{k+1}} \dot{x}(t)\, dt
$$

Simpson 求积公式（3 阶精度，用 3 个等距样本点）：

$$
\int_{t_k}^{t_{k+1}} g(t)\, dt \approx \frac{h}{6}\bigl(g(t_k) + 4\,g(t_\text{mid}) + g(t_{k+1})\bigr).
$$

把 $g = \dot{x} = f(x,u)$ 代进去：

$$
\boxed{\;x_{k+1} - x_k - \tfrac{h}{6}\bigl(f_k + 4\,f_\text{mid} + f_{k+1}\bigr) = 0\;}
$$

这就是 Hermite-Simpson collocation defect，写成代码就是：

```python
defect = x_kp1 - x_k - (h / 6.0) * (f_k + 4.0 * f_mid + f_kp1)
```

`f_mid = f(x_mid, u_k)` 是动力学在多项式中点的求值。

### 3.4 为什么是 3 阶精度？

- Hermite 三次插值的残差是 $O(h^4)$；
- Simpson 求积公式对 5 次多项式精确，对一般光滑函数误差是 $O(h^5)$；
- 把两者组合在一起，单段的 local truncation error 是 $O(h^4)$，整段累计 (global error) 是 $O(h^3)$；
- 这就是 "Hermite-Simpson is 3rd-order accurate" 的来历。

实操上：分段不需要太密。例如 $h = 1\text{s}$ 时，单段动力学积分误差大约 $10^{-3}$ 量级，远小于工程容差。

### 3.5 控制怎么离散？

我们选了**段常数控制 (piecewise constant)**：每段一个 $u_k$，段内三个评估点（左端、中点、右端）都用同一个值。

这样有两个工程好处：

1. 决策变量数和原来的 multiple shooting 一致 (`N` 段控制 × 3 维)，前端不用改输出协议；
2. 飞行员/自动驾驶的"指令"本来就是段常数的（每段都是一个 "把推力设为 X" 之类的命令）。

代价是控制的最高精度被压到了 2 阶（一阶导数在节点处不连续）。如果要把整体精度推到 4 阶，可以改用段线性 + Hermite-Simpson 分离形式，或者高阶 Radau 配点（Lobatto IIIA 系）。

---

## 4. Free-Final-Time：在 direct collocation 里几乎免费

把 $T$ 升级成决策变量。整段 NLP 写成：

$$
\begin{aligned}
\min_{x_{0:N},\,u_{0:N-1},\,T} \quad &
\frac{T}{T_\text{max}} + \lambda\,\frac{1}{N}\sum_{k=0}^{N-1}\|u_k\|^2_W \\
\text{s.t.}\quad & x_{k+1} - x_k - \tfrac{h}{6}(f_k + 4f_\text{mid} + f_{k+1}) = 0,\; h = T/N,\;\forall k \\
& x_0 = x_\text{start},\; x_N = x_\text{target} \\
& \underline{x} \le x_k \le \overline{x},\; \underline{u} \le u_k \le \overline{u} \\
& T_\text{min} \le T \le T_\text{max}
\end{aligned}
$$

为什么这在 direct collocation 里"几乎免费"？

- defect 公式里 $h$ 只是个符号，IPOPT 对它自动求导没有任何额外代价；
- 雅可比新增了一列（对 $T$ 的偏导），稀疏结构不变；
- 单次求解给出 $(x^*, u^*, T^*)$，**不需要外层 bisection**。

对照 multiple shooting：$h$ 出现在 RK4 的递推
$k_1 = f(x_k), k_2 = f(x_k + \tfrac{h}{2}k_1), \ldots$ 里面，对 $T$ 的链式求导会把雅可比塞得很满。再加上 RK4 步数本身也可能跟 $h$ 联动，工程上常常干脆把 $T$ 固定、外层 bisection 找最优——`CasadiOptimizer.optimize_time_to_target` 走的就是这条路。

我们的 `optimize_free_time` 实现了一个**两阶段 warm-start**：

1. 先用同一套节点、固定 $T = T_\text{max}$ 解一遍 fixed-time NLP（这本身就是一个可行的 direct collocation 问题）；
2. 把这个解作为 free-time NLP 的初值（加一个 $T_0 = T_\text{max}$），让 IPOPT 沿可行流形向更短的 $T$ 收缩。

冷启动（线性插值的状态轨迹）通常会让 IPOPT 一上来就在 restoration phase 里挣扎；warm-start 把 inf_pr 从 ~40 直接压到 ~0。代价是多解一次 NLP，但因为 fixed-time NLP 收敛极快（动力学约束已经被 IPOPT 之前的解记住了），总耗时仍然显著低于 bisection。

---

## 5. 两种方法对照表

| 维度 | Multiple Shooting | Direct Collocation (HS) |
|---|---|---|
| 段内动力学怎么满足 | 数值积分 (RK4) | 多项式 + 配点 defect |
| 是否需要积分器调用 | 是，每段 RK4 多步 | 否，只在 3 个点求 $f$ |
| 单次 NLP 求解的雅可比稀疏度 | 中等（积分链式导数稠密） | 稀疏（带状） |
| 阶数 | 与积分器相同（RK4 = 4） | 3（HS 紧凑形） |
| $T$ 作为决策变量 | 困难（积分器内有 $h$） | 自然（$h$ 仅出现在 defect） |
| 实现复杂度 | 中等（仔细处理 substeps） | 偏低（公式直接展开） |
| 对差初值的鲁棒性 | 较好（积分先把轨迹"驱动"成可行） | 较差（冷启动可能远离可行流形） |
| 段长 $h$ 增大时的精度衰减 | $h^4$ | $h^3$ |
| 与微分方程不连续点的兼容 | 较差（积分器跨断点会假阳性） | 较好（每段独立、可放断点） |

工程经验法则：

- 飞行轨迹这类连续动力学、需要灵活 $T$、终端约束硬的问题，**Hermite-Simpson direct collocation 通常更好用**；
- 控制空间很离散、动力学带断点（比如档位切换）的问题，multiple shooting + 事件检测可能更自然；
- 大规模 (>100 段) 时 direct collocation 的稀疏雅可比会显著领先；
- 对初值敏感时考虑 warm-start：先 fixed-time，再 free-time（就是本项目的策略）。

---

## 6. 与 RK4 / 单次打靶的关系

| 方法 | 动力学 | 决策变量 | 终端约束 |
|---|---|---|---|
| 单次打靶 (single shooting) | 全段 RK4 一次性积分 | 只有 $u_{0:N-1}$（和 $T$） | 终端通过积分隐式得到 |
| 多重打靶 (multiple shooting) | 每段独立 RK4 | $x_k, u_k$ 全是变量 | 显式 defect + 终端等式 |
| 直接配点 (direct collocation) | 多项式 + 几个点上的 ODE 残差 | $x_k, u_k$ 全是变量（HS 紧凑形不含中点） | 显式 defect + 终端等式 |

直观上：从 single shooting → multiple shooting → direct collocation，"显式约束"在变多，"隐式积分"在变少。

显式约束变多有两个直接好处：

1. 雅可比稀疏，IPOPT 内点法的牛顿系统好解；
2. 可以为每段单独地塞控制盒子、状态盒子、避障盒子等约束，不会被积分器"遮挡"。

代价：决策变量也变多了。所以 mesh refinement（自适应加点）是常见后续优化方向。

---

## 7. 与本项目代码的对应关系

| 公式 / 概念 | 代码位置 |
|---|---|
| 连续 RHS $f(x,u)$ | `aerodynamic_model.casadi_simulator.make_dynamics_model()` |
| HS defect 公式 | `casadi_direct_collocation_optimizer.hermite_simpson_defect_expr` |
| 固定时间 NLP | `make_direct_collocation_solver` |
| 自由时间 NLP（含 $T$） | `make_direct_collocation_solver_free_time` |
| 段常数控制 + 段端状态决策 | NLP 构造里的 `seg_controls`, `seg_states` |
| 几何/状态边界 | `make_state_bounds`, `make_control_bounds` |
| 两阶段 warm-start | `CasadiDirectCollocationOptimizer._build_free_time_initial_guess` |
| Fixed-ENU 坐标系 | `_geodetic_state_to_enu_decision` / `_enu_decision_to_geodetic_state` |
| 曲率误差量化（论证 fixed-ENU 可用） | `fixed_enu_frame_error.py` |

固定 ENU 这一选择是 direct collocation 的硬性要求：每个 defect 公式假设 $f(x,u)$ 是连续可微的状态函数。原来 `make_geo_step_from_enu_integrator` 每一步重新锚定 ENU 框架，那个"换锚"是离散跳变，没法塞进 defect 里——所以 collocation 必须用单个固定 ENU 框架，由此引入的曲率误差由 `fixed_enu_frame_error.py` 量化，在 5 km 范围内不超过 ~2 m / 0.06°，足够工程使用。

---

## 8. 进一步阅读

- Betts, J. T. *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming.* SIAM, 2010. （direct collocation 的经典教材；HS 推导在第 4 章）
- Hargraves, C. R. and Paris, S. W. *Direct Trajectory Optimization Using Nonlinear Programming and Collocation.* J. Guidance, Control, and Dynamics, 1987. （HS 方法在飞行力学里的第一次系统应用）
- Kelly, M. *An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation.* SIAM Review, 2017. （非常好读的入门，公式对应代码可以直接照抄）
- Rawlings, J. B., Mayne, D. Q., Diehl, M. *Model Predictive Control: Theory, Computation, and Design.* 2nd ed., 2020. （从 OCP 离散化角度的现代综述）
- 本项目其他文档：`casadi_ipopt_implementation_guide.zh.md`（multiple shooting 在本项目的实现路径）、`transcription_optimizer_speedup_plan.zh.md`（原 SLSQP 路径的对照）。
