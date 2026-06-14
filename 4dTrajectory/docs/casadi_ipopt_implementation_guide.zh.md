# CasADi/IPOPT 轨迹优化实现指南

本文面向 `4dTrajectory/optimization` 中的下一代优化器实现。它不是代码清单，而是一份实现逻辑教学：目标是让初学者理解“为什么这样建模、每一块负责什么、实现时应该按什么顺序推进”。

当前项目已经有基于 SciPy/SLSQP 的 `TranscriptionOptimizor`。CasADi/IPOPT 方案可以理解为它的升级版：

- 仍然使用 direct transcription / multiple shooting 思想；
- 仍然优化一段 4D 进近轨迹；
- 仍然满足飞机动力学、控制边界和终端状态约束；
- 主要变化是：用 CasADi 构造可自动求导的非线性规划问题，再交给 IPOPT 求解。

---

## 1. 先理解要解决的问题

轨迹优化不是“画一条平滑曲线”，而是在问：

> 给定飞机初始状态和目标状态，能否找到一组控制量，让飞机按照动力学模型飞到目标，同时满足速度、高度、姿态、推力等限制，并让某个代价最小？

在本项目的外部接口里，一个状态可以先用 6 个量表示：

```text
x = [lat, lon, alt, V, psi, gamma]
```

含义：

| 变量 | 含义 | 单位 |
|---|---|---|
| `lat` | 纬度 | degree |
| `lon` | 经度 | degree |
| `alt` | 高度 | m |
| `V` | 真空速或模型速度 | m/s |
| `psi` | 航向角 | rad |
| `gamma` | 航迹角 | rad |

控制量可以先沿用当前优化器的 3 个输入：

```text
u = [thrust, bank, attack]
```

含义：

| 变量 | 含义 | 单位 |
|---|---|---|
| `thrust` | 推力 | N |
| `bank` | 坡度角 / 滚转角 | rad |
| `attack` | 迎角 | rad |

质量 `m` 在当前动力学里满足 `dmdt = 0`，也就是不随时间变化。因此在第一版 CasADi/IPOPT 方案中，建议把质量当作固定参数，而不是优化变量。这样问题更小，也能避免重复约束带来的数值退化。

后面会建议优化器内部使用局部 ENU 坐标：

```text
x_internal = [east, north, alt, V, psi, gamma]
```

这不改变 backend/front-end 看到的接口。可以把它理解成：

```text
外部输入输出: lat/lon
优化器内部: east/north
```

这样既能保持项目 API 稳定，又能让 CasADi 动力学表达更简单。

---

## 2. 为什么要从 SLSQP 迁移到 CasADi/IPOPT

当前 SciPy/SLSQP 方案能跑通，但它有几个天然限制：

1. SLSQP 主要依赖数值差分估计梯度和约束 Jacobian。
2. 每次差分都要多次调用 simulator，计算成本很高。
3. 如果变量尺度差异大，SLSQP 很容易受到数值病态影响。
4. 动力学、约束、边界越复杂，有限差分越慢、越不稳定。

CasADi/IPOPT 的核心优势是：

| 能力 | 为什么重要 |
|---|---|
| 符号表达式 | 动力学、目标函数和约束都能变成计算图 |
| 自动求导 | Jacobian/Hessian 不需要手写，也不需要有限差分 |
| 稀疏矩阵 | multiple shooting 的约束结构很稀疏，IPOPT 可以利用 |
| 大规模 NLP | 更适合多段轨迹、多约束、多控制变量的问题 |

初学者可以这样理解：

```text
SLSQP:
    你给它一个黑盒函数，它不断试探。

CasADi/IPOPT:
    你把问题结构告诉它，它利用导数和稀疏结构更聪明地搜索。
```

---

## 3. 总体实现路线

建议新建一个并列优化器，例如：

```text
4dTrajectory/optimization/casadi_ipopt_optimizor.py
```

它的职责不要一开始做得太大。第一版只需要对齐当前 `TranscriptionOptimizor` 的核心能力：

1. 接收 `initial_state`、`target_state`、`n_segments`、`dt`、`max_iterations`。
2. 构造优化变量：总时间、每段控制、每个节点状态。
3. 构造符号动力学传播器。
4. 构造 defect constraints。
5. 构造 terminal constraints。
6. 构造目标函数。
7. 调用 IPOPT。
8. 把结果转换成当前 backend 能返回的 `final_time, node_control, node_state`。

推荐先保留现有 backend API。也就是说，前端仍然发送同样的 payload，只是 `optimizer` 增加一个新选项，例如：

```text
optimizer = "casadiIpopt"
```

这样可以让新旧优化器并存，方便比较结果和性能。

---

## 4. 选择 multiple shooting，而不是一开始做 collocation

CasADi 常见最优控制建模方式有两类：

| 方法 | 特点 |
|---|---|
| Multiple shooting | 每段开始/结束状态作为变量，中间用积分器传播 |
| Direct collocation | 在段内加入 collocation 点，同时约束多项式动力学 |

本项目第一版建议用 multiple shooting，原因是：

1. 它和当前 `TranscriptionOptimizor` 概念一致，迁移成本低。
2. 它更容易复用“每段控制保持常值”的思路。
3. 它更容易和现有返回格式对齐：每段一个 control，每段一个 node state。
4. 初学者更容易调试：每个 defect 就是“模拟出来的下一状态”和“优化变量里的下一状态”的差。

Direct collocation 长期也很有价值，但第一版不建议直接上。它会引入更多节点类型、更多方程和更复杂的结果解释，不利于先把 CasADi/IPOPT 跑通。

---

## 5. 决定 NLP 的变量布局

CasADi/IPOPT 最终求解的是一个非线性规划问题：

```text
minimize    J(z)
subject to  g_lower <= g(z) <= g_upper
            z_lower <= z    <= z_upper
```

这里的 `z` 是所有优化变量拼起来的大向量。

推荐变量布局和当前 transcription optimizer 保持一致：

```text
z =
[
    final_time,
    control_0,
    control_1,
    ...,
    control_{N-1},
    state_1,
    state_2,
    ...,
    state_N
]
```

其中，`state_i` 的前两个位置分量取决于内部建模选择：

```text
control_i = [thrust_i, bank_i, attack_i]
state_i   = [east_i, north_i, alt_i, V_i, psi_i, gamma_i]   # 推荐的内部形式
          或 [lat_i, lon_i, alt_i, V_i, psi_i, gamma_i]     # 兼容旧思路
```

注意这里的 `state_1 ... state_N` 是每段末端状态。初始状态 `state_0` 来自请求输入，是固定参数，不放进优化变量。

变量数量为：

```text
1 + N * control_dim + N * state_dim
= 1 + N * 3 + N * 6
= 1 + 9N
```

例如 `N = 10` 时，一共 `91` 个变量。

---

## 6. 先做尺度归一化

这是 CasADi/IPOPT 方案里非常重要的一步。

原始物理变量的数量级差异很大：

| 变量 | 典型数量级 |
|---|---:|
| `final_time` | 10 到 1000 |
| `thrust` | 1000 到 1000000 |
| `alt` | 100 到 10000 |
| `V` | 30 到 200 |
| `bank` / `attack` / `gamma` | 0 到 1 |
| `lat` / `lon` | 几十度 |

如果直接把这些物理量喂给 IPOPT，求解器会看到一个尺度很不均匀的问题。更好的做法是：

```text
优化器内部变量 = 物理变量 / scale
```

也就是说，IPOPT 优化的是 normalized variable，但动力学计算前再转换回 physical variable。

推荐第一版使用下列表：

| 物理变量 | scale 示例 |
|---|---:|
| `final_time` | `100 s` |
| `thrust` | `100000 N` |
| `bank` | `1 rad` |
| `attack` | `0.1 rad` |
| `lat` | `1 deg` 或 local delta scale |
| `lon` | `1 deg` 或 local delta scale |
| `alt` | `1000 m` |
| `V` | `100 m/s` |
| `psi` | `1 rad` |
| `gamma` | `0.1 rad` |

对初学者来说，重点不是 scale 表一次写完美，而是形成习惯：

```text
IPOPT 看到的变量最好都在 0.1 到 10 附近。
```

后续如果发现收敛慢或约束残差异常，再调整 scale。

---

## 7. 建立 CasADi 版本的动力学方程

这是整个迁移中最关键的一步。

现有 `GeodeticSimulator.step()` 是普通 Python + math + numpy 代码。CasADi 不能直接对这种黑盒 Python simulator 自动求导。因此第一版需要在 CasADi 里重新表达一份符号动力学。

### 7.1 第一版建议用局部平面动力学

当前 geodetic simulator 每一步都会做 WGS84/ENU 坐标转换。这个过程实现较复杂，直接符号化成本高。

为了先把 CasADi/IPOPT 跑通，建议第一版使用局部平面近似：

```text
east/north/up local frame
```

内部状态可以转换为：

```text
x_local = [east_m, north_m, alt_m, V, psi, gamma]
```

优化完成后，再把 `east/north` 转回 `lat/lon` 输出。

这样做的理由：

1. 终端空域范围通常是几十公里，局部 ENU 近似足够作为第一版优化模型。
2. 动力学方程更简单，CasADi 表达更清晰。
3. 能避免一开始就把 WGS84 转换和优化器纠缠在一起。

如果必须保持变量为 `lat/lon`，也可以用近似换算：

```text
dlat/dt = north_velocity / metres_per_degree_lat
dlon/dt = east_velocity / metres_per_degree_lon
```

但教学和调试上，局部 ENU 更清楚。

### 7.2 点质量动力学

局部平面下，状态导数可以表达为：

```text
east_dot  = V * cos(gamma) * cos(psi)
north_dot = V * cos(gamma) * sin(psi)
alt_dot   = V * sin(gamma)
```

气动力：

```text
rho = ISA_density(alt)
CL  = CL0 + CL_alpha * attack
CD  = CD0 + k * CL^2
L   = 0.5 * rho * V^2 * S * CL
D   = 0.5 * rho * V^2 * S * CD
```

速度和角度变化：

```text
V_dot     = (thrust * cos(attack) - D) / m - g * sin(gamma)
psi_dot   = (L + thrust * sin(attack)) * sin(bank) / (m * V * cos(gamma))
gamma_dot = ((L + thrust * sin(attack)) * cos(bank) - m * g * cos(gamma)) / (m * V)
```

这些方程和当前 `aerodynamic_model/simulator.py` 的核心模型一致，只是位置部分换成了局部 ENU。

### 7.3 避免动力学奇异点

动力学里有两个危险分母：

```text
m * V
m * V * cos(gamma)
```

因此 bounds 必须保证：

```text
V >= min_speed
abs(gamma) < pi / 2
m > 0
```

质量如果是固定参数，也要在进入优化前验证它大于 0。

---

## 8. 构造积分器

Multiple shooting 每段都需要一个传播器：

```text
state_{i+1, predicted} = F(state_i, control_i, dt_segment)
```

其中：

```text
dt_segment = final_time / N
```

CasADi 有两种常用方式：

| 方式 | 说明 |
|---|---|
| 手写 RK4 | 简单、可控、适合第一版 |
| CasADi integrator | 可调用 CVODES 等积分器，但配置更复杂 |

第一版建议手写固定步长 RK4。逻辑是：

```text
one_segment_integrator:
    h = dt_segment / n_substeps
    repeat n_substeps:
        k1 = f(x, u)
        k2 = f(x + h/2 * k1, u)
        k3 = f(x + h/2 * k2, u)
        k4 = f(x + h   * k3, u)
        x  = x + h/6 * (k1 + 2k2 + 2k3 + k4)
    return x
```

这不是“为了替代真实 simulator”，而是为了在优化问题内部提供可求导、稳定、速度可控的传播器。

建议：

- 第一版每段内部 `n_substeps = 1` 或 `2`。
- 如果轨迹误差大，再增加到 `4`。
- 不要把 `dt` 也做成优化变量；先让每段时长由 `final_time / N` 决定。

---

## 9. 构造 defect constraints

Defect constraint 是 multiple shooting 的核心。

假设第 `i` 段开始状态是 `x_i`，控制是 `u_i`，优化变量里给出的段末状态是 `x_{i+1}`。动力学传播器预测出：

```text
x_{i+1,predicted} = F(x_i, u_i, dt_segment)
```

那么这一段的 defect 是：

```text
defect_i = x_{i+1,predicted} - x_{i+1}
```

约束要求：

```text
defect_i = 0
```

初学者可以把它理解成：

> 优化器不能随便填节点状态。每个节点状态必须能由上一节点和上一段控制真实飞出来。

第 0 段的开始状态是固定输入 `initial_state`。之后每段的开始状态就是前一个优化节点：

```text
segment 0: initial_state -> state_1
segment 1: state_1      -> state_2
segment 2: state_2      -> state_3
...
```

Defect 约束数量：

```text
N * state_dim
```

如果 `N = 10` 且 `state_dim = 6`，就是 `60` 条等式约束。

---

## 10. 构造 terminal constraints

Terminal constraint 负责让最后一个节点命中目标状态：

```text
state_N = target_state
```

但不一定每个分量都必须 hard constraint。第一版建议：

```text
hard constraints:
    position: east/north/alt
    speed: V
    heading: psi
    flight path: gamma
```

也就是 6 个状态全部约束。

如果发现问题太难收敛，可以临时放松：

```text
hard constraints:
    position + altitude

soft objective:
    speed + heading + gamma
```

这样做的含义是：

> 先确保飞到目标点，再尽量匹配目标速度和姿态。

等第一版稳定后，再把更多终端状态改回 hard constraint。

### 航向角要用 wrap 差值

航向角有周期性。`179 deg` 和 `-179 deg` 实际只差 `2 deg`，不能直接相减。

终端航向误差和 defect 中的航向误差都应该使用“最短角差”：

```text
angle_error = atan2(sin(left - right), cos(left - right))
```

这比普通减法更适合优化。

---

## 11. 构造目标函数

目标函数决定“在所有满足约束的轨迹里，哪一个更好”。

第一版可以从简单目标开始：

```text
J = final_time
```

含义：

> 在满足所有动力学和终端约束的前提下，让飞行时间尽量短。

但只最小化时间可能导致控制量贴边，例如最大推力、大坡度、大迎角。为了让轨迹更温和，可以逐步加入正则项：

```text
J =
    w_time * final_time_scaled
  + w_thrust * sum(thrust_normalized^2)
  + w_bank * sum(bank^2)
  + w_attack * sum(attack^2)
  + w_smooth * sum((u_i - u_{i-1})^2)
```

建议实现顺序：

1. 先只用 `final_time` 跑通。
2. 如果控制量频繁贴边，再加控制正则。
3. 如果相邻段控制跳变明显，再加平滑正则。

不要一开始把目标函数写得很复杂。目标越复杂，越难判断失败原因。

---

## 12. 设置边界

边界是优化器的护栏。没有边界，IPOPT 会探索大量物理上不可能的状态。

### 12.1 时间边界

```text
1 s <= final_time <= 1000 s
```

后续可以根据航段距离和机型速度动态设置更窄范围。

### 12.2 控制边界

第一版可以沿用当前优化器的思路：

```text
0 <= thrust <= aircraft_max_thrust
-bank_max <= bank <= bank_max
attack_min <= attack <= attack_max
```

建议不要默认允许 `bank = +-90 deg`。优化器通常不需要探索这么极端的坡度角。对进近轨迹而言，`+-30 deg` 更适合作为第一版边界。

### 12.3 状态边界

```text
alt >= 0
V >= min_speed
V <= max_speed
abs(gamma) <= gamma_max
```

如果使用局部 ENU 状态，还需要给 `east/north` 一个合理范围，例如机场附近几十公里：

```text
-100000 m <= east/north <= 100000 m
```

如果使用 `lat/lon`，则至少限制在合法经纬度范围：

```text
-90 <= lat <= 90
-180 <= lon <= 180
```

但更推荐局部范围限制，因为优化器不应该为了一个进近任务探索半个地球。

---

## 13. 构造初始猜测

初始猜测决定 IPOPT 从哪里开始搜索。好的初值能让求解器很快进入可行区域；坏的初值会导致失败或耗时很长。

### 13.1 时间初值

不要固定写死 `100 s`。更稳妥的逻辑是：

```text
distance = initial 到 target 的三维距离
average_speed = max(initial.V, target.V, aircraft_nominal_speed)
final_time_guess = distance / average_speed * safety_factor
```

其中 `safety_factor` 可以先取 `1.2` 到 `1.5`。

### 13.2 状态初值

最简单有效的方式是线性插值：

```text
state_i_guess = initial + (target - initial) * i / N
```

对于角度变量：

- `psi` 应使用最短角差插值；
- `gamma` 通常可以直接插值；
- 如果从 `lat/lon` 转到 ENU，应在 ENU 空间里插值位置。

### 13.3 控制初值

控制初值不要全零。推荐：

```text
thrust = aircraft default thrust guess
bank = 0
attack = trim attack guess
```

其中 trim attack 的直觉是：

```text
lift(alpha) + thrust * sin(alpha) ~= m * g * cos(gamma)
```

它不是飞控律，只是让初始轨迹不要一开始就严重下坠或爬升。现有 `TranscriptionOptimizor` 已经有类似逻辑，可以作为设计参考。

---

## 14. 组织 CasADi 表达式

实现时建议把逻辑分层，而不是把所有表达式堆在一个函数里。

推荐结构：

```text
CasadiIpoptOptimizor
    validate inputs
    build scaling metadata
    build initial guess
    build variable bounds
    build dynamics function
    build segment integrator
    build objective and constraints
    call solver
    unpack and format result
```

几个关键点：

1. `build_dynamics_function` 只负责 `x_dot = f(x, u, params)`。
2. `build_integrator` 只负责从 `x_i` 传播到 `x_{i+1,predicted}`。
3. `build_nlp` 只负责把变量、目标和约束拼起来。
4. `unpack_solution` 只负责把 IPOPT 返回的大向量拆回 `final_time, controls, states`。

这样写的好处是：当结果不对时，可以单独测试动力学、积分器、变量布局和约束维度。

---

## 15. IPOPT 的约束上下界

IPOPT 不使用 SciPy 那种 `{'type': 'eq', 'fun': ...}` 格式。它需要一个大约束向量 `g(z)`，以及对应的下界和上界：

```text
lbg <= g(z) <= ubg
```

等式约束：

```text
g_i(z) = 0
lbg_i = 0
ubg_i = 0
```

不等式约束：

```text
lower <= g_i(z) <= upper
```

第一版中，defect constraints 和 terminal constraints 都可以作为等式约束。

变量边界则单独放在：

```text
lbx <= z <= ubx
```

初学者最容易出错的地方是：`g`、`lbg`、`ubg` 的长度必须完全一致；`z`、`lbx`、`ubx` 的长度也必须完全一致。

---

## 16. 结果转换回项目格式

如果内部用 ENU 优化，求解完成后要转回当前项目使用的地理状态：

```text
east/north/alt -> lat/lon/alt
```

返回给 backend 的形状建议保持不变：

```text
final_time: float
node_control: shape (N, 3)
node_state: shape (N, 6)
```

其中 `node_state` 仍然是：

```text
[lat, lon, alt, V, psi, gamma]
```

这样 `aeroviz_backend/optimization_backend.py` 只需要最小改动：识别新 optimizer 名称，然后调用新类。

---

## 17. 验证顺序

不要等整个优化器写完才测试。建议按下面顺序验证。

### Step 1：动力学符号函数

给一个普通 A320 进近状态和控制量，检查：

```text
x_dot finite
V_dot finite
psi_dot finite
gamma_dot finite
```

如果这里已经出现 NaN 或 inf，后面一定失败。

### Step 2：RK4 积分器

用同一个初始状态和控制量传播 1 秒，检查：

```text
next state finite
altitude >= 0
speed >= min_speed
```

再和当前 Python simulator 做一个粗略对比。第一版不要求完全相同，但方向应一致：

- 推力大时速度不应无故下降很多；
- 正 gamma 时高度应上升；
- 左右 bank 应改变 heading 方向。

### Step 3：变量维度

固定 `N = 3`，检查：

```text
number of variables = 1 + 9N
number of defect constraints = 6N
number of terminal constraints = 6
```

维度正确比一开始追求收敛更重要。

### Step 4：初始点约束残差

在调用 IPOPT 之前，先计算初始猜测的约束残差：

```text
defect residual norm
terminal residual norm
```

如果初始残差极大，说明 state guess 或 control guess 不合理。IPOPT 不是魔法，它也需要一个能看懂的问题。

### Step 5：小问题求解

先用非常容易的问题测试：

```text
initial 和 target 距离很近
高度差很小
航向差很小
N = 3 或 4
```

成功后再恢复真实 RW05L payload。

### Step 6：和现有 SLSQP 结果对比

比较：

| 指标 | 说明 |
|---|---|
| 是否成功 | IPOPT status |
| final time | 是否合理 |
| terminal residual | 是否命中目标 |
| control 是否贴边 | 是否过度使用最大推力/最大坡度 |
| wall time | 是否比 SLSQP 更快 |
| replay error | 用现有 simulator 回放后是否仍接近目标 |

---

## 18. 真实 simulator 回放很重要

CasADi 内部动力学如果用了局部 ENU 近似，就和现有 `GeodeticSimulator` 不完全相同。因此优化结束后，建议增加一个 replay 验证：

```text
initial_state
    + optimized control sequence
    + final_time / N
    -> GeodeticSimulator.step replay
```

然后比较 replay 最终状态和 target：

```text
position error
altitude error
speed error
heading error
gamma error
```

这一步非常关键。它告诉你：

> 优化器内部的近似模型，和项目真实播放/验证模型是否足够一致。

如果 replay error 太大，优先检查：

1. ENU 和 lat/lon 转换是否正确；
2. heading convention 是否一致；
3. `psi = 0` 是指 East 还是 North；
4. RK4 substeps 是否太少；
5. CasADi 动力学参数是否和 `Simulator` 一致。

---

## 19. 常见失败原因

### 19.1 IPOPT 一开始就 infeasible

常见原因：

- target 太远，但 final time 上界太小；
- 速度/高度/姿态边界过窄；
- 初始状态或目标状态本身不在边界内；
- terminal constraints 太严格。

处理顺序：

1. 检查 initial/target 是否满足 bounds。
2. 放宽 final time 上界。
3. 降低 terminal hard constraint 数量。
4. 增加 `N` 或 RK4 substeps。

### 19.2 解出来但控制量全部贴边

常见原因：

- 目标函数只最小化时间；
- 时间上界或终端状态过于激进；
- 控制正则权重太小。

处理方式：

- 增加控制使用代价；
- 增加控制平滑项；
- 给 final time 一个合理下界；
- 检查 target 是否物理可达。

### 19.3 求解很慢

常见原因：

- 变量没有缩放；
- `N` 太大；
- RK4 substeps 太多；
- 终端约束过硬；
- 初始猜测离可行轨迹太远。

处理方式：

- 先用 `N = 4` 跑通；
- 检查 normalized variables 是否在合理范围；
- 使用上一次结果 warm start；
- 粗网格求解后再细化。

### 19.4 IPOPT 成功，但前端轨迹看起来不对

常见原因：

- internal ENU 到 lat/lon 转换方向错；
- `east/north` 和 `x/y` 约定混淆；
- `psi` 的方向定义和 Cesium/后端理解不同；
- 返回的 node states 少了初始点或顺序错。

处理方式：

- 打印第一个、最后一个节点；
- 在地图上只画 initial、target、optimized final 三个点；
- 单独验证 ENU 转 geodetic；
- 用 replay 结果和 optimized node result 对比。

---

## 20. 建议的实现阶段

### Phase 1：最小可运行 CasADi/IPOPT optimizer

目标：

```text
能在一个简单 payload 上成功返回 final_time、controls、states。
```

范围：

- 局部 ENU 状态；
- fixed mass；
- multiple shooting；
- RK4 integrator；
- final time objective；
- hard terminal state constraint；
- 基础 bounds；
- 与 backend API 对接。

不做：

- 风场；
- 地形约束；
- 复杂程序约束；
- direct collocation；
- 多阶段 refine；
- 前端 UI 大改。

### Phase 2：数值质量

目标：

```text
真实 RW05L 默认 payload 稳定成功，控制量不过度贴边。
```

范围：

- 完整变量缩放；
- control regularization；
- control smoothness；
- 更合理的 aircraft-specific bounds；
- 更好的 final time guess；
- replay validation。

### Phase 3：性能和交互体验

目标：

```text
前端调用速度明显优于 SLSQP，且连续调参体验更稳定。
```

范围：

- warm start；
- coarse-to-fine 求解；
- IPOPT 参数调优；
- debug metadata 返回；
- benchmark 脚本。

### Phase 4：扩展约束

目标：

```text
服务 thesis 中更真实的 4D approach trajectory 场景。
```

可逐步加入：

- altitude window；
- speed schedule；
- CTA / required time of arrival；
- runway threshold crossing constraints；
- OCS / terrain clearance soft or hard constraints；
- procedure waypoint corridor constraints；
- bank rate / vertical speed / load factor constraints。

---

## 21. IPOPT 参数建议

第一版不需要过度调参。可以先关注几个常用项：

| 参数 | 用途 |
|---|---|
| `ipopt.max_iter` | 最大迭代次数 |
| `ipopt.tol` | 总体收敛容差 |
| `ipopt.constr_viol_tol` | 约束违反容差 |
| `ipopt.print_level` | IPOPT 日志详细程度 |
| `print_time` | 是否打印 CasADi 求解时间 |

调试阶段建议打开较详细日志。接入前端后，应降低日志噪声。

不要一开始就通过放宽容差来“制造成功”。如果约束残差很大，即使 IPOPT 返回了某种结果，轨迹也未必有物理意义。

---

## 22. 推荐的测试清单

至少准备这些测试：

| 测试 | 保护什么 |
|---|---|
| optimizer import test | 新依赖和路径没有破坏测试环境 |
| variable layout test | `z` 的拆包顺序正确 |
| dynamics finite test | 符号动力学不会产生 NaN/inf |
| integrator sanity test | RK4 传播方向合理 |
| small payload solve test | 最小问题能成功 |
| backend optimizer selection test | `optimizer="casadiIpopt"` 能被识别 |
| response shape test | 返回 `controls/states` 数量正确 |
| replay validation test | 优化控制用真实 simulator 回放后误差可接受 |

测试不要一开始就只跑真实大问题。真实 payload 慢、失败原因多，不适合作为唯一反馈。

---

## 23. 初学者实现时的心法

1. **先建小问题，再建真实问题。** `N = 3` 的近距离目标，比完整进近场景更适合调试。
2. **先看维度，再看数值。** 变量、约束、上下界长度不一致，是最常见的第一类错误。
3. **先让初始点有限。** 如果 `x0` 代入动力学就 NaN，IPOPT 没法救。
4. **先用简单目标函数。** 等约束稳定后再加入控制平滑和复杂代价。
5. **每个角度误差都要考虑 wrap。** 特别是航向角。
6. **优化模型和真实 simulator 要回放对比。** 只有 IPOPT 成功不等于系统可信。
7. **不要把质量作为第一版优化变量。** 当前模型质量不变，固定参数更清楚。
8. **不要一开始引入太多航空规则。** 先跑通动力学 NLP，再加程序约束、地形约束、CTA。

---

## 24. 一句话总结

CasADi/IPOPT 方案的核心不是“换一个优化器名字”，而是把当前黑盒式 SLSQP 问题重写成结构清晰、可自动求导、尺度良好的非线性规划：

```text
状态节点 + 控制节点 + 动力学 defect
    + 终端约束 + 物理边界 + 合理初值
    -> CasADi symbolic NLP
    -> IPOPT solve
    -> replay validation
    -> AeroViz-4D trajectory output
```

第一版只要把这条链路跑通，就已经完成了最重要的迁移。后续的性能、约束丰富度和前端体验，都可以在这个基础上逐步增加。
