# Transcription Optimizer 数值错误排查指南

这份文档记录一次真实排查过程：前端调用 `/optimization/run` 后，后端先后出现：

```text
Required step size is less than spacing between numbers.
Optimization failed: Singular matrix C in LSQ subproblem
```

目标不是只记住这两个错误的修法，而是学会一种通用排查思想：

1. 先把 HTTP 错误还原成一个本地、可重复、可打印的优化问题；
2. 再把优化器黑盒拆成目标函数、约束函数、初值、边界和 Jacobian；
3. 最后用动力学方程解释数值症状，而不是猜参数。

---

## 1. 先判断错误来自哪一层

后端日志是：

```text
[aeroviz-backend] error status=400 method=POST path=/optimization/run message=...
```

这个 `400` 本身不是根因。`aeroviz_backend/http_server.py` 会把 `ValueError` 包装成 HTTP 400，所以第一步要问：

| 看到的现象 | 应该继续追问 |
|---|---|
| HTTP 400 | 是输入 schema 错，还是优化器抛了 `ValueError`？ |
| `Optimization failed: ...` | 是 SciPy optimizer 失败，还是 simulator 失败被 optimizer 包起来？ |
| `solve_ivp` message | 是 ODE 本身病态，还是 optimizer 给了不可能的状态？ |

这次真正的链路是：

```text
POST /optimization/run
  -> OptimizationBackend.optimize()
  -> TranscriptionOptimizor.optimize_trajectory()
  -> scipy.optimize.minimize(method="SLSQP")
  -> defect_constraints()
  -> GeodeticSimulator.step()
  -> Simulator.simulate()
  -> scipy.integrate.solve_ivp()
```

所以排查要落到 `TranscriptionOptimizor`，不能只盯着 HTTP 层。

---

## 2. 建立一个不依赖前端的复现闭环

前端默认请求可以直接在 Python 里复现。这样做有三个好处：

- 不需要浏览器交互；
- 可以 monkeypatch `minimize`；
- 可以打印优化变量和约束矩阵。

本项目 Python 命令要用 conda `aviation` 环境。脚本建议直接使用环境解释器：

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python3.13 - <<'PY'
from aeroviz_backend.optimization_backend import OptimizationBackend

payload = {
    "initialState": {
        "lon": -78.7873,
        "lat": 35.878659,
        "altM": 1000,
        "speedMps": 120,
        "headingDeg": 0,
        "flightPathDeg": 0,
        "massKg": 78000,
        "aircraftType": "A320",
    },
    "targetState": {
        "lon": -78.802,
        "lat": 35.874,
        "altM": 111.86,
        "speedMps": 70,
        "headingDeg": 45,
        "flightPathDeg": -4,
        "massKg": 78000,
        "aircraftType": "A320",
    },
    "targetControl": {"attackDeg": 4},
    "nSegments": 10,
}

print(OptimizationBackend().optimize(payload))
PY
```

如果这段脚本复现了后端日志里的错误，就说明已经有了一个可靠反馈闭环。

---

## 3. 第一类错误：积分器步长小到机器精度以下

错误：

```text
Required step size is less than spacing between numbers.
```

这是 `solve_ivp` 的数值积分失败。它通常不是“时间步长设置错了”这么简单，而是 ODE 右端函数在某些输入状态下接近奇异。

查看 `aerodynamic_model/simulator.py` 的动力学：

```text
dpsidt  = (...) / (m * V * cos(gamma))
dgamadt = (...) / (m * V)
```

因此如果 optimizer 给 simulator 的候选状态出现下面情况，ODE 就可能炸掉：

| 危险状态 | 为什么危险 |
|---|---|
| `V = 0` 或非常小 | 航向角/航迹角导数分母接近 0 |
| `m = 0` 或非常小 | 加速度和角速度分母接近 0 |
| `gamma` 接近 `+-90 deg` | `cos(gamma)` 接近 0 |
| 高度为负 | geodetic simulator 会直接拒绝 |

这次最早的版本用了全零 `node_state_guess`，等价于把 `V=0, m=0` 的节点喂给 simulator。修复思路不是吞掉异常，而是：

1. 用 `initial_state -> target_state` 插值生成节点初值；
2. 给状态加物理边界，例如 `V >= 1`、`m >= 1`、`alt >= 0`；
3. 对 SLSQP 探测到的无效候选点返回“大残差”，让它知道这个点不可行。

---

## 4. 第二类错误：SLSQP 的 LSQ 子问题矩阵奇异

错误：

```text
Optimization failed: Singular matrix C in LSQ subproblem
```

这个错误来自 SLSQP。直观解释是：它在当前点线性化等式约束后，发现约束 Jacobian 不可用，无法形成稳定的最小二乘子问题。

不要靠猜。直接打印约束向量和约束 Jacobian 的秩。

### 4.1 用 monkeypatch 拦截 `minimize`

下面这段调试脚本不会改源码。它把 `transcription_optimizor.minimize` 临时替换成检查函数，然后在真正优化前计算：

- 变量数；
- 约束数；
- 初始约束残差范数；
- 约束是否全是 `1e9` 的 invalid residual；
- 数值 Jacobian 的 rank；
- 最小奇异值。

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python3.13 - <<'PY'
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from aeroviz_backend.optimization_backend import OptimizationBackend
import transcription_optimizor as module

payload = {
    "initialState": {
        "lon": -78.7873,
        "lat": 35.878659,
        "altM": 1000,
        "speedMps": 120,
        "headingDeg": 0,
        "flightPathDeg": 0,
        "massKg": 78000,
        "aircraftType": "A320",
    },
    "targetState": {
        "lon": -78.802,
        "lat": 35.874,
        "altM": 111.86,
        "speedMps": 70,
        "headingDeg": 45,
        "flightPathDeg": -4,
        "massKg": 78000,
        "aircraftType": "A320",
    },
    "targetControl": {"attackDeg": 4},
    "nSegments": 10,
}

original_minimize = module.minimize

def inspecting_minimize(fun, x0, bounds, constraints, method, options):
    del fun, bounds, method, options

    def all_constraints(z):
        return np.concatenate([
            constraint["fun"](z)
            for constraint in constraints
        ])

    c0 = all_constraints(x0)
    jac = approx_derivative(all_constraints, x0, method="2-point", rel_step=1e-6)
    singular_values = np.linalg.svd(jac, compute_uv=False)

    print("variables:", len(x0))
    print("constraints:", len(c0))
    print("initial residual norm:", np.linalg.norm(c0))
    print("all invalid residual:", np.all(c0 == 1e9))
    print("jacobian shape:", jac.shape)
    print("jacobian rank:", np.linalg.matrix_rank(jac, tol=1e-8))
    print("smallest singular values:", singular_values[-8:])

    raise SystemExit

module.minimize = inspecting_minimize
try:
    OptimizationBackend().optimize(payload)
finally:
    module.minimize = original_minimize
PY
```

`scipy.optimize._numdiff.approx_derivative` 是 SciPy 的内部调试工具，不建议放进生产代码；但排查时非常有用。

### 4.2 如何解读输出

#### 情况 A：约束全是 `1e9`

如果看到：

```text
all invalid residual: True
jacobian rank: very small
```

说明初始点一进 `defect_constraints()` 就被判定为不可模拟，或者 simulator 第一步就失败了。

这次的直接原因是：control 初值用 `alpha = 0`，A320 在低速/大质量下升力不足，第一步模拟会快速下沉，约束函数返回常量 `1e9`。常量残差几乎没有导数信息，SLSQP 得不到有效搜索方向。

修复思想：

```text
不要让第一个优化探测点落到常量惩罚面上。
```

具体做法是给每个 shooting segment 的起点估算一个 trim alpha：

```text
lift(alpha) + thrust * sin(alpha) ~= m * g * cos(gamma)
```

这不是自动驾驶控制律，只是初始化启发式。它的作用是让第一轮数值 Jacobian 可计算。

#### 情况 B：约束不是全 invalid，但 rank 少 1

如果看到：

```text
constraints: 77
jacobian rank: 76
```

说明不是 simulator 第一时间失败，而是约束本身有冗余。

这次冗余来自 mass：

- 动力学模型里 `dmdt = 0`；
- 每段 defect 已经强制 `m_i = m_{i-1}`；
- final state 又要求 `m_final = m_target`；
- 如果 target mass 本来就等于 initial mass，这条 final mass 约束就是前面 defect 约束的线性组合。

SLSQP 对等式约束 rank 很敏感，冗余约束会让 LSQ 子问题矩阵奇异。

修复思想：

```text
如果一个状态量没有动力学自由度，不要在 terminal constraint 里重复约束它。
```

因此 final state constraint 只约束：

```text
lat, lon, alt, V, psi, gamma
```

不再约束 `m`。质量仍然由 defect 约束保持常量。

---

## 5. 把诊断变成回归测试

调试脚本只能帮你找到问题，不能防止问题回来。修复后至少要把两个关键发现变成测试。

### 5.1 初始猜测必须在 simulator domain 内

测试思想：

```text
用真实 GeodeticSimulator + 前端默认 RW05L payload 构造 x0；
调用 defect constraint；
断言 defect 不是全 1e9。
```

它保护的是：

- control guess 不能再回到全零 alpha；
- 初始点不能直接落入 invalid residual 平面；
- 第一次 Jacobian 至少有真实动力学信息。

### 5.2 terminal constraint 不要重复约束 mass

测试思想：

```text
final_state_constraint 的 shape 应该是 state_dim - 1；
也就是 6 个终端状态约束，而不是 7 个。
```

它保护的是：

- `m` 不会重新被加入 terminal equality；
- SLSQP 约束 Jacobian 不会因为这个结构性冗余再少秩。

---

## 6. 一套可复用的排查流程

遇到优化器报错时，可以照这个顺序走：

### Step 1：复现请求

把前端请求 payload 固定下来，用 Python 直接调用 backend。

成功标准：

```text
不用前端，也能稳定复现同一个错误。
```

### Step 2：分类错误

| 错误来源 | 典型文本 | 下一步 |
|---|---|---|
| ODE 积分器 | `Required step size...` | 查状态奇异点、控制初值、边界 |
| SLSQP 子问题 | `Singular matrix C...` | 查约束 Jacobian rank |
| 约束不可达 | `Positive directional derivative...` | 查目标是否可达、约束尺度 |
| 迭代超限 | `Iteration limit reached` | 查缩放、初值、性能和收敛阈值 |

### Step 3：检查初始点

对 `x0` 问四个问题：

1. 每个状态是否物理上可模拟？
2. 每个控制是否在合理范围？
3. `defect_constraints(x0)` 是否有限？
4. 残差是不是常量惩罚值？

### Step 4：检查约束 Jacobian

如果 SLSQP 报奇异矩阵，直接算 rank：

```text
rank == number_of_constraints  -> 约束结构大概率没退化，继续查尺度/可达性
rank < number_of_constraints   -> 约束有冗余或某些约束在当前点没有导数
```

### Step 5：回到物理方程解释

不要停在“rank 少 1”这个数学结论。继续问：

```text
哪个状态量没有自由度？
哪个约束是别的约束推出的？
哪个分母可能接近 0？
哪个控制初值让 simulator 进入不可行域？
```

这一步能把修复从“调参数”变成“改建模”。

### Step 6：写回归测试

每个根因至少一个测试：

| 根因 | 测试 |
|---|---|
| 初值落到 invalid residual | 默认 payload 的 defect 不全是 `1e9` |
| terminal mass 冗余 | final constraint shape 是 `state_dim - 1` |
| `V=0/m=0` 奇异 | endpoint validation 拒绝不可模拟状态 |
| simulator failure 冒泡 | defect 把 simulator failure 转成 infeasible residual |

---

## 7. 这次修复后的验证命令

相关 Python 回归：

```bash
conda run -n aviation pytest \
  /Users/liudongxu/Desktop/studys/thesis/4dTrajectory/optimization/tests \
  /Users/liudongxu/Desktop/studys/thesis/aerodynamic_model/tests \
  /Users/liudongxu/Desktop/studys/thesis/aeroviz_backend/tests \
  -q
```

真实 payload 验证：

```bash
/Users/liudongxu/opt/miniconda3/envs/aviation/bin/python3.13 - <<'PY'
from aeroviz_backend.optimization_backend import OptimizationBackend

payload = {
    "initialState": {
        "lon": -78.7873,
        "lat": 35.878659,
        "altM": 1000,
        "speedMps": 120,
        "headingDeg": 0,
        "flightPathDeg": 0,
        "massKg": 78000,
        "aircraftType": "A320",
    },
    "targetState": {
        "lon": -78.802,
        "lat": 35.874,
        "altM": 111.86,
        "speedMps": 70,
        "headingDeg": 45,
        "flightPathDeg": -4,
        "massKg": 78000,
        "aircraftType": "A320",
    },
    "targetControl": {"attackDeg": 4},
    "nSegments": 10,
}

result = OptimizationBackend().optimize(payload)
print(result["ok"], result["finalTimeS"], len(result["controls"]), len(result["states"]))
print(result["states"][-1])
PY
```

这次真实请求成功后，末端状态命中 runway target，说明：

- 积分器不再被初始无效状态击穿；
- SLSQP 约束 Jacobian 不再因为 terminal mass 冗余而退化；
- backend 的 `/optimization/run` 可以返回有效轨迹。

---

## 8. 排查心法

最重要的经验不是某个 SciPy 参数，而是这几句话：

1. **先复现，再解释。** 没有稳定复现，就不要急着改参数。
2. **把优化器拆开看。** `x0`、bounds、constraints、Jacobian 比错误字符串更诚实。
3. **数学症状要回到物理模型。** rank 退化、常量残差、ODE 步长失败，背后通常对应冗余约束、不可模拟状态或动力学奇异点。
4. **惩罚残差要谨慎。** 它能避免异常冒泡，但如果初始点就在惩罚平面上，优化器会失去导数信息。
5. **修复必须变成测试。** 调试脚本负责发现问题，单元测试负责防止问题回来。

