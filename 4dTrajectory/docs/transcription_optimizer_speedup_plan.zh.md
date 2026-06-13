# Transcription Optimizer 加速方案

这份文档记录 `/optimization/run` 当前变慢的原因，以及后续可以按阶段实施的加速方案。目标不是一次性把优化器重写，而是在不破坏现有物理模型和前端工作流的前提下，逐步降低等待时间。

当前真实 RW05L 默认请求已经可以成功求解，但一次优化大约需要 1 到 2 分钟。这个速度对验证算法可以接受，对前端交互偏慢。

---

## 1. 当前瓶颈在哪里

`TranscriptionOptimizor` 使用 SciPy `minimize(method="SLSQP")` 做 multiple shooting。以默认 `nSegments = 10` 为例：

```text
变量数量约为：
1 个 final_time
+ 10 段 * 3 个 control
+ 10 个节点 * 7 个 state
= 101 个变量

等式约束数量约为：
10 段 * 7 个 defect constraint
+ 6 个 terminal state constraint
= 76 个约束
```

SLSQP 默认用数值差分估计梯度/Jacobian。每次估计 Jacobian 时，它会反复扰动优化变量并调用约束函数。

而每次调用 `defect_constraints()` 又会：

```text
for each segment:
    GeodeticSimulator.step()
        -> Simulator.simulate()
            -> scipy.integrate.solve_ivp()
```

也就是说，一次 SLSQP 迭代可能触发大量 ODE 积分。优化慢的核心不是 Python 函数调用本身，而是：

```text
数值差分次数 * shooting segments * solve_ivp 积分成本
```

---

## 2. 加速原则

加速时要保持三个原则：

1. **先减少问题规模，再考虑换算法。** 变量和约束越少，数值 Jacobian 越便宜。
2. **先优化交互体验，再追求高精度。** 前端可以先给用户快速预览，必要时再提高 segments。
3. **不要让加速破坏物理一致性。** 如果使用近似传播器，要明确它只是优化阶段近似，最终验证仍需用真实 simulator。

---

## 3. 优先级最高的改动

### 3.1 降低前端默认 `nSegments`

当前默认值是 `10`。可以考虑把默认值改成 `4` 或 `6`。

收益：

- 改动最小；
- 变量和约束立刻减少；
- `solve_ivp` 调用次数成比例下降；
- 前端交互会明显变快。

代价：

- 轨迹分辨率降低；
- 控制序列更粗；
- 对复杂转弯或大高度变化的拟合能力下降。

建议：

```text
默认预览：nSegments = 4 或 6
精细求解：用户手动调到 10 或更高
```

可以在 UI 上把它理解成：

```text
Fast preview  -> lower segments
Detailed path -> higher segments
```

---

### 3.2 从优化变量中彻底移除 mass

当前模型里：

```text
dmdt = 0
```

质量不会随时间变化。上一轮已经移除了 terminal mass equality，但每个 node state 里仍然保留 `m` 作为优化变量。

更进一步，可以把 optimizer 内部 state 从：

```text
[lat, lon, alt, V, psi, gamma, m]
```

改为：

```text
[lat, lon, alt, V, psi, gamma]
```

返回结果时再把固定 `initial_state.m` 补回。

收益：

- 每个 node 少 1 个变量；
- 每个 defect 少 1 个约束；
- 对 `nSegments = 10`，变量减少 10 个，约束减少 10 个；
- Jacobian 更小；
- 避免将来 mass 再次引入冗余约束。

代价：

- 需要小范围改 `unpack_z()`、state guess、state bounds、defect constraint 和返回值；
- 测试要覆盖返回 state 仍然包含 `massKg`。

适合现在做，因为当前动力学确实没有燃油消耗。

---

### 3.3 做变量尺度归一化

当前优化变量的数量级差异很大：

| 变量 | 典型数量级 |
|---|---:|
| lat/lon | 10 到 100 |
| altitude | 100 到 1000 |
| speed | 70 到 150 |
| heading/gamma | 0 到 1 radians |
| mass | 78000 |
| thrust | 10000 到 80000 |

SLSQP 对变量尺度比较敏感。尺度差异过大时，数值差分和线性子问题都会更病态。

优化器内部可以使用 normalized variables，例如：

```text
z_scaled = z_physical / scale
```

示例 scale：

| 变量 | scale |
|---|---:|
| final_time | 100 |
| thrust | 10000 |
| bank | 1 |
| alpha | 0.1 |
| lat/lon delta | 0.01 |
| altitude | 1000 |
| speed | 100 |
| psi | 1 |
| gamma | 0.1 |

收益：

- 可能减少 SLSQP 迭代次数；
- 数值 Jacobian 更稳定；
- 有助于降低 `Singular matrix`、`Positive directional derivative` 这类数值问题概率。

代价：

- 需要写清楚 physical space 和 scaled space 的转换；
- bounds 和 initial guess 都要同步缩放；
- defect/final constraints 最好也按状态尺度做归一化，否则约束残差仍然尺度不均。

建议作为第二阶段做。它比降低 segments 和移除 mass 更容易引入实现错误。

---

## 4. 中期加速方案

### 4.1 优化阶段使用 fixed-step propagator

现在每个 segment 都调用 `solve_ivp`。`solve_ivp` 是通用自适应积分器，稳定但比较重。

优化阶段可以考虑使用固定步长 RK4：

```text
segment propagation:
    split dt into k substeps
    run RK4 on local point-mass dynamics
```

最终播放和验证仍然使用 `GeodeticSimulator.step()`。

收益：

- 每段传播成本更可控；
- 没有自适应步长开销；
- 对 SLSQP 的大量重复调用更友好。

风险：

- RK4 近似结果可能和 `solve_ivp` 不完全一致；
- 如果 substeps 太少，优化出来的轨迹在真实 simulator 里可能偏离；
- 需要单独测试 RK4 propagator 和 `solve_ivp` 的误差。

建议：

```text
先保留 solve_ivp 作为默认；
新增 optimizer-only fast propagator；
通过参数或策略切换。
```

---

### 4.2 Warm start 上一次优化结果

前端交互中，用户经常只是小幅改变：

- target speed；
- target gamma；
- target runway；
- segments；
- initial state。

如果每次都从插值初值重新开始，SLSQP 会重复做很多工作。

可以把上一次结果作为下一次 initial guess：

```text
previous controls/states/final_time
    -> resample to current nSegments
    -> use as next x0
```

收益：

- 对连续调参非常有效；
- 用户体验接近“即时更新”；
- 不改变动力学模型。

风险：

- 如果目标变化很大，旧解可能反而误导优化器；
- 需要 fallback：warm start 失败时回到 cold start。

建议：

```text
前端或 backend session 层保存 last successful optimization。
当 payload 变化小的时候使用 warm start。
```

---

### 4.3 分两阶段求解

可以先用粗 segments 求一个初解，再升到细 segments：

```text
Stage 1: nSegments = 4
Stage 2: resample Stage 1 result to nSegments = 10
Stage 3: refine
```

收益：

- 初始解更接近可行轨迹；
- 细网格优化更容易收敛；
- 可以先把 Stage 1 结果返回给前端作为预览。

风险：

- 实现比单次求解复杂；
- 前端要区分 preview result 和 refined result；
- 如果 Stage 1 太粗，可能给 Stage 2 一个错误拓扑的路径。

---

## 5. 不建议优先做的方向

### 5.1 盲目提高 SLSQP 容差

比如把 `ftol` 调大，或者降低 `maxiter`，确实可能更快返回，但可能只是更早失败或返回质量较差的轨迹。

可以作为最后的 UI 超时策略，不应作为核心加速方案。

### 5.2 一开始就换成复杂最优控制库

例如直接迁移到 CasADi / IPOPT / direct collocation 框架，长期可能更强，但短期成本高：

- 需要重写模型接口；
- 需要显式 Jacobian 或自动微分；
- 需要重新设计测试和部署。

当前项目更适合先把现有 transcription optimizer 做小步加速。

### 5.3 把物理模型简化到失真

如果为了速度完全绕开升力、阻力、转弯动力学，优化速度会很快，但结果不能代表当前 simulator。这样会伤害 thesis visualization 的可信度。

---

## 6. 推荐实施路线

### Phase 1：低风险快速收益

目标：把前端等待从分钟级降到更可接受的范围。

任务：

1. 将前端默认 `nSegments` 从 `10` 降到 `4` 或 `6`。
2. optimizer 内部移除 mass 变量。
3. 保持返回 API 不变，仍然返回 `massKg`。
4. 增加性能基准脚本，记录默认 RW05L payload 的耗时。

验证：

```text
Python tests pass
默认 payload 成功
返回 states/controls 数量正确
terminal target 仍然命中
耗时有明确下降
```

---

### Phase 2：数值稳定和迭代减少

目标：减少 SLSQP 迭代次数，提高不同 target 的稳定性。

任务：

1. 引入 scaled optimization variables。
2. 对 constraints residual 做尺度归一化。
3. 记录每次优化的迭代次数、函数调用次数、约束残差。

验证：

```text
默认 payload 成功
多个 runway/target speed payload 成功
迭代次数下降或稳定
不再出现明显尺度导致的 SLSQP error
```

---

### Phase 3：交互式体验优化

目标：用户小幅修改参数时不需要完整冷启动。

任务：

1. 支持 warm start。
2. 支持粗到细两阶段优化。
3. 前端显示 preview/refined 状态。
4. 失败时 fallback 到 cold start。

验证：

```text
连续调整 target speed/gamma 时，第二次以后明显更快
warm start 失败不影响 cold start
前端不会显示 stale trajectory
```

---

### Phase 4：可选 fast propagator

目标：减少每次 defect constraint 的传播成本。

任务：

1. 新增 optimizer-only RK4 propagator。
2. 写对比测试：RK4 vs solve_ivp 在典型 segment 上误差可接受。
3. 增加策略开关：accurate / fast。

验证：

```text
fast mode 明显更快
最终轨迹用 solve_ivp replay 后仍然接近 target
误差在 thesis visualization 可接受范围内
```

---

## 7. 性能基准应该怎么写

不要只凭感觉说“快了”。建议加一个小的 benchmark 脚本或测试辅助函数，固定 payload，记录：

| 指标 | 含义 |
|---|---|
| wall time | 用户实际等待时间 |
| `nSegments` | 问题规模 |
| iterations | SLSQP 迭代次数 |
| function evaluations | 目标函数调用次数 |
| constraint evaluations | 约束调用次数 |
| final residual norm | 最终约束误差 |
| success/message | 优化状态 |

最小版本可以先打印：

```text
segments=10
success=True
final_time=181.55
wall_time_s=...
states=10
controls=10
```

后续再把 SciPy result 里的 `nit`、`nfev` 等字段暴露到 debug 输出。

---

## 8. 建议的第一批代码改动

如果马上开始实现，我建议第一批只做：

1. 前端默认 segments 改成 `6`；
2. optimizer 内部 state 维度从 7 改成 6，mass 固定；
3. 增加一个本地 benchmark 脚本或测试辅助；
4. 保持 HTTP response schema 不变。

这批改动的优点是：

- 不改变 aerodynamic model；
- 不引入新依赖；
- 不需要换优化器；
- 性能收益应该能直接观察到；
- 风险可通过现有 optimizer/backend tests 覆盖。

---

## 9. 一句话总结

当前优化慢的主要原因是：

```text
SLSQP 数值差分 Jacobian
  * 多 shooting segment
  * 每个约束评估都要 solve_ivp
```

最稳妥的加速路线是：

```text
先减少 segments 和变量维度
再做尺度归一化
然后加 warm start
最后才考虑 fast propagator 或更换优化框架
```

