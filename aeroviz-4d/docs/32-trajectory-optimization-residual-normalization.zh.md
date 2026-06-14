# 轨迹优化中的 Residual 归一化与 State 归一化

## 状态

日期：2026-06-14

本文档说明 multiple shooting 优化里 residual normalization 的目的、当前实现方式，以及后续如果要做 state / decision variable normalization 应该怎么设计。

这里讨论的是数值优化工程问题，不是修改飞机动力学模型。Simulator 仍然使用真实物理单位。

## 一句话结论

当前阶段推荐做的是：

- 保持 optimizer variables 使用真实单位。
- 保持 simulator 输入输出使用真实单位。
- 只在 equality constraint 返回 residual 前做归一化。

也就是说：

```text
真实 state/control -> simulator -> 真实 predicted state
真实 predicted state - 真实 node state -> scaled residual -> SLSQP
```

这一步可以改善 SLSQP 的约束雅可比数值条件，同时改动范围很小。

## 为什么 residual 需要归一化

multiple shooting 的 state 是：

```text
[lat, lon, altitude, V, psi, gamma]
```

它们的单位和数量级完全不同：

- `lat/lon`: degree
- `altitude`: metre
- `V`: m/s
- `psi/gamma`: radian

如果直接返回：

```python
predicted_state - node_state
```

SLSQP 会同时看到：

- `0.001 deg` 的经纬度误差
- `100 m` 的高度误差
- `10 m/s` 的速度误差
- `0.01 rad` 的角度误差

这些数字本身不能直接比较大小。优化器不会自动知道 `0.001 deg` 其实约等于几十到一百米，也不会知道 `0.05 rad` 的航迹角误差已经很大。

结果是 constraint Jacobian 的行尺度很差，可能造成：

- 收敛慢
- search direction 不稳定
- `Singular matrix C in LSQ subproblem`
- 某些 state 误差被数值尺度掩盖

## 当前 residual scaling 的思路

当前实现只改变 constraint 返回值，不改变优化变量本身。

state residual 被转成下面的尺度：

| State | 原始单位 | Residual 单位 | 直觉 |
|---|---:|---:|---|
| latitude | degree | km | 1 km lateral miss -> 约 1 |
| longitude | degree | km | 按平均纬度换算成 km |
| altitude | metre | 100 m | 100 m vertical miss -> 1 |
| speed | m/s | 10 m/s | 10 m/s speed miss -> 1 |
| heading `psi` | rad | 10 deg | 10 deg heading miss -> 1 |
| flight path `gamma` | rad | 2 deg | 2 deg gamma miss -> 1 |

这样 SLSQP 看到的 residual 大致都在相近数量级。

## 为什么 heading 要特殊处理

Heading 是周期角。直接相减会出错：

```text
179 deg - (-179 deg) = 358 deg
```

但物理上这两个 heading 只差 `2 deg`。

所以 heading residual 必须 wrap 到 `[-pi, pi)`：

```python
(left - right + pi) % (2*pi) - pi
```

这一步只适用于角度残差，不适用于 lat/lon。

## 当前代码落点

当前 multiple shooting 的 residual scaling 位于：

- `4dTrajectory/optimization/transcription_optimizor.py`

核心函数：

```python
state_constraint_error(left_state, right_state)
```

它被两个 constraint 使用：

```python
defect_constraints()
final_state_constraint()
```

其中：

- `defect_constraints()` 比较 simulator 积分结果和当前 node state。
- `final_state_constraint()` 比较最后一个 node state 和 target state。

两者都返回 scaled residual。

## 这不是完整的 state normalization

Residual normalization 只是在返回 constraint residual 的最后一步做缩放。

它没有改变：

- `z` 中 state 的单位
- `node_state` 的单位
- `bounds` 的单位
- `build_state_guess()` 的单位
- simulator 的输入输出单位

所以这是一个低风险修正。

## 后续如果要做 state normalization

完整的 state / decision variable normalization 是另一件事。

它的目标是让 optimizer 的 decision variables 本身也处在相近数量级，而不是只缩放 constraint residual。

当前 `z` 大致是：

```text
[final_time, controls..., states...]
```

其中 states 使用真实单位：

```text
lat degree, lon degree, altitude m, speed m/s, psi rad, gamma rad
```

如果做 state normalization，`z` 里的 state 应该变成 normalized state，例如：

```text
lat_km, lon_km, altitude_100m, speed_10mps, heading_10deg, gamma_2deg
```

然后在进入 simulator 前还原：

```text
normalized z -> denormalized GeodeticState -> simulator
```

## 推荐的未来结构

未来可以加两个明确函数：

```python
normalize_state(state_array, reference_state)
denormalize_state(normalized_state, reference_state)
```

以及对应的 batch helper：

```python
normalize_node_states(node_state, reference_state)
denormalize_node_states(normalized_node_state, reference_state)
```

设计原则：

- optimizer 内部的 `z` 使用 normalized state。
- simulator 边界处统一 denormalize。
- 返回给 frontend/backend 的状态仍然是真实单位。
- bounds 也要同步换成 normalized bounds。

## State normalization 会牵涉哪些代码

如果未来要完整实现，需要同时改：

- `unpack_z()`
- `build_state_guess()`
- `build_state_bounds()`
- `defect_constraints()`
- `final_state_constraint()`
- `array_to_geodetic_state()`
- 单元测试里的 expected `x0` 和 bounds

因此不要把完整 state normalization 当成小修小补。它是一次 optimizer variable encoding 的改动。

## 不建议做的事情

不要在 simulator 里做 normalization。

原因：

- simulator 是物理模型，应该只接受真实单位。
- normalization 是 optimizer 数值工程问题，不应该污染动力学接口。
- 如果 simulator 接受 normalized state，后续调试会很难判断某个值到底是真实物理量还是优化变量。

也不要在多个地方重复写缩放公式。应该集中在 optimizer 的 helper 函数里。

## 当前检查清单

每次修改 residual scaling 后，至少检查：

- `defect_constraints()` 返回 shape 仍是 `n_segments * state_dim`。
- `final_state_constraint()` 返回 shape 仍是 `state_dim`。
- heading residual 使用 wrapped angle difference。
- simulator 调用仍然使用真实单位。
- backend 返回给前端的 state 仍然是真实单位。

当前对应测试：

- `4dTrajectory/optimization/tests/test_transcription_optimizor_import.py`

