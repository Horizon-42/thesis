# 飞机相关优化参数开发说明

## 状态

日期：2026-06-14

本文档记录 AeroViz-4D 中哪些轨迹优化参数必须按飞机类型设计。它面向后续开发，不是飞行手册，也不代表真实机型认证数据。这里的数值优先服务于当前简化点质量动力学模型和 SLSQP 优化器的数值稳定性。

## 背景

当前系统已经有飞机预设：

- `A320`: narrow body, `mass_kg=78000`, `wing_area_m2=122.6`
- `B77W`: wide body, `mass_kg=351530`, `wing_area_m2=436.8`
- `C172`: general aviation, `mass_kg=1157`, `wing_area_m2=16.2`

但是优化器和前端仍有一些固定值是按喷气客机直觉写的，例如：

- single shooting 初始推力猜测固定为 `12000 N`
- optimizer 推力上界固定到 `1000000 N`
- 前端默认初始速度固定为 `120 m/s`
- 前端 runway threshold target speed 固定为 `70 m/s`
- 切换 aircraft 时只更新 `aircraftType` 和 `massKg`，不更新速度、推力、目标速度

这会导致两个典型问题：

- 小飞机如 `C172` 被套用喷气客机级别的速度/推力，优化初始点就可能飞出合理包线。
- 大飞机虽然更不容易直接数值爆炸，但如果终端状态只是 soft objective 而不是 hard constraint，结果可能明显偏离 target state。

## 原则

### 1. 真实规格和仿真调参分开

真实规格描述飞机本体：

- `mass_kg`
- `wing_area_m2`
- 未来可能加入 `max_thrust_n`
- 未来可能加入更真实的气动系数

仿真调参描述“这个机型在当前简化模型中怎么初始化和限制优化器”：

- 默认初始速度
- 默认跑道入口速度
- 默认推力猜测
- 推力上下界
- 速度上下界
- 合理迎角范围
- 合理坡度角范围
- 用于 final time guess 的巡航/进近速度

不要把仿真调参写死在 optimizer 或 UI 常量里。它们应该来自 aircraft config，或者至少由 aircraft category 派生。

### 2. 前端默认值必须跟飞机联动

用户从 `A320` 切到 `C172` 时，不应该继续保留客机默认速度和推力，除非用户已经手动改过并明确想保留。

最低要求：

- 初始速度随机型更新。
- target speed 随机型更新。
- manual control 的 thrust 输入范围随机型更新。
- 默认 control thrust 随机型更新。

### 3. 优化器初始点必须可模拟

SLSQP 会通过有限差分反复探测目标函数。如果初始点或初始点附近大量候选都会导致：

- altitude below 0
- speed 非物理
- gamma 接近 `+/-90 deg`
- ISA density 计算溢出或非有限值

优化器就会拿不到有意义的梯度，表现为失败、迭代耗尽，或者返回明显不贴 target 的局部结果。

因此每个机型至少需要一组“温和、可模拟”的默认控制猜测。

## 建议的数据模型

建议扩展 `aerodynamic_model/aircraft_sets.py` 中的 `AircraftSpec`，不要把这些值散落在 frontend/backend/optimizer。

```python
@dataclass(frozen=True, slots=True)
class AircraftSpec:
    code: str
    name: str
    category: str
    wing_area_m2: float
    mass_kg: float

    # Simulation/optimization tuning, not certified aircraft data.
    default_initial_speed_mps: float
    default_target_speed_mps: float
    default_thrust_guess_n: float
    max_thrust_n: float
    min_speed_mps: float
    max_speed_mps: float
    max_bank_deg: float
    min_attack_deg: float
    max_attack_deg: float
```

如果暂时不想改 dataclass，也可以先建立一个独立的 `AIRCRAFT_OPTIMIZATION_PRESETS` 字典。但长期看，把它和 `AircraftSpec` 放在一起更不容易漏传。

## 每个参数为什么需要按飞机设计

| 参数 | 用在哪里 | 为什么不能固定 |
|---|---|---|
| `default_initial_speed_mps` | 前端 initial aircraft 默认值 | `120 m/s` 对 A320 合理，对 C172 过快 |
| `default_target_speed_mps` | runway threshold target 默认值 | A320/B77W threshold 速度通常约 `65-85 m/s`，C172 通常约 `30-40 m/s` |
| `default_thrust_guess_n` | single/multiple shooting 初始控制猜测 | `12000 N` 对 C172 会把飞机推到极端爬升，对 B77W 又偏小 |
| `max_thrust_n` | optimizer bounds 和 UI input max | `1000000 N` 对 C172 完全不现实，会让 SLSQP 探测大量坏点 |
| `min_speed_mps` | state validation / optimizer bounds | 不同机型 stall margin 不同；统一 `1 m/s` 只是在防除零，不是飞行包线 |
| `max_speed_mps` | UI 和 optimizer bounds | 防止小飞机被优化到客机速度，或大飞机被优化到荒谬高速 |
| `max_bank_deg` | manual control / optimizer bounds | 轨迹优化通常不需要 `90 deg` bank；大角度会造成数值不稳定 |
| `min_attack_deg`, `max_attack_deg` | control bounds / trim guess | 当前模型线性 `CL = CL0 + CL_alpha * alpha`，超出范围没有 stall 行为，必须限制 |
| `nominal_approach_gamma_deg` | target 默认值 | 通常 `-3 deg` 可做通用默认，但将来可按机型或程序微调 |
| `final_time_speed_mps` | final time guess | 初始时间猜测应使用该机型合理速度，否则过短/过长都会影响收敛 |

## 初始建议值

下面是“让当前简化模型更容易跑起来”的初始建议，不是 POH/FCOM/AFM 数值。

| 机型 | initial speed | target speed | thrust guess | max thrust | speed bounds | bank bound | alpha bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C172` | `55-65 m/s` | `32-40 m/s` | `500-1500 N` | `4000 N` | `25-80 m/s` | `30 deg` | `-6..14 deg` |
| `A320` | `110-130 m/s` | `70-80 m/s` | `40000-80000 N` | `250000 N` | `55-180 m/s` | `30 deg` | `-6..18 deg` |
| `B77W` | `120-145 m/s` | `75-85 m/s` | `90000-180000 N` | `1000000 N` | `60-210 m/s` | `30 deg` | `-6..18 deg` |

说明：

- `C172` 的螺旋桨模型在真实世界不是“恒定推力 N”这么简单。当前 simulator 只有 `thrust`，所以这里的数值是简化模型里的等效推力。
- `B77W max thrust` 可以接近当前 `1000000 N`，但这个值不应被 C172/A320 共用。
- `bank bound` 对 trajectory optimization 建议先用 `30 deg`，manual pilot UI 可以允许更大，但 optimizer 不需要默认探索极端滚转。

## 代码落点

### Aircraft preset

主位置：

- `aerodynamic_model/aircraft_sets.py`

这里应该成为飞机相关参数的 single source of truth。

### Backend catalog

主位置：

- `aeroviz_backend/simulation_backend.py`

`aircraft_catalog()` 需要把前端会用到的 aircraft tuning 字段返回出去，例如：

- `defaultInitialSpeedMps`
- `defaultTargetSpeedMps`
- `defaultThrustN`
- `maxThrustN`
- `minSpeedMps`
- `maxSpeedMps`
- `maxBankDeg`
- `minAttackDeg`
- `maxAttackDeg`

### Frontend state defaults

主位置：

- `aeroviz-4d/src/components/PilotPanel.tsx`
- `aeroviz-4d/src/pilot/pilotClient.ts`

需要改的行为：

- `makeDefaultInitialState()` 使用 aircraft config 的默认速度。
- `makeDefaultTrajectoryTarget()` 使用 aircraft config 的默认 target speed。
- `updateAircraftType()` 切换飞机时同步更新速度、target speed、control thrust 和 input bounds。
- manual control 的 thrust max 不再固定 `60000`。

### Optimizers

主位置：

- `4dTrajectory/optimization/single_shooting_optimizor.py`
- `4dTrajectory/optimization/transcription_optimizor.py`

需要改的行为：

- `_DEFAULT_THRUST_GUESS_N` 改为从 `self.sim.simulator.aircraft` 读取。
- thrust bounds 从 aircraft `max_thrust_n` 读取。
- attack bounds 和 bank bounds 从 aircraft preset 读取，或者至少按 category 读取。
- final time guess 使用 aircraft 的合理速度，而不是固定 `50 m/s` 或固定 `100 s`。
- multiple shooting 的 trim attack guess 也应使用 aircraft-specific thrust guess。

## Single shooting 和 multiple shooting 的差异

### Single shooting

Single shooting 只优化：

- final time
- 每个 segment 的 control

中间 state 全靠 simulator 从 initial state 一路积分出来。因此它对初始控制猜测和控制边界非常敏感。

如果 C172 被给了客机级别推力，第一轮模拟就可能把飞机推到极端姿态；此时 SLSQP 的有限差分会在坏区域里打转。

### Multiple shooting

Multiple shooting 同时优化：

- final time
- 每个 segment 的 control
- 每个 segment 末端的 state

它有 defect constraints 来约束“模拟出来的下一状态”和“优化变量里的下一状态”一致，所以更容易表达终端状态约束。但它也需要合理的 state bounds 和 control bounds，否则一样会不可行。

## 终端状态：soft objective 还是 hard constraint

如果目标是“尽量接近 target”，可以只用 objective：

```python
norm(endpoint_error_vector(final_state, target_state))
```

如果目标是“必须到达 target”，必须加 equality constraint：

```python
endpoint_error_vector(final_state, target_state) == 0
```

注意：hard constraint 只有在 target state 物理可行时才应该启用。比如要求 C172 以客机速度、客机下降剖面、客机推力边界到达跑道入口，就是一个配置错误，不是 optimizer 应该硬解的问题。

开发建议：

- UI 上明确 optimizer 的语义：soft target 或 hard terminal constraint。
- backend error message 区分“数值失败”和“目标可能不在当前飞机包线内”。
- target state 默认值随 aircraft 改，避免用户一切换飞机就生成不可行问题。

## 开发检查清单

新增或修改飞机类型时，至少检查：

- `aircraft_catalog()` 返回的字段是否完整。
- 前端 aircraft dropdown 切换后，initial speed 是否变化。
- target speed 是否变化。
- manual thrust input max 是否变化。
- single shooting 初始控制 replay 是否不会立刻 altitude below 0 或 gamma 接近 `+/-90 deg`。
- optimizer thrust bounds 是否使用 aircraft max thrust。
- A320/B77W/C172 各跑一次最小优化用例，失败时错误信息是否能说明是 infeasible 还是 iteration limit。

一个很有用的快速 sanity check：

1. 用默认 initial state。
2. 用默认 control guess。
3. 不优化，只 simulator replay 20 秒。
4. 检查 altitude、speed、gamma 是否还在合理范围。

如果这一步都不合理，优化器大概率不会稳定。

## 当前已知技术债

- `targetControl.attackDeg` 已经由前端发送，但 backend optimizer 当前没有真正使用它。
- single shooting 和 multiple shooting 的 control bounds 仍有固定值。
- 前端默认速度和推力仍有固定值。
- state scaling / defect scaling 还不一致，SLSQP 会受到单位尺度影响。
- simulator 的 ISA density 在极端高度下可能产生非有限值；这通常是 optimizer 探测到坏候选点后的症状，根因仍应优先从 aircraft bounds 和初始猜测处理。

