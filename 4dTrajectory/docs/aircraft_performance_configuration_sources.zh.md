# 机型性能配置依据：推力、末端速度与初始放置区域

本文档用于指导 `4dTrajectory` 中 `AircraftSpec` 和前端初始化设置的实现。目标不是做真实飞行签派或飞控计算，而是给轨迹优化器一个物理量级正确、来源可追溯、初学者能理解的机型配置。

## 1. 为什么不能继续用一套默认值

现在 least-squares 里出现 `alpha` 打满、`thrust` 基本不变、轨迹像直线的问题，核心原因不只在优化算法，也在输入问题本身：

- A320、B777-300ER、C172 的推力能力差异巨大，不适合共用一个全局 thrust 上限或默认值。
- 62 kt 作为 A320 的末端速度过低。按当前简化空气动力模型，即使 `alpha` 到 18 deg，也难以产生足够升力，所以优化器会把迎角顶到上限。
- 初始飞机如果离跑道太近、但高度很高，会强迫优化器走出极陡下滑路径。之前默认位置到 KRDU RW05L 约 1.4 km，但高度差约 886 m，需要约 -32 deg 下滑角；正常最终进近通常按约 3 deg 下滑角理解，这种初始条件本身就不可行。

所以后续实现应该把“机型事实配置”和“优化初值/前端放置引导”分开处理。

## 2. 可追溯来源

| 配置问题 | 来源 | 可追溯内容 | 对实现的影响 |
| --- | --- | --- | --- |
| 进近速度分类 | [14 CFR § 97.3, FAA/eCFR via Cornell LII](https://www.law.cornell.edu/cfr/text/14/97.3) | Aircraft approach category is based on VREF, or 1.3 Vs0 if VREF is unavailable. Category C is 121 kt to less than 141 kt; Category D is 141 kt to less than 166 kt. | 不能把大型喷气机末端速度设成 62 kt。A320 应在 Cat C 附近建模，B777-300ER 应在 Cat D 附近建模。 |
| A320 最终进近速度概念 | [Gulf Air A320 accident report PDF, SKYbrary bookshelf](https://skybrary.aero/bookshelf/books/1020.pdf) | 报告引用 A320 FCOM 逻辑：速度低于 VAPP - 5 kt 或高于 target + 10 kt 时需要喊话；并说明 VAPP = VLS + 1/3 headwind component。报告还描述 A320 在约 5 NM final approach fix 附近建立最终进近构型。 | A320 的终端速度应按 VAPP/VREF 思路设置，不应使用 62 kt。前端初始放置区域可以从约 5 NM 以外的 final approach 区域开始。 |
| A320ceo 发动机推力量级 | [CFM International CFM56 official page](https://www.cfmaeroengines.com/cfm56) | CFM56 family thrust range is 18,500 to 32,000 lb, and powers A320ceo / B737NG families. | A320ceo 的总最大推力量级应是两台发动机合计约 240 kN 到 285 kN，而不是和 C172 或 B777 共用一个随意值。 |
| B777-300ER 发动机推力量级 | [GE Aerospace GE90 official page](https://www.geaerospace.com/commercial/aircraft-engines/ge90) | GE90-115B powers Boeing 777-300ER and has sea-level max power about 115,300 lbf. | B777-300ER 两台发动机总最大推力量级约 1,026 kN。若后端仍把 thrust clamp 到 80 kN，会严重低估 B777。 |
| C172 速度与动力量级 | [Textron Cessna Skyhawk official page](https://cessna.txtav.com/en/piston/cessna-skyhawk) | Skyhawk max cruise 124 ktas, stall speed 48 KCAS, engine 180 hp. | C172 的末端速度可从 1.3 * stall speed 得到约 62 kt；动力需要用功率换算为简化模型的等效 thrust，不能和喷气机同一套推力定义直接比较。 |

## 3. 推荐写入 AircraftSpec 的字段

后续建议把这些字段放进 `AircraftSpec`，由后端和前端共同读取：

```text
max_thrust_n
approach_thrust_guess_n
terminal_speed_kt
terminal_speed_min_kt
terminal_speed_max_kt
final_approach_min_nm
final_approach_max_nm
final_approach_lateral_half_width_nm
final_approach_glide_angle_deg
threshold_crossing_height_m
```

字段含义：

- `max_thrust_n`：优化控制量的上界，必须随机型变化。
- `approach_thrust_guess_n`：优化初始猜测使用的推力，不代表真实油门设定，只是让求解器从合理数量级开始。
- `terminal_speed_kt`：前端默认末端速度，也是后端目标状态的默认值。
- `terminal_speed_min_kt` / `terminal_speed_max_kt`：前端滑块或输入框范围，防止用户无意间给 A320 设置 C172 速度。
- `final_approach_*`：前端在地图上高亮“建议初始放置区域”时使用。
- `threshold_crossing_height_m`：跑道入口上方参考高度，建议先用 15 m 作为简化默认值。

前端初始飞机速度不单独增加字段，先用 `terminal_speed_kt + 25 kt` 作为 final approach 外侧的初始猜测。这只是初始化启发式，用来避免 A320 默认 233 kt 进入最终进近；真正的末端速度仍由 `terminal_speed_kt` 约束。

## 4. 推荐初始配置值

这些值是“优化模型默认值”，不是飞行手册数据。实现时可以先作为 aircraft catalog 的默认配置，后续再根据模型标定调整。

| 机型 | `max_thrust_n` | `approach_thrust_guess_n` | `terminal_speed_kt` | 合理末端速度范围 | 建议初始放置区域 |
| --- | ---: | ---: | ---: | ---: | --- |
| A320 | 240000 | 40000 | 145 | 135-155 kt | 跑道入口外 5-10 NM，横向半宽 0.8 NM，3 deg 下滑 |
| B77W | 1026000 | 140000 | 155 | 145-165 kt | 跑道入口外 6-12 NM，横向半宽 1.0 NM，3 deg 下滑 |
| C172 | 3200 | 800 | 65 | 60-75 kt | 跑道入口外 2-5 NM，横向半宽 0.5 NM，3 deg 下滑 |

说明：

- A320 的 `max_thrust_n` 采用两台约 27,000 lbf 级发动机的保守量级；如果后续要模拟 CFM56 family 上界，可提高到约 285,000 N。
- B77W 的 `max_thrust_n` 来自两台 GE90-115B：`2 * 115300 lbf * 4.4482216 = 1025759.9 N`。
- C172 没有喷气式 thrust。这里用 `T = eta * P / V` 做简化等效换算：`eta = 0.8`，`P = 180 hp`，`V = 65 kt`，得到约 3,200 N。这个值只是为了当前点质量模型能用“力”的形式优化。
- A320 的 145 kt 看起来比某些真实 VAPP 略高，是因为当前模型还没有襟翼/起落架/高升力构型。用 62 kt 会让模型在升力上明显不可行。

## 5. 前端初始放置区域怎么画

前端不要再让用户随便把飞机放在机场中心附近，然后强行优化到跑道入口。更合理的做法是：选定跑道后，在最终进近方向上画一个环状条带。

几何逻辑：

```text
runway_heading = 目标跑道入口方向
inbound_back_bearing = runway_heading + 180 deg
inner_distance = final_approach_min_nm
outer_distance = final_approach_max_nm
lateral_half_width = final_approach_lateral_half_width_nm
```

条带含义：

- 沿跑道延长线向外，从 `inner_distance` 到 `outer_distance` 之间是建议放置区域。
- 条带左右各 `lateral_half_width`，避免用户必须点在一条极细的线上。
- 用户点击条带内某个位置时，可以自动给一个建议高度：

```text
suggested_altitude_m =
  runway_threshold_alt_m
  + threshold_crossing_height_m
  + tan(final_approach_glide_angle_deg) * distance_to_threshold_m
```

举例：3 deg 下滑角、跑道入口高度 113 m、入口上方参考高度 15 m，在 5 NM final 的建议高度约为：

```text
113 + 15 + tan(3 deg) * 9260 = 613 m
```

这比“离跑道 1.4 km 但高度 1000 m”的默认初始状态合理得多。

## 6. 后续实现顺序

建议按以下顺序改，避免一次性牵动太多逻辑：

1. 扩展 Python `AircraftSpec` 和 aircraft catalog，把上面的推力、末端速度、final approach 放置区域写进去。
2. 后端优化接口从 selected aircraft spec 读取 `max_thrust_n`、`approach_thrust_guess_n`、默认末端速度，不再使用全局 magic number。
3. 前端 aircraft 设置面板拉取这些字段；切换机型时自动更新默认末端速度和输入范围。
4. 前端地图在选择目标跑道后绘制 final approach 建议条带，并把初始飞机默认放到条带中线附近。
5. 再评估 least-squares 结果。如果仍然 thrust 不动，需要继续检查控制量尺度、残差权重、积分步长以及当前气动模型是否缺少构型变量。
