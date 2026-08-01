# 弧长几何损失：最简基线与单航迹容量实验

日期：2026-08-01

## 1. 实验边界

本轮只研究按水平弧长对齐的局部状态误差，不使用走廊、曲率、DTW 或第二次
dynamics rollout。所有容量实验固定使用同一条锁定的 outer-train 航迹：

`KSJC:ASA956_30L_a1f7fb_20260719T014743Z`

训练和回放都只打开这一条 outer-train 航迹。outer-val 和 outer-test 只参与锁定划分
的身份计数，没有读取其轨迹值。所有结果中的
`test_policy.outer_test_tracks_opened` 均为 `false`。

共同设置：iTransformer、N=64、transport-chart-velocity dynamics、0.5 s 积分、
factorized duration、1000 epochs、学习率 1e-4、终端位置权重 1.0、终端速度权重
1.0、control effort/smoothness 权重 0。

## 2. 对齐方法

预测曲线由 anchor 和 64 个 control 段末端状态组成。参考曲线由同一 anchor、固定
2 s 参考点和精确终端点组成。

两条有序曲线分别计算水平累计弧长，再把各自弧长归一化到 0 到 1，并采样 64 个
等间隔弧长位置。同一个插值方案同时重采样位置和 ENU 速度，损失比较的是相同弧长
进度下的位置、水平速度和垂直速度，不比较相同物理时刻下的状态。

位置仍按训练集 state scale 归一化后计算 SmoothL1。默认局部速度把 E/N 两分量误差
的模除以显式水平速度尺度，垂直速度使用 U 分量绝对误差和独立尺度。只有两侧原始
速度节点都具备真实监督的弧长采样点才参与局部速度项；尾部拟合占位速度和人为补出
的终端速度不会被当作监督目标。

本轮在同一个 objective 内加入三个正交消融开关，没有增加新的公开 loss objective：

1. 终端跑道分量：把终端位置与速度误差旋转为沿跑道、横向、垂直绝对误差，并按
   1/3/5 加权求和。ENU rollout 单独携带真实跑道航向；它不会误用 ENU frame 的零旋转角。
2. 切向与速率：把局部水平 E/N 向量误差替换为 `1-cos` 切向误差和水平速率绝对误差，
   垂直速度项不变。
3. 末段位置加权：弧长进度从 0 到 1 时，原始线性权重从 1 增至 4，再除以均值 2.5，
   实际相对权重约为 0.4 到 1.6；它重新分配梯度，但不放大几何项的平均尺度。

该对齐允许飞机以不同速度通过相同几何路径，因此与可变 control 段时长不冲突。
终端位置、终端速度和最终时间仍由独立项约束。

## 3. 解耦结构

- `arc_length_geometry.py`：负责共享的弧长插值方案、位置/速度重采样、可靠性掩码和
  诊断指标。
- `control_loss_components.py`：通过 objective registry 组合弧长局部状态、终端状态和
  最终时间项。
- `train.py`：通过 validation-selection registry 选择与训练 recipe 对应的 checkpoint。
- `config.py`：每个公开模式、权重和尺度都有显式序列化字段；旧派生 checkpoint 缺字段时
  直接拒绝并要求重建，不提供兼容回退。
- `terminal_state_loss.py`：共享训练与验证的终端速度目标、ENU→跑道轴旋转和分量指标。
- CLI 和 pipeline 把三项参数化、权重与尺度纳入 artifact identity 和用户可见标签。

当前仍只有一个弧长 objective：

1. `arc-length-geometry`：在同一水平弧长进度上监督归一化位置、水平局部速度和垂直
   局部速度，并保留独立的终端位置、终端速度及最终时间约束。终端表达、局部速度
   表达和位置进度权重是该 objective 的三个可独立切换参数，不是新增 objective。

`arc-length-distance` 和 `arc-length-geometry-path` 是本轮已经完成的失败消融，现已从
配置、CLI、训练 registry 和 validation selection 中移除。下面仍列出它们的历史结果，
用于说明为什么没有保留这些模式；历史输出目录不代表当前仍支持对应 recipe。

## 4. 单航迹结果

| 模式 | 关键权重 | best epoch | common-grid ADE | FDE | 终端速度误差 | 弧长几何/距离结果 |
|---|---:|---:|---:|---:|---:|---:|
| 既有 terminal-state | dense=0.25 | 999 | 2993.184 m | 59.857 m | 1.58 m/s（旧报告） | 未记录 |
| arc normalized | geometry=0.25 | 953 | 8692.543 m | 48.868 m | 2.768 m/s | normalized loss 0.6044；水平均值 9305 m |
| arc normalized | geometry=0.75 | 866 | 3194.271 m | 10.560 m | 0.109 m/s | normalized loss 0.0798；水平均值 2369 m |
| arc 3D distance | geometry=0.25 / 1000 m | 903 | 7417.249 m | 10.803 m | 0.373 m/s | 3D 距离均值 6872 m |
| arc normalized + path | geometry=0.75，path=0.25 | 768 | 5410.584 m | 13.482 m | 3.975 m/s | 3D 距离 6728 m；长度比 1.471 |
| arc position + local velocity | geometry=0.75，horizontal velocity=0.25，vertical velocity=0.25 | 966 | 7682.184 m | 14.458 m | 0.434 m/s | 水平速度 MAE 93.230 m/s；垂直速度 MAE 15.937 m/s |
| 方法 2：终端跑道分量 | 终端沿/横/垂直=1/3/5 | 975 | 5199.519 m | 23.289 m | 0.260 m/s | 弧长距离 5098.747 m；长度比 0.953 |
| 方法 3：切向+速率 | tangent=0.25，speed=0.25 | 966 | 8120.501 m | 13.918 m | 0.878 m/s | 水平向量速度 MAE 98.334 m/s；速率 MAE 64.452 m/s |
| 方法 4：末段位置加权 | end weight=4（均值归一） | 892 | 5837.878 m | 19.633 m | 0.794 m/s | 弧长距离 5408.721 m；长度比 1.287 |
| 方法 2+4 | 终端 1/3/5，end weight=4 | 917 | **4511.445 m** | 14.931 m | 0.718 m/s | 弧长距离 **4191.711 m**；长度比 0.937 |

各次完整结果位于：

- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_arc_length_geometry_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_arc_length_geometry_g0p75_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_arc_length_distance_s1000_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_arc_length_geometry_g0p75_path0p25_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_transport_chart_velocity_arc_length_state_velocity_g0p75_hv0p25_vv0p25_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_arc_ablation_terminal_runway_c3_u5_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_arc_ablation_tangent_speed_t0p25_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_arc_ablation_progress_end4_single_flight_overfit_1000`
- `4dTrajectory/outputs/KSJC/ts_control_fixed_dt_ASA956_arc_ablation_terminal_runway_c3_u5_progress_end4_single_flight_overfit_1000`

## 5. 结论

方法 2 单独把 ADE 从 7682 m 降到 5200 m，并把路径长度比从 1.465 修正到 0.953，
但 FDE 增至 23.3 m，终端误差主要为 21.7 m 横向偏差。方法 4 单独把 ADE 降到
5838 m，但 FDE 增至 19.6 m，终端误差主要为 18.1 m 垂直偏差。方法 3 只有 FDE
小幅改善 0.54 m，ADE、局部速度和终端速度均退化，因此没有进入组合实验。

方法 2+4 具有实际互补效果：ADE 降至 4511 m，比 local-velocity 基线改善 41.3%；
弧长距离降至 4192 m，路径长度比为 0.937。FDE 为 14.93 m，只比基线高 0.47 m；
终端横向和垂直误差分别降至 0.37 m 和 1.38 m，剩余误差几乎全部是 14.86 m 的沿
跑道偏移。水平/垂直局部速度 MAE 也从 93.23/15.94 m/s 改善到 55.27/11.95 m/s。

不过 4511 m 仍未达到既有 terminal-state 单航迹 ADE 2993 m，也未达到此前不含局部
速度的 arc normalized 3194 m。它证明终端跑道分量与末段权重能修复当前
local-velocity recipe 的一部分坏局部解，但还不足以证明应该进入五机场 train/val。

失败不是因为 loss 没有下降。几种模式都能把总 loss 大幅压低，但仍能形成“终点正确、
中段走错”的控制解：

- normalized SmoothL1 在数公里误差区间内梯度过早减弱；提高权重有帮助但不充分。
- 直接米制距离产生更强梯度，却没有消除单次长时域 shooting 的坏局部解。
- 总路径长度只约束一个全局标量。长度接近参考不代表局部方向正确；对应消融停在
  1.471 倍参考长度，并与终端速度/几何项发生梯度竞争。
- 把 E/N 水平向量改成切向和速率并没有解决问题；方向/速率解耦后，模型仍可能找到
  终点较好但中段路径和速度都错误的控制解。
- 方法 2+4 把横向、垂直落点误差压低后，剩余终端问题集中为沿跑道偏移。这比一个
  不可解释的三维 FDE 更容易诊断，但中段约 4.5 km ADE 仍是主要容量瓶颈。

## 6. 下一步建议

本轮不再搜索更多权重，也不启用方法 1 的 staged schedule。若继续 loss 研究，优先用
方法 2+4 作为当前 local-velocity 基线，先诊断沿跑道终端偏移和中段速度项为何仍停在
55.27/11.95 m/s；不要把方法 3 叠加进去，也不要新增公开 objective。只有单航迹容量
明显优于 2993 m 的既有 terminal-state 基线，才值得进入五机场 train/val。outer-test
继续保持冻结。

## 7. 2+4 新基线与多航迹容量诊断

按本轮消融结论，`arc-length-geometry` 的默认 recipe 已改为方法 2+4：

- `geometry weight = 0.75`；
- 局部水平速度仍使用 E/N 向量分量，不叠加失败的方法 3；
- 终端位置与速度采用跑道轴分量，沿跑道/横向/垂直相对权重为 1/3/5；
- 弧长位置权重从起点 1 线性增加到终点 4，并按平均值归一化；
- 终端位置、终端速度仍各为 1.0，control effort/smoothness 仍为 0。

这组默认值同时落在 `TSConfig`、pipeline 和单航迹 overfit CLI 中。它仍是同一个
`arc-length-geometry` objective，只改变正交参数默认值，没有增加或复制 loss 模式。

随后用固定哈希规则选择了 5 条 outer-train 航迹，保持 iTransformer、N=64、学习率
1e-4、seed=1337 和最多 1000 epochs 不变。validation 和 outer-test 航迹值均未打开。

| 机场 | 时域 | epochs | common ADE | FDE | 弧长水平 MAE | 弧长垂直 MAE | 终端速度误差 | 路径长度比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KSTL | 107.7 s | 1000 | 226.3 m | 5.81 m | 10.2 m | 142.9 m | 22.50 m/s | 1.000 |
| KSMF | 154.5 s | 1000 | 512.7 m | 1.78 m | 241.1 m | 227.1 m | 32.64 m/s | 0.982 |
| KSJC | 179.9 s | 1000 | 313.1 m | 1.42 m | 30.9 m | 110.3 m | 7.67 m/s | 1.000 |
| KMSY | 428.0 s | 561 | 6000.7 m | 21793.5 m | 6000.4 m | 394.1 m | 141.49 m/s | 0.727 |
| KRDU | 580.0 s | 462 | 4649.4 m | 1240.1 m | 7031.3 m | 510.7 m | 47.75 m/s | 2.671 |

完整交互图见
[`2026-08-01_arc24_multiflight_capacity_diagnosis.zh.html`](2026-08-01_arc24_multiflight_capacity_diagnosis.zh.html)。
页面含每条航迹的 E-N 平面、高度、物理时间误差、速度、位置差分速度与状态速度的
一致性、加速度、三类控制、64 段时长、终端跑道轴分解和 loss 曲线。它由
`build_multiflight_capacity_report.py` 从显式实验结果生成，生成前会验证配方一致以及
`outer_test_tracks_opened=false`、`validation_tracks_opened=false`。

诊断产物使用 v3 clock-aligned schema：segment durations、offsets 和 endpoint states
来自同一次 observed-supervision rollout。报告按完整实验 config 校验兼容性，并把
`reference_fully_measured=false` 的 fitted-tail 速度及相关区间显示为断点，不把零权重
占位速度解释为观测。

### 7.1 主要发现

1. 2+4 可以在三条 180 s 内航迹上把 FDE 压到 1.4--5.8 m，但 ADE 仍为
   226--513 m。它首先解决了“到点”，尚未解决“沿正确剖面、以正确速度到点”。
2. 三条短航迹中，终端剩余速度误差几乎都集中在沿跑道方向；KSTL、KSMF 分别为
   22.50、32.64 m/s。单纯继续放大横向或垂直终端权重不会处理这个问题。
3. 五条航迹的弧长垂直 MAE 都超过 100 m，说明垂直剖面是跨机场重复出现的瓶颈，
   不是某一条水平转弯造成的 ADE 假象。
4. 两条长航迹进入不同坏局部解。KMSY 路径缩到参考的 0.727，终端沿跑道少飞
   21.79 km；KRDU 路径扩到 2.671，终端主要是 1.21 km 垂直误差。因此不能用一个
   固定积分比例误差解释，长链优化和 64 段表达分辨率都需要进一步隔离。
5. 控制并未普遍贴上下界：两条长航迹三类控制的贴边率均为 0%。失败更像梯度/坏
   局部解和目标冲突，不像 bounded head 饱和。
6. 更重要的是，五条参考航迹的位置差分速度与存储速度状态 RMSE 为
   4.76--10.23 m/s，参考加速度 p95 为 11.8--32.2 m/s²。可行动力学强制
   `position derivative = velocity`，而当前 geometry + velocity loss 要同时匹配这两套
   不完全自洽的标签，因而存在不可约张力。HTML 中的自一致性曲线可定位这些误差在
   每条航迹上何时出现。

因此，2+4 作为后续 loss 实验的新相对基线是成立的，但当前结果还不能证明它已达到
“正确降落”的绝对容量。下一步应先在 train 数据上隔离参考 state 自一致性问题，并用
不同 N 或更短时域判断长航迹失败究竟来自 64 段分辨率还是单次长链优化；outer-test
继续冻结。
