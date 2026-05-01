
# AeroViz-4D：4D 轨迹可视化与验证设计说明（Markdown）

> 版本：v1.0  
> 目标：基于 AeroViz-4D 项目说明，以及 FAA（Federal Aviation Administration）《Order 8260.58D: United States Standard for Performance Based Navigation (PBN) Instrument Procedure Design》中的相关条款，给出一个**可直接落地**的三维（3D）与二维（2D）联动可视化设计方案，重点覆盖：
> 1. 机场场景与轨迹播放；
> 2. 跑道中心的二维剖面（profile）验证；
> 3. RNAV（Area Navigation）/RNP（Required Navigation Performance）procedure 的 segment 呈现；
> 4. 以 OEA（Obstacle Evaluation Area）、XTT（Cross Track Tolerance）、ATT（Along-Track Tolerance）、PFAF（Precise Final Approach Fix）、LTP（Landing Threshold Point）/FTP（Fictitious Threshold Point）、MAP（Missed Approach Point）等为依据的验证叠加层。

---

## 1. 文档定位与设计边界

AeroViz-4D 的核心不是“飞行操作界面”，而是“研究验证界面”。因此，本设计优先满足以下问题：

1. **飞机在哪里**：当前位置、姿态、速度、航迹、历史轨迹、预测轨迹。
2. **它什么时候在那里**：所有视图都必须由同一个仿真时间驱动。
3. **它与 procedure 几何关系如何**：是否沿着 RNAV（Area Navigation）/RNP（Required Navigation Performance）procedure 的几何中心线、RF（Radius-to-Fix）弧段、TF（Track-to-Fix）直段、最终进近（final approach）与复飞（missed approach）区域运动。
4. **它是否合理**：横向是否偏离、垂向是否不合理、转弯是否过急、是否穿越 procedure 水平板或近似 tunnel（通道）、是否与跑道/障碍物/地形上下文一致。

**非目标**：

- 不把本系统做成航图替代品；
- 不作为运行中的导航软件；
- 不在界面中直接暗示“合规判定结论”，而是提供“可审查的证据链”。

---

## 2. 总体设计原则

### 2.1 一切围绕“单一时间轴”
三维（3D）场景、平面图（x-y）、垂直剖面图（x-z）、轨迹列表、fix（航路点）信息、segment（航段）高亮，全部绑定同一个 simulation time（仿真时间）。

### 2.2 一切围绕“选中跑道阈值”
二维（2D）剖面不使用全机场坐标，而使用**跑道阈值局部坐标系**：

- 原点：选中 runway threshold（跑道阈值）；
- `+x`：沿最终进近中心线方向，从阈值指向外侧进近空域；
- `+y`：面对进近方向时，航迹右侧；
- `z`：MSL（Mean Sea Level）高度。

这样可以把程序、航迹、障碍物、地形切片都投影到一个稳定、可比较的 frame（坐标框架）中。

### 2.3 procedure 几何必须和“验证层”分开
建议把显示对象分成三层：

1. **Nominal Geometry（名义几何）**  
   procedure 中心线、fix、RF（Radius-to-Fix）弧、TF（Track-to-Fix）段、跑道、中心线。
2. **Protected / Evaluation Geometry（保护/评估几何）**  
   OEA（Obstacle Evaluation Area）主区（primary area）、次区（secondary area）、最终进近面、复飞扩展区、近似 procedure tunnel（程序通道）。
3. **Observed / Predicted Motion（观测/预测运动）**  
   真实回放轨迹、预测轨迹、历史尾迹、当前位置、时间窗。

这样用户可以分辨：  
“procedure 本身长什么样” 和 “飞机是否偏离”。

### 2.4 研究验证优先于视觉花哨
Cesium（CesiumJS）中的 3D 场景不应为了“好看”而牺牲几何可读性。所有透明体、表面、线宽、标签优先保证：
- 不遮挡轨迹；
- 不遮挡跑道入口与阈值；
- 不遮挡关键 fix；
- 支持一键切换“分析模式”。

---

## 3. 信息架构（Information Architecture）

建议将应用划分为 5 个主要工作区：

1. **3D Airport Scene（机场三维场景）**
2. **2D Plan Profile（二维平面剖面，x-y）**
3. **2D Vertical Profile（二维垂直剖面，x-z）**
4. **Procedure Inspector（程序详情检查器）**
5. **Timeline + State Panel（时间轴与状态面板）**

推荐布局：

- 左侧：图层与 procedure 控件
- 中央：3D Airport Scene（机场三维场景）
- 右侧上：2D Plan Profile（二维平面剖面）
- 右侧下：2D Vertical Profile（二维垂直剖面）
- 底部：timeline（时间轴）+ aircraft state（航空器状态）+ active segment（当前 segment）

---

## 4. 3D 可视化设计

## 4.1 图层栈（由低到高）

### L0. Terrain（地形）
- 使用 Cesium World Terrain（Cesium 世界地形）或预处理地形栅格；
- 默认低饱和度、低对比度；
- 支持 hillshade（地形阴影）开关；
- 在分析模式中允许提高地形对比度，以便观察 final approach（最终进近）与 missed approach（复飞）路径附近的相对高差。

### L1. Airport Base（机场基础设施）
- runway polygon（跑道面）；
- runway threshold（跑道阈值）；
- runway centerline（跑道中心线）；
- displaced threshold（如有，错移阈值）；
- airport reference point（机场基准点）；
- 关键标注：runway identifier（跑道编号）、threshold elevation（阈值高程）。

### L2. Fixes / Waypoints（航路点）
- IAF（Initial Approach Fix）、IF（Intermediate Fix）、PFAF（Precise Final Approach Fix）、FAF（Final Approach Fix）、MAP（Missed Approach Point）、MA fix（missed approach fix，复飞点）；
- 普通 waypoint（航路点）使用小圆点；
- 关键点使用带 ring（环）的点符号；
- 当前选中 branch（分支）的 fix 标签常显，未选中分支的 fix 标签缩减显示。

### L3. Procedure Centerlines（程序中心线）
对不同 segment（航段）类别使用明确的颜色与线型：

- Feeder Segment（引导段）：浅蓝色，虚线；
- Initial Segment（初始进近段）：蓝青色，实线；
- Intermediate Segment（中间进近段）：中蓝色，实线加轻微外发光；
- Final Segment（最终进近段）：绿色，较粗实线；
- Missed Approach Segment（复飞段）：橙色，实线；
- Transition / Branch（转换/分支）：根据父 segment 颜色，透明度降低。

### L4. Protected Surfaces / Tunnel（保护区与近似通道）
- OEA（Obstacle Evaluation Area）主区（primary area）：半透明实色；
- OEA（Obstacle Evaluation Area）次区（secondary area）：更浅的同色带；
- 最终进近 OCS（Obstacle Clearance Surface，障碍物净空面）与复飞扩展：可切换为轮廓线模式；
- Approximate Procedure Tunnel（近似程序通道）：默认关闭，仅在“验证模式”打开。

### L5. Aircraft Motion（航空器运动）
- 当前飞机：带 heading（航向）的小型飞机符号；
- 历史尾迹：按时间淡出；
- 预测轨迹：虚线或半透明线；
- 真实回放 vs 预测：用双层线对比；
- 时间窗内未来位置点：小节点序列，支持 5 秒 / 10 秒 / 30 秒采样。

### L6. Alerts / Deviations（偏差与告警）
- 超出 horizontal plate（水平板）或 tunnel（通道）时，局部线段变红；
- 高度异常时，在飞机附近显示垂向箭头；
- 转弯异常时，在 RF（Radius-to-Fix）/FB（Fly-by）/FO（Fly-over）过渡处显示 turn-rate（转弯率）异常提示。

---

## 4.2 3D 视角预设（Camera Presets）

建议提供以下一键视角：

1. **Airport Overview（机场总览）**
   - 显示全部 runway（跑道）、procedure（程序）、terrain（地形）、obstacle（障碍物）；
   - 用于选 procedure 与 runway。

2. **Runway-Aligned Approach View（跑道对正进近视角）**
   - 相机位于 final approach（最终进近）外侧，沿跑道中心线看向阈值；
   - 用于检查 final segment（最终航段）对正关系。

3. **Top Orthographic Validation View（俯视验证视角）**
   - 近似正射；
   - 用于观察 horizontal plate（水平板）、branch（分支）、RF（Radius-to-Fix）弧段。

4. **Obstacle Context View（障碍物上下文视角）**
   - 自动框选当前 segment（航段）附近 obstacle（障碍物）与 terrain（地形）；
   - 适合答辩演示“为什么这个段看起来不合理”。

5. **Follow Aircraft View（跟随视角）**
   - 相机位于飞机后上方；
   - 不用于精确验证，只用于理解动态运动。

---

## 4.3 3D 中的 RNAV（Area Navigation）/RNP（Required Navigation Performance）segment 呈现规范

## 4.3.1 按 flight phase（飞行阶段）分类

### Feeder Segment（引导段）
用途：连接 en route（航路）结构与 IAF（Initial Approach Fix）/TAA（Terminal Arrival Area）边界。  
呈现：
- 浅蓝虚线；
- 低权重；
- 在选中时显示连接来源 airway（航路）或 entry fix（进入点）；
- 若与 TAA（Terminal Arrival Area）相关，可在平面图上显示“扇区扇面 + 引导线”。

### Initial Segment（初始进近段）
用途：把飞机从 IAF（Initial Approach Fix）带到 IF（Intermediate Fix）或 course reversal（反向程序）入口。  
呈现：
- 蓝青色实线；
- 可显示 HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）或 racetrack（竞赛道）结构；
- 对于 Basic T（基础 T 型）结构，左右分支要明显区分，但默认只高亮当前选中分支。

### Intermediate Segment（中间进近段）
用途：把飞机稳定引导到 final approach（最终进近）入口。  
呈现：
- 中蓝色实线；
- 如果在 PFAF（Precise Final Approach Fix）前 2 NM（Nautical Mile，海里）到后 1 NM 存在 OEA（Obstacle Evaluation Area）连接过渡，应显示 taper（收敛/过渡）区域；
- 若 intermediate（中间进近）与 final（最终进近）不共线，应显示 offset construction（偏置连接）。

### Final Segment（最终进近段）
用途：跑道对正验证的核心。  
呈现：
- 绿色粗实线；
- 线宽略大于其他 segment（航段）；
- 阈值、LTP（Landing Threshold Point）/FTP（Fictitious Threshold Point）、TCH（Threshold Crossing Height）与 glidepath（下滑路径）要联动显示；
- 对于 LNAV（Lateral Navigation）、LNAV/VNAV（Lateral Navigation / Vertical Navigation）、LP（Localizer Performance）、LPV（Localizer Performance with Vertical Guidance）/GLS（GBAS Landing System），可在右侧属性面板显示 final type（最终进近类型）。

### Missed Approach Segment（复飞段）
用途：验证“到 MAP（Missed Approach Point）后如何扩展”。  
呈现：
- 橙色实线；
- section 1（第一段）与 section 2（第二段）要视觉区分；
- 复飞第一段一般从 final approach（最终进近）尾端延伸，需强调 splay（扩展）与 climb surface（爬升面）。

---

## 4.3.2 按 leg type（航段类型）分类

### TF（Track-to-Fix）
- 用直线段表示；
- 端点显示 fix（航路点）；
- 是最常见的基准 leg（航段）。

### RF（Radius-to-Fix）
- 用真实圆弧表示，不要用折线近似；
- 鼠标悬停时显示 radius（半径）、arc length（弧长）、turn direction（转弯方向）；
- 可选显示 turn center（转弯中心）小标记，仅在调试模式启用。

### DF（Direct-to-Fix）
- 用较细的虚线表示“直飞连接”；
- 不应与 TF（Track-to-Fix）混淆；
- 在轨迹验证中用于识别自动导航系统可能产生的“直切”。

### FB（Fly-by）
- 不只显示折点，还要显示 anticipation region（提前转弯区域）；
- 建议在角点处增加一个淡色扇形或 curved preview（预转弯曲线）；
- 这样用户能理解飞机为何在 fix（航路点）前开始转。

### FO（Fly-over）
- 必须明确标出“先飞越 fix（航路点），再转弯”；
- 建议用“方形 fix + 延后转弯弧”；
- 因为它和 FB（Fly-by）的验证语义完全不同。

### HILPT（Hold-In-Lieu-of-Procedure Turn）
- 用标准 racetrack（竞赛道）图形；
- 进出方向与最低等待高度需在详情面板说明；
- 适合在 Procedure Details（程序详情）页中单独展开。

---

## 4.4 3D 中的近似 procedure tunnel（程序通道）设计

由于 AeroViz-4D 是研究可视化原型，而不是 FAA（Federal Aviation Administration）正式设计工具，因此建议采用**两级 tunnel（通道）模型**：

### Level A：Simplified Horizontal Plate（简化水平板）
- 仅用 segment（航段） footprint（平面投影）表达横向保护区；
- 对 TF（Track-to-Fix）使用带状 polygon（多边形）；
- 对 RF（Radius-to-Fix）使用环形弧带；
- primary area（主区）和 secondary area（次区）分色；
- 适合常规浏览。

### Level B：Approximate 3D Validation Tunnel（近似三维验证通道）
- 在 final approach（最终进近）与关键 initial/intermediate（初始/中间）segment（航段）上，沿 centerline（中心线）挤出形成透明通道；
- 横向半宽：
  - 默认使用 primary area（主区）半宽；
  - 可切换为 primary + secondary（主区 + 次区）总宽；
- 垂向高度：
  - 对 final segment（最终航段）：以下滑路径、stepdown altitude（下降台阶高度）、DA（Decision Altitude）/MDA（Minimum Descent Altitude）约束构成上下边界；
  - 对 initial/intermediate（初始/中间进近）：用 published altitude constraint（公布高度约束）与 projected vertical path（投影垂向路径）构成近似；
  - 对 missed approach（复飞）：用 section 1（第一段）平面 + sloping surface（坡面）近似。

**建议**：  
默认只显示 horizontal plate（水平板）；当用户点击“验证模式”时，再显示 3D tunnel（程序通道），否则三维场景会过于拥挤。

---

## 4.5 轨迹本体渲染

### 4.5.1 真实回放轨迹
- 实线；
- 颜色随时间变化，近时刻更亮；
- 当前时间点有飞机符号；
- 支持尾迹长度 slider（滑块）。

### 4.5.2 预测轨迹
- 虚线；
- 可显示 uncertainty envelope（不确定性包络），例如：
  - 横向半透明带；
  - 高度置信区间阴影；
- 与真实轨迹重叠时，允许自动错开线宽或用 outline（外描边）区分。

### 4.5.3 segment attribution（航段归属）
每一个轨迹采样点都应被标注：
- 当前 procedure id（程序标识）；
- 当前 branch id（分支标识）；
- 当前 segment class（Feeder / Initial / Intermediate / Final / Missed）；
- 当前 leg type（TF / RF / DF / FB / FO / HILPT）；
- 到 centerline（中心线）的横向偏差；
- 沿轨距离；
- 与 altitude constraint（高度约束）的差值。

这会直接驱动右侧 2D profile（二维剖面）与状态面板。

---

## 5. 2D Profile（二维剖面）设计

## 5.1 统一的跑道阈值局部坐标系

建议所有二维（2D）验证视图统一使用一个 local runway frame（跑道局部坐标系）：

- 原点：选中 threshold（阈值）；
- `+x`：沿 final approach course（最终进近航向）从阈值向外；
- `+y`：面对进近方向时向右；
- `z`：MSL（Mean Sea Level）高度。

优点：
- final approach（最终进近）直接可比；
- 不同 procedure（程序）可叠加；
- 预测轨迹与回放轨迹可在同一尺度比较；
- 与 FAA（Federal Aviation Administration）文档中的 runway-centered（跑道中心）评估逻辑更一致。

---

## 5.2 2D Plan Profile（二维平面剖面，x-y）

### 5.2.1 用途
回答“飞机横向是否合理”。

### 5.2.2 必须显示的元素
1. runway threshold（跑道阈值）与 runway centerline（跑道中心线）；
2. selected procedure（选中程序）的 nominal route（名义航迹）；
3. branch（分支）；
4. fix（航路点）与名称；
5. horizontal plate（水平板）/ primary area（主区）/ secondary area（次区）；
6. aircraft trajectory（航空器轨迹）；
7. current position（当前位置）；
8. cross-track deviation（横向偏差）颜色编码。

### 5.2.3 横向偏差表达
建议在 plan profile（平面剖面）中采用两种同时存在的表达：

- **点/线着色**：偏差越大，颜色越暖；
- **背景带**：
  - 绿色：在 primary area（主区）内；
  - 黄色：在 secondary area（次区）内；
  - 红色：超出保护区。

### 5.2.4 RF（Radius-to-Fix）段呈现
RF（Radius-to-Fix）段在 plan profile（平面剖面）中是最重要的几何对象之一，建议：
- 画出真实圆弧；
- 选中时显示 arc center（弧心）和半径；
- 若轨迹在 RF（Radius-to-Fix）段内切或外鼓，显示“inside / outside of arc”（偏内/偏外）提示；
- 若 procedure 采用 RF（Radius-to-Fix）到 PFAF（Precise Final Approach Fix），则在 PFAF（Precise Final Approach Fix）附近叠加 final transition（最终进近过渡）轮廓。

### 5.2.5 FB（Fly-by）与 FO（Fly-over）在平面图中的区别
- FB（Fly-by）：fix（航路点）之前出现弯曲过渡；
- FO（Fly-over）：必须先到达 fix（航路点）再转向，图上应有清晰的延迟转弯形态。

这对研究中判断“模型预测为什么提前切弯”非常重要。

---

## 5.3 2D Vertical Profile（二维垂直剖面，x-z）

### 5.3.1 用途
回答“飞机垂向是否合理”。

### 5.3.2 必须显示的元素
1. runway threshold elevation（阈值高程）；
2. glidepath（下滑路径）/ VDA（Vertical Descent Angle）；
3. PFAF（Precise Final Approach Fix）、FAF（Final Approach Fix）、MAP（Missed Approach Point）、stepdown fixes（下降台阶点）；
4. published altitude constraints（公布高度约束）；
5. actual / predicted altitude trace（真实/预测高度曲线）；
6. terrain profile（地形剖面）；
7. obstacle stems（障碍物竖杆）；
8. missed approach climb line（复飞爬升线）。

### 5.3.3 x-z 图的横轴定义
推荐横轴不是“时间”，而是：
- **distance from threshold（距阈值距离）**；
- 以 `x = 0` 为 threshold（阈值）；
- approach（进近）在 `x > 0`；
- threshold（阈值）之后若仍需显示，可允许 `x < 0`，但默认只显示 approach side（进近侧）与初始 missed approach（复飞初段）。

### 5.3.4 final approach（最终进近）专用叠加
对 final segment（最终航段）建议叠加：
- nominal glidepath（名义下滑路径）；
- TCH（Threshold Crossing Height）位置；
- DA（Decision Altitude）/MDA（Minimum Descent Altitude）标记；
- 若是 LNAV（Lateral Navigation）则强调 stepdown altitude（下降台阶高度）；
- 若是 LNAV/VNAV（Lateral Navigation / Vertical Navigation）或 LPV（Localizer Performance with Vertical Guidance）/GLS（GBAS Landing System），则强调 continuous descent（连续下降）。

### 5.3.5 复飞（missed approach）垂直表达
建议在 x-z 图中将 missed approach（复飞）画成两部分：

- **Section 1（第一段）**  
  从 MAP（Missed Approach Point）/DA（Decision Altitude）/MDA（Minimum Descent Altitude）附近开始，显示 height loss（高度损失）和 flat / transition（平段/过渡段）；
- **Section 2（第二段）**  
  显示 standard climb gradient（标准爬升梯度）或指定 climb gradient（爬升梯度）。

如果轨迹没有执行复飞，可把复飞面只显示为背景基准，不突出。

---

## 5.4 3D 与 2D 联动规则

### 5.4.1 Time cursor（时间游标）联动
拖动 timeline（时间轴）时：
- 3D 飞机位置更新；
- plan profile（平面剖面）当前位置点更新；
- vertical profile（垂直剖面）当前位置点更新；
- 当前 active segment（活动航段）在所有视图中高亮。

### 5.4.2 Brushing（刷选）联动
在 2D 视图中框选一个区间：
- 3D 中仅高亮该时间范围轨迹；
- 状态栏显示该区间最大偏差、最大爬升率、最大横滚估计、segment 切换点。

### 5.4.3 Fix / Segment hover（航路点/航段悬停）
悬停 fix（航路点）或 segment（航段）时：
- 3D、plan、vertical 三处同时高亮；
- 右侧面板显示：
  - segment class（航段类别）；
  - leg type（航段类型）；
  - altitude constraints（高度约束）；
  - XTT（Cross Track Tolerance）/ATT（Along-Track Tolerance）；
  - 是否属于 A-RNP（Advanced Required Navigation Performance） / RNP APCH（Required Navigation Performance Approach） / RNP AR APCH（Authorization Required Approach）。

---

## 6. RNAV（Area Navigation）procedure 的 segment 呈现细化规范

## 6.1 Segment（航段）层级模型

建议在数据结构中把 procedure 表达为：

```text
Procedure
  ├─ RunwayGroup
  │   ├─ Branch
  │   │   ├─ Segment (Feeder / Initial / Intermediate / Final / Missed)
  │   │   │   ├─ Leg (TF / RF / DF / CA / HILPT / ...)
  │   │   │   ├─ Fix list
  │   │   │   ├─ Constraints
  │   │   │   ├─ Protected geometry
  │   │   │   └─ Metadata
```

其中：

- Segment（航段）是用户理解 procedure（程序）的主层级；
- Leg（航段类型）是几何与验证实现的主层级；
- Branch（分支）是 UI（User Interface，用户界面）控制与 route filtering（航路筛选）的主层级。

## 6.2 Segment（航段）可视化编码表

| Segment（航段） | 主颜色 | 线型 | 默认显示层级 | 重点标签 | 验证重点 |
|---|---:|---|---|---|---|
| Feeder Segment（引导段） | 浅蓝 | 虚线 | 中 | entry fix（进入点） | 与 TAA（Terminal Arrival Area）/IAF（Initial Approach Fix）连接 |
| Initial Segment（初始进近段） | 蓝青 | 实线 | 高 | IAF（Initial Approach Fix）、IF（Intermediate Fix） | 分支选择、course reversal（反向程序） |
| Intermediate Segment（中间进近段） | 蓝 | 实线 | 很高 | IF（Intermediate Fix）、PFAF（Precise Final Approach Fix） | 与 final（最终进近）对接、横向收敛 |
| Final Segment（最终进近段） | 绿 | 粗实线 | 最高 | PFAF（Precise Final Approach Fix）、MAP（Missed Approach Point）、threshold（阈值） | 跑道对正、垂向合理性 |
| Missed Approach Segment（复飞段） | 橙 | 实线 | 高 | MAP（Missed Approach Point）、MA fix（复飞点） | 扩展区、爬升面 |

## 6.3 Segment（航段）切换点必须显式显示
以下点位必须在界面中清楚标出来：
- Feeder → Initial
- Initial → Intermediate
- Intermediate → Final（特别是 PFAF（Precise Final Approach Fix））
- Final → Missed（特别是 MAP（Missed Approach Point））

原因：  
研究中大量“看起来不合理”的轨迹，其实都发生在 segment transition（航段切换）附近。

## 6.4 A-RNP（Advanced Required Navigation Performance）/RNP AR APCH（Authorization Required Approach）特殊呈现
A-RNP（Advanced Required Navigation Performance）与 RNP AR APCH（Authorization Required Approach）通常意味着：
- 更小的 XTT（Cross Track Tolerance）；
- 某些 phase（阶段）没有 secondary area（次区）；
- width change（宽度变化）可能更敏感；
- RF（Radius-to-Fix）更关键。

因此建议：
- 在图例中单独加一个“precision / reduced-width mode（高精度/窄保护区模式）”；
- 对无 secondary area（次区）的段，直接使用单层 primary envelope（主区包络），避免误导用户；
- 对 RNP AR APCH（Authorization Required Approach） final（最终进近）和 missed（复飞），在属性面板中明确写出“Secondary area not applied（不使用次区）”。

---

## 7. 设计-依据对照表（便于验证）

| 设计项 | 设计建议 | 依据 |
|---|---|---|
| 用单一时间轴驱动 3D + 2D | 所有视图共享一个 simulation time（仿真时间） | 项目说明中明确要求 4D trajectory playback（四维轨迹播放） |
| 采用 runway-threshold-centered（跑道阈值中心）二维坐标 | 2D x-y / x-z 全部围绕选中 threshold（阈值）构建 | 项目说明中明确提到 runway-scoped 2D trajectory profile（跑道范围二维剖面） |
| 以 segment（航段）为一级呈现单元 | Feeder / Initial / Intermediate / Final / Missed 分色分层 | FAA（Federal Aviation Administration） Order 8260.58D 对各阶段分章定义，见 [R6][R7][R9] |
| TF（Track-to-Fix）/RF（Radius-to-Fix）/FB（Fly-by）/FO（Fly-over）分开绘制 | 几何语义不能混用 | [R3] |
| Intermediate（中间进近）到 Final（最终进近）显示 taper（收敛/过渡） | PFAF（Precise Final Approach Fix）前后存在明确连接规则 | [R6][R7] |
| LNAV（Lateral Navigation） final（最终进近）显示 ±0.6 NM（Nautical Mile，海里）主区 + 0.3 NM 次区 | 作为默认 final（最终进近）验证基准 | [R7] |
| LNAV/VNAV（Lateral Navigation / Vertical Navigation）显示 glidepath（下滑路径）与垂向限制 | final（最终进近）是 3D guidance（立体引导） | [R8] |
| RNP AR APCH（Authorization Required Approach） final（最终进近）不显示 secondary area（次区） | 误差保护语义不同 | [R9] |
| Missed Approach（复飞）要分 section 1 / section 2 | section 1（第一段）与后续扩展的几何逻辑不同 | [R10] 与 [R11] |
| HILPT（Hold-In-Lieu-of-Procedure Turn）单独呈现 | 属于初始进近的重要结构 | [R5] |

---

## 8. 可落地的实现建议（前端与数据）

## 8.1 数据预处理输出建议

建议预处理脚本输出以下 JSON（JavaScript Object Notation，JavaScript 对象表示法）或 CZML（Cesium Language，Cesium 标记语言）资产：

### `airport_scene.json`
- airport id
- runways
- thresholds
- centerlines
- terrain tiles references
- obstacle list

### `procedure_catalog.json`
- procedure id
- runway group
- branch ids
- segment list
- leg list
- fix definitions
- constraints
- source metadata
- chart links

### `procedure_geometry.json`
对每个 leg（航段类型）预先计算：
- geodetic polyline（大地线折线）；
- local runway frame coordinates（跑道局部坐标）；
- OEA（Obstacle Evaluation Area） primary polygon（主区多边形）；
- OEA（Obstacle Evaluation Area） secondary polygon（次区多边形）；
- RF（Radius-to-Fix） center + radius + arc angles；
- approximate tunnel mesh（近似程序通道网格，可选）。

### `trajectory_samples.parquet/json`
每个 sample（采样点）包含：
- time
- lon / lat / alt
- heading / groundspeed / vertical rate
- trajectory source（prediction / replay，预测/回放）
- mapped procedure / branch / segment / leg
- x / y / z in runway frame
- cross-track error（横向偏差）
- along-track distance（沿轨距离）
- altitude delta to nominal（相对名义高度差）

---

## 8.2 坐标计算建议

### 8.2.1 跑道局部坐标变换
对每个点 `P`：
1. 以 threshold（阈值）为 origin（原点）；
2. 取 runway final approach bearing（跑道最终进近方位）构造单位向量 `ux`；
3. 用右手法则构造 `uy`；
4. 通过 ENU（East-North-Up，东-北-天）局部坐标或地理投影得到：
   - `x = dot(P - T, ux)`
   - `y = dot(P - T, uy)`
   - `z = alt_msl`

### 8.2.2 segment attribution（航段归属）算法
对每个 sample（采样点）：
1. 先根据 procedure id（程序标识）与时间窗筛 candidate branches（候选分支）；
2. 计算到每个 leg（航段类型）中心线的距离；
3. 若落在 RF（Radius-to-Fix）段，则使用圆弧参数判断沿弧位置；
4. 根据 along-track position（沿轨位置）和 phase order（阶段顺序）确定 active segment（活动航段）；
5. 若跨 branch（分支）歧义，则以最近 fix（航路点）和 heading consistency（航向一致性）解歧。

---

## 8.3 Cesium（CesiumJS）渲染建议

### 几何类型
- runway / OEA footprint（跑道 / 保护区足迹）：GroundPrimitive（贴地几何）
- procedure centerlines（程序中心线）：PolylineCollection（折线集合）或 Entity（实体）
- tunnel（通道）：Primitive（网格体）
- obstacles（障碍物）：billboard（广告牌精灵） + vertical line（竖线）
- aircraft（飞机）：Model / Billboard（模型 / 精灵）
- timeline-linked cursor（时间联动游标）：动态 entity（动态实体）

### 性能策略
- branch（分支）未选中时只显示 centerline（中心线），不显示 full OEA（完整保护区）；
- 2D 视图优先使用预投影坐标；
- RF（Radius-to-Fix）段不要实时重算 arc mesh（圆弧网格），预处理后缓存；
- 大机场 obstacle（障碍物）要分 level-of-detail（细节层级）。

---

## 9. 典型验证工作流

## 9.1 工作流 A：验证 final approach（最终进近）横向合理性
1. 选择 runway（跑道）与 procedure（程序）；
2. 打开 2D Plan Profile（二维平面剖面）；
3. 高亮 final segment（最终航段）；
4. 打开 primary / secondary area（主区/次区）；
5. 播放 trajectory（轨迹）或拖动时间轴；
6. 观察轨迹是否在 final centerline（最终中心线）附近收敛；
7. 若在 PFAF（Precise Final Approach Fix）附近发生偏折，则检查前一 intermediate segment（中间进近）是否 offset（偏置）或 RF（Radius-to-Fix）入 final（最终进近）。

## 9.2 工作流 B：验证垂向合理性
1. 切换 2D Vertical Profile（二维垂直剖面）；
2. 显示 published altitude constraints（公布高度约束）；
3. 显示 glidepath（下滑路径）/ stepdown fixes（下降台阶点）；
4. 对比 actual / predicted altitude（真实/预测高度）；
5. 若 final（最终进近）过高或过低，查看：
   - 是否提前下沉；
   - 是否晚捕获 glidepath（下滑路径）；
   - 是否在复飞起始处高度异常。

## 9.3 工作流 C：验证复飞（missed approach）
1. 进入 MAP（Missed Approach Point）附近；
2. 打开 missed approach（复飞） section 1（第一段）与 section 2（第二段）；
3. 观察 trajectory（轨迹）是否沿 FAC（Final Approach Course，最终进近航迹）延伸后再扩展；
4. 检查 turn-at-altitude（到高转弯）是否过早；
5. 对照 obstacle / terrain（障碍物/地形）判断路径是否具有研究上的 plausibility（合理性）。

## 9.4 工作流 D：验证 RF（Radius-to-Fix）弧段
1. 在 Procedure Details（程序详情）页选中 RF（Radius-to-Fix） leg（航段类型）；
2. 查看 arc radius（弧半径）、entry / exit fix（入/出点）、arc angle（弧角）；
3. 在 2D Plan Profile（二维平面剖面）检查轨迹是否内切/外鼓；
4. 在 3D 中切换顶视与斜视，确认与 terrain（地形）/obstacle（障碍物）的关系。

---

## 10. 最低可行版本（Minimum Viable Product，最小可行版本）建议

若论文时间有限，建议按以下优先级实现：

### Phase 1（第一阶段）
- runway（跑道）/threshold（阈值）/centerline（中心线）
- procedure centerline（程序中心线）
- 3D trajectory playback（轨迹播放）
- 2D plan profile（二维平面剖面）
- 2D vertical profile（二维垂直剖面）
- segment（航段）高亮
- fix（航路点）信息

### Phase 2（第二阶段）
- primary / secondary area（主区/次区）
- RF（Radius-to-Fix）真实弧段
- branch（分支）切换
- obstacle stems（障碍物竖杆）
- terrain profile（地形剖面）
- trajectory deviation（轨迹偏差）着色

### Phase 3（第三阶段）
- approximate 3D tunnel（近似三维程序通道）
- final / missed surfaces（最终进近/复飞面）
- uncertainty envelope（不确定性包络）
- automated validation badges（自动验证徽章）
- screenshot presets（截图预设）与 thesis export（论文导出）

---

## 11. 论文与答辩导向的展示建议

为了服务 thesis（学位论文）与 defense（答辩），建议每个 procedure（程序）都能生成四张固定图：

1. **3D 全局图**：机场、跑道、procedure、trajectory、terrain。
2. **2D 平面图**：runway-centered（跑道中心）水平板 + trajectory。
3. **2D 垂直剖面图**：glidepath（下滑路径）、constraints（约束）、trajectory。
4. **segment detail（航段细节图）**：关键 RF（Radius-to-Fix）或 final / missed transition（最终进近/复飞过渡）。

这样一套图可以直接进入论文主体或附录。

---

## 12. 风险与注意事项

1. **不要把近似 tunnel（通道）当成官方保护面**  
   它只是研究可视化辅助层，不是 FAA（Federal Aviation Administration）正式设计输出。

2. **不要把 projected altitude（投影高度）直接当成 aircraft intent（航空器真实意图）**  
   它更适合作为 validation reference（验证参考）。

3. **不要在 3D 中同时开太多透明层**  
   否则轨迹可读性会迅速下降。

4. **对 RF（Radius-to-Fix）段必须保真**  
   RF（Radius-to-Fix）若被 polyline（折线）粗略替代，会显著削弱研究可信度。

5. **论文中要写清楚“研究用途”**  
   项目说明已经明确指出：不是 certified navigation software（认证导航软件），也不是 official chart（官方航图）替代。

---

## 13. 结论

对于 AeroViz-4D，最合理的可视化策略不是“单纯做一个漂亮的 Cesium（CesiumJS）三维场景”，而是建立一个**以 runway-centered（跑道中心）分析坐标为核心、以 segment（航段）语义为主线、以 3D + 2D 联动为证据链**的研究验证系统。

一句话总结本设计：

> **3D 用来理解空间上下文，2D 用来做可审查验证，segment（航段）层级用来组织 RNAV（Area Navigation）/RNP（Required Navigation Performance）procedure 的语义。**

---

## 14. 参考文献与依据（References）

> 说明：以下参考均来自你提供的 PDF（Portable Document Format）文档  
> **FAA（Federal Aviation Administration） Order 8260.58D, effective date 2025-01-15**。  
> 我在此同时给出 **章节号** 与 **PDF 页码 / 订单内页码**，便于你在原文中核对。

### [R1] PBN（Performance Based Navigation）与 RNAV（Area Navigation）/RNP（Required Navigation Performance）定义
- Order 8260.58D, Section 1-1, Section 1-2.4
- PDF p.1, p.6–7 / printed p.1-1, p.1-2–1-3
- 用途：界定项目中 procedure-aware visualization（程序感知可视化）的术语边界。

### [R2] 不同 flight phase（飞行阶段）的导航精度与 XTT（Cross Track Tolerance）/ATT（Along-Track Tolerance）
- Table 1-2-1. Navigation Accuracy by NavSpec / Flight Phase
- PDF p.7 / printed p.1-3
- 用途：决定不同 segment（航段）的宽度与验证阈值显示。

### [R3] OEA（Obstacle Evaluation Area）与 flight path construction（航迹构造）
- Section 1-2.5
- PDF p.8–27 / printed p.1-4–1-23
- 重点：
  - geodesic path（大地线）构造；
  - alignment tolerance（对正容差）±0.03°；
  - XTT（Cross Track Tolerance）与 ATT（Along-Track Tolerance）；
  - 30 NM（Nautical Mile，海里）模式转换；
  - FB（Fly-by）/FO（Fly-over）/RF（Radius-to-Fix） turn construction（转弯构造）。
- 用途：决定程序几何、通道、转弯段与保护区如何可视化。

### [R4] 各飞行阶段 projected altitude（投影高度）与 turn parameters（转弯参数）
- Section 1-2.5(c)
- PDF p.15–20 / printed p.1-11–1-16
- 重点：
  - final（最终进近）/intermediate（中间进近）/initial（初始进近）/feeder（引导段）/missed（复飞）垂向路径；
  - 250 ft/NM、400 ft/NM、500 ft/NM 等假定垂向增长率；
  - KIAS（Knots Indicated Airspeed，指示空速）、KTAS（Knots True Airspeed，真空速）、turn radius（转弯半径）。
- 用途：决定 2D vertical profile（二维垂直剖面）与转弯合理性分析。

### [R5] Basic T（基础 T 型）、HILPT（Hold-In-Lieu-of-Procedure Turn）、intermediate segment（中间进近段）默认长度
- Section 1-3.1(c)–(f)
- PDF p.29–31 / printed p.1-24–1-27
- 重点：
  - initial segment（初始进近段）可由 TF（Track-to-Fix）、RF（Radius-to-Fix）或 HILPT（Hold-In-Lieu-of-Procedure Turn）构成；
  - Basic T（基础 T 型）；
  - intermediate segment（中间进近段）默认长度；
  - final fixes（最终进近关键点）与 VDP（Visual Descent Point，目视下降点）。
- 用途：决定程序详情页与 branch（分支）可视化。

### [R6] RNAV（GPS） Approach（进近）的一般 segment（航段）定义
- Chapter 3, Section 3-1
- PDF p.51–57 / printed p.3-1–3-6
- 重点：
  - Feeder Segment（引导段）；
  - Initial Segment（初始进近段）；
  - Intermediate Segment（中间进近段）；
  - Intermediate-to-final（中间到最终）过渡；
  - Final Segment（最终进近段）只允许 TF（Track-to-Fix），不允许 RF（Radius-to-Fix）和 TF turn（TF 转弯）。
- 用途：决定 RNAV（GPS） procedure 的 segment 呈现总逻辑。

### [R7] LNAV（Lateral Navigation） final segment（最终进近段）OEA（Obstacle Evaluation Area）
- Section 3-2
- PDF p.60–64 / printed p.3-9–3-13
- 重点：
  - final OEA（最终进近保护区）从 PFAF（Precise Final Approach Fix）前 0.3 NM 到 LTP（Landing Threshold Point）/FTP（Fictitious Threshold Point）后 0.3 NM；
  - 主区半宽 ±0.6 NM；
  - 每侧 0.3 NM 次区；
  - tapering area（收敛区）计算。
- 用途：决定 2D 平面图和 3D final tunnel（最终进近通道）默认宽度。

### [R8] LNAV/VNAV（Lateral Navigation / Vertical Navigation） final segment（最终进近段）与对正条件
- Section 3-3
- PDF p.64–65 / printed p.3-14–3-15
- 重点：
  - LNAV/VNAV（Lateral Navigation / Vertical Navigation）是 vertically guided（垂直引导）程序；
  - TF-TF turns not allowed in FAS（Final Approach Segment，最终进近段）；
  - final course（最终航迹）最多可 offset（偏置）15°，但需满足与 runway centerline（跑道中心线）的交会条件。
- 用途：决定 final alignment（最终对正）验证逻辑。

### [R9] RNP AR APCH（Authorization Required Approach） feeder / initial / intermediate / final
- Chapter 4, Section 4-1, Section 4-2
- PDF p.118–119 / printed p.4-2–4-3
- 重点：
  - feeder（引导段）可使用 RNAV 1（Area Navigation 1）/RNP 1（Required Navigation Performance 1）/A-RNP（Advanced Required Navigation Performance）；
  - initial（初始进近）/intermediate（中间进近）/final（最终进近）使用 RNP AR APCH（Authorization Required Approach）；
  - secondary areas do not apply（不使用次区）；
  - final（最终进近）最小 HAT（Height Above Threshold，阈值上高）250 ft。
- 用途：决定高精度 procedure 的视觉差异化。

### [R10] RNP AR APCH（Authorization Required Approach） missed approach segment（复飞段）
- Section 4-3
- PDF p.127–132 / printed p.4-12–4-16
- 重点：
  - Default MAS（Missed Approach Segment，默认复飞段）；
  - RNAV MAS（Area Navigation 复飞段）；
  - Reduced RNP MAS（缩减 RNP 复飞段）；
  - straight missed approach（直线复飞）至少以 15° 扩展到 ±2 NM。
- 用途：决定 RNP AR APCH（Authorization Required Approach）复飞呈现。

### [R11] RNAV（GPS） missed approach（复飞）一般规则
- Chapter 3, Section 3-5, Section 3-6, Section 3-7
- PDF p.85–91 / printed p.3-34–3-40（已检索到关键片段）
- 重点：
  - CA（Course-to-Altitude，航向到高度） leg（航段）作为 RNAV（Area Navigation） missed approach（复飞）的首段；
  - section 1（第一段）与后续扩展；
  - climb gradient（爬升梯度）与 OEA（Obstacle Evaluation Area）扩展。
- 用途：决定普通 RNAV（GPS） 复飞层的表达。

---

## 15. 建议的后续动作

1. 先把 **runway-centered（跑道中心）二维坐标转换** 做对；
2. 再实现 **segment（航段）级 procedure 几何数据结构**；
3. 再实现 **3D centerline（中心线）+ 2D plan / vertical（平面/垂直剖面）联动**；
4. 最后再上 **primary / secondary area（主区/次区）与 approximate tunnel（近似程序通道）**。

如果论文周期紧，这个顺序最稳。
