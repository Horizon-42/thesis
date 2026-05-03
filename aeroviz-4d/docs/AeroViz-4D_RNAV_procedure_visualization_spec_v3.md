# AeroViz-4D：RNAV（Area Navigation）Procedure（Procedure）可视化与 Segment（Segment）呈现设计说明（重写增强版）

> 版本：v3.0  
> 目标：本版只聚焦 **RNAV（Area Navigation）/RNP（Required Navigation Performance）procedure（Procedure）本体的可视化**，不再展开整体页面布局、通用坐标系统推导或既有 tunnel（Tunnel，通道体）方案。  
> 核心要求：把 procedure（Procedure）从“fix（Fix，航路点）连线”提升为“**按 segment（Segment，航段）规范、leg（Leg，航迹腿）类型、保护区与垂向面**共同表达的研究验证对象”。  
> 主要依据：FAA（Federal Aviation Administration）《Order 8260.58D: United States Standard for Performance Based Navigation (PBN, Performance Based Navigation) Instrument Procedure Design》。fileciteturn0file0

---

## 1. 本版文档的边界

### 1.1 本版只回答一个问题

**RNAV（Area Navigation）procedure（Procedure）在 AeroViz-4D 中到底应该怎样被“正确地画出来”。**

这里的“正确”不是指视觉上漂亮，而是指：

1. 能区分 **segment（Segment，航段）级别** 的不同语义；
2. 能区分 **leg（Leg，航迹腿）类型** 与转弯构型；
3. 能体现 **OEA（Obstacle Evaluation Area，障碍物评估区）/OCS（Obstacle Clearance Surface，障碍物净空面）** 这类“保护/评估几何”；
4. 能支持你把预测轨迹或回放轨迹，和 published procedure geometry（published procedure geometry，已公布程序几何）进行逐项对照验证。  

### 1.2 本版明确不展开的内容

1. 不再写通用页面信息架构；
2. 不再把重点放在跑道局部坐标系推导；
3. 不再默认沿用你现在的 tunnel（Tunnel，通道体）呈现；
4. 不把 thesis（Thesis，论文）里的“验证结论”提前硬编码成单一红绿判断，而是提供可核查的几何证据链。

---

## 2. 结论先行：RNAV（Area Navigation）procedure（Procedure）不能再被画成“折线 + 点”

对于 AeroViz-4D，你要可视化的不是一条 polyline（Polyline，折线），而是一个**程序图结构**：

- 上层是 **procedure（Procedure）/transition（Transition，过渡）/branch（Branch，分支）**；
- 中层是 **segment（Segment，航段）**；
- 下层是 **leg（Leg，航迹腿）**；
- 附属层是 **fix（Fix，航路点）/constraint（Constraint，约束）/protected area（protected area，保护区）/vertical surface（vertical surface，垂向面）**。

因此，建议把 RNAV（Area Navigation）procedure（Procedure）拆成四类可视化对象：

1. **Nominal Path（Nominal Path，名义航迹）**  
   即 pilot（Pilot，飞行员）/FMS（Flight Management System，飞行管理系统）应该飞的中心线几何；
2. **Segment Envelope（Segment Envelope，航段包络）**  
   即按 XTT（Cross Track Tolerance，横向容差）/ATT（Along-Track Tolerance，沿航迹容差）与章节规则生成的 lateral protection（lateral protection，横向保护）；
3. **Vertical Evaluation Geometry（Vertical Evaluation Geometry，垂向评估几何）**  
   包括 glidepath（Glidepath，下滑路径）、MDA（Minimum Descent Altitude，最低下降高度）、DA（Decision Altitude，决断高度）、OCS（Obstacle Clearance Surface，障碍物净空面）、missed approach（Missed Approach，复飞）爬升面；
4. **Operational Branch Structure（Operational Branch Structure，运行分支结构）**  
   用来表达 Basic T（Basic T，基础 T 型）、HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）、不同 IAF（Initial Approach Fix，初始进近定位点）分支、不同 missed approach（Missed Approach，复飞）路径分歧。  

这四类对象必须同时存在，才能让可视化真正服务于验证。  

---

## 3. 建议的核心设计：从 “Tunnel（Tunnel，通道体）” 改为 “Ribbon（Ribbon，带状包络）+ Surface（Surface，评估面）+ Probe（Probe，剖切探针）”

### 3.1 为什么不建议把 tunnel（Tunnel，通道体）作为主表达

你现有的 tunnel（Tunnel，通道体）表达方式有一个优点：直观。但它有三个研究验证层面的缺点：

1. 它容易把“名义路径”和“保护边界”混成一个厚实体；
2. 它很容易遮挡 runway threshold（runway threshold，跑道入口）/PFAF（Precise Final Approach Fix，精确最后进近定位点）/MAP（Missed Approach Point，复飞点）等关键点；
3. 它不适合表达 FAA（Federal Aviation Administration）Order 8260.58D 中很多其实是 **平面宽度变化 + 垂向面变化** 的规则，而不是统一截面的体管道。  

### 3.2 更适合本课题的 3D（三维，Three-Dimensional）主表达

建议把 procedure（Procedure）可视化改成三层主表达：

#### A. Centerline（Centerline，中心线层）
- 只画 leg（Leg，航迹腿）真实几何；
- TF（Track-to-Fix，航迹到定位点）画直线；
- RF（Radius-to-Fix，半径到定位点）画真圆弧，不要用粗采样折线冒充；
- CA（Course-to-Altitude，航向到高度）与 DF（Direct-to-Fix，直飞到定位点）要作为不同 leg（Leg，航迹腿）类型画出来；
- 当前 active leg（active leg，当前激活航迹腿）加粗并发光，其余 leg（Leg，航迹腿）降亮度。

#### B. Protection Ribbon（Protection Ribbon，保护带层）
- 不做封闭 tunnel（Tunnel，通道体）；
- 只画贴合 nominal path（nominal path，名义航迹）的半透明 ribbon（Ribbon，带状包络）；
- ribbon（Ribbon，带状包络）宽度按 XTT（Cross Track Tolerance，横向容差）/secondary area（secondary area，次区）规则变化；
- 只表达 lateral protection（lateral protection，横向保护），不伪装成“飞行器真实能活动的体积”。

#### C. Surface Cards（Surface Cards，评估面层）
- final segment（final segment，最后进近航段）和 missed approach（Missed Approach，复飞）不要再靠 tunnel（Tunnel，通道体）表示；
- 直接显示 FAA（Federal Aviation Administration）规范里的 OCS（Obstacle Clearance Surface，障碍物净空面）/section（Section，分段）面：
  - LNAV（Lateral Navigation，横向导航）/LP（Localizer Performance，航向道性能） final（Final，最后进近）用 final OEA（Obstacle Evaluation Area，障碍物评估区）带状面；
  - LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）用 LNAV（Lateral Navigation，横向导航）横向带 + sloping OCS（sloping OCS，倾斜净空面）；
  - LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）用 W/X/Y surfaces（W/X/Y surfaces，W/X/Y 评估面）；
  - missed approach（Missed Approach，复飞）用 section 1（Section 1，第一段）与 section 2（Section 2，第二段）分开画。

### 3.3 验证时才启用的剖切表达

为了避免 3D（三维，Three-Dimensional）遮挡，建议增加 **Probe Plane（Probe Plane，剖切探针平面）**：

- 在 PFAF（Precise Final Approach Fix，精确最后进近定位点）、FAF（Final Approach Fix，最后进近定位点）、MAP（Missed Approach Point，复飞点）、RF（Radius-to-Fix，半径到定位点）中点、missed approach（Missed Approach，复飞）section（Section，分段）转换点等位置，生成可移动切面；
- 3D（三维，Three-Dimensional）只看全局结构；
- 精确检查靠 2D（二维，Two-Dimensional）剖切。  

**结论**：对你的 thesis（Thesis，论文）场景，更好的 3D（三维，Three-Dimensional）方案不是 tunnel（Tunnel，通道体），而是：  
**Centerline（Centerline，中心线） + Protection Ribbon（Protection Ribbon，保护带） + Surface Cards（Surface Cards，评估面） + Probe Plane（Probe Plane，剖切探针平面）**。

---

## 4. RNAV（Area Navigation）procedure（Procedure）对象模型：建议按 segment（Segment，航段）驱动渲染

建议把每个 procedure（Procedure）对象规范化成如下结构：

```text
Procedure
  ├─ Transition / Branch
  │    ├─ Segment
  │    │    ├─ Leg[TF / RF / DF / CA / HILPT sequence]
  │    │    ├─ Start Fix / End Fix
  │    │    ├─ NavSpec
  │    │    ├─ XTT / ATT
  │    │    ├─ Secondary area applies?
  │    │    ├─ Vertical rule set
  │    │    └─ Geometry bundle
  │    └─ Merge / Diverge relation
  └─ Missed Approach Structure
```

### 4.1 Segment（Segment，航段）级别必须是渲染主单位

每个 segment（Segment，航段）至少要带这些字段：

- `segmentType`：feeder（feeder，引导航段）/initial（initial，初始航段）/intermediate（intermediate，中间航段）/final（final，最后进近航段）/missed_s1（missed_s1，复飞第一段）/missed_s2（missed_s2，复飞第二段）；
- 若 CIFP/detail export 只能给出 `route` 语义，应按 branch role 归一为 `TRANSITION_ROUTE`（transition branch 的连接航路）或 `PROCEDURE_ROUTE`（procedure branch 内尚未细分的程序航路），不能降级为 `UNKNOWN`。
- `navSpec`：例如 RNAV 1（Area Navigation 1）、RNP APCH（Required Navigation Performance Approach，所需导航性能进近）、A-RNP（Advanced Required Navigation Performance，高级所需导航性能）、RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）；
- `xtt`：XTT（Cross Track Tolerance，横向容差）；
- `att`：ATT（Along-Track Tolerance，沿航迹容差）；
- `secondaryEnabled`：是否有 secondary area（secondary area，次区）；
- `legList`：内部 leg（Leg，航迹腿）序列；
- `segmentEnvelope`：本段 lateral envelope（lateral envelope，横向包络）；
- `verticalSurfaces`：本段 vertical surfaces（vertical surfaces，垂向面）；
- `constraints`：altitude（Altitude，高度）/speed（Speed，速度）/turn（Turn，转弯）限制；
- `constructionNotes`：是否有 offset construction（offset construction，偏置构造）、taper（taper，收敛段）、splay（splay，张角扩展）等。  

### 4.2 不要只按 fix（Fix，航路点）渲染

fix（Fix，航路点）只是控制点，不是主体。  
真正要画的是：

- fix（Fix，航路点）前后的 ATT（Along-Track Tolerance，沿航迹容差）影响；
- fix（Fix，航路点）处是否为 FB（Fly-by，提前转弯）或 FO（Fly-over，飞越后转弯）；
- fix（Fix，航路点）后是否触发 width change（width change，宽度变化）、turn construction（turn construction，转弯构造）、section split（section split，分段切换）。

---

## 5. Segment（Segment，航段）呈现规范：按 FAA（Federal Aviation Administration）Order 8260.58D 重新定义

## 5.1 Feeder Segment（Feeder Segment，引导航段）

### 5.1.1 几何来源与规则

对于 RNAV（Area Navigation）(GPS) approach（Global Positioning System approach，全球定位系统进近），feeder segment（Feeder Segment，引导航段）按 Chapter 2（Chapter 2，第二章）与 §3-1-2 构造，可由一个或多个 TF（Track-to-Fix，航迹到定位点）或 RF（Radius-to-Fix，半径到定位点）leg（Leg，航迹腿）组成；默认 NavSpec（Navigation Specification，导航规范）是 RNAV 1（Area Navigation 1），XTT（Cross Track Tolerance，横向容差）为 1.00；在满足表注条件时可按 effective XTT（effective XTT，有效横向容差）2.00 处理；secondary area（secondary area，次区）适用，但 A-RNP（Advanced Required Navigation Performance，高级所需导航性能）除外。见 §3-1-2。fileciteturn0file0

### 5.1.2 可视化要求

Feeder Segment（Feeder Segment，引导航段）不应只作为“远端连线”弱化处理，而应明确表达：

1. 它是 procedure（Procedure）的一部分；
2. 它的 lateral protection（lateral protection，横向保护）通常宽于 final（Final，最后进近）航段；
3. 它可能承接 en route（en route，航路）模式与 terminal（terminal，终端）模式的过渡。  

### 5.1.3 建议画法

- **中心线**：中等线宽；
- **Protection Ribbon（Protection Ribbon，保护带）**：显示 primary area（primary area，主区）+ optional secondary area（optional secondary area，可选次区）；
- **标签**：显示进入点、IAF（Initial Approach Fix，初始进近定位点）连接关系；
- **分支关系**：若同一 runway（runway，跑道）存在多个 feeder（feeder，引导）入口，默认全部显示，但只高亮当前 branch（Branch，分支）。

### 5.1.4 研究验证意义

Feeder Segment（Feeder Segment，引导航段）可用于验证：

- 预测轨迹是否过早“吸附”到中间进近；
- RF（Radius-to-Fix，半径到定位点）入口是否被模型错误近似成直线；
- 轨迹是否在 30 NM（Nautical Mile，海里）模式切换前后出现非合理横向跳变。  

---

## 5.2 Initial Segment（Initial Segment，初始进近航段）

### 5.2.1 几何来源与规则

Initial Segment（Initial Segment，初始进近航段）按 §1-3-1(d) 与 §3-1-3 构造，可由一个或多个 TF（Track-to-Fix，航迹到定位点）或 RF（Radius-to-Fix，半径到定位点）leg（Leg，航迹腿）组成；默认 NavSpec（Navigation Specification，导航规范）为 RNP APCH（Required Navigation Performance Approach，所需导航性能进近），XTT（Cross Track Tolerance，横向容差）为 1.00，可选 A-RNP（Advanced Required Navigation Performance，高级所需导航性能）0.30 或 helicopter（helicopter，直升机）RNP 0.3（Required Navigation Performance 0.3，所需导航性能 0.3）。Basic T（Basic T，基础 T 型）和 HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）属于这一层结构。见 §1-3-1(d)、§3-1-3。fileciteturn0file0

### 5.2.2 可视化重点

Initial Segment（Initial Segment，初始进近航段）的关键不是“画出三条进近路”，而是要画出**结构关系**：

- 左/右 IAF（Initial Approach Fix，初始进近定位点）是否组成 Basic T（Basic T，基础 T 型）；
- IF（Intermediate Fix，中间定位点）/IAF（Initial Approach Fix，初始进近定位点）是否同时承担 HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）；
- holding（holding，等待）是否与后续 TF（Track-to-Fix，航迹到定位点）或 RF（Radius-to-Fix，半径到定位点）切线相接。  

### 5.2.3 建议画法

- 左右初始分支使用同一色相不同明度；
- HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）不要画成普通 racetrack（racetrack，赛道形），而要单独标记“course reversal（course reversal，程序反向）”；
- holding（holding，等待）入口箭头、出航段、入航段与后续 leg（Leg，航迹腿）切线关系应可见；
- 若后续是 RF（Radius-to-Fix，半径到定位点），则 HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）终止方向与 RF（Radius-to-Fix，半径到定位点）起点切线关系需要在 2D（二维，Two-Dimensional）plan view（plan view，平面视图）中明确显示。  

### 5.2.4 不建议的画法

- 不要把 HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）和普通 branch（Branch，分支）混成一条线；
- 不要把初始 segment（segment，航段）与 intermediate segment（intermediate segment，中间进近航段）在视觉上等同。  

---

## 5.3 Intermediate Segment（Intermediate Segment，中间进近航段）

### 5.3.1 几何来源与规则

Intermediate Segment（Intermediate Segment，中间进近航段）是本课题中最需要被重新认真表达的段。  

根据 §1-3-1(e) 与 §3-1-4：

- 可由一个或多个 TF（Track-to-Fix，航迹到定位点）或 RF（Radius-to-Fix，半径到定位点）leg（Leg，航迹腿）组成；
- 末段通向 PFAF（Precise Final Approach Fix，精确最后进近定位点）；
- 对于 RNAV（Area Navigation）(GPS) approach（Global Positioning System approach，全球定位系统进近），在 PFAF（Precise Final Approach Fix，精确最后进近定位点）处，ATT（Along-Track Tolerance，沿航迹容差）按 applicable final navigation accuracy（applicable final navigation accuracy，适用最后进近导航精度）处理；
- intermediate-to-final（intermediate-to-final，中间进近到最后进近）连接不是简单“相接”，而是一个 taper（taper，收敛）/offset construction（offset construction，偏置构造）问题。  

特别是 §3-1-4(d) 到 §3-1-4(d)(4)-(6) 给了非常明确的可视化依据：

1. **LNAV（Lateral Navigation，横向导航）/LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）**：primary area（primary area，主区）与 secondary area（secondary area，次区）从 PFAF（Precise Final Approach Fix，精确最后进近定位点）前 2 NM（Nautical Mile，海里）到后 1 NM（Nautical Mile，海里）均匀连接；
2. **LP（Localizer Performance，航向道性能）/LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）**：在同样的前 2 NM（Nautical Mile，海里）到 PFAF（Precise Final Approach Fix，精确最后进近定位点）后 1 NM（Nautical Mile，海里）区间连接到 final boundary（final boundary，最后进近边界）或 X/OCS（Obstacle Clearance Surface，障碍物净空面）边界；
3. IF（Intermediate Fix，中间定位点）转弯若导致 standard FB turn construction（standard FB turn construction，标准提前转弯构造）超出正常宽度，可用直线连接 inside turn boundary（inside turn boundary，内侧转弯边界）；
4. 若 intermediate course（intermediate course，中间进近航向）与 final course（final course，最后进近航向）不共线，则必须执行 offset construction（offset construction，偏置构造），不是随便把线接上。  

见 §3-1-4(d) 及 Figures 3-1-1 到 3-1-6。fileciteturn0file0

### 5.3.2 可视化要求

Intermediate Segment（Intermediate Segment，中间进近航段）建议拆成三个同时显示的对象：

1. **中间进近中心线**；
2. **PFAF（Precise Final Approach Fix，精确最后进近定位点）前 2 NM（Nautical Mile，海里）至后 1 NM（Nautical Mile，海里）的 taper zone（taper zone，收敛区）**；
3. **若存在 offset construction（offset construction，偏置构造），则显示专门的 inside/outside connection geometry（inside/outside connection geometry，内外侧连接几何）**。  

### 5.3.3 建议画法

- 用一层较细中心线画 leg（Leg，航迹腿）；
- 另用半透明带显式画出 taper zone（taper zone，收敛区）；
- 若发生 offset construction（offset construction，偏置构造），在 2D（二维，Two-Dimensional）plan view（plan view，平面视图）中把 line A、line B、连接线、切线弧明确画出；
- 在 3D（三维，Three-Dimensional）中不必把辅助构造线常显，但应允许“construction debug（construction debug，构造调试）”开关。  

### 5.3.4 对 thesis（Thesis，论文）最关键的原因

很多 trajectory prediction（trajectory prediction，轨迹预测）模型在进入 final（Final，最后进近）前会出现“过早对正”或“横向捷径”，这恰好就体现在 intermediate-to-final（intermediate-to-final，中间进近到最后进近）连接区。  

因此，中间进近不能只画中心线；**必须把 taper（taper，收敛）与 offset construction（offset construction，偏置构造）显示出来**，否则无法判断模型偏差究竟发生在“final（Final，最后进近）内部”，还是“进入 final（Final，最后进近）之前”。

---

## 5.4 Final Segment（Final Segment，最后进近航段）：必须按 procedure type（procedure type，程序类型）区分画法

## 5.4.1 统一原则

Final Segment（Final Segment，最后进近航段）是验证最重要的一段，但它**不是单一模板**。至少要分四类：

1. LNAV（Lateral Navigation，横向导航）；
2. LP（Localizer Performance，航向道性能）；
3. LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）；
4. LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）；
5. 另外若支持 RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近），final（Final，最后进近）还要单独建模。  

## 5.4.2 RNAV（Area Navigation）(GPS) final（Final，最后进近）总体限制

根据 §3-1-5：

- RNAV（Area Navigation）(GPS) final segment（final segment，最后进近航段）由一个或多个 **TF（Track-to-Fix，航迹到定位点）** leg（Leg，航迹腿）构成；
- **RF（Radius-to-Fix，半径到定位点）leg（Leg，航迹腿）和 TF（Track-to-Fix，航迹到定位点）turn（turn，转弯）在 final（Final，最后进近）内不允许**；
- secondary area（secondary area，次区）适用。  

这意味着：**如果你的 procedure（Procedure）数据里在 final（Final，最后进近）内部仍显示 RF（Radius-to-Fix，半径到定位点）或明显转折，渲染层必须将其标为异常或前处理错误，而不是直接画出来。**fileciteturn0file0

---

### 5.4.3 LNAV（Lateral Navigation，横向导航）Final（Final，最后进近）

根据 §3-2：

- OEA（Obstacle Evaluation Area，障碍物评估区）从 PFAF（Precise Final Approach Fix，精确最后进近定位点）前 0.3 NM（Nautical Mile，海里）开始，到 LTP（Landing Threshold Point，着陆入口点）/FTP（Fictitious Threshold Point，虚拟入口点）后 0.3 NM（Nautical Mile，海里）结束；
- 从 PFAF（Precise Final Approach Fix，精确最后进近定位点）前 0.3 NM（Nautical Mile，海里）到后 1 NM（Nautical Mile，海里）存在 taper（taper，收敛）连接；
- 从 taper（taper，收敛）结束后到 LTP（Landing Threshold Point，着陆入口点）/FTP（Fictitious Threshold Point，虚拟入口点）后 0.3 NM（Nautical Mile，海里），primary area（primary area，主区）固定为中心线两侧 ±0.6 NM（Nautical Mile，海里），secondary area（secondary area，次区）每侧 0.3 NM（Nautical Mile，海里）。见 §3-2-3。fileciteturn0file0

- taper（taper，收敛）宽度必须按 Formula 3-2-1 从 taper end（PFAF 后 1 NM）向回计算：`primary half-width = 0.6 + 1.4 * Dtaper / 3`，`secondary area width = 0.3 + 0.7 * Dtaper / 3`。因此 final OEA 在 PFAF 前 0.3 NM 附近更宽，向跑道方向收敛到 PFAF 后 1 NM；之后到跑道入口后 0.3 NM 为固定宽度，并不是整段 final 一直缩窄。

#### 建议画法

- **中心线**：最粗、最稳定；
- **primary ribbon（primary ribbon，主区带）**：明确画成固定宽度主体；
- **secondary ribbon（secondary ribbon，次区带）**：画成外侧浅色带；
- **PFAF（Precise Final Approach Fix，精确最后进近定位点）前 0.3 NM（Nautical Mile，海里）到后 1 NM（Nautical Mile，海里）的 taper（taper，收敛）** 必须可见；
- **MAP（Missed Approach Point，复飞点）/LTP（Landing Threshold Point，着陆入口点）** 要显式显示，不能被带状面遮住。  

#### 研究价值

这套画法可以直接回答：

- 预测轨迹是否在 final（Final，最后进近）中仍左右摆动；
- 模型是否在接近 runway threshold（runway threshold，跑道入口）前才突然收敛到中心线；
- 轨迹偏差是在 primary area（primary area，主区）内，还是已经进入 secondary area（secondary area，次区）甚至外部。  

---

### 5.4.4 LP（Localizer Performance，航向道性能）Final（Final，最后进近）

LP（Localizer Performance，航向道性能）final（Final，最后进近）的横向几何不是 LNAV（Lateral Navigation，横向导航）的常数宽度版本，而是基于阈值附近固定值并向外扩展。根据 §3-2-3(b)：

- primary area（primary area，主区）在 LTP（Landing Threshold Point，着陆入口点）附近为 700 ft（foot，英尺）半宽；
- secondary area（secondary area，次区）在 LTP（Landing Threshold Point，着陆入口点）附近为 300 ft（foot，英尺）宽；
- 再按公式向外扩展。  

#### 建议画法

- 不要偷懒复用 LNAV（Lateral Navigation，横向导航）带状模板；
- 用“近阈值窄、向外逐步扩张”的 ribbon（Ribbon，带状包络）表达；
- 阈值附近横向精度更高，应在视觉上体现“收窄到 runway-aligned channel（runway-aligned channel，跑道对正通道）”。  

---

### 5.4.5 LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）Final（Final，最后进近）

根据 §3-3：

- 横向 OEA（Obstacle Evaluation Area，障碍物评估区）沿用 LNAV（Lateral Navigation，横向导航）尺寸逻辑；
- 但 obstacle clearance（obstacle clearance，障碍物净空）要同时考虑 level OCS（level OCS，水平净空面）与 sloping OCS（sloping OCS，倾斜净空面）；
- DA（Decision Altitude，决断高度）受最低 HAT（Height Above Touchdown，接地区以上高度）、offset（offset，偏置）、控制障碍物共同约束。  

#### 建议画法

LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）final（Final，最后进近）在 3D（三维，Three-Dimensional）里应当有两层：

1. **横向 ribbon（Ribbon，带状包络）**：与 LNAV（Lateral Navigation，横向导航）相同；
2. **垂向 OCS（Obstacle Clearance Surface，障碍物净空面）**：
   - 近阈值水平面；
   - 向外上升的 sloping OCS（sloping OCS，倾斜净空面）；
   - glidepath（Glidepath，下滑路径）单独作为细线显示。  

#### 为什么这比 tunnel（Tunnel，通道体）好

因为 LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）的核心不是“飞机应在一个厚管里飞”，而是：

- 横向要满足 OEA（Obstacle Evaluation Area，障碍物评估区）；
- 垂向要相对 glidepath（Glidepath，下滑路径）与 OCS（Obstacle Clearance Surface，障碍物净空面）合理。  

用 ribbon（Ribbon，带状包络）+ sloping surface（sloping surface，倾斜评估面）能把这两个问题分开看清。  

---

### 5.4.6 LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）Final（Final，最后进近）

根据 §3-4：

- OEA（Obstacle Evaluation Area，障碍物评估区）从 LTP（Landing Threshold Point，着陆入口点）前 200 ft（foot，英尺）起始，向外扩展到 PFAF（Precise Final Approach Fix，精确最后进近定位点）后 40 m（meter，米）；
- OCS（Obstacle Clearance Surface，障碍物净空面）由 **W/X/Y surfaces（W/X/Y surfaces，W/X/Y 评估面）** 组成；
- W surface（W surface，W 面）沿航迹上升，X surface（X surface，X 面）和 Y surface（Y surface，Y 面）在横向继续上升。见 §3-4-3、§3-4-4。fileciteturn0file0

#### 建议画法

LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）final（Final，最后进近）不要用 tunnel（Tunnel，通道体）；建议改成：

- **中心 glidepath（Glidepath，下滑路径）**：一条明确细线；
- **W surface（W surface，W 面）**：主透明面；
- **X surface（X surface，X 面）**：左右两侧稍高面；
- **Y surface（Y surface，Y 面）**：再外侧过渡面；
- obstacle（Obstacle，障碍物）若穿入 X/Y surface（X/Y surface，X/Y 面），用垂直投影线与 penetration marker（penetration marker，穿透标记）指出。  

#### 对论文展示的好处

- 你可以明确展示“某个障碍物为什么不是直接和中心线比较，而是先按 X/Y surface（X/Y surface，X/Y 面）做横向高度调整”；
- 这比 tunnel（Tunnel，通道体）更符合规范文本，也更容易答辩解释。  

---

## 5.5 RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）Final（Final，最后进近）应单独实现

如果你的 AeroViz-4D 计划支持 RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近），不要把它并入 RNAV（Area Navigation）(GPS) final（Final，最后进近）模板。  

根据 Chapter 4（Chapter 4，第四章）：

- initial（initial，初始航段）/intermediate（intermediate，中间航段）/final（final，最后进近航段）都可使用更小的 RNP（Required Navigation Performance，所需导航性能）值；
- **secondary area（secondary area，次区）不适用**；
- turns in FAS（turns in FAS，最后进近航段内转弯）可存在，但若 final（Final，最后进近）中需要转弯，应采用 RF（Radius-to-Fix，半径到定位点）leg（Leg，航迹腿），并在 minimum FROP（Final Rollout Point，最小最后拉直点）距离前转入符合对正要求的 TF（Track-to-Fix，航迹到定位点）leg（Leg，航迹腿）；
- OEA（Obstacle Evaluation Area，障碍物评估区）宽度为中心线两侧 2 × XTT（Cross Track Tolerance，横向容差）；
- OCS（Obstacle Clearance Surface，障碍物净空面）基于 VEB（Vertical Error Budget，垂向误差预算）。见 §4-1-1、§4-1-4、§4-2-1 到 §4-2-4。fileciteturn0file0

### 5.5.1 建议画法

- RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）使用独立色系；
- 不要绘制 secondary ribbon（secondary ribbon，次区带）；
- 若 FAS（Final Approach Segment，最后进近航段）含 RF（Radius-to-Fix，半径到定位点）leg（Leg，航迹腿），要把 RF（Radius-to-Fix，半径到定位点）与 minimum FROP（Final Rollout Point，最小最后拉直点）标出来；
- 在属性面板明确提示“FO（Fly-over，飞越后转弯） fixes（fixes，定位点） that require turn construction（turn construction，转弯构造） are not authorized（not authorized，不允许）”。  

---

## 5.6 Missed Approach（Missed Approach，复飞）必须拆成 Section 1（Section 1，第一段）与 Section 2（Section 2，第二段）

这是现有很多研究原型最容易做错的地方。  

根据 Chapter 3（Chapter 3，第三章）§3-5、§3-6、§3-7：

1. missed approach（Missed Approach，复飞）不能只画成从 MAP（Missed Approach Point，复飞点）往外的一条线；
2. 首 leg（Leg，航迹腿）是 **CA（Course-to-Altitude，航向到高度）**，必须沿 FAC（Final Approach Course，最后进近航道）延长线；
3. CA（Course-to-Altitude，航向到高度）后必须接 **DF（Direct-to-Fix，直飞到定位点）**；
4. section 1（Section 1，第一段）与 section 2（Section 2，第二段）有不同的宽度与 OCS（Obstacle Clearance Surface，障碍物净空面）构造；
5. turning missed approach（turning missed approach，转弯复飞）还要区分 turn-at-altitude（turn-at-altitude，到达高度再转弯）与 turn-at-fix（turn-at-fix，到定位点再转弯），以及 early/inside turn（early/inside turn，提前/内侧转弯）与 late/outside turn（late/outside turn，延后/外侧转弯）构造。  

### 5.6.1 Section 1（Section 1，第一段）建议画法

#### 非垂直引导程序

根据 §3-6-1：

- section 1（Section 1，第一段）从 final segment（final segment，最后进近航段）ATT（Along-Track Tolerance，沿航迹容差） prior to MAP（prior to MAP，复飞点前）开始；
- 延伸到 SOC（Start of Climb，开始爬升点）或 projected 400 ft（projected 400 ft，投影到 400 英尺）高度点；
- 存在 FSL（Flat Surface Length，平面长度）与 extension（extension，延伸）；
- width（width，宽度）按 15°（degree，度）splay（splay，张角扩展）逐步达到 2 XTT（Cross Track Tolerance，横向容差）/3 XTT（Cross Track Tolerance，横向容差）；
- OCS（Obstacle Clearance Surface，障碍物净空面）由 flat surface（flat surface，平面）与 sloping extension（sloping extension，倾斜延伸面）构成。  

#### 渲染建议

- section 1（Section 1，第一段）单独着色；
- 平面部分与倾斜部分颜色不同；
- line C-D、line J-K、line A-B 若在 debug mode（debug mode，调试模式）中可显示，会非常有助于验证。  

### 5.6.2 LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）Section 1（Section 1，第一段）

根据 §3-6-2：

- 仍沿用 non-vertically guided（non-vertically guided，无垂直引导）结构；
- 但 FSL（Flat Surface Length，平面长度）按 15 秒规则；
- HMAS（Height at Missed Approach Surface，复飞面起始高度）基于 base DA（base DA，基准决断高度）与 ROC（Required Obstacle Clearance，所需障碍物净空）计算。  

#### 渲染建议

- 仍维持 section 1（Section 1，第一段）分段显示；
- 但 vertical panel（vertical panel，垂直剖面面板）中要显示 flat-to-slope（flat-to-slope，平面到斜面）转换点。  

### 5.6.3 LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）Section 1（Section 1，第一段）

根据 §3-6-3：

- section 1（Section 1，第一段）从 base DA（base DA，基准决断高度）开始到 SOC（Start of Climb，开始爬升点）结束；
- 再分成 section 1a（Section 1a，第一段 a）与 section 1b（Section 1b，第一段 b）；
- section 1a（Section 1a，第一段 a）延续 FAS（Final Approach Segment，最后进近航段）OCS（Obstacle Clearance Surface，障碍物净空面）；
- section 1b（Section 1b，第一段 b）则扩展成 1bW/1bX/1bY surfaces（1bW/1bX/1bY surfaces，1bW/1bX/1bY 面）。  

#### 渲染建议

- 不建议简化成一段统一色体；
- 至少要显示：
  - 1a continuation（1a continuation，1a 延续面）；
  - 1bW（1bW，1bW 面）；
  - 1bX/1bY（1bX/1bY，1bX/1bY 面）横向上升结构。  

### 5.6.4 Section 2（Section 2，第二段）建议画法

根据 §3-7-1 与 §3-7-2：

- section 2（Section 2，第二段）从 section 1（Section 1，第一段）末端开始；
- 直复飞时，OEA（Obstacle Evaluation Area，障碍物评估区）按 15°（degree，度）splay（splay，张角扩展）直到 full width（full width，全宽）；
- turn-at-altitude（turn-at-altitude，到达高度再转弯）与 turn-at-fix（turn-at-fix，到定位点再转弯）构造不同；
- turning missed approach（turning missed approach，转弯复飞）需要显示 TIA（Turn Initiation Area，转弯起始区）、early turn baseline（early turn baseline，提前转弯基线）、late turn baseline（late turn baseline，延后转弯基线）、wind spiral（wind spiral，风螺旋）等。  

#### 渲染建议

- 直复飞 section 2（Section 2，第二段）可以用简化 ribbon（Ribbon，带状包络）+ OCS（Obstacle Clearance Surface，障碍物净空面）表示；
- turning missed approach（turning missed approach，转弯复飞）建议默认只显示 nominal path（nominal path，名义航迹）与最终 OEA（Obstacle Evaluation Area，障碍物评估区）外边界；
- 但必须提供 debug mode（debug mode，调试模式）显示 early/late baselines（early/late baselines，提前/延后基线）和 wind spirals（wind spirals，风螺旋），否则难以验证几何实现是否正确。  

---

## 6. Leg（Leg，航迹腿）级别呈现规范：不是所有 leg（Leg，航迹腿）都能被同样处理

## 6.1 TF（Track-to-Fix，航迹到定位点）

### 呈现规则
- 必须是严格 geodesic line（geodesic line，测地线）；
- 起终点 fix（Fix，航路点）明确；
- 若前后有 ATT（Along-Track Tolerance，沿航迹容差）影响，端点前后要留出 ATT（Along-Track Tolerance，沿航迹容差）可视范围。  

### 不可接受的实现
- 不要仅以 web mercator（web mercator，网络墨卡托）屏幕空间直线替代；
- 不要把 TF（Track-to-Fix，航迹到定位点）和 DF（Direct-to-Fix，直飞到定位点）混同。  

## 6.2 RF（Radius-to-Fix，半径到定位点）

### 呈现规则
- 用真实圆弧；
- 支持显示 turn center（turn center，转弯中心）、radius（radius，半径）、arc angle（arc angle，圆弧角）、arc length（arc length，弧长）；
- OEA（Obstacle Evaluation Area，障碍物评估区）边界应为平行弧，而不是沿折线偏移；
- 当 RF（Radius-to-Fix，半径到定位点）用于 intermediate（intermediate，中间进近）或 RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）final（Final，最后进近）时，必须保持 RF（Radius-to-Fix，半径到定位点）身份，不要预先离散成若干 TF（Track-to-Fix，航迹到定位点）段。  

## 6.3 CA（Course-to-Altitude，航向到高度）

### 呈现规则
- 只在 missed approach（Missed Approach，复飞）首段出现时高亮；
- 不能把它当成普通 TF（Track-to-Fix，航迹到定位点）leg（Leg，航迹腿）；
- 需要同时画出“按航向延伸的水平路径”和“达到转弯高度前的垂向爬升要求”。  

## 6.4 DF（Direct-to-Fix，直飞到定位点）

### 呈现规则
- 必须和 CA（Course-to-Altitude，航向到高度）后接关系一起看；
- 在 turning missed approach（turning missed approach，转弯复飞）里，DF（Direct-to-Fix，直飞到定位点）常决定 inside/outside turn（inside/outside turn，内侧/外侧转弯）构造；
- plan view（plan view，平面视图）中，DF（Direct-to-Fix，直飞到定位点）需要和 outbound TF（outbound TF，离场 TF 航迹腿）分开着色。  

## 6.5 FB（Fly-by，提前转弯）/FO（Fly-over，飞越后转弯）

### 呈现规则
- fix（Fix，航路点）符号必须不同；
- FB（Fly-by，提前转弯）要强调 DTA（Distance of Turn Anticipation，转弯提前距离）；
- FO（Fly-over，飞越后转弯）要强调 reaction-and-roll distance（reaction-and-roll distance，反应与滚转距离）与转弯后扩张区。  

### 对研究验证的意义
- 预测轨迹往往会在这些点显示出“过早转弯”或“过迟转弯”；
- 若你不把 FB（Fly-by，提前转弯）/FO（Fly-over，飞越后转弯）区分显示，模型误差会被误判成“路径偏差”而不是“转弯模式偏差”。

---

## 7. 2D（二位，Two-Dimensional）视图如何服务 RNAV（Area Navigation）procedure（Procedure）可视化

## 7.1 Plan View（Plan View，平面视图）

Plan View（Plan View，平面视图）是 RNAV（Area Navigation）procedure（Procedure）几何表达的主场。应重点显示：

1. branch（Branch，分支）层次；
2. segment（Segment，航段）边界；
3. TF（Track-to-Fix，航迹到定位点）/RF（Radius-to-Fix，半径到定位点）/CA（Course-to-Altitude，航向到高度）/DF（Direct-to-Fix，直飞到定位点）差异；
4. taper（taper，收敛）/offset construction（offset construction，偏置构造）/splay（splay，张角扩展）；
5. primary area（primary area，主区）/secondary area（secondary area，次区）/W-X-Y surfaces（W-X-Y surfaces，W-X-Y 面）的投影边界。  

### 建议

- Plan View（Plan View，平面视图）默认开启 OEA（Obstacle Evaluation Area，障碍物评估区）边界；
- 允许切换“只看 nominal path（nominal path，名义航迹）”与“看 full protection（full protection，全保护几何）”；
- PFAF（Precise Final Approach Fix，精确最后进近定位点）附近的前 2 NM（Nautical Mile，海里）至后 1 NM（Nautical Mile，海里）连接区要永远可见。  

## 7.2 Vertical Profile（Vertical Profile，垂直剖面）

Vertical Profile（Vertical Profile，垂直剖面）不适合表达全部 lateral rule（lateral rule，横向规则），但非常适合表达：

- intermediate altitude（intermediate altitude，中间进近高度）到 glidepath intercept（glidepath intercept，下滑道截获点）；
- PFAF（Precise Final Approach Fix，精确最后进近定位点）位置；
- stepdown fixes（stepdown fixes，梯降定位点）；
- TCH（Threshold Crossing Height，入口越障高）/DA（Decision Altitude，决断高度）/MDA（Minimum Descent Altitude，最低下降高度）；
- missed approach（Missed Approach，复飞）section 1（Section 1，第一段）平面与倾斜面转换。  

### 建议

- vertical profile（vertical profile，垂直剖面）中不要塞入过多 lateral label（lateral label，横向标签）；
- 只显示与当前 active segment（active segment，当前激活航段）相关的垂向对象；
- LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）时优先显示 W surface（W surface，W 面）与 glidepath（Glidepath，下滑路径）关系。  

---

## 8. 交互规则：用户应当如何“读”一个 RNAV（Area Navigation）procedure（Procedure）

## 8.1 三个模式

### 模式 A：Procedure Structure（Procedure Structure，程序结构）
- 只看 branch（Branch，分支）、segment（Segment，航段）、leg（Leg，航迹腿）、fix（Fix，航路点）；
- 不看 OEA（Obstacle Evaluation Area，障碍物评估区）；
- 用于熟悉 procedure（Procedure）拓扑。  

### 模式 B：Protected Geometry（Protected Geometry，保护几何）
- 显示 primary area（primary area，主区）、secondary area（secondary area，次区）、W/X/Y surfaces（W/X/Y surfaces，W/X/Y 面）、missed approach（Missed Approach，复飞）section（Section，分段）面；
- 用于验证 published geometry（published geometry，公布几何）自身实现。  

### 模式 C：Trajectory Validation（Trajectory Validation，轨迹验证）
- 显示轨迹与当前 active segment（active segment，当前激活航段）的相对关系；
- 当前时刻只高亮 aircraft（aircraft，航空器）所在 segment（Segment，航段）；
- 超出 primary area（primary area，主区）/secondary area（secondary area，次区）时用分段着色，不做简单整条报警。  

## 8.2 高亮策略

- 高亮必须以 segment（Segment，航段）为单位；
- 当 aircraft（aircraft，航空器）处于 PFAF（Precise Final Approach Fix，精确最后进近定位点）连接区时，应同时高亮 intermediate（intermediate，中间进近）末段和 final（Final，最后进近）起段；
- 当 aircraft（aircraft，航空器）进入 missed approach（Missed Approach，复飞）时，先高亮 CA（Course-to-Altitude，航向到高度），后再切换到 DF（Direct-to-Fix，直飞到定位点）与 section 2（Section 2，第二段）。  

---

## 9. 实现优先级：先把哪些 segment（Segment，航段）做好

### 第一优先级

1. intermediate-to-final（intermediate-to-final，中间进近到最后进近）连接区；
2. LNAV（Lateral Navigation，横向导航） final（Final，最后进近）primary/secondary ribbons（primary/secondary ribbons，主区/次区带）；
3. missed approach（Missed Approach，复飞）section 1（Section 1，第一段）与 section 2（Section 2，第二段）拆分。  

### 第二优先级

4. HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）与 Basic T（Basic T，基础 T 型）初始结构；
5. RF（Radius-to-Fix，半径到定位点）leg（Leg，航迹腿）真圆弧与 parallel OEA（parallel OEA，平行评估区）构造；
6. LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统） W/X/Y surfaces（W/X/Y surfaces，W/X/Y 面）。  

### 第三优先级

7. turning missed approach（turning missed approach，转弯复飞）debug mode（debug mode，调试模式）；
8. RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）专门模板。  

---

## 10. 最终建议：AeroViz-4D 应该怎样重新定义 RNAV（Area Navigation）procedure（Procedure）呈现

可以把最终设计浓缩成一句话：

> **RNAV（Area Navigation）procedure（Procedure）在 AeroViz-4D 中，不应再被画成 fixes（fixes，航路点）与折线，而应被实现为“segment-aware（segment-aware，航段感知）、leg-typed（leg-typed，航迹腿分型）、protection-visible（protection-visible，保护区可见）、vertical-surface-aware（vertical-surface-aware，垂向面感知）”的可验证几何对象。**

具体落地为：

1. 以 segment（Segment，航段）为主对象，不以 fix（Fix，航路点）为主对象；
2. 以 ribbon（Ribbon，带状包络）+ surface（Surface，评估面）替代 tunnel（Tunnel，通道体）；
3. intermediate-to-final（intermediate-to-final，中间进近到最后进近）连接区必须专门实现；
4. final（Final，最后进近）必须按 LNAV（Lateral Navigation，横向导航）/LP（Localizer Performance，航向道性能）/LNAV-VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）/LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）/RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）区分模板；
5. missed approach（Missed Approach，复飞）必须按 CA（Course-to-Altitude，航向到高度）+ DF（Direct-to-Fix，直飞到定位点）+ section 1（Section 1，第一段）+ section 2（Section 2，第二段）表达。  

---

## 11. 参考依据清单（便于逐条核对）

下面这张清单只保留与“RNAV（Area Navigation）procedure（Procedure）可视化与 segment（Segment，航段）呈现”直接相关的条目，便于你对照 PDF（Portable Document Format，便携式文档格式）验证。

### 11.1 通用构造规则

1. **§1-2-5**：OEA（Obstacle Evaluation Area，障碍物评估区）与 flight path construction（flight path construction，飞行路径构造）总则（PDF（Portable Document Format，便携式文档格式）第 8-27 页）  
   - geodesic course（geodesic course，测地航迹）  
   - XTT（Cross Track Tolerance，横向容差）/ATT（Along-Track Tolerance，沿航迹容差）  
   - FB（Fly-by，提前转弯）/FO（Fly-over，飞越后转弯）/RF（Radius-to-Fix，半径到定位点）构造  
   - width change（width change，宽度变化）与 30 NM（Nautical Mile，海里）模式切换  
   - 对应 PDF（Portable Document Format，便携式文档格式）前部 Chapter 1（Chapter 1，第一章）相关页。fileciteturn0file0

2. **§1-3-1(d)**：initial segment（initial segment，初始进近航段）（PDF（Portable Document Format，便携式文档格式）第 29-30 页）  
   - Basic T（Basic T，基础 T 型）  
   - HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）  
   - holding（holding，等待）与后续 TF（Track-to-Fix，航迹到定位点）/RF（Radius-to-Fix，半径到定位点）关系。fileciteturn0file0

3. **§1-3-1(e)**：intermediate segment（intermediate segment，中间进近航段）（PDF（Portable Document Format，便携式文档格式）第 30-31 页）  
   - 末 leg（Leg，航迹腿）进入 PFAF（Precise Final Approach Fix，精确最后进近定位点）的长度与对正要求。fileciteturn0file0

4. **§1-3-1(f)**：final segment（final segment，最后进近航段）垂向路径共性（PDF（Portable Document Format，便携式文档格式）第 31-33 页）  
   - stepdown fix（stepdown fix，梯降定位点）  
   - TCH（Threshold Crossing Height，入口越障高）  
   - PFAF（Precise Final Approach Fix，精确最后进近定位点）定位。fileciteturn0file0

### 11.2 RNAV（Area Navigation）(GPS) approach（Global Positioning System approach，全球定位系统进近）

5. **§3-1-2**：feeder segment（feeder segment，引导航段）规则（PDF（Portable Document Format，便携式文档格式）第 51 页）  
6. **§3-1-3**：initial segment（initial segment，初始进近航段）规则（PDF（Portable Document Format，便携式文档格式）第 51 页）  
7. **§3-1-4**：intermediate segment（intermediate segment，中间进近航段）规则（PDF（Portable Document Format，便携式文档格式）第 51-56 页）  
   - PFAF（Precise Final Approach Fix，精确最后进近定位点）连接区  
   - offset construction（offset construction，偏置构造）  
   - Figures 3-1-1 ~ 3-1-6。fileciteturn0file0

8. **§3-1-5**：final segment（final segment，最后进近航段）总限制（PDF（Portable Document Format，便携式文档格式）第 56 页）  
   - final（Final，最后进近）只允许 TF（Track-to-Fix，航迹到定位点）  
   - RF（Radius-to-Fix，半径到定位点）与 TF turn（TF turn，TF 转弯）不允许。fileciteturn0file0

9. **§3-2**：LNAV（Lateral Navigation，横向导航）/LP（Localizer Performance，航向道性能） non-vertically guided（non-vertically guided，无垂直引导） final（Final，最后进近）（PDF（Portable Document Format，便携式文档格式）第 59-63 页）  
   - LNAV（Lateral Navigation，横向导航） final OEA（Obstacle Evaluation Area，障碍物评估区）长度与宽度  
   - LP（Localizer Performance，航向道性能） final OEA（Obstacle Evaluation Area，障碍物评估区）宽度公式。fileciteturn0file0

10. **§3-3**：LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航） final（Final，最后进近）（PDF（Portable Document Format，便携式文档格式）第 64-72 页）  
    - sloping OCS（sloping OCS，倾斜净空面）  
    - level OCS（level OCS，水平净空面）  
    - DA（Decision Altitude，决断高度）逻辑。fileciteturn0file0

11. **§3-4**：LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统） final（Final，最后进近）（PDF（Portable Document Format，便携式文档格式）第 73-83 页）  
    - W/X/Y surfaces（W/X/Y surfaces，W/X/Y 面）  
    - OEA（Obstacle Evaluation Area，障碍物评估区）扩展逻辑。fileciteturn0file0

12. **§3-5**：missed approach（Missed Approach，复飞）总则（PDF（Portable Document Format，便携式文档格式）第 84-85 页）  
    - CA（Course-to-Altitude，航向到高度）首 leg（Leg，航迹腿）  
    - DF（Direct-to-Fix，直飞到定位点）后接关系。fileciteturn0file0

13. **§3-6**：missed approach section 1（missed approach section 1，复飞第一段）（PDF（Portable Document Format，便携式文档格式）第 86-93 页）  
    - LNAV（Lateral Navigation，横向导航）/LP（Localizer Performance，航向道性能）  
    - LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）  
    - LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）。fileciteturn0file0

14. **§3-7**：missed approach section 2（missed approach section 2，复飞第二段）（PDF（Portable Document Format，便携式文档格式）第 94-115 页）  
    - straight missed approach（straight missed approach，直复飞）  
    - turning missed approach（turning missed approach，转弯复飞）  
    - TIA（Turn Initiation Area，转弯起始区）  
    - wind spiral（wind spiral，风螺旋）与 inside/outside turn（inside/outside turn，内/外侧转弯）构造。fileciteturn0file0

### 11.3 RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）

15. **§4-1-1**：RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）总设计原则（PDF（Portable Document Format，便携式文档格式）第 116-117 页）  
    - RF（Radius-to-Fix，半径到定位点）优先  
    - FO（Fly-over，飞越后转弯） turn construction（turn construction，转弯构造）不允许。fileciteturn0file0

16. **§4-1-2 ~ §4-1-4**：feeder（feeder，引导）/initial（initial，初始）/intermediate（intermediate，中间）规则（PDF（Portable Document Format，便携式文档格式）第 117-118 页）  
    - secondary area（secondary area，次区）差异  
    - PFAF（Precise Final Approach Fix，精确最后进近定位点）前 FB（Fly-by，提前转弯）限制。fileciteturn0file0

17. **§4-2-1 ~ §4-2-4**：RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近） final（Final，最后进近）（PDF（Portable Document Format，便携式文档格式）第 118-121 页）  
    - FROP（Final Rollout Point，最后拉直点）  
    - 2 × XTT（Cross Track Tolerance，横向容差） final width（final width，最后进近宽度）  
    - VEB（Vertical Error Budget，垂向误差预算）与 OCS（Obstacle Clearance Surface，障碍物净空面）。fileciteturn0file0

---

## 12. 给实现阶段的直接落地建议

如果你马上就要重构现有 RNAV（Area Navigation）procedure（Procedure）可视化，我建议按下面顺序改：

1. **先把数据模型改成 segment（Segment，航段）-first（first，优先）**；
2. **再实现 intermediate-to-final（intermediate-to-final，中间进近到最后进近）连接区**；
3. **再把 final（Final，最后进近）从 tunnel（Tunnel，通道体）改成 ribbon（Ribbon，带状包络）+ surface（Surface，评估面）**；
4. **最后再补 missed approach（Missed Approach，复飞）section 1（Section 1，第一段）/section 2（Section 2，第二段）与 debug mode（debug mode，调试模式）**。  

这样改完之后，AeroViz-4D 的 procedure（Procedure）呈现才会真正从“演示型几何”升级为“可验证几何”。


---

## 13. 数据 Schema（Schema，数据模式）：把“程序文本/点线数据”升级为“可构造、可调试、可验证”的几何数据

这一部分的目标不是再定义一套抽象数据库，而是给前端与几何模块一个**可直接落地的中间层**。  
建议把数据分成四层：

1. **Source Layer（Source Layer，源数据层）**：来自 CIFP（Coded Instrument Flight Procedures，编码仪表飞行程序）、机场数据、跑道数据、障碍物数据、chart metadata（chart metadata，图表元数据）；
2. **Procedure Semantic Layer（Procedure Semantic Layer，程序语义层）**：把原始记录归一化成 procedure（Procedure，程序）、branch（Branch，分支）、segment（Segment，航段）、leg（Leg，航迹腿）、constraint（Constraint，约束）；
3. **Derived Geometry Layer（Derived Geometry Layer，派生几何层）**：几何算法输出的 centerline（Centerline，中心线）、envelope（Envelope，包络）、surface（Surface，评估面）、debug primitives（debug primitives，调试图元）；
4. **Validation Layer（Validation Layer，验证层）**：轨迹投影、偏差、包络内外判定、事件标记、截图/报告记录。  

### 13.1 Schema（Schema，数据模式）设计原则

1. **procedure（Procedure，程序）与 renderable（renderable，可渲染对象）分离**  
   原始程序对象只保存语义与参数，不保存 Cesium（CesiumJS，三维地球引擎）实体；
2. **segment（Segment，航段）是主键级对象**  
   一切渲染、过滤、统计、验证都应以 segment（Segment，航段）为主，不以 fix（Fix，航路点）为主；
3. **每一段都必须同时保留“输入参数”和“构造结果”**  
   这样才能做 debug（debug，调试）与复核；
4. **每一种 final（Final，最后进近）/missed approach（Missed Approach，复飞）类型都必须有独立 surface model（surface model，评估面模型）**；
5. **每个几何结果都要能回溯到 FAA（Federal Aviation Administration，美国联邦航空管理局）条款或公式来源**。  

### 13.2 顶层对象建议

```ts
export interface ProcedurePackage {
  packageId: string;
  airportId: string;
  runwayId: string;
  procedureId: string;
  procedureName: string;
  procedureFamily:
    | 'RNAV_GPS'
    | 'RNP_AR_APCH'
    | 'SID'
    | 'STAR';
  sourceMeta: SourceMeta;
  branches: ProcedureBranch[];
  sharedFixes: ProcedureFix[];
  validationConfig: ValidationConfig;
}
```

```ts
export interface SourceMeta {
  cifpCycle: string;
  sourceFiles: string[];
  chartLinks: string[];
  notes?: string[];
  authority: 'FAA_8260_58D';
}
```

### 13.3 Fix（Fix，航路点）与约束对象

```ts
export interface ProcedureFix {
  fixId: string;
  ident: string;
  role?: FixRole[];
  latDeg: number;
  lonDeg: number;
  altFtMsl?: number;
  isFlyOver?: boolean; // FO（Fly-over，飞越后转弯）
  isFlyBy?: boolean;   // FB（Fly-by，提前转弯）
  annotations?: string[];
}

export type FixRole =
  | 'IAF'   // IAF（Initial Approach Fix，初始进近定位点）
  | 'IF'    // IF（Intermediate Fix，中间定位点）
  | 'PFAF'  // PFAF（Precise Final Approach Fix，精确最后进近定位点）
  | 'FAF'   // FAF（Final Approach Fix，最后进近定位点）
  | 'MAP'   // MAP（Missed Approach Point，复飞点）
  | 'MAHF'  // MAHF（Missed Approach Holding Fix，复飞等待定位点）
  | 'RWY'
  | 'FROP'; // FROP（Final Rollout Point，最后拉直点）
```

```ts
export interface AltitudeConstraint {
  kind: 'AT' | 'AT_OR_ABOVE' | 'AT_OR_BELOW' | 'WINDOW';
  minFtMsl?: number;
  maxFtMsl?: number;
  sourceText?: string;
}

export interface SpeedConstraint {
  maxKias?: number; // KIAS（Knots Indicated Airspeed，指示空速节）
  minKias?: number;
  sourceText?: string;
}
```

### 13.4 Leg（Leg，航迹腿）输入语义对象

```ts
export type LegType =
  | 'TF'     // TF（Track-to-Fix，航迹到定位点）
  | 'RF'     // RF（Radius-to-Fix，半径到定位点）
  | 'DF'     // DF（Direct-to-Fix，直飞到定位点）
  | 'CA'     // CA（Course-to-Altitude，航向到高度）
  | 'HM'     // HM（Hold-to-Manual，等待到人工终止）
  | 'HA'     // HA（Hold-to-Altitude，等待到高度）
  | 'HF';    // HF（Hold-to-Fix，等待到定位点）

export interface ProcedureLeg {
  legId: string;
  segmentId: string;
  legType: LegType;
  startFixId?: string;
  endFixId?: string;
  inboundCourseDeg?: number;
  outboundCourseDeg?: number;
  turnDirection?: 'LEFT' | 'RIGHT';
  arcRadiusNm?: number;              // RF（Radius-to-Fix，半径到定位点）半径
  centerLatDeg?: number;             // RF（Radius-to-Fix，半径到定位点）圆心
  centerLonDeg?: number;
  requiredAltitude?: AltitudeConstraint;
  requiredSpeed?: SpeedConstraint;
  navSpecAtLeg: NavSpecCode;
  xttNm: number;                     // XTT（Cross Track Tolerance，横向容差）
  attNm: number;                     // ATT（Along-Track Tolerance，沿航迹容差）
  secondaryEnabled: boolean;
  notes?: string[];
  sourceRefs: SourceRef[];
}
```

```ts
export type NavSpecCode =
  | 'RNAV_1'
  | 'RNAV_2'
  | 'RNP_APCH'
  | 'A_RNP_1'
  | 'A_RNP_0_3'
  | 'RNP_AR_0_3'
  | 'RNP_AR_0_2'
  | 'RNP_AR_0_1';
```

### 13.5 Segment（Segment，航段）语义对象

```ts
export type SegmentType =
  | 'FEEDER'
  | 'INITIAL'
  | 'INTERMEDIATE'
  | 'TRANSITION_ROUTE'
  | 'PROCEDURE_ROUTE'
  | 'FINAL_LNAV'
  | 'FINAL_LP'
  | 'FINAL_LNAV_VNAV'
  | 'FINAL_LPV'
  | 'FINAL_GLS'
  | 'FINAL_RNP_AR'
  | 'MISSED_S1'
  | 'MISSED_S2'
  | 'HOLDING';

export interface ProcedureSegment {
  segmentId: string;
  branchId: string;
  segmentType: SegmentType;
  navSpec: NavSpecCode;
  startFixId?: string;
  endFixId?: string;
  legIds: string[];
  xttNm: number;
  attNm: number;
  secondaryEnabled: boolean;
  widthChangeMode?: 'LINEAR_TAPER' | 'ABRUPT' | 'SPLAY_30' | 'NONE';
  transitionRule?: TransitionRule;
  verticalRule?: VerticalRule;
  constructionFlags?: ConstructionFlags;
  sourceRefs: SourceRef[];
}
```

```ts
export interface TransitionRule {
  kind:
    | 'INTERMEDIATE_TO_FINAL_LNAV'
    | 'INTERMEDIATE_TO_FINAL_LP'
    | 'INTERMEDIATE_TO_FINAL_LNAV_VNAV'
    | 'INTERMEDIATE_TO_FINAL_LPV_GLS'
    | 'RNP_CHANGE_ABRUPT'
    | 'MODE_CHANGE_30NM'
    | 'MISSED_SECTION_SPLIT';
  anchorFixId?: string; // 常用于 PFAF（Precise Final Approach Fix，精确最后进近定位点）
  beforeNm?: number;
  afterNm?: number;
  notes?: string[];
}
```

```ts
export interface VerticalRule {
  kind:
    | 'NONE'
    | 'LEVEL_ROC'
    | 'BARO_GLIDEPATH'
    | 'LPV_GLS_SURFACES'
    | 'MISSED_CLIMB_SURFACE'
    | 'RNP_AR_VERTICAL';
  gpaDeg?: number;      // GPA（Glidepath Angle，下滑路径角）
  tchFt?: number;       // TCH（Threshold Crossing Height，入口越障高）
  mdaFtMsl?: number;    // MDA（Minimum Descent Altitude，最低下降高度）
  daFtMsl?: number;     // DA（Decision Altitude，决断高度）
  climbGradientFtPerNm?: number;
}
```

```ts
export interface ConstructionFlags {
  hasOffsetConstruction?: boolean;
  hasTurnAtIf?: boolean;
  hasRfToPfaf?: boolean;
  isBasicT?: boolean;
  hasHilpt?: boolean; // HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）
  isTurningMissedApproach?: boolean;
  foTurnNotAllowed?: boolean;
}
```

### 13.6 Branch（Branch，分支）与程序图结构

```ts
export interface ProcedureBranch {
  branchId: string;
  runwayId: string;
  branchName: string;
  branchRole:
    | 'LEFT_IAF'
    | 'RIGHT_IAF'
    | 'STRAIGHT_IN'
    | 'MISSED'
    | 'TRANSITION'
    | 'HOLDING';
  segmentIds: string[];
  mergeToBranchId?: string;
  divergesFromBranchId?: string;
}
```

### 13.7 派生几何对象：前端真正渲染的结构

```ts
export interface SegmentGeometryBundle {
  segmentId: string;
  centerline: PolylineGeometry3D;
  stationAxis: StationAxis; // 供 2D（Two-Dimensional，二维）剖面与投影使用
  primaryEnvelope?: LateralEnvelopeGeometry;
  secondaryEnvelope?: LateralEnvelopeGeometry;
  transitionGeometry?: TransitionGeometry;
  verticalSurfaces?: VerticalSurfaceGeometry[];
  debugPrimitives?: DebugPrimitive[];
  qaFootnotes?: string[];
}
```

```ts
export interface PolylineGeometry3D {
  worldPositions: CartesianPoint[];
  geodesicLengthNm: number;
  isArc: boolean;
}

export interface CartesianPoint {
  x: number;
  y: number;
  z: number;
}
```

```ts
export interface LateralEnvelopeGeometry {
  geometryId: string;
  envelopeType: 'PRIMARY' | 'SECONDARY';
  leftBoundary: CartesianPoint[];
  rightBoundary: CartesianPoint[];
  halfWidthNmSamples: WidthSample[];
}

export interface WidthSample {
  stationNm: number;
  halfWidthNm: number;
}
```

```ts
export interface TransitionGeometry {
  transitionId: string;
  kind:
    | 'TAPER'
    | 'OFFSET_CONNECTOR'
    | 'FB_TURN_CONSTRUCTION'
    | 'FO_TURN_CONSTRUCTION'
    | 'RF_PARALLEL_ARC'
    | 'MODE_CHANGE_SPLAY';
  guideLines?: CartesianPoint[][];
  boundaryPolylines?: CartesianPoint[][];
  anchorStationsNm?: number[];
}
```

```ts
export interface VerticalSurfaceGeometry {
  surfaceId: string;
  surfaceType:
    | 'LNAV_LEVEL_OEA'
    | 'LNAV_VNAV_LEVEL_OCS'
    | 'LNAV_VNAV_SLOPING_OCS'
    | 'LPV_W'
    | 'LPV_X'
    | 'LPV_Y'
    | 'MISSED_SECTION1_FLAT'
    | 'MISSED_SECTION1_SLOPING'
    | 'MISSED_SECTION2_OEA'
    | 'RNP_AR_OCS';
  meshVertices: CartesianPoint[];
  meshIndices: number[];
  metadata?: Record<string, number | string | boolean>;
}
```

### 13.8 验证层对象

```ts
export interface TrajectoryValidationRecord {
  recordId: string;
  trajectoryId: string;
  procedureId: string;
  branchId?: string;
  segmentAssessments: SegmentAssessment[];
  globalAssessment: GlobalAssessment;
}

export interface SegmentAssessment {
  segmentId: string;
  alongTrackProgressNm: number[];
  crossTrackErrorNm: number[];
  verticalErrorFt: number[];
  inPrimaryFlags: boolean[];
  inSecondaryFlags: boolean[];
  outsideFlags: boolean[];
  eventMarkers: ValidationEvent[];
}
```

```ts
export interface ValidationEvent {
  eventType:
    | 'EARLY_TURN'
    | 'LATE_TURN'
    | 'LEFT_ENVELOPE'
    | 'OFFSET_INTERCEPT'
    | 'FINAL_NOT_ALIGNED'
    | 'MISSED_TRIGGER'
    | 'VERTICAL_BELOW_OCS'
    | 'BRANCH_SWITCH';
  stationNm: number;
  summary: string;
}
```

### 13.9 SourceRef（SourceRef，来源引用）建议

```ts
export interface SourceRef {
  docId: 'FAA_ORDER_8260_58D';
  chapter?: string;
  section?: string;
  paragraph?: string;
  figure?: string;
  formula?: string;
  pdfPage?: number;
}
```

### 13.10 最小必需字段清单

若你希望先做可运行版本，最少要保证以下字段存在，否则很多 FAA（Federal Aviation Administration，美国联邦航空管理局）规则无法做成真正的 segment-aware（segment-aware，航段感知）渲染：

1. `legType`
2. `xttNm`
3. `attNm`
4. `navSpec`
5. `segmentType`
6. `isFlyOver / isFlyBy`
7. `turnDirection`
8. `arcRadiusNm / centerLatDeg / centerLonDeg`（RF（Radius-to-Fix，半径到定位点）必须）
9. `verticalRule`
10. `sourceRefs`

---

## 14. 几何算法接口：把 FAA（Federal Aviation Administration，美国联邦航空管理局）构造规则拆成稳定、可测、可替换的模块

### 14.1 几何算法总体流水线

建议把整个 procedure（Procedure，程序）构造拆成九个步骤：

1. **Normalize（Normalize，归一化）**  
   CIFP（Coded Instrument Flight Procedures，编码仪表飞行程序）/人工补充字段 -> 语义对象；
2. **Build Nominal Leg Geometry（Build Nominal Leg Geometry，构造名义 leg 几何）**  
   先得到 TF（Track-to-Fix，航迹到定位点）/RF（Radius-to-Fix，半径到定位点）/DF（Direct-to-Fix，直飞到定位点）/CA（Course-to-Altitude，航向到高度）中心线；
3. **Compute Station Axis（Compute Station Axis，生成里程轴）**  
   为 2D（Two-Dimensional，二维）剖面、取样、误差投影准备 station（station，里程）；
4. **Build Lateral Envelopes（Build Lateral Envelopes，构造横向包络）**  
   primary area（primary area，主区）、secondary area（secondary area，次区）、ATT（Along-Track Tolerance，沿航迹容差）扩展；
5. **Build Transition Geometry（Build Transition Geometry，构造过渡几何）**  
   包括 FB（Fly-by，提前转弯）、FO（Fly-over，飞越后转弯）、RF（Radius-to-Fix，半径到定位点）、intermediate-to-final（intermediate-to-final，中间进近到最后进近）offset/taper（offset/taper，偏置/收敛）；
6. **Build Vertical Surfaces（Build Vertical Surfaces，构造垂向面）**  
   LNAV（Lateral Navigation，横向导航）/LP（Localizer Performance，航向道性能）/LNAV-VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）/LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）/RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）分别实现；
7. **Attach Debug Guides（Attach Debug Guides，挂接调试构造线）**  
   line A / line B / tangent point（tangent point，切点）/bisector（bisector，角平分线）/turn center（turn center，转弯圆心）；
8. **Build Validation Projections（Build Validation Projections，构造验证投影）**  
   将实际轨迹投影到 station axis（station axis，里程轴）；
9. **Emit Render Bundles（Emit Render Bundles，输出渲染包）**。  

### 14.2 顶层接口

```ts
export interface GeometryBuildContext {
  ellipsoid: 'WGS84';
  samplingStepNm: number;
  defaultSegmentStepNm: number;
  enableDebugPrimitives: boolean;
  localFrameMode: 'SEGMENT_ENU' | 'RUNWAY_ENU';
}
```

```ts
export interface ProcedureGeometryBuilder {
  buildProcedure(
    pkg: ProcedurePackage,
    ctx: GeometryBuildContext
  ): ProcedureRenderBundle;
}

export interface ProcedureRenderBundle {
  procedureId: string;
  branchBundles: BranchGeometryBundle[];
  diagnostics: BuildDiagnostic[];
}
```

### 14.3 基础几何接口

```ts
export interface LegGeometryBuilder {
  buildTfLeg(leg: ProcedureLeg, fixes: Map<string, ProcedureFix>): PolylineGeometry3D;
  buildRfLeg(leg: ProcedureLeg, fixes: Map<string, ProcedureFix>): PolylineGeometry3D;
  buildDfLeg(leg: ProcedureLeg, fixes: Map<string, ProcedureFix>): PolylineGeometry3D;
  buildCaLeg(leg: ProcedureLeg, fixes: Map<string, ProcedureFix>): PolylineGeometry3D;
}
```

#### 说明

- `buildTfLeg` 必须按 geodesic（geodesic，测地线）构造，而不是简单经纬度线性插值；依据 §1-2-5.a(1)。  
- `buildRfLeg` 必须输出真圆弧，并保留半径、圆心、起止方位；依据 §1-2-5.b(1)(b)、§1-2-5.d(3)。  
- `buildDfLeg` 不等同于 TF（Track-to-Fix，航迹到定位点）；它应保留“直飞截获”的语义标签。  
- `buildCaLeg` 必须同时输出平面方向与垂向终止条件，因为 missed approach（Missed Approach，复飞）首段常由 CA（Course-to-Altitude，航向到高度）开始；依据 §3-5-2。  

### 14.4 横向包络接口

```ts
export interface EnvelopeBuilder {
  buildStraightEnvelope(
    leg: ProcedureLeg,
    centerline: PolylineGeometry3D,
    ctx: GeometryBuildContext
  ): SegmentEnvelopeResult;

  buildFbTurnEnvelope(
    inboundLeg: ProcedureLeg,
    outboundLeg: ProcedureLeg,
    turnFix: ProcedureFix,
    ctx: GeometryBuildContext
  ): SegmentEnvelopeResult;

  buildFoTurnEnvelope(
    inboundLeg: ProcedureLeg,
    outboundLeg: ProcedureLeg,
    turnFix: ProcedureFix,
    ctx: GeometryBuildContext
  ): SegmentEnvelopeResult;

  buildRfEnvelope(
    leg: ProcedureLeg,
    ctx: GeometryBuildContext
  ): SegmentEnvelopeResult;

  buildAbruptWidthChange(
    prevSegment: ProcedureSegment,
    nextSegment: ProcedureSegment,
    atFix: ProcedureFix,
    ctx: GeometryBuildContext
  ): TransitionGeometry;

  buildThirtyNmModeChange(
    segment: ProcedureSegment,
    arpLatDeg: number,
    arpLonDeg: number,
    ctx: GeometryBuildContext
  ): TransitionGeometry;
}
```

#### 14.4.1 规则映射建议

- `buildStraightEnvelope`  
  适用于 TF（Track-to-Fix，航迹到定位点）直段。primary area（primary area，主区）默认为中心线两侧各 2 × XTT（Cross Track Tolerance，横向容差），secondary area（secondary area，次区）为 primary area（primary area，主区）外各 1 × XTT（Cross Track Tolerance，横向容差）；依据 §1-2-5.b(2)(a)。  
- `buildFbTurnEnvelope`  
  依据 §1-2-5.d(1) 五步构造，必须显式保留 bisector（bisector，角平分线）、turn radius（turn radius，转弯半径）、outer boundary arc（outer boundary arc，外边界圆弧）。  
- `buildFoTurnEnvelope`  
  必须包含 reaction and roll distance（reaction and roll distance，反应与滚转距离）`Drr`；依据公式 1-2-12 与 §1-2-5.d(2)。  
- `buildRfEnvelope`  
  必须支持 Case 1（Case 1，情况 1）与 Case 2（Case 2，情况 2）；依据 §1-2-5.d(3) 与 Figures 1-2-10、1-2-11。  
- `buildAbruptWidthChange`  
  适用于 A-RNP（Advanced Required Navigation Performance，高级所需导航性能）/RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）RNP（Required Navigation Performance，所需导航性能）变化“在 fix（Fix，航路点）处突变”的规则；依据 §1-2-5.b(2)(b)、§4-1-1。  
- `buildThirtyNmModeChange`  
  适用于 30 NM（Nautical Mile，海里）模式切换的 30-degree splay（30-degree splay，30 度张角扩展）；依据 §1-2-5.b(2)(b) 及公式 1-2-5、1-2-6。  

### 14.5 intermediate-to-final（intermediate-to-final，中间进近到最后进近）连接接口

这是整套系统里最值得单独实现的接口：

```ts
export interface IntermediateFinalConnectorBuilder {
  buildAlignedLnavConnector(
    intermediate: ProcedureSegment,
    finalSegment: ProcedureSegment,
    pfafFix: ProcedureFix,
    ctx: GeometryBuildContext
  ): TransitionGeometry;

  buildAlignedLpOrLpvConnector(
    intermediate: ProcedureSegment,
    finalSegment: ProcedureSegment,
    pfafFix: ProcedureFix,
    ctx: GeometryBuildContext
  ): TransitionGeometry;

  buildOffsetLnavConnector(
    intermediate: ProcedureSegment,
    finalSegment: ProcedureSegment,
    pfafFix: ProcedureFix,
    ctx: GeometryBuildContext
  ): TransitionGeometry;

  buildOffsetLpOrLpvConnector(
    intermediate: ProcedureSegment,
    finalSegment: ProcedureSegment,
    pfafFix: ProcedureFix,
    ctx: GeometryBuildContext
  ): TransitionGeometry;
}
```

#### 14.5.1 接口约束

1. 所有 connector（connector，连接器）都必须显式返回：
   - line A（line A，A 线）
   - line B（line B，B 线）
   - inside boundary（inside boundary，内侧边界）
   - outside boundary（outside boundary，外侧边界）
   - tangent connection（tangent connection，切线连接）
2. aligned case（aligned case，对正情况）与 offset case（offset case，偏置情况）必须分开实现；
3. LNAV（Lateral Navigation，横向导航）/LNAV-VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）与 LP（Localizer Performance，航向道性能）/LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）必须分开实现，因为 FAA（Federal Aviation Administration，美国联邦航空管理局）在 PFAF（Precise Final Approach Fix，精确最后进近定位点）前 2 NM（Nautical Mile，海里）和后 1 NM（Nautical Mile，海里）的边界连接方式不同；依据 §3-1-4(d)(1)-(4)。  

### 14.6 final（Final，最后进近）垂向面接口

```ts
export interface FinalSurfaceBuilder {
  buildLnavFinalOea(
    finalSegment: ProcedureSegment,
    pfafFix: ProcedureFix,
    mapFix: ProcedureFix,
    ctx: GeometryBuildContext
  ): VerticalSurfaceGeometry[];

  buildLpFinalOea(
    finalSegment: ProcedureSegment,
    pfafFix: ProcedureFix,
    ctx: GeometryBuildContext
  ): VerticalSurfaceGeometry[];

  buildLnavVnavFinalSurfaces(
    finalSegment: ProcedureSegment,
    ctx: GeometryBuildContext
  ): VerticalSurfaceGeometry[];

  buildLpvGlsFinalSurfaces(
    finalSegment: ProcedureSegment,
    ctx: GeometryBuildContext
  ): VerticalSurfaceGeometry[];

  buildRnpArFinalSurfaces(
    finalSegment: ProcedureSegment,
    ctx: GeometryBuildContext
  ): VerticalSurfaceGeometry[];
}
```

#### 14.6.1 各接口最少应输出什么

- `buildLnavFinalOea`  
  输出 final segment（final segment，最后进近航段）primary area（primary area，主区）/secondary area（secondary area，次区）边界带；依据 §3-2-3。  
- `buildLpFinalOea`  
  输出 LP（Localizer Performance，航向道性能）final boundary（final boundary，最后进近边界）与横向扩展逻辑；依据 §3-2-3。  
- `buildLnavVnavFinalSurfaces`  
  至少输出 level OCS（level OCS，水平净空面）与 sloping OCS（sloping OCS，倾斜净空面）两类面；依据 §3-3-1、§3-3-4。  
- `buildLpvGlsFinalSurfaces`  
  至少输出 W surface（W surface，W 面）、X surface（X surface，X 面）、Y surface（Y surface，Y 面）；依据 §3-4-4。  
- `buildRnpArFinalSurfaces`  
  输出 final area（final area，最后进近区）、FROP（Final Rollout Point，最后拉直点）参考、OCS（Obstacle Clearance Surface，障碍物净空面）；依据 §4-2。  

### 14.7 missed approach（Missed Approach，复飞）接口

```ts
export interface MissedApproachBuilder {
  buildMissedSection1(
    segment: ProcedureSegment,
    finalSegment: ProcedureSegment,
    ctx: GeometryBuildContext
  ): VerticalSurfaceGeometry[];

  buildMissedSection2Straight(
    segment: ProcedureSegment,
    ctx: GeometryBuildContext
  ): SegmentGeometryBundle;

  buildMissedSection2Turning(
    segment: ProcedureSegment,
    ctx: GeometryBuildContext
  ): SegmentGeometryBundle;
}
```

#### 14.7.1 关键约束

1. `buildMissedSection1` 必须按 final type（final type，最后进近类型）区分，因为 LNAV（Lateral Navigation，横向导航）/LP（Localizer Performance，航向道性能）、LNAV-VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）、LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）section 1（section 1，第一段）起始面和扩展方式不同；依据 §3-6。  
2. `buildMissedSection2Turning` 必须显式支持 TIA（Turn Initiation Area，转弯起始区）、wind spiral（wind spiral，风螺旋）、inside/outside turn（inside/outside turn，内/外侧转弯）debug primitives（debug primitives，调试图元）；依据 §3-7。  
3. missed approach（Missed Approach，复飞）首段若为 CA（Course-to-Altitude，航向到高度），则渲染与验证对象必须保留“到高度终止”的语义，不得误标为普通 TF（Track-to-Fix，航迹到定位点）。  

### 14.8 轨迹验证接口

```ts
export interface TrajectoryProjectionService {
  projectToSegmentAxis(
    trajectory: CartesianPoint[],
    segmentGeometry: SegmentGeometryBundle
  ): SegmentAssessment;
}
```

```ts
export interface ContainmentService {
  classifySamples(
    samples: CartesianPoint[],
    segmentGeometry: SegmentGeometryBundle
  ): {
    inPrimaryFlags: boolean[];
    inSecondaryFlags: boolean[];
    outsideFlags: boolean[];
  };
}
```

```ts
export interface VerticalClearanceService {
  computeVerticalError(
    samples: CartesianPoint[],
    surfaces: VerticalSurfaceGeometry[]
  ): number[]; // feet（feet，英尺），负值表示低于要求面
}
```

### 14.9 BuildDiagnostic（BuildDiagnostic，构造诊断）建议

```ts
export interface BuildDiagnostic {
  severity: 'INFO' | 'WARN' | 'ERROR';
  segmentId?: string;
  legId?: string;
  code:
    | 'RF_RADIUS_MISSING'
    | 'RNP_CHANGE_INSIDE_FAS'
    | 'FO_NOT_ALLOWED_RNP_AR'
    | 'FINAL_HAS_TURN'
    | 'CONNECTOR_NOT_CONSTRUCTIBLE'
    | 'SECONDARY_DISABLED_BY_RULE'
    | 'SOURCE_INCOMPLETE';
  message: string;
  sourceRefs?: SourceRef[];
}
```

这会直接决定系统是否真正“可验收”。  
没有诊断层，你只能看到画出来的结果，却不知道它是不是按规范构造的。

---

## 15. 可验收标准：不是“看起来差不多”，而是“几何、语义、验证三层都能过”

建议把验收分成三组：

1. **几何正确性验收**
2. **可视化语义验收**
3. **验证工作流验收**

### 15.1 几何正确性验收

| 编号 | 验收项 | 标准 | 对应依据 |
|---|---|---|---|
| G-01 | TF（Track-to-Fix，航迹到定位点）leg（Leg，航迹腿）构造 | 采用 geodesic（geodesic，测地线）而非简单平面直线；任一点逆算航向连续、长度误差不超过 0.01 NM（Nautical Mile，海里） | §1-2-5.a(1) |
| G-02 | RF（Radius-to-Fix，半径到定位点）leg（Leg，航迹腿）构造 | 圆弧半径、起止方位、圆心可回溯；采样点到理论圆弧径向误差不超过 0.005 NM（Nautical Mile，海里） | §1-2-5.b(1)(b), §1-2-5.d(3) |
| G-03 | XTT（Cross Track Tolerance，横向容差）/ATT（Along-Track Tolerance，沿航迹容差）宽度生成 | primary area（primary area，主区）与 secondary area（secondary area，次区）宽度计算符合 2 × XTT（Cross Track Tolerance，横向容差）与 1 × XTT（Cross Track Tolerance，横向容差）规则；检查点误差不超过 0.01 NM（Nautical Mile，海里） | §1-2-5.b(2)(a) |
| G-04 | FB（Fly-by，提前转弯）构造 | bisector（bisector，角平分线）、R（Radius，半径）、内外边界弧可显示；边界不发生自交 | §1-2-5.d(1) |
| G-05 | FO（Fly-over，飞越后转弯）构造 | `Drr`（Reaction and Roll Distance，反应与滚转距离）已计算并用于基线后移；边界连接与 30° taper（30° taper，30 度收敛）完整 | §1-2-5.d(2), 公式 1-2-12 |
| G-06 | RF（Radius-to-Fix，半径到定位点）包络 | 支持 Case 1（Case 1，情况 1）与 Case 2（Case 2，情况 2）；Case 2（Case 2，情况 2）内侧几何不崩溃 | §1-2-5.d(3), Figures 1-2-10/1-2-11 |
| G-07 | 30 NM（Nautical Mile，海里）模式切换 | 宽度切换采用 30-degree splay（30-degree splay，30 度张角扩展），而不是瞬时跳变 | §1-2-5.b(2)(b), 公式 1-2-5/1-2-6 |
| G-08 | A-RNP（Advanced Required Navigation Performance，高级所需导航性能）/RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）RNP（Required Navigation Performance，所需导航性能）变化 | 宽度变化发生在 leg（Leg，航迹腿）起点 fix（Fix，航路点）并为 abrupt（abrupt，突变） | §1-2-5.b(2)(b), §4-1-1 |
| G-09 | intermediate-to-final（intermediate-to-final，中间进近到最后进近）aligned connector（aligned connector，对正连接器） | PFAF（Precise Final Approach Fix，精确最后进近定位点）前 2 NM（Nautical Mile，海里）和后 1 NM（Nautical Mile，海里）锚点正确；边界连续无断裂 | §3-1-4(d)(1)-(2) |
| G-10 | intermediate-to-final（intermediate-to-final，中间进近到最后进近）offset connector（offset connector，偏置连接器） | line A（line A，A 线）、line B（line B，B 线）、inside boundary（inside boundary，内侧边界）、outside boundary（outside boundary，外侧边界）可回显 | §3-1-4(d)(4), Figures 3-1-3 ~ 3-1-6 |
| G-11 | LNAV（Lateral Navigation，横向导航）final（Final，最后进近）OEA（Obstacle Evaluation Area，障碍物评估区） | 起点为 PFAF（Precise Final Approach Fix，精确最后进近定位点）前 0.3 NM（Nautical Mile，海里），终点为 LTP/FTP（Landing Threshold Point / Fictitious Threshold Point，着陆入口点/虚拟入口点）后 0.3 NM（Nautical Mile，海里）；1.0 NM（Nautical Mile，海里）后半宽稳定到 0.6 NM（Nautical Mile，海里） | §3-2-3 |
| G-12 | LNAV-VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）final（Final，最后进近） | 同时存在 level OCS（level OCS，水平净空面）和 sloping OCS（sloping OCS，倾斜净空面）两类面 | §3-3-1, §3-3-4 |
| G-13 | LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）final（Final，最后进近） | W surface（W surface，W 面）、X surface（X surface，X 面）、Y surface（Y surface，Y 面）为独立可切换对象 | §3-4-4 |
| G-14 | RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）final（Final，最后进近） | FAS（Final Approach Segment，最后进近航段）内不出现 RNP（Required Navigation Performance，所需导航性能）变化；FO（Fly-over，飞越后转弯）转弯构造被禁用 | §4-1-1 |
| G-15 | missed approach（Missed Approach，复飞）分段 | section 1（section 1，第一段）与 section 2（section 2，第二段）必须是两个独立几何对象 | §3-6, §3-7 |

### 15.2 可视化语义验收

| 编号 | 验收项 | 标准 |
|---|---|---|
| V-01 | 程序结构 | 至少能从 UI（User Interface，用户界面）中看出 branch（Branch，分支）-> segment（Segment，航段）-> leg（Leg，航迹腿）层级 |
| V-02 | leg（Leg，航迹腿）分型 | TF（Track-to-Fix，航迹到定位点）、RF（Radius-to-Fix，半径到定位点）、DF（Direct-to-Fix，直飞到定位点）、CA（Course-to-Altitude，航向到高度）视觉上可区分 |
| V-03 | protection（protection，保护）与 centerline（centerline，中心线）区分 | 用户能单独开关 nominal path（nominal path，名义航迹）与 envelope（envelope，包络） |
| V-04 | final（Final，最后进近）分型 | LNAV（Lateral Navigation，横向导航）、LP（Localizer Performance，航向道性能）、LNAV-VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航）、LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）、GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）、RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）至少在图例与面对象层面可区分 |
| V-05 | transition（transition，过渡）可读性 | intermediate-to-final（intermediate-to-final，中间进近到最后进近）连接区不是一条普通折线，而是单独区域对象 |
| V-06 | missed approach（Missed Approach，复飞）可读性 | section 1（section 1，第一段）/section 2（section 2，第二段）切换点可见，CA（Course-to-Altitude，航向到高度）首段可见 |
| V-07 | debug（debug，调试）能力 | 可打开构造线，显示 line A（line A，A 线）、line B（line B，B 线）、turn center（turn center，转弯圆心）、bisector（bisector，角平分线）、tangent point（tangent point，切点） |

### 15.3 验证工作流验收

| 编号 | 验收项 | 标准 |
|---|---|---|
| W-01 | 轨迹投影 | 任意一条 replay（replay，回放）/prediction（prediction，预测）轨迹可投影到 segment（Segment，航段）里程轴 |
| W-02 | 横向偏差 | 能输出 cross-track error（cross-track error，横向偏差）曲线，单位为 NM（Nautical Mile，海里） |
| W-03 | 垂向偏差 | 对 final（Final，最后进近）和 missed approach（Missed Approach，复飞）可输出 vertical error（vertical error，垂向偏差），单位为 ft（feet，英尺） |
| W-04 | 包络判定 | 能逐采样点判定 in primary（in primary，位于主区）、in secondary（in secondary，位于次区）、outside（outside，越界） |
| W-05 | 关键事件标记 | 能自动标记 early turn（early turn，提前转弯）、late turn（late turn，迟转弯）、offset intercept（offset intercept，偏置截获）、below OCS（below OCS，低于净空面） |
| W-06 | 截图与复核 | 对任何一个事件点，用户能一键跳到 3D（三维，Three-Dimensional）视图、plan view（plan view，平面视图）、vertical profile（vertical profile，垂直剖面）三联查看 |
| W-07 | 来源追溯 | 选中任何 segment（Segment，航段）或 surface（surface，评估面），都能看到来自哪个 FAA（Federal Aviation Administration，美国联邦航空管理局）章节/条款/图号 |

### 15.4 最小验收测试集

为了避免“只有一条程序能跑”的假通过，建议至少准备以下 10 个测试样例：

1. **Case A**：纯 TF（Track-to-Fix，航迹到定位点） feeder（feeder，引导）+ initial（initial，初始）+ aligned LNAV（Lateral Navigation，横向导航） final（Final，最后进近）  
2. **Case B**：Basic T（Basic T，基础 T 型）+ HILPT（Hold-In-Lieu-of-Procedure Turn，等待代替程序转弯）  
3. **Case C**：IF（Intermediate Fix，中间定位点）处 FB（Fly-by，提前转弯）进入 LNAV（Lateral Navigation，横向导航） final（Final，最后进近）  
4. **Case D**：offset intermediate（offset intermediate，偏置中间进近）进入 LNAV-VNAV（Lateral Navigation / Vertical Navigation，横向/垂向导航） final（Final，最后进近）  
5. **Case E**：RF（Radius-to-Fix，半径到定位点）进入 LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）/GLS（Ground Based Augmentation System Landing System，地基增强着陆系统）  
6. **Case F**：A-RNP（Advanced Required Navigation Performance，高级所需导航性能）宽度 abrupt（abrupt，突变）变化  
7. **Case G**：RNP AR APCH（Required Navigation Performance Authorization Required Approach，需特殊批准的所需导航性能进近）final（Final，最后进近）  
8. **Case H**：LNAV（Lateral Navigation，横向导航） missed approach（Missed Approach，复飞）section 1（section 1，第一段）+ straight section 2（straight section 2，直线第二段）  
9. **Case I**：LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能） missed approach（Missed Approach，复飞）section 1（section 1，第一段）  
10. **Case J**：turning missed approach（turning missed approach，转弯复飞）section 2（section 2，第二段）  

### 15.5 失败即不通过的红线项

以下任一项出现，都应判定当前实现**不通过验收**：

1. final（Final，最后进近）仍被渲染为“fix（Fix，航路点）连线 + 粗 tube（tube，管状体）”而无 segment（Segment，航段）/surface（surface，评估面）语义；
2. RF（Radius-to-Fix，半径到定位点）被折线近似但无半径/圆心可追溯；
3. intermediate-to-final（intermediate-to-final，中间进近到最后进近）连接区没有单独对象；
4. section 1（section 1，第一段）与 section 2（section 2，第二段）复飞未拆分；
5. 选中对象后无法看到来源条款；
6. 轨迹验证只能显示“是否重合”，不能给出 lateral（lateral，横向）/vertical（vertical，垂向）偏差。  

### 15.6 交付时建议附带的验收材料

建议每次阶段性交付都附带四类材料：

1. **数据样例**：至少 3 个完整 procedure（Procedure，程序）JSON（JavaScript Object Notation，JavaScript 对象表示法）；
2. **几何截图**：每个关键 case（case，案例）至少一张 3D（三维，Three-Dimensional）图、一张 plan view（plan view，平面视图）、一张 vertical profile（vertical profile，垂直剖面）；
3. **debug 截图**：至少展示一次 line A（line A，A 线）/line B（line B，B 线）/turn center（turn center，转弯圆心）；
4. **误差报告**：以表格列出关键测点的半宽、半径、偏差、是否在 primary area（primary area，主区）或 secondary area（secondary area，次区）内。  

---

## 16. 这一版增强后，你可以直接进入实现的最短路径

如果按本版继续往下做，我建议直接进入下面三件事：

1. 先实现 **Schema（Schema，数据模式）归一化器**，把你现有 procedure（Procedure，程序）数据整理成 `ProcedurePackage`；
2. 再实现 **IntermediateFinalConnectorBuilder（IntermediateFinalConnectorBuilder，中间进近到最后进近连接器构造器）** 和 **FinalSurfaceBuilder（FinalSurfaceBuilder，最后进近评估面构造器）**；
3. 最后实现 **TrajectoryProjectionService（TrajectoryProjectionService，轨迹投影服务）** 与 **ContainmentService（ContainmentService，包络归属判定服务）**。  

这样你会最快得到一个真正可研究、可调试、可验收的 RNAV（Area Navigation，区域导航）procedure（Procedure，程序）可视化系统，而不是只把图形画得更复杂。

---

## 17. 从现有 AeroViz-4D 迁移到 v3 设计的实施计划

本节把前面的目标设计转成可执行迁移计划。结论是：**v3 的 segment-first（segment-first，航段优先）数据结构明显优于当前 route/fix-first（route/fix-first，路线/航路点优先）方案，应作为长期目标重构；但迁移不应一次性推翻现有 `procedure-details` 数据与 UI，而应先增加归一化适配层，再逐步替换几何和渲染层。**

### 17.1 当前结构与 v3 目标结构的分歧

当前 AeroViz-4D 已经有可用基础：

- `ProcedureDetailDocument` 保存 airport（Airport，机场）、runway（Runway，跑道）、procedure（Procedure，程序）、fixes（Fixes，航路点）、branches（Branches，分支）、legs（Legs，航迹腿）、verticalProfiles（Vertical Profiles，垂直剖面）；
- `ProcedureRouteViewModel` 把 branch（Branch，分支）压平成 route（Route，路线），供 3D procedure layer、2D runway profile、Procedure Details plan/profile 使用；
- `useProcedureLayer` 当前渲染 centerline（Centerline，中心线）+ fixes（Fixes，航路点）+ approximate tunnel（approximate tunnel，近似通道体）；
- `useOcsLayer` 当前能从 FAF（Final Approach Fix，最后进近定位点）到 threshold（Threshold，跑道入口）生成简化 primary/secondary OCS（Obstacle Clearance Surface，障碍物净空面）。

但这些结构与 v3 目标有本质差异：

| 维度 | 当前结构 | v3 目标 | 迁移判断 |
|---|---|---|---|
| 主对象 | branch/route/fix | segment/leg/protection/surface | 必须重构为 segment-first |
| 几何表达 | polyline + tunnel | centerline + ribbon + surface + probe/debug | tunnel 只保留为 legacy/对照层 |
| leg 支持 | 主要 IF/TF，其他 leg 作为 skipped/simplified warning | TF/RF/DF/CA/HM/HA/HF 明确分型 | 分阶段补齐，先支持 TF/LNAV final |
| 保护区 | tunnel half-width + 简化 OCS | XTT/ATT、primary/secondary、final/missed surface model | 需要新增 geometry bundle |
| 验证 | horizontal plate inclusion + 2D profile | segment projection、cross-track、vertical error、event markers | 需要新增 validation layer |
| 来源追溯 | warning/sourceLine 局部存在 | 每个 segment/surface/debug primitive 可追溯 FAA 条款 | 需要规范 `SourceRef` |

因此，v3 schema 不应只是“另一个文档 schema”；它应成为新的 canonical intermediate model（canonical intermediate model，规范中间模型）。当前 `ProcedureDetailDocument` 可以保留为 source-semantic export（source-semantic export，源语义导出），但渲染和验证应逐步改为消费 `ProcedurePackage` 与 `ProcedureRenderBundle`。

### 17.2 迁移原则

1. **先适配，后替换**  
   不直接删除 `ProcedureDetailDocument`、`ProcedureRouteViewModel`、`useProcedureLayer` 的现有路径。先新增 adapter（Adapter，适配器）把现有文档转换为 `ProcedurePackage`，等新 geometry/render/validation 通过验收后再切换默认路径。

2. **保留当前用户可见能力**  
   Procedure Panel、Procedure Details、2D runway profile 当前可用功能不能因为 schema 重构中断。新模型上线前，旧 route view model 继续服务现有 UI。

3. **v3 schema 作为新核心，不反向压低设计**  
   如果当前数据缺少 RF center、FB/FO、XTT/ATT、sourceRefs 等字段，不能把 v3 schema 改回 route/fix-first；应在 adapter 中填充 conservative defaults（conservative defaults，保守默认值）并产生 `BuildDiagnostic`。

4. **每一阶段都有可验收输出**  
   每阶段都必须产生：测试、样例 JSON、可视化截图或可查询 debug primitive。不能只完成类型定义。

5. **研究准确性优先于视觉复杂度**  
   如果某条 procedure 缺少构造所需数据，应显示 warning/diagnostic，不应画一个看似完整但无法追溯的几何。

### 17.3 阶段 0：基线冻结与回归保护

目标：在开始 schema/geometry 重构前，先固定当前行为，避免迁移过程中丢失现有功能。

工作项：

1. 为当前 KRDU procedure layer、Procedure Details、Runway Trajectory Profile 建立基线测试清单；
2. 保留当前 `ProcedureDetailDocument` schema 的 fixture（fixture，测试样例）；
3. 保留当前 `ProcedureRouteViewModel` 生成结果的 snapshot-style（snapshot-style，快照式）测试；
4. 在 docs 中明确当前 tunnel 是 `legacy visual approximation`，不是 official containment。

验收：

- `npm test -- --run` 通过；
- `npm run build` 通过；
- 至少一个 KRDU procedure document 能被加载并转换为当前 route view model；
- 当前 Procedure Details plan/profile 与 2D runway profile 仍可打开。

### 17.4 阶段 1：新增 v3 schema 类型与归一化适配器

目标：不改现有渲染，先让当前 procedure documents 可以稳定转换成 `ProcedurePackage`。

建议新增文件：

```text
src/data/procedurePackage.ts
src/data/procedurePackageAdapter.ts
src/data/__tests__/procedurePackageAdapter.test.ts
```

适配规则：

1. `ProcedureDetailDocument.procedure` -> `ProcedurePackage` 顶层 metadata；
2. `ProcedureDetailFix` -> `ProcedureFix`；
3. `ProcedureDetailBranch` -> `ProcedureBranch`；
4. `ProcedureDetailLeg` -> `ProcedureLeg`；
5. 按 `leg.segmentType`、`roleAtEnd`、branch role、procedure modes 推导初版 `ProcedureSegment`；
6. 对当前缺失字段填入保守默认：
   - RNAV(GPS) approach 默认 `navSpec = RNP_APCH`；
   - final LNAV 初版默认 `xttNm = 0.3`，但必须标记 `SOURCE_INCOMPLETE`，直到 parser 能确认；
   - `attNm` 初版可按 rule profile default 填充，同样带 diagnostic；
   - `isFlyOver/isFlyBy` 未知时不猜测，保持 undefined 并产生 warning；
   - RF 所需 `arcRadiusNm/centerLatDeg/centerLonDeg` 缺失时不得构造 RF，只产生 `RF_RADIUS_MISSING`。

优化设计：

- `ProcedurePackage` 不直接替换 `ProcedureDetailDocument` 的 source record 角色。更稳妥的结构是：

```text
ProcedureDetailDocument  ->  normalizeProcedurePackage(...)  ->  ProcedurePackage
                           source semantic layer                canonical procedure model
```

- 后续 Python exporter 可以直接输出 `procedure-package.json`，但第一步不强制更改生成管线。

验收：

- KRDU 当前所有 procedure documents 都能归一化为 `ProcedurePackage`；
- 每个 package 至少包含 procedure、branch、segment、leg、fix 四层；
- 缺失 XTT/ATT/FB/FO/RF 参数时产生 diagnostic，而不是静默默认；
- 现有 UI 与测试不受影响。

### 17.5 阶段 2：建立 segment geometry kernel，但暂不替换 UI

目标：把几何计算从现有 tunnel helper 拆出来，建立 v3 的独立 geometry kernel（geometry kernel，几何内核）。

建议新增文件：

```text
src/utils/procedureSegmentGeometry.ts
src/utils/procedureEnvelopeGeometry.ts
src/utils/procedureSurfaceGeometry.ts
src/utils/procedureValidationGeometry.ts
src/utils/__tests__/procedureSegmentGeometry.test.ts
src/utils/__tests__/procedureEnvelopeGeometry.test.ts
```

第一批实现范围：

1. `buildTfLeg(...)`：geodesic TF centerline；
2. `computeStationAxis(...)`：segment station axis；
3. `buildStraightEnvelope(...)`：TF primary/secondary envelope；
4. `buildLnavFinalOea(...)`：LNAV final OEA primary/secondary ribbon；
5. `buildAlignedLnavConnector(...)`：PFAF 前 2 NM / 后 1 NM aligned connector 的最小版本；
6. `BuildDiagnostic` 输出与聚合。

暂不实现：

- RF 真圆弧；
- HILPT；
- LPV W/X/Y；
- turning missed approach wind spiral；
- RNP AR final template。

这些不是降低 v3 设计，而是为了先通过第一优先级中最关键的 LNAV final 与 intermediate-to-final 验证闭环。

验收：

- G-01、G-03、G-09、G-11 的最小版本可用单元测试覆盖；
- geometry kernel 输出 `SegmentGeometryBundle`，不依赖 Cesium；
- 对无法构造的 segment 返回 diagnostic，不返回伪完整几何；
- 与现有 `buildTunnelSections` 并存。

### 17.6 阶段 3：新增 render bundle 与 legacy/new 双通道渲染

目标：在 3D 与 Procedure Details 中接入 `ProcedureRenderBundle`，但保留 legacy tunnel 作为 fallback。

建议新增或改造：

```text
src/data/procedureRenderBundle.ts
src/hooks/useProcedureSegmentLayer.ts
src/components/ProcedureVisualizationModeControl.tsx
src/components/ProcedureDetailsPage.tsx
```

渲染策略：

1. 新增三种 mode：
   - `Procedure Structure`
   - `Protected Geometry`
   - `Trajectory Validation`
2. `Procedure Structure` 显示 branch -> segment -> leg -> fix；
3. `Protected Geometry` 显示 centerline + primary/secondary ribbon + final OEA surface；
4. `Trajectory Validation` 保留当前 aircraft/profile 能力，同时准备 segment assessment overlay；
5. legacy tunnel 默认关闭或标记为 `Legacy Approximation`，只作为对照层打开。

与当前设计的兼容：

- 当前 Procedure Details 的 branch/fix focus 可以保留，但 focus target 应从 fix/branch 升级为 segment/leg/fix；
- 当前 plan view 与 vertical profile 可以继续使用 SVG，但数据源从 `ProcedureBranchPolyline` 逐步切换到 `SegmentGeometryBundle`；
- 当前 2D runway profile 的 horizontal plate 判断后续应改为使用 segment envelope，而不是 route half-width。

验收：

- V-01、V-03、V-05 初步通过；
- 用户能在 UI 中明确看到 segment 层级；
- 用户能单独开关 nominal path 与 envelope；
- tunnel 不再是默认主表达。

### 17.7 阶段 4：把 final 与 intermediate-to-final 做到 thesis validation 可用

目标：优先完成 v3 第一优先级中最有研究价值的几何。

实现范围：

1. aligned LNAV final；
2. offset LNAV final；
3. LNAV/VNAV final 的横向 ribbon 复用 LNAV，垂向先输出 placeholder diagnostic；
4. PFAF connector debug primitives：
   - line A；
   - line B；
   - inside boundary；
   - outside boundary；
   - anchor stations；
5. Procedure Details plan view 永远显示 PFAF 前 2 NM / 后 1 NM 的连接区。

验收：

- G-09、G-10、G-11 达到可截图复核；
- V-05 达到可读；
- W-01、W-02 可针对 final/intermediate segments 输出 station 与 cross-track error；
- 每个 connector 的 sourceRefs 能指向 §3-1-4(d)。

### 17.8 阶段 5：迁移 trajectory validation 到 segment assessment

目标：把 2D runway profile 从“是否在 RNAV horizontal plate”升级为“在哪个 segment、偏差多少、属于 primary/secondary/outside”。

实现范围：

1. `TrajectoryProjectionService.projectToSegmentAxis(...)`；
2. `ContainmentService.classifySamples(...)`；
3. 当前 2D runway profile 中增加 segment-aware overlay；
4. 对 aircraft sample 输出：
   - activeSegmentId；
   - stationNm；
   - crossTrackErrorNm；
   - inPrimary/inSecondary/outside；
5. Procedure Details 与 2D profile 使用同一 `SegmentAssessment` 数据结构。

验收：

- W-01、W-02、W-04 初步通过；
- 当前 “No aircraft are inside the RNAV horizontal plate” 逻辑可由 segment containment 替代；
- 一条 replay/prediction trajectory 能在 plan view 和 vertical profile 中同步定位。

### 17.9 阶段 6：missed approach section 化

目标：把 missed approach 从当前 branch/line 表达升级为 CA + DF + section 1 + section 2。

实现范围：

1. parser/adapter 保留 CA、DF、missed section boundary；
2. `MISSED_S1` / `MISSED_S2` segment type 落地；
3. straight missed section 1 / section 2 的 first implementation；
4. Procedure Details plan/profile 显示 section split；
5. `BuildDiagnostic` 标记无法构造的 turning missed approach。

验收：

- G-15、V-06 初步通过；
- missed approach 不再只是一条从 MAP 向外的 line；
- CA leg 不能被误标为 TF。

### 17.10 阶段 7：补齐高级 leg 与高级 surface

目标：在基础验证链路稳定后，补齐 v3 第二/第三优先级。

实现顺序：

1. RF centerline + RF envelope；
2. FB/FO turn construction；
3. HILPT / Basic T 结构识别；
4. LPV/GLS W/X/Y surfaces；
5. RNP AR final template；
6. turning missed approach debug mode。

验收：

- G-02、G-04、G-05、G-06、G-12、G-13、G-14 逐项补齐；
- V-02、V-04、V-07 通过；
- 最小验收测试集 Case B、E、G、I、J 能运行。

### 17.11 Python exporter 的迁移策略

短期不要立即让 Python exporter 直接生成完整 v3 derived geometry。更合理的分工是：

```text
Python exporter:
  CIFP/chart/source data -> source semantic JSON

TypeScript normalizer:
  source semantic JSON -> ProcedurePackage

TypeScript geometry kernel:
  ProcedurePackage -> ProcedureRenderBundle / SegmentAssessment
```

原因：

- 几何算法需要被前端 2D/3D/validation 重复使用，放在 TypeScript 内核更容易共享；
- Python 适合做 fixed-width parsing、chart asset publishing、source metadata；
- TypeScript 适合做 interactive debug、unit test、visual regression、runtime diagnostics。

中长期可以让 Python 额外输出 `procedure-package.json` 作为缓存，但不应把 Cesium-ready render primitives 固化在 Python 输出里。

### 17.12 文件级迁移地图

| 当前文件 | 迁移动作 | 目标 |
|---|---|---|
| `src/data/procedureDetails.ts` | 保留 source-semantic 类型，增加 v3 package adapter | 不直接承载 derived geometry |
| `src/data/procedureRoutes.ts` | 保留 legacy route view model，新增从 `ProcedurePackage` 到 legacy route 的兼容 adapter | 迁移期维持旧 UI |
| `src/utils/procedureGeometry.ts` | 标记为 legacy tunnel geometry，新增 segment geometry utils | tunnel 不再是主表达 |
| `src/utils/ocsGeometry.ts` | 拆分/升级为 final surface builders | 支持 final type 分型 |
| `src/hooks/useProcedureLayer.ts` | 保留 legacy，新增 `useProcedureSegmentLayer` | 双通道渲染 |
| `src/hooks/useOcsLayer.ts` | 迁移到 `FinalSurfaceBuilder` 输出 | 不再从 tunnel width 推 OCS |
| `src/components/ProcedureDetailsPage.tsx` | focus model 从 fix/branch 扩展为 segment/leg/fix | Procedure Details 成为验证工作台 |
| `src/components/RunwayTrajectoryProfilePanel.tsx` | horizontal plate 判断迁移到 segment containment | 输出 station/cross-track/containment |
| `python/preprocess_procedures.py` | 补充更多 ARINC leg 字段与 sourceRefs | 不再只输出 IF/TF 可画点 |

### 17.13 每阶段的分支策略

建议使用 feature flags（feature flags，功能开关）避免大规模重构一次性上线：

```ts
procedureVisualizationEngine: 'legacy-route' | 'segment-v3'
procedureProtectionDisplay: 'legacy-tunnel' | 'ribbon-surface'
procedureValidationMode: 'horizontal-plate' | 'segment-assessment'
```

默认策略：

1. 阶段 1-2：默认 `legacy-route`；
2. 阶段 3：开发环境可切换 `segment-v3`；
3. 阶段 4：Procedure Details 默认使用 segment-v3 plan view；
4. 阶段 5：2D runway profile 默认使用 segment-assessment；
5. 阶段 6 之后：legacy tunnel 改为手动打开的 comparison layer。

### 17.14 迁移完成定义

当以下条件同时满足时，才认为从当前方案迁移到 v3 方案完成：

1. `ProcedurePackage` 是 procedure 可视化与验证的主数据入口；
2. `ProcedureRenderBundle` 是 3D、Procedure Details、2D profile 的共享几何来源；
3. final approach 不再默认显示 tunnel，而是显示 centerline + primary/secondary ribbon + final surface；
4. intermediate-to-final connector 是独立对象，并可显示 debug primitives；
5. missed approach 至少拆成 `MISSED_S1` 与 `MISSED_S2`；
6. trajectory validation 输出 segment-level station、cross-track error、containment flags；
7. 选中任意 segment/surface/debug primitive 可看到 sourceRefs 与 diagnostics；
8. v3 验收表中第一优先级相关项全部通过；
9. legacy route/tunnel 路径只作为 fallback 或 comparison，不再作为默认主表达。
