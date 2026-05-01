# AeroViz-4D：RNAV（Area Navigation）Procedure（Procedure）可视化与 Segment（Segment）呈现设计说明（重写版）

> 版本：v2.0  
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
