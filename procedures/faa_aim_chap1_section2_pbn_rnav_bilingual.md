# FAA AIM Chapter 1 Section 2：Performance-Based Navigation (PBN) and Area Navigation (RNAV)｜中英对照翻译

> 来源：FAA（Federal Aviation Administration，美国联邦航空管理局）AIM（Aeronautical Information Manual，《航空信息手册》）Chapter 1, Section 2：Performance-Based Navigation (PBN) and Area Navigation (RNAV)  
> 原文链接：https://www.faa.gov/air_traffic/publications/atpubs/aim_html/chap1_section_2.html  
> 说明：本文为学习参考用中英对照翻译。运行、飞行计划、放行、训练和合规判断应以 FAA（Federal Aviation Administration，美国联邦航空管理局）官方现行文本、航空器批准文件、公司运行规范和适用法规为准。

---

## 缩写与术语速览

| 缩写 | 全称 | 中文译名 |
|---|---|---|
| FAA | Federal Aviation Administration | 美国联邦航空管理局 |
| AIM | Aeronautical Information Manual | 航空信息手册 |
| PBN | Performance-Based Navigation | 基于性能的导航 / 性能导航 |
| RNAV | Area Navigation | 区域导航 |
| RNP | Required Navigation Performance | 所需导航性能 |
| NavSpec / NavSpecs | Navigation Specification(s) | 导航规范 |
| ICAO | International Civil Aviation Organization | 国际民用航空组织 |
| AC | Advisory Circular | 咨询通告 |
| RAIM | Receiver Autonomous Integrity Monitoring | 接收机自主完好性监视 |
| DP | Departure Procedure | 离场程序 |
| STAR | Standard Terminal Arrival | 标准终端进场程序 |
| ATC | Air Traffic Control | 空中交通管制 |
| GPS | Global Positioning System | 全球定位系统 |
| WAAS | Wide Area Augmentation System | 广域增强系统 |
| TSO | Technical Standard Order | 技术标准令 |
| CDI | Course Deviation Indicator | 航道偏离指示器 |
| DME | Distance Measuring Equipment | 测距设备 |
| VOR | VHF Omni-directional Range | 甚高频全向信标 |
| VHF | Very High Frequency | 甚高频 |
| TACAN | Tactical Air Navigation | 战术空中导航 |
| VORTAC | VOR/TACAN | 甚高频全向信标 / 战术空中导航合装台 |
| NDB | Non-directional Beacon | 无方向信标 |
| ADF | Automatic Direction Finder | 自动定向仪 |
| FMS | Flight Management System | 飞行管理系统 |
| LOC | Localizer | 航向道 |
| IRU | Inertial Reference Unit | 惯性基准组件 |
| OBPMA | Onboard Performance Monitoring and Alerting | 机载性能监视与告警 |
| AFM | Aircraft Flight Manual | 飞机飞行手册 |
| APCH | Approach | 进近 |
| LNAV | Lateral Navigation | 横向导航 |
| VNAV | Vertical Navigation | 垂直导航 |
| LPV | Localizer Performance with Vertical Guidance | 带垂直引导的航向道性能 |
| LP | Localizer Performance | 航向道性能 |
| SBAS | Space-Based Augmentation System | 星基增强系统 |
| GBAS | Ground-Based Augmentation System | 地基增强系统 |
| GLS | GBAS Landing System | 地基增强系统着陆系统 |
| AR | Authorization Required | 需要授权 |
| A-RNP | Advanced Required Navigation Performance | 高级所需导航性能 |
| ILS | Instrument Landing System | 仪表着陆系统 |
| IAF | Initial Approach Fix | 初始进近定位点 |
| EPU | Estimated Position Uncertainty | 估算位置不确定度 |
| ANP | Actual Navigation Performance | 实际导航性能 |
| EPE | Estimated Position Error | 估算位置误差 |
| NAS | National Airspace System | 国家空域系统 |
| NAVAID | Navigation Aid | 导航设施 / 助航设备 |
| NOTAM | Notice to Airmen / Notice to Air Missions | 航行通告 |
| AIRAC | Aeronautical Information Regulation and Control | 航空资料规章与管制周期 |
| VFR | Visual Flight Rules | 目视飞行规则 |
| IFR | Instrument Flight Rules | 仪表飞行规则 |
| FDE | Fault Detection and Exclusion | 故障探测与排除 |
| IAP | Instrument Approach Procedure | 仪表进近程序 |
| MDA | Minimum Descent Altitude | 最低下降高度 |
| DA | Decision Altitude | 决断高度 |
| TAWS | Terrain Awareness and Warning System | 地形感知与警告系统 |
| ADS-B | Automatic Dependent Surveillance-Broadcast | 广播式自动相关监视 |
| ATM | Air Traffic Management | 空中交通管理 |
| CPDLC | Controller-Pilot Data Link Communications | 管制员-飞行员数据链通信 |
| MON | Minimum Operational Network | 最低运行网络 |
| OEM | Original Equipment Manufacturer | 原始设备制造商 |
| PFD | Primary Flight Display | 主飞行显示器 |
| ND | Navigation Display | 导航显示器 |
| NM | Nautical Mile | 海里 |

---

## Section 2. Performance-Based Navigation (PBN) and Area Navigation (RNAV)  
## 第 2 节：基于性能的导航（PBN，Performance-Based Navigation）与区域导航（RNAV，Area Navigation）

---

## 1-2-1. General｜总则

### 1. Introduction to PBN｜PBN（Performance-Based Navigation，基于性能的导航）简介

| English | 中文 |
|---|---|
| As air travel has evolved, methods of navigation have improved to give operators more flexibility. PBN exists under the umbrella of area navigation (RNAV). The term RNAV in this context, as in procedure titles, just means “area navigation,” regardless of the equipment capability of the aircraft. (See FIG 1-2-1.) Many operators have upgraded their systems to obtain the benefits of PBN. | 随着航空运输的发展，导航方法也不断改进，为运行人提供了更大的灵活性。PBN（Performance-Based Navigation，基于性能的导航）属于 RNAV（Area Navigation，区域导航）的范畴。在这一语境中，RNAV（Area Navigation，区域导航）一词如同程序标题中的用法一样，仅表示“区域导航”，不论航空器的设备能力如何。许多运行人已经升级其系统，以获得 PBN（Performance-Based Navigation，基于性能的导航）的优势。 |
| Within PBN there are two main categories of navigation methods or specifications: area navigation (RNAV) and required navigation performance (RNP). In this context, the term RNAV x means a specific navigation specification with a specified lateral accuracy value. For an aircraft to meet the requirements of PBN, a specified RNAV or RNP accuracy must be met 95 percent of the flight time. | 在 PBN（Performance-Based Navigation，基于性能的导航）中，导航方法或规范主要分为两类：RNAV（Area Navigation，区域导航）和 RNP（Required Navigation Performance，所需导航性能）。在此语境下，“RNAV x”表示一种具有指定横向精度值的特定导航规范。航空器如要满足 PBN（Performance-Based Navigation，基于性能的导航）要求，必须在 95% 的飞行时间内达到指定的 RNAV（Area Navigation，区域导航）或 RNP（Required Navigation Performance，所需导航性能）精度。 |
| RNP is a PBN system that includes onboard performance monitoring and alerting capability (for example, Receiver Autonomous Integrity Monitoring (RAIM)). PBN also introduces the concept of navigation specifications (NavSpecs) which are a set of aircraft and aircrew requirements needed to support a navigation application within a defined airspace concept. | RNP（Required Navigation Performance，所需导航性能）是一种包含机载性能监视与告警能力的 PBN（Performance-Based Navigation，基于性能的导航）系统，例如 RAIM（Receiver Autonomous Integrity Monitoring，接收机自主完好性监视）。PBN（Performance-Based Navigation，基于性能的导航）还引入了 NavSpecs（Navigation Specifications，导航规范）的概念；导航规范是在特定空域概念中支持某项导航应用所需的一组航空器和机组要求。 |
| For both RNP and RNAV NavSpecs, the numerical designation refers to the lateral navigation accuracy in nautical miles which is expected to be achieved at least 95 percent of the flight time by the population of aircraft operating within the airspace, route, or procedure. | 对 RNP（Required Navigation Performance，所需导航性能）和 RNAV（Area Navigation，区域导航）NavSpecs（Navigation Specifications，导航规范）而言，其数字标识指横向导航精度，单位为 NM（Nautical Mile，海里）；在相关空域、航路或程序内运行的航空器群体预计至少在 95% 的飞行时间内达到该精度。 |
| This information is detailed in International Civil Aviation Organization's (ICAO) Doc 9613, Performance-based Navigation (PBN) Manual and the latest FAA AC 90-105, Approval Guidance for RNP Operations and Barometric Vertical Navigation in the U.S. National Airspace System and in Remote and Oceanic Airspace. | 这些信息详见 ICAO（International Civil Aviation Organization，国际民用航空组织）Doc 9613《Performance-based Navigation (PBN) Manual》（《基于性能的导航手册》）以及最新版 FAA（Federal Aviation Administration，美国联邦航空管理局）AC（Advisory Circular，咨询通告）90-105《Approval Guidance for RNP Operations and Barometric Vertical Navigation in the U.S. National Airspace System and in Remote and Oceanic Airspace》。 |

### FIG 1-2-1｜图 1-2-1

| English | 中文 |
|---|---|
| Navigation Specifications | 导航规范 |

> 图示说明：原文图示展示了导航规范的分类：RNP（Required Navigation Performance，所需导航性能）规范与 RNAV（Area Navigation，区域导航）规范。RNP（Required Navigation Performance，所需导航性能）规范包含机载性能监视与告警要求；RNAV（Area Navigation，区域导航）规范不包含该要求。

---

### 2. Area Navigation (RNAV)｜区域导航（RNAV，Area Navigation）

#### 1. General｜概述

| English | 中文 |
|---|---|
| RNAV is a method of navigation that permits aircraft operation on any desired flight path within the coverage of ground- or space-based navigation aids or within the limits of the capability of self-contained aids, or a combination of these. In the future, there will be an increased dependence on the use of RNAV in lieu of routes defined by ground-based navigation aids. | RNAV（Area Navigation，区域导航）是一种导航方法，允许航空器在地基或天基导航设施覆盖范围内，或在机载自主导航设备能力限制范围内，或在二者组合条件下，沿任意期望航迹运行。未来，将越来越依赖 RNAV（Area Navigation，区域导航），以替代由地基导航设施定义的航路。 |
| RNAV routes and terminal procedures, including departure procedures (DPs) and standard terminal arrivals (STARs), are designed with RNAV systems in mind. There are several potential advantages of RNAV routes and procedures: | RNAV（Area Navigation，区域导航）航路和终端程序，包括 DP（Departure Procedure，离场程序）和 STAR（Standard Terminal Arrival，标准终端进场程序），均按 RNAV（Area Navigation，区域导航）系统的使用来设计。RNAV（Area Navigation，区域导航）航路和程序具有若干潜在优势： |
| 1. Time and fuel savings; | 1. 节省时间和燃油； |
| 2. Reduced dependence on radar vectoring, altitude, and speed assignments allowing a reduction in required ATC radio transmissions; and | 2. 减少对雷达引导、高度指配和速度指配的依赖，从而减少所需 ATC（Air Traffic Control，空中交通管制）无线电通信；以及 |
| 3. More efficient use of airspace. | 3. 更高效地使用空域。 |
| In addition to information found in this manual, guidance for domestic RNAV DPs, STARs, and routes may also be found in AC 90-100, U.S. Terminal and En Route Area Navigation (RNAV) Operations. | 除本手册中的信息外，有关美国国内 RNAV（Area Navigation，区域导航）DP（Departure Procedure，离场程序）、STAR（Standard Terminal Arrival，标准终端进场程序）和航路的指导，也可参见 AC（Advisory Circular，咨询通告）90-100《U.S. Terminal and En Route Area Navigation (RNAV) Operations》。 |

#### 2. RNAV Operations｜RNAV（Area Navigation，区域导航）运行

| English | 中文 |
|---|---|
| RNAV procedures, such as DPs and STARs, demand strict pilot awareness and maintenance of the procedure centerline. Pilots should possess a working knowledge of their aircraft navigation system to ensure RNAV procedures are flown in an appropriate manner. In addition, pilots should have an understanding of the various waypoint and leg types used in RNAV procedures; these are discussed in more detail below. | RNAV（Area Navigation，区域导航）程序，例如 DP（Departure Procedure，离场程序）和 STAR（Standard Terminal Arrival，标准终端进场程序），要求飞行员保持高度情景意识，并严格保持程序中心线。飞行员应具备航空器导航系统的实用知识，以确保按适当方式飞行 RNAV（Area Navigation，区域导航）程序。此外，飞行员还应了解 RNAV（Area Navigation，区域导航）程序中使用的各种航路点和航段类型；下文将对此作进一步说明。 |

##### 1. Waypoints｜航路点

| English | 中文 |
|---|---|
| A waypoint is a predetermined geographical position that is defined in terms of latitude/longitude coordinates. Waypoints may be a simple named point in space or associated with existing navaids, intersections, or fixes. A waypoint is most often used to indicate a change in direction, speed, or altitude along the desired path. RNAV procedures make use of both fly-over and fly-by waypoints. | 航路点是一个预先确定的地理位置，以经纬度坐标定义。航路点可以是空间中一个简单命名点，也可以与现有 NAVAID（Navigation Aid，导航设施）、交叉点或定位点相关联。航路点最常用于表示沿期望航迹的方向、速度或高度变化。RNAV（Area Navigation，区域导航）程序同时使用飞越航路点和飞越前转弯航路点。 |
| Fly-by waypoints. Fly-by waypoints are used when an aircraft should begin a turn to the next course prior to reaching the waypoint separating the two route segments. This is known as turn anticipation. | 飞越前转弯航路点。当前后两个航段之间的分界航路点尚未到达，而航空器应提前开始转向下一航向时，使用飞越前转弯航路点。这称为转弯预置。 |
| Fly-over waypoints. Fly-over waypoints are used when the aircraft must fly over the point prior to starting a turn. | 飞越航路点。当航空器必须先飞越该点再开始转弯时，使用飞越航路点。 |
| NOTE: FIG 1-2-2 illustrates several differences between a fly-by and a fly-over waypoint. | 注：图 1-2-2 展示了飞越前转弯航路点与飞越航路点之间的若干差异。 |

### FIG 1-2-2｜图 1-2-2

| English | 中文 |
|---|---|
| Fly-by and Fly-over Waypoints | 飞越前转弯航路点与飞越航路点 |

##### 2. RNAV Leg Types｜RNAV（Area Navigation，区域导航）航段类型

| English | 中文 |
|---|---|
| A leg type describes the desired path proceeding, following, or between waypoints on an RNAV procedure. Leg types are identified by a two-letter code that describes the path (e.g., heading, course, track, etc.) and the termination point (e.g., the path terminates at an altitude, distance, fix, etc.). Leg types used for procedure design are included in the aircraft navigation database, but not normally provided on the procedure chart. | 航段类型描述 RNAV（Area Navigation，区域导航）程序中通向、离开或位于航路点之间的期望航迹。航段类型以两个字母代码标识，用于描述航迹形式（例如航向、航线、航迹等）和终止点（例如航迹终止于某一高度、距离、定位点等）。用于程序设计的航段类型包含在航空器导航数据库中，但通常不会显示在程序图上。 |
| The narrative depiction of the RNAV chart describes how a procedure is flown. The “path and terminator concept” defines that every leg of a procedure has a termination point and some kind of path into that termination point. Some of the available leg types are described below. | RNAV（Area Navigation，区域导航）图的文字描述说明程序应如何飞行。“航迹与终止点概念”规定，程序的每个航段都有一个终止点，并有某种通向该终止点的航迹。以下描述部分可用航段类型。 |
| Track to Fix. A Track to Fix (TF) leg is intercepted and acquired as the flight track to the following waypoint. Track to a Fix legs are sometimes called point-to-point legs for this reason. Narrative: “direct ALPHA, then on course to BRAVO WP.” See FIG 1-2-3. | 航迹至定位点。TF（Track to Fix，航迹至定位点）航段被截获并建立为飞向后续航路点的飞行航迹。因此，TF（Track to Fix，航迹至定位点）航段有时也称为点对点航段。文字描述：“直飞 ALPHA，然后沿航线飞向 BRAVO WP（Waypoint，航路点）。”见图 1-2-3。 |
| Direct to Fix. A Direct to Fix (DF) leg is a path described by an aircraft's track from an initial area direct to the next waypoint. Narrative: “turn right direct BRAVO WP.” See FIG 1-2-4. | 直飞定位点。DF（Direct to Fix，直飞定位点）航段是航空器从初始区域直接飞向下一航路点的航迹。文字描述：“右转直飞 BRAVO WP（Waypoint，航路点）。”见图 1-2-4。 |

### FIG 1-2-3｜图 1-2-3

| English | 中文 |
|---|---|
| Track to Fix Leg Type | 航迹至定位点航段类型 |

### FIG 1-2-4｜图 1-2-4

| English | 中文 |
|---|---|
| Direct to Fix Leg Type | 直飞定位点航段类型 |

| English | 中文 |
|---|---|
| Course to Fix. A Course to Fix (CF) leg is a path that terminates at a fix with a specified course at that fix. Narrative: “on course 150 to ALPHA WP.” See FIG 1-2-5. | 航线至定位点。CF（Course to Fix，航线至定位点）航段是一条以指定航线终止于某一定位点的航迹。文字描述：“沿 150 航线飞向 ALPHA WP（Waypoint，航路点）。”见图 1-2-5。 |

### FIG 1-2-5｜图 1-2-5

| English | 中文 |
|---|---|
| Course to Fix Leg Type | 航线至定位点航段类型 |

| English | 中文 |
|---|---|
| Radius to Fix. A Radius to Fix (RF) leg is defined as a constant radius circular path around a defined turn center that terminates at a fix. See FIG 1-2-6. | 半径至定位点。RF（Radius to Fix，半径至定位点）航段定义为围绕某一既定转弯中心的恒定半径圆弧航迹，并终止于某一定位点。见图 1-2-6。 |

### FIG 1-2-6｜图 1-2-6

| English | 中文 |
|---|---|
| Radius to Fix Leg Type | 半径至定位点航段类型 |

| English | 中文 |
|---|---|
| Heading. A Heading leg may be defined as, but not limited to, a Heading to Altitude (VA), Heading to DME range (VD), and Heading to Manual Termination, i.e., Vector (VM). Narrative: “climb heading 350 to 1500”, “heading 265, at 9 DME west of PXR VORTAC, right turn heading 360”, “fly heading 090, expect radar vectors to DRYHT INT.” | 航向。航向航段可定义为但不限于：VA（Heading to Altitude，航向至高度）、VD（Heading to DME range，航向至 DME 距离）以及 VM（Heading to Manual Termination，航向至人工终止，即雷达引导）。文字描述：“以 350 航向爬升至 1500”、“航向 265，在 PXR VORTAC（VOR/TACAN，甚高频全向信标 / 战术空中导航合装台）以西 9 DME（Distance Measuring Equipment，测距设备）处，右转航向 360”、“飞航向 090，预期雷达引导至 DRYHT INT（Intersection，交叉点）。” |

##### 3. Navigation Issues｜导航问题

| English | 中文 |
|---|---|
| Pilots should be aware of their navigation system inputs, alerts, and annunciations in order to make better-informed decisions. In addition, the availability and suitability of particular sensors/systems should be considered. | 飞行员应了解其导航系统输入、告警和显示提示，以便做出信息更充分的决策。此外，还应考虑特定传感器 / 系统的可用性和适用性。 |
| GPS/WAAS. Operators using TSO-C129(), TSO-C196(), TSO-C145() or TSO-C146() systems should ensure departure and arrival airports are entered to ensure proper RAIM availability and CDI sensitivity. | GPS（Global Positioning System，全球定位系统）/ WAAS（Wide Area Augmentation System，广域增强系统）。使用 TSO（Technical Standard Order，技术标准令）-C129()、TSO（Technical Standard Order，技术标准令）-C196()、TSO（Technical Standard Order，技术标准令）-C145() 或 TSO（Technical Standard Order，技术标准令）-C146() 系统的运行人，应确保输入离场和到达机场，以保证 RAIM（Receiver Autonomous Integrity Monitoring，接收机自主完好性监视）可用性和 CDI（Course Deviation Indicator，航道偏离指示器）灵敏度正确。 |
| DME/DME. Operators should be aware that DME/DME position updating is dependent on navigation system logic and DME facility proximity, availability, geometry, and signal masking. | DME（Distance Measuring Equipment，测距设备）/ DME（Distance Measuring Equipment，测距设备）。运行人应了解，DME/DME 位置更新取决于导航系统逻辑，以及 DME（Distance Measuring Equipment，测距设备）设施的距离、可用性、几何分布和信号遮蔽情况。 |
| VOR/DME. Unique VOR characteristics may result in less accurate values from VOR/DME position updating than from GPS or DME/DME position updating. | VOR（VHF Omni-directional Range，甚高频全向信标）/ DME（Distance Measuring Equipment，测距设备）。VOR（VHF Omni-directional Range，甚高频全向信标）的独有特性可能导致 VOR/DME 位置更新的精度低于 GPS（Global Positioning System，全球定位系统）或 DME/DME 位置更新。 |
| Inertial Navigation. Inertial reference units and inertial navigation systems are often coupled with other types of navigation inputs, e.g., DME/DME or GPS, to improve overall navigation system performance. | 惯性导航。IRU（Inertial Reference Unit，惯性基准组件）和惯性导航系统常与其他类型的导航输入耦合使用，例如 DME/DME（Distance Measuring Equipment / Distance Measuring Equipment，双测距设备）或 GPS（Global Positioning System，全球定位系统），以提高整体导航系统性能。 |
| NOTE: Specific inertial position updating requirements may apply. | 注：可能适用特定的惯性位置更新要求。 |

##### 4. Flight Management System (FMS)｜飞行管理系统（FMS，Flight Management System）

| English | 中文 |
|---|---|
| An FMS is an integrated suite of sensors, receivers, and computers, coupled with a navigation database. These systems generally provide performance and RNAV guidance to displays and automatic flight control systems. | FMS（Flight Management System，飞行管理系统）是由传感器、接收机和计算机组成的综合系统，并与导航数据库相结合。这些系统通常向显示器和自动飞行控制系统提供性能信息和 RNAV（Area Navigation，区域导航）引导。 |
| Inputs can be accepted from multiple sources such as GPS, DME, VOR, LOC and IRU. These inputs may be applied to a navigation solution one at a time or in combination. Some FMSs provide for the detection and isolation of faulty navigation information. | 输入可来自多个来源，例如 GPS（Global Positioning System，全球定位系统）、DME（Distance Measuring Equipment，测距设备）、VOR（VHF Omni-directional Range，甚高频全向信标）、LOC（Localizer，航向道）和 IRU（Inertial Reference Unit，惯性基准组件）。这些输入可单独或组合用于导航解算。有些 FMS（Flight Management System，飞行管理系统）具备故障导航信息的探测与隔离能力。 |
| When appropriate navigation signals are available, FMSs will normally rely on GPS and/or DME/DME (that is, the use of distance information from two or more DME stations) for position updates. Other inputs may also be incorporated based on FMS system architecture and navigation source geometry. | 当适当的导航信号可用时，FMS（Flight Management System，飞行管理系统）通常依靠 GPS（Global Positioning System，全球定位系统）和 / 或 DME/DME（Distance Measuring Equipment / Distance Measuring Equipment，双测距设备；即使用来自两个或更多 DME 台站的距离信息）进行位置更新。也可根据 FMS（Flight Management System，飞行管理系统）系统架构和导航源几何关系纳入其他输入。 |
| NOTE: DME/DME inputs coupled with one or more IRU(s) are often abbreviated as DME/DME/IRU or D/D/I. | 注：DME/DME（Distance Measuring Equipment / Distance Measuring Equipment，双测距设备）输入与一个或多个 IRU（Inertial Reference Unit，惯性基准组件）耦合时，通常缩写为 DME/DME/IRU（Distance Measuring Equipment / Distance Measuring Equipment / Inertial Reference Unit）或 D/D/I（DME/DME/IRU）。 |

##### 5. RNAV Navigation Specifications (Nav Specs)｜RNAV（Area Navigation，区域导航）导航规范（Nav Specs，Navigation Specifications）

| English | 中文 |
|---|---|
| Nav Specs are a set of aircraft and aircrew requirements needed to support a navigation application within a defined airspace concept. For both RNP and RNAV designations, the numerical designation refers to the lateral navigation accuracy in nautical miles which is expected to be achieved at least 95 percent of the flight time by the population of aircraft operating within the airspace, route, or procedure. (See FIG 1-2-1.) | Nav Specs（Navigation Specifications，导航规范）是在特定空域概念中支持某项导航应用所需的一组航空器和机组要求。对于 RNP（Required Navigation Performance，所需导航性能）和 RNAV（Area Navigation，区域导航）标识，数字标识均指以 NM（Nautical Mile，海里）为单位的横向导航精度；在相关空域、航路或程序内运行的航空器群体预计至少在 95% 的飞行时间内达到该精度。见图 1-2-1。 |
| RNAV 1. Typically RNAV 1 is used for DPs and STARs and appears on the charts. Aircraft must maintain a total system error of not more than 1 NM for 95 percent of the total flight time. | RNAV（Area Navigation，区域导航）1。RNAV 1 通常用于 DP（Departure Procedure，离场程序）和 STAR（Standard Terminal Arrival，标准终端进场程序），并显示在航图上。航空器在总飞行时间的 95% 内，必须保持总系统误差不超过 1 NM（Nautical Mile，海里）。 |
| RNAV 2. Typically RNAV 2 is used for en route operations unless otherwise specified. T-routes and Q-routes are examples of this Nav Spec. Aircraft must maintain a total system error of not more than 2 NM for 95 percent of the total flight time. | RNAV（Area Navigation，区域导航）2。除非另有规定，RNAV 2 通常用于航路运行。T 航路和 Q 航路属于这一 NavSpec（Navigation Specification，导航规范）的示例。航空器在总飞行时间的 95% 内，必须保持总系统误差不超过 2 NM（Nautical Mile，海里）。 |
| RNAV 10. Typically RNAV 10 is used in oceanic operations. See paragraph 4-7-1 for specifics and explanation of the relationship between RNP 10 and RNAV 10 terminology. | RNAV（Area Navigation，区域导航）10。RNAV 10 通常用于远洋运行。有关具体内容以及 RNP（Required Navigation Performance，所需导航性能）10 与 RNAV（Area Navigation，区域导航）10 术语之间关系的说明，见第 4-7-1 段。 |

---

## 1-2-2. Required Navigation Performance (RNP)｜所需导航性能（RNP，Required Navigation Performance）

### 1. General｜概述

| English | 中文 |
|---|---|
| While both RNAV navigation specifications (NavSpecs) and RNP NavSpecs contain specific performance requirements, RNP is RNAV with the added requirement for onboard performance monitoring and alerting (OBPMA). RNP is also a statement of navigation performance necessary for operation within a defined airspace. | 虽然 RNAV（Area Navigation，区域导航）导航规范（NavSpecs，Navigation Specifications）和 RNP（Required Navigation Performance，所需导航性能）导航规范都包含特定性能要求，但 RNP（Required Navigation Performance，所需导航性能）是在 RNAV（Area Navigation，区域导航）基础上增加 OBPMA（Onboard Performance Monitoring and Alerting，机载性能监视与告警）要求。RNP（Required Navigation Performance，所需导航性能）也是在特定空域内运行所必需的导航性能声明。 |
| A critical component of RNP is the ability of the aircraft navigation system to monitor its achieved navigation performance, and to identify for the pilot whether the operational requirement is, or is not, being met during an operation. OBPMA capability therefore allows a lessened reliance on air traffic control intervention and/or procedural separation to achieve the overall safety of the operation. | RNP（Required Navigation Performance，所需导航性能）的一个关键组成部分，是航空器导航系统能够监视其已达到的导航性能，并向飞行员识别运行过程中是否满足运行要求。因此，OBPMA（Onboard Performance Monitoring and Alerting，机载性能监视与告警）能力可降低对 ATC（Air Traffic Control，空中交通管制）干预和 / 或程序间隔的依赖，以实现运行的整体安全。 |
| RNP capability of the aircraft is a major component in determining the separation criteria to ensure that the overall containment of the operation is met. The RNP capability of an aircraft will vary depending upon the aircraft equipment and the navigation infrastructure. For example, an aircraft may be eligible for RNP 1, but may not be capable of RNP 1 operations due to limited NAVAID coverage or avionics failure. | 航空器的 RNP（Required Navigation Performance，所需导航性能）能力是确定间隔标准的重要组成部分，用于确保满足运行的整体包容要求。航空器的 RNP（Required Navigation Performance，所需导航性能）能力会因航空器设备和导航基础设施而异。例如，某航空器可能具备 RNP 1 资格，但由于 NAVAID（Navigation Aid，导航设施）覆盖有限或航空电子设备故障，可能无法执行 RNP 1 运行。 |
| The Aircraft Flight Manual (AFM) or avionics documents for your aircraft should specifically state the aircraft's RNP eligibilities. Contact the manufacturer of the avionics or the aircraft if this information is missing or incomplete. NavSpecs should be considered different from one another, not “better” or “worse” based on the described lateral navigation accuracy. It is this concept that requires each NavSpec eligibility to be listed separately in the avionics documents or AFM. | 航空器的 AFM（Aircraft Flight Manual，飞机飞行手册）或航空电子设备文件应明确列明航空器的 RNP（Required Navigation Performance，所需导航性能）资格。如该信息缺失或不完整，应联系航空电子设备或航空器制造商。各 NavSpec（Navigation Specification，导航规范）应视为彼此不同，而不是仅依据所描述的横向导航精度来判断“更好”或“更差”。正是这一概念要求每一项 NavSpec（Navigation Specification，导航规范）资格必须在航空电子设备文件或 AFM（Aircraft Flight Manual，飞机飞行手册）中分别列明。 |
| For example, RNP 1 is different from RNAV 1, and an RNP 1 eligibility does NOT mean automatic RNP 2 or RNAV 1 eligibility. As a safeguard, the FAA requires that aircraft navigation databases hold only those procedures that the aircraft maintains eligibility for. If you look for a specific instrument procedure in your aircraft's navigation database and cannot find it, it's likely that procedure contains PBN elements your aircraft is ineligible for or cannot compute and fly. | 例如，RNP（Required Navigation Performance，所需导航性能）1 不同于 RNAV（Area Navigation，区域导航）1；具备 RNP 1 资格并不表示自动具备 RNP 2 或 RNAV 1 资格。作为安全保障，FAA（Federal Aviation Administration，美国联邦航空管理局）要求航空器导航数据库仅包含该航空器保持资格的程序。如果你在航空器导航数据库中查找某一特定仪表程序而找不到，该程序很可能包含航空器不具备资格、无法计算或无法飞行的 PBN（Performance-Based Navigation，基于性能的导航）要素。 |
| Further, optional capabilities such as Radius-to-fix (RF) turns or scalability should be described in the AFM or avionics documents. Use the capabilities of your avionics suite to verify the appropriate waypoint and track data after loading the procedure from your database. | 此外，RF（Radius-to-Fix，半径至定位点）转弯或可缩放性等可选能力，也应在 AFM（Aircraft Flight Manual，飞机飞行手册）或航空电子设备文件中说明。从数据库加载程序后，应使用航空电子套件的能力核实相应航路点和航迹数据。 |

### 2. PBN Operations｜PBN（Performance-Based Navigation，基于性能的导航）运行

#### 1. Lateral Accuracy Values｜横向精度值

| English | 中文 |
|---|---|
| Lateral Accuracy values are applicable to a selected airspace, route, or procedure. The lateral accuracy value is a value typically expressed as a distance in nautical miles from the intended centerline of a procedure, route, or path. RNP applications also account for potential errors at some multiple of lateral accuracy value (for example, twice the RNP lateral accuracy values). | 横向精度值适用于所选空域、航路或程序。横向精度值通常以距程序、航路或航迹预期中心线的距离表示，单位为 NM（Nautical Mile，海里）。RNP（Required Navigation Performance，所需导航性能）应用还会考虑横向精度值某一倍数范围内的潜在误差，例如 RNP（Required Navigation Performance，所需导航性能）横向精度值的两倍。 |
| RNP NavSpecs. U.S. standard NavSpecs supporting typical RNP airspace uses are as specified below. Other NavSpecs may include different lateral accuracy values as identified by ICAO or other states. (See FIG 1-2-1.) | RNP（Required Navigation Performance，所需导航性能）NavSpecs（Navigation Specifications，导航规范）。支持典型 RNP（Required Navigation Performance，所需导航性能）空域使用的美国标准 NavSpecs（Navigation Specifications，导航规范）如下。其他 NavSpecs（Navigation Specifications，导航规范）可能包含 ICAO（International Civil Aviation Organization，国际民用航空组织）或其他国家确定的不同横向精度值。见图 1-2-1。 |
| RNP Approach (RNP APCH). In the U.S., RNP APCH procedures are titled RNAV(GPS) and offer several lines of minima to accommodate varying levels of aircraft equipage: either lateral navigation (LNAV), LNAV/vertical navigation (LNAV/VNAV), Localizer Performance with Vertical Guidance (LPV), and Localizer Performance (LP). GPS with or without Space-Based Augmentation System (SBAS) (for example, WAAS) can provide the lateral information to support LNAV minima. | RNP（Required Navigation Performance，所需导航性能）进近（RNP APCH，Required Navigation Performance Approach）。在美国，RNP APCH 程序标题为 RNAV(GPS)（Area Navigation / Global Positioning System，区域导航 / 全球定位系统），并提供多条最低标准，以适应不同航空器设备水平：LNAV（Lateral Navigation，横向导航）、LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向导航 / 垂直导航）、LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）和 LP（Localizer Performance，航向道性能）。带或不带 SBAS（Space-Based Augmentation System，星基增强系统）（例如 WAAS（Wide Area Augmentation System，广域增强系统））的 GPS（Global Positioning System，全球定位系统），均可提供支持 LNAV（Lateral Navigation，横向导航）最低标准的横向信息。 |
| LNAV/VNAV incorporates LNAV lateral with vertical path guidance for systems and operators capable of either barometric or SBAS vertical. Pilots are required to use SBAS to fly to the LPV or LP minima. RF turn capability is optional in RNP APCH eligibility. This means that your aircraft may be eligible for RNP APCH operations, but you may not fly an RF turn unless RF turns are also specifically listed as a feature of your avionics suite. | LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向导航 / 垂直导航）将 LNAV（Lateral Navigation，横向导航）横向引导与垂直航径引导结合，适用于具备气压垂直或 SBAS（Space-Based Augmentation System，星基增强系统）垂直能力的系统和运行人。飞行员必须使用 SBAS（Space-Based Augmentation System，星基增强系统）飞至 LPV（Localizer Performance with Vertical Guidance，带垂直引导的航向道性能）或 LP（Localizer Performance，航向道性能）最低标准。RF（Radius-to-Fix，半径至定位点）转弯能力在 RNP APCH（Required Navigation Performance Approach，所需导航性能进近）资格中属于可选项。这意味着你的航空器可能具备 RNP APCH 运行资格，但除非 RF（Radius-to-Fix，半径至定位点）转弯也明确列为航空电子套件的一项功能，否则不得飞 RF 转弯。 |
| GBAS Landing System (GLS) procedures are also constructed using RNP APCH NavSpecs and provide precision approach capability. RNP APCH has a lateral accuracy value of 1 in the terminal and missed approach segments and essentially scales to RNP 0.3 (or 40 meters with SBAS) in the final approach. (See paragraph 5-4-18, RNP AR (Authorization Required) Instrument Procedures.) | GLS（GBAS Landing System，地基增强系统着陆系统）程序也使用 RNP APCH（Required Navigation Performance Approach，所需导航性能进近）NavSpecs（Navigation Specifications，导航规范）构建，并提供精密进近能力。RNP APCH（Required Navigation Performance Approach，所需导航性能进近）在终端和复飞航段的横向精度值为 1，并在最后进近中实质上缩放至 RNP（Required Navigation Performance，所需导航性能）0.3，或在使用 SBAS（Space-Based Augmentation System，星基增强系统）时为 40 米。参见第 5-4-18 段 RNP AR（Required Navigation Performance Authorization Required，所需导航性能-需要授权）仪表程序。 |
| RNP Authorization Required Approach (RNP AR APCH). In the U.S., RNP AR APCH procedures are titled RNAV (RNP). These approaches have stringent equipage and pilot training standards and require special FAA authorization to fly. Scalability and RF turn capabilities are mandatory in RNP AR APCH eligibility. RNP AR APCH vertical navigation performance is based upon barometric VNAV or SBAS. RNP AR is intended to provide specific benefits at specific locations. | RNP（Required Navigation Performance，所需导航性能）需要授权进近（RNP AR APCH，Required Navigation Performance Authorization Required Approach）。在美国，RNP AR APCH 程序标题为 RNAV (RNP)（Area Navigation / Required Navigation Performance，区域导航 / 所需导航性能）。这些进近具有严格的设备和飞行员训练标准，并要求获得 FAA（Federal Aviation Administration，美国联邦航空管理局）特殊授权后方可飞行。可缩放性和 RF（Radius-to-Fix，半径至定位点）转弯能力是 RNP AR APCH 资格的强制要求。RNP AR APCH 垂直导航性能基于气压 VNAV（Vertical Navigation，垂直导航）或 SBAS（Space-Based Augmentation System，星基增强系统）。RNP AR（Required Navigation Performance Authorization Required，所需导航性能-需要授权）旨在在特定地点提供特定效益。 |
| It is not intended for every operator or aircraft. RNP AR capability requires specific aircraft performance, design, operational processes, training, and specific procedure design criteria to achieve the required target level of safety. RNP AR APCH has lateral accuracy values that can range below 1 in the terminal and missed approach segments and essentially scale to RNP 0.3 or lower in the final approach. | 它并非面向所有运行人或航空器。RNP AR（Required Navigation Performance Authorization Required，所需导航性能-需要授权）能力要求特定航空器性能、设计、运行流程、训练以及特定程序设计标准，以达到所需目标安全水平。RNP AR APCH（Required Navigation Performance Authorization Required Approach，所需导航性能-需要授权进近）在终端和复飞航段的横向精度值可低于 1，并在最后进近中实质上缩放至 RNP（Required Navigation Performance，所需导航性能）0.3 或更低。 |
| Before conducting these procedures, operators should refer to the latest AC 90-101, Approval Guidance for RNP Procedures with AR. (See paragraph 5-4-18.) | 执行这些程序前，运行人应参阅最新版 AC（Advisory Circular，咨询通告）90-101《Approval Guidance for RNP Procedures with AR》。参见第 5-4-18 段。 |
| RNP Authorization Required Departure (RNP AR DP). Similar to RNP AR approaches, RNP AR departure procedures have stringent equipage and pilot training standards and require special FAA authorization to fly. Scalability and RF turn capabilities is mandatory in RNP AR DP eligibility. RNP AR DP is intended to provide specific benefits at specific locations. It is not intended for every operator or aircraft. | RNP（Required Navigation Performance，所需导航性能）需要授权离场（RNP AR DP，Required Navigation Performance Authorization Required Departure）。与 RNP AR（Required Navigation Performance Authorization Required，所需导航性能-需要授权）进近类似，RNP AR（Required Navigation Performance Authorization Required，所需导航性能-需要授权）离场程序具有严格的设备和飞行员训练标准，并要求获得 FAA（Federal Aviation Administration，美国联邦航空管理局）特殊授权后方可飞行。可缩放性和 RF（Radius-to-Fix，半径至定位点）转弯能力是 RNP AR DP（Required Navigation Performance Authorization Required Departure，所需导航性能-需要授权离场）资格的强制要求。RNP AR DP 旨在在特定地点提供特定效益，并非面向所有运行人或航空器。 |
| RNP AR DP capability requires specific aircraft performance, design, operational processes, training, and specific procedure design criteria to achieve the required target level of safety. RNP AR DP has lateral accuracy values that can scale to no lower than RNP 0.3 in the initial departure flight path. Before conducting these procedures, operators should refer to the latest AC 90-101, Approval Guidance for RNP Procedures with AR. (See paragraph 5-4-18.) | RNP AR DP（Required Navigation Performance Authorization Required Departure，所需导航性能-需要授权离场）能力要求特定航空器性能、设计、运行流程、训练和特定程序设计标准，以达到所需目标安全水平。RNP AR DP（Required Navigation Performance Authorization Required Departure，所需导航性能-需要授权离场）在初始离场航迹中的横向精度值可缩放，但不得低于 RNP（Required Navigation Performance，所需导航性能）0.3。执行这些程序前，运行人应参阅最新版 AC（Advisory Circular，咨询通告）90-101《Approval Guidance for RNP Procedures with AR》。参见第 5-4-18 段。 |
| Advanced RNP (A-RNP). Advanced RNP is a NavSpec with a minimum set of mandatory functions enabled in the aircraft's avionics suite. In the U.S., these minimum functions include capability to calculate and perform RF turns, scalable RNP, and parallel offset flight path generation. Higher continuity (such as dual systems) may be required for certain oceanic and remote continental airspace. | 高级 RNP（A-RNP，Advanced Required Navigation Performance）。A-RNP（Advanced Required Navigation Performance，高级所需导航性能）是一种 NavSpec（Navigation Specification，导航规范），要求航空器航空电子套件启用一组最低强制功能。在美国，这些最低功能包括计算和执行 RF（Radius-to-Fix，半径至定位点）转弯的能力、可缩放 RNP（Required Navigation Performance，所需导航性能）能力，以及平行偏置航迹生成能力。某些远洋和偏远大陆空域可能要求更高连续性，例如双套系统。 |
| Other “advanced” options for use in the en route environment (such as fixed radius transitions and Time of Arrival Control) are optional in the U.S. Typically, an aircraft eligible for A-RNP will also be eligible for operations comprising: RNP APCH, RNP/RNAV 1, RNP/RNAV 2, RNP 4, and RNP/RNAV 10. A-RNP allows for scalable RNP lateral navigation values (either 1.0 or 0.3) in the terminal environment. | 其他用于航路环境的“高级”选项，例如固定半径过渡和到达时间控制，在美国属于可选项。通常，具备 A-RNP（Advanced Required Navigation Performance，高级所需导航性能）资格的航空器，也将具备以下运行资格：RNP APCH（Required Navigation Performance Approach，所需导航性能进近）、RNP/RNAV（Required Navigation Performance / Area Navigation，所需导航性能 / 区域导航）1、RNP/RNAV 2、RNP（Required Navigation Performance，所需导航性能）4，以及 RNP/RNAV 10。A-RNP（Advanced Required Navigation Performance，高级所需导航性能）允许在终端环境中使用可缩放 RNP（Required Navigation Performance，所需导航性能）横向导航值，即 1.0 或 0.3。 |
| Use of these reduced lateral accuracies will normally require use of the aircraft's autopilot and/or flight director. See the latest AC 90-105 for more information on A-RNP, including NavSpec bundling options, eligibility determinations, and operations approvals. | 使用这些较小横向精度值通常要求使用航空器自动驾驶和 / 或飞行指引仪。关于 A-RNP（Advanced Required Navigation Performance，高级所需导航性能）的更多信息，包括 NavSpec（Navigation Specification，导航规范）组合选项、资格确定和运行批准，见最新版 AC（Advisory Circular，咨询通告）90-105。 |
| NOTE: A-RNP eligible aircraft are NOT automatically eligible for RNP AR APCH or RNP AR DP operations, as RNP AR eligibility requires a separate determination process and special FAA authorization. | 注：具备 A-RNP（Advanced Required Navigation Performance，高级所需导航性能）资格的航空器并不会自动具备 RNP AR APCH（Required Navigation Performance Authorization Required Approach，所需导航性能-需要授权进近）或 RNP AR DP（Required Navigation Performance Authorization Required Departure，所需导航性能-需要授权离场）运行资格，因为 RNP AR（Required Navigation Performance Authorization Required，所需导航性能-需要授权）资格需要单独的确定流程和 FAA（Federal Aviation Administration，美国联邦航空管理局）特殊授权。 |
| RNP 1. RNP 1 requires a lateral accuracy value of 1 for arrival and departure in the terminal area, and the initial and intermediate approach phase when used on conventional procedures with PBN segments (for example, an ILS with a PBN feeder, IAF, or missed approach). RF turn capability is optional in RNP 1 eligibility. | RNP（Required Navigation Performance，所需导航性能）1。RNP 1 要求在终端区域进场和离场中，以及在带有 PBN（Performance-Based Navigation，基于性能的导航）航段的传统程序（例如含 PBN（Performance-Based Navigation，基于性能的导航）进场引导航段、IAF（Initial Approach Fix，初始进近定位点）或复飞的 ILS（Instrument Landing System，仪表着陆系统））的初始和中间进近阶段中，横向精度值为 1。RF（Radius-to-Fix，半径至定位点）转弯能力在 RNP 1 资格中属于可选项。 |
| This means that your aircraft may be eligible for RNP 1 operations, but you may not fly an RF turn unless RF turns are also specifically listed as a feature of your avionics suite. | 这意味着你的航空器可能具备 RNP（Required Navigation Performance，所需导航性能）1 运行资格，但除非 RF（Radius-to-Fix，半径至定位点）转弯也明确列为航空电子套件的一项功能，否则不得飞 RF 转弯。 |
| RNP 2. RNP 2 will apply to both domestic and oceanic/remote operations with a lateral accuracy value of 2. | RNP（Required Navigation Performance，所需导航性能）2。RNP 2 适用于国内和远洋 / 偏远地区运行，横向精度值为 2。 |
| RNP 4. RNP 4 will apply to oceanic and remote operations only with a lateral accuracy value of 4. RNP 4 eligibility will automatically confer RNP 10 eligibility. | RNP（Required Navigation Performance，所需导航性能）4。RNP 4 仅适用于远洋和偏远地区运行，横向精度值为 4。具备 RNP 4 资格将自动获得 RNP 10 资格。 |
| RNP 10. The RNP 10 NavSpec applies to certain oceanic and remote operations with a lateral accuracy of 10. In such airspace, the RNAV 10 NavSpec will be applied, so any aircraft eligible for RNP 10 will be deemed eligible for RNAV 10 operations. Further, any aircraft eligible for RNP 4 operations is automatically qualified for RNP 10/ RNAV 10 operations. (See also the latest AC 91-70, Oceanic and Remote Continental Airspace Operations, for more information on oceanic RNP/RNAV operations.) | RNP（Required Navigation Performance，所需导航性能）10。RNP 10 NavSpec（Navigation Specification，导航规范）适用于某些远洋和偏远地区运行，横向精度为 10。在此类空域中，将适用 RNAV（Area Navigation，区域导航）10 NavSpec（Navigation Specification，导航规范），因此任何具备 RNP 10 资格的航空器均视为具备 RNAV 10 运行资格。此外，任何具备 RNP 4 运行资格的航空器，也自动具备 RNP 10 / RNAV 10 运行资格。有关远洋 RNP/RNAV（Required Navigation Performance / Area Navigation，所需导航性能 / 区域导航）运行的更多信息，另见最新版 AC（Advisory Circular，咨询通告）91-70《Oceanic and Remote Continental Airspace Operations》。 |
| RNP 0.3. The RNP 0.3 NavSpec requires a lateral accuracy value of 0.3 for all authorized phases of flight. RNP 0.3 is not authorized for oceanic, remote, or the final approach segment. Use of RNP 0.3 by slow-flying fixed-wing aircraft is under consideration, but the RNP 0.3 NavSpec initially will apply only to rotorcraft operations. RF turn capability is optional in RNP 0.3 eligibility. | RNP（Required Navigation Performance，所需导航性能）0.3。RNP 0.3 NavSpec（Navigation Specification，导航规范）要求在所有获授权飞行阶段中横向精度值为 0.3。RNP 0.3 未获准用于远洋、偏远地区或最后进近航段。慢速固定翼航空器使用 RNP 0.3 正在研究中，但 RNP 0.3 NavSpec（Navigation Specification，导航规范）最初仅适用于旋翼航空器运行。RF（Radius-to-Fix，半径至定位点）转弯能力在 RNP 0.3 资格中属于可选项。 |
| This means that your aircraft may be eligible for RNP 0.3 operations, but you may not fly an RF turn unless RF turns are also specifically listed as a feature of your avionics suite. | 这意味着你的航空器可能具备 RNP（Required Navigation Performance，所需导航性能）0.3 运行资格，但除非 RF（Radius-to-Fix，半径至定位点）转弯也明确列为航空电子套件的一项功能，否则不得飞 RF 转弯。 |
| NOTE: On terminal procedures or en route charts, do not confuse a charted RNP value of 0.30, or any standard final approach course segment width of 0.30, with the NavSpec title “RNP 0.3.” Charted RNP values of 0.30 or below should contain two decimal places (for example, RNP 0.15, or 0.10, or 0.30) whereas the NavSpec title will only state “RNP 0.3.” | 注：在终端程序或航路图上，不要将航图标注的 RNP（Required Navigation Performance，所需导航性能）值 0.30，或任何标准最后进近航线航段宽度 0.30，与 NavSpec（Navigation Specification，导航规范）标题“RNP 0.3”混淆。航图标注的 RNP（Required Navigation Performance，所需导航性能）值为 0.30 或更低时，应包含两位小数，例如 RNP 0.15、0.10 或 0.30；而 NavSpec（Navigation Specification，导航规范）标题只会写作“RNP 0.3”。 |

#### 2. Application of Standard Lateral Accuracy Values｜标准横向精度值的应用

| English | 中文 |
|---|---|
| U.S. standard lateral accuracy values typically used for various routes and procedures supporting RNAV operations may be based on use of a specific navigational system or sensor such as GPS, or on multi-sensor RNAV systems having suitable performance. | 美国用于支持 RNAV（Area Navigation，区域导航）运行的各类航路和程序的标准横向精度值，通常可基于特定导航系统或传感器（例如 GPS（Global Positioning System，全球定位系统））的使用，或基于具备适当性能的多传感器 RNAV（Area Navigation，区域导航）系统。 |

#### 3. Depiction of PBN Requirements｜PBN（Performance-Based Navigation，基于性能的导航）要求的表示

| English | 中文 |
|---|---|
| In the U.S., PBN requirements like Lateral Accuracy Values or NavSpecs applicable to a procedure will be depicted on affected charts and procedures. In the U.S., a specific procedure's Performance-Based Navigation (PBN) requirements will be prominently displayed in separate, standardized notes boxes. | 在美国，适用于某一程序的 PBN（Performance-Based Navigation，基于性能的导航）要求，例如横向精度值或 NavSpecs（Navigation Specifications，导航规范），将在相关航图和程序中表示。在美国，特定程序的 PBN（Performance-Based Navigation，基于性能的导航）要求将以单独、标准化的注释框突出显示。 |
| For procedures with PBN elements, the “PBN box” will contain the procedure's NavSpec(s); and, if required: specific sensors or infrastructure needed for the navigation solution, any additional or advanced functional requirements, the minimum RNP value, and any amplifying remarks. Items listed in this PBN box are REQUIRED to fly the procedure's PBN elements. For example, an ILS with an RNAV missed approach would require a specific capability to fly the missed approach portion of the procedure. | 对含 PBN（Performance-Based Navigation，基于性能的导航）要素的程序，“PBN（Performance-Based Navigation，基于性能的导航）框”将包含该程序的 NavSpec（Navigation Specification，导航规范）；如有要求，还会列出导航解算所需的特定传感器或基础设施、任何附加或高级功能要求、最低 RNP（Required Navigation Performance，所需导航性能）值，以及任何补充说明。PBN（Performance-Based Navigation，基于性能的导航）框中列出的项目，是飞行该程序 PBN 要素的必要条件。例如，含 RNAV（Area Navigation，区域导航）复飞的 ILS（Instrument Landing System，仪表着陆系统）程序，将要求具备飞行该程序复飞部分的特定能力。 |
| That required capability will be listed in the PBN box. The separate Equipment Requirements box will list ground-based equipment and/or airport specific requirements. On procedures with both PBN elements and ground-based equipment requirements, the PBN requirements box will be listed first. (See FIG 5-4-1.) | 该必要能力将列在 PBN（Performance-Based Navigation，基于性能的导航）框中。单独的设备要求框将列出地基设备和 / 或机场特定要求。在同时包含 PBN（Performance-Based Navigation，基于性能的导航）要素和地基设备要求的程序上，PBN（Performance-Based Navigation，基于性能的导航）要求框将列在前面。见图 5-4-1。 |

### 3. Other RNP Applications Outside the U.S.｜美国以外的其他 RNP（Required Navigation Performance，所需导航性能）应用

| English | 中文 |
|---|---|
| The FAA and ICAO member states have led initiatives in implementing the RNP concept to oceanic operations. For example, RNP-10 routes have been established in the northern Pacific (NOPAC) which has increased capacity and efficiency by reducing the distance between tracks to 50 NM. (See paragraph 4-7-1.) | FAA（Federal Aviation Administration，美国联邦航空管理局）和 ICAO（International Civil Aviation Organization，国际民用航空组织）成员国一直引领将 RNP（Required Navigation Performance，所需导航性能）概念应用于远洋运行的举措。例如，北太平洋（NOPAC，Northern Pacific）已建立 RNP（Required Navigation Performance，所需导航性能）-10 航路，将航迹间距减小至 50 NM（Nautical Mile，海里），从而提高容量和效率。见第 4-7-1 段。 |

### 4. Aircraft and Airborne Equipment Eligibility for RNP Operations｜航空器及机载设备的 RNP（Required Navigation Performance，所需导航性能）运行资格

| English | 中文 |
|---|---|
| Aircraft eligible for RNP operations will have an appropriate entry including special conditions and limitations in its AFM, avionics manual, or a supplement. Operators of aircraft not having specific RNP eligibility statements in the AFM or avionics documents may be issued operational approval including special conditions and limitations for specific RNP eligibilities. | 具备 RNP（Required Navigation Performance，所需导航性能）运行资格的航空器，其 AFM（Aircraft Flight Manual，飞机飞行手册）、航空电子设备手册或补充文件中将有相应条目，包括特殊条件和限制。对于 AFM（Aircraft Flight Manual，飞机飞行手册）或航空电子设备文件中没有具体 RNP（Required Navigation Performance，所需导航性能）资格声明的航空器，运行人可获得运行批准，其中包括针对特定 RNP（Required Navigation Performance，所需导航性能）资格的特殊条件和限制。 |
| NOTE: Some airborne systems use Estimated Position Uncertainty (EPU) as a measure of the current estimated navigational performance. EPU may also be referred to as Actual Navigation Performance (ANP) or Estimated Position Error (EPE). | 注：某些机载系统使用 EPU（Estimated Position Uncertainty，估算位置不确定度）作为当前估算导航性能的度量。EPU（Estimated Position Uncertainty，估算位置不确定度）也可称为 ANP（Actual Navigation Performance，实际导航性能）或 EPE（Estimated Position Error，估算位置误差）。 |

### TBL 1-2-1｜表 1-2-1：U.S. Standard RNP Levels｜美国标准 RNP（Required Navigation Performance，所需导航性能）等级

| RNP Level | Typical Application | Primary Route Width (NM) - Centerline to Boundary | RNP 等级 | 典型应用 | 主航路宽度（NM，Nautical Mile，海里）——中心线至边界 |
|---|---|---|---|---|---|
| 0.1 to 1.0 | RNP AR Approach Segments | 0.1 to 1.0 | 0.1 至 1.0 | RNP AR（Required Navigation Performance Authorization Required，所需导航性能-需要授权）进近航段 | 0.1 至 1.0 |
| 0.3 to 1.0 | RNP Approach Segments | 0.3 to 1.0 | 0.3 至 1.0 | RNP（Required Navigation Performance，所需导航性能）进近航段 | 0.3 至 1.0 |
| 1 | Terminal and En Route | 1.0 | 1 | 终端和航路 | 1.0 |
| 2 | En Route | 2.0 | 2 | 航路 | 2.0 |
| 4 | Oceanic/remote areas where performance-based horizontal separation is applied. | 4.0 | 4 | 采用基于性能的水平间隔的远洋 / 偏远地区。 | 4.0 |
| 10 | Oceanic/remote areas where performance-based horizontal separation is applied. | 10.0 | 10 | 采用基于性能的水平间隔的远洋 / 偏远地区。 | 10.0 |

---

## 1-2-3. Use of Suitable Area Navigation (RNAV) Systems on Conventional Procedures and Routes｜在传统程序和航路上使用适当的区域导航（RNAV，Area Navigation）系统

### 1. Discussion｜讨论

| English | 中文 |
|---|---|
| This paragraph sets forth policy, while providing operational and airworthiness guidance regarding the suitability and use of RNAV systems when operating on, or transitioning to, conventional, non-RNAV routes and procedures within the U.S. National Airspace System (NAS): | 本段阐明政策，并就航空器在美国 NAS（National Airspace System，国家空域系统）内运行于或过渡至传统非 RNAV（Area Navigation，区域导航）航路和程序时，RNAV（Area Navigation，区域导航）系统的适用性和使用提供运行与适航指导： |
| Use of a suitable RNAV system as a Substitute Means of Navigation when a Very-High Frequency (VHF) Omni-directional Range (VOR), Distance Measuring Equipment (DME), Tactical Air Navigation (TACAN), VOR/TACAN (VORTAC), VOR/DME, Non-directional Beacon (NDB), or compass locator facility including locator outer marker and locator middle marker is out-of-service (that is, the navigation aid (NAVAID) information is not available); an aircraft is not equipped with an Automatic Direction Finder (ADF) or DME; or the installed ADF or DME on an aircraft is not operational. | 当 VOR（VHF Omni-directional Range，甚高频全向信标）、DME（Distance Measuring Equipment，测距设备）、TACAN（Tactical Air Navigation，战术空中导航）、VORTAC（VOR/TACAN，甚高频全向信标 / 战术空中导航合装台）、VOR/DME（VHF Omni-directional Range / Distance Measuring Equipment，甚高频全向信标 / 测距设备）、NDB（Non-directional Beacon，无方向信标）或指南针定位台设施（包括定位外指点标和定位中指点标）停用时，也就是 NAVAID（Navigation Aid，导航设施）信息不可用时；或者航空器未配备 ADF（Automatic Direction Finder，自动定向仪）或 DME（Distance Measuring Equipment，测距设备）；或者航空器已安装的 ADF（Automatic Direction Finder，自动定向仪）或 DME（Distance Measuring Equipment，测距设备）不可用时，可使用适当的 RNAV（Area Navigation，区域导航）系统作为替代导航手段。 |
| For example, if equipped with a suitable RNAV system, a pilot may hold over an out-of-service NDB. | 例如，如果航空器配备适当的 RNAV（Area Navigation，区域导航）系统，飞行员可以在停用的 NDB（Non-directional Beacon，无方向信标）上空等待。 |
| Use of a suitable RNAV system as an Alternate Means of Navigation when a VOR, DME, VORTAC, VOR/DME, TACAN, NDB, or compass locator facility including locator outer marker and locator middle marker is operational and the respective aircraft is equipped with operational navigation equipment that is compatible with conventional navaids. For example, if equipped with a suitable RNAV system, a pilot may fly a procedure or route based on operational VOR using that RNAV system without monitoring the VOR. | 当 VOR（VHF Omni-directional Range，甚高频全向信标）、DME（Distance Measuring Equipment，测距设备）、VORTAC（VOR/TACAN，甚高频全向信标 / 战术空中导航合装台）、VOR/DME（VHF Omni-directional Range / Distance Measuring Equipment，甚高频全向信标 / 测距设备）、TACAN（Tactical Air Navigation，战术空中导航）、NDB（Non-directional Beacon，无方向信标）或指南针定位台设施（包括定位外指点标和定位中指点标）可用，并且相关航空器配备与传统 NAVAID（Navigation Aid，导航设施）兼容的可用导航设备时，可使用适当的 RNAV（Area Navigation，区域导航）系统作为备用导航手段。例如，如果航空器配备适当的 RNAV（Area Navigation，区域导航）系统，飞行员可使用该 RNAV 系统飞行基于可用 VOR（VHF Omni-directional Range，甚高频全向信标）的程序或航路，而无需监视 VOR。 |
| NOTE 1: Additional information and associated requirements are available in Advisory Circular 90-108 titled “Use of Suitable RNAV Systems on Conventional Routes and Procedures.” | 注 1：更多信息及相关要求见 AC（Advisory Circular，咨询通告）90-108《Use of Suitable RNAV Systems on Conventional Routes and Procedures》。 |
| NOTE 2: Good planning and knowledge of your RNAV system are critical for safe and successful operations. | 注 2：良好计划以及对 RNAV（Area Navigation，区域导航）系统的了解，是安全且成功运行的关键。 |
| NOTE 3: Pilots planning to use their RNAV system as a substitute means of navigation guidance in lieu of an out-of-service NAVAID may need to advise ATC of this intent and capability. | 注 3：飞行员如计划使用 RNAV（Area Navigation，区域导航）系统作为替代导航引导手段，以替代停用的 NAVAID（Navigation Aid，导航设施），可能需要向 ATC（Air Traffic Control，空中交通管制）告知这一意图和能力。 |
| NOTE 4: The navigation database should be current for the duration of the flight. If the AIRAC cycle will change during flight, operators and pilots should establish procedures to ensure the accuracy of navigation data, including suitability of navigation facilities used to define the routes and procedures for flight. To facilitate validating database currency, the FAA has developed procedures for publishing the amendment date that instrument approach procedures were last revised. | 注 4：导航数据库在整个飞行期间应保持现行有效。如果 AIRAC（Aeronautical Information Regulation and Control，航空资料规章与管制周期）将在飞行期间更换，运行人和飞行员应建立程序，确保导航数据准确，包括用于定义飞行航路和程序的导航设施是否适用。为便于验证数据库现行性，FAA（Federal Aviation Administration，美国联邦航空管理局）已制定程序，公布仪表进近程序上次修订的修订日期。 |
| The amendment date follows the amendment number, e.g., Amdt 4 14Jan10. Currency of graphic departure procedures and STARs may be ascertained by the numerical designation in the procedure title. If an amended chart is published for the procedure, or the procedure amendment date shown on the chart is on or after the expiration date of the database, the operator must not use the database to conduct the operation. | 修订日期跟在修订编号之后，例如 Amdt 4 14Jan10。图形化离场程序和 STAR（Standard Terminal Arrival，标准终端进场程序）的现行性，可通过程序标题中的数字标识确定。如果某程序发布了修订航图，或航图上显示的程序修订日期等于或晚于数据库失效日期，运行人不得使用该数据库执行运行。 |

### 2. Types of RNAV Systems that Qualify as a Suitable RNAV System｜可作为适当 RNAV（Area Navigation，区域导航）系统的 RNAV 系统类型

| English | 中文 |
|---|---|
| When installed in accordance with appropriate airworthiness installation requirements and operated in accordance with applicable operational guidance (for example, aircraft flight manual and Advisory Circular material), the following systems qualify as a suitable RNAV system: | 当按适用适航安装要求安装，并按适用运行指导运行时（例如 AFM（Aircraft Flight Manual，飞机飞行手册）和 AC（Advisory Circular，咨询通告）材料），以下系统符合适当 RNAV（Area Navigation，区域导航）系统的条件： |
| An RNAV system with TSO-C129/ -C145/-C146 equipment, installed in accordance with AC 20-138, Airworthiness Approval of Global Positioning System (GPS) Navigation Equipment for Use as a VFR and IFR Supplemental Navigation System, and authorized for instrument flight rules (IFR) en route and terminal operations (including those systems previously qualified for “GPS in lieu of ADF or DME” operations), or | 配备 TSO（Technical Standard Order，技术标准令）-C129 / -C145 / -C146 设备的 RNAV（Area Navigation，区域导航）系统，该系统按 AC（Advisory Circular，咨询通告）20-138《Airworthiness Approval of Global Positioning System (GPS) Navigation Equipment for Use as a VFR and IFR Supplemental Navigation System》安装，并获准用于 IFR（Instrument Flight Rules，仪表飞行规则）航路和终端运行，包括此前符合“以 GPS（Global Positioning System，全球定位系统）代替 ADF（Automatic Direction Finder，自动定向仪）或 DME（Distance Measuring Equipment，测距设备）”运行条件的系统；或 |
| An RNAV system with DME/DME/IRU inputs that is compliant with the equipment provisions of AC 90-100A, U.S. Terminal and En Route Area Navigation (RNAV) Operations, for RNAV routes. A table of compliant equipment is available at the following website: https://www.faa.gov/about/office_org/headquarters_offices/avs/offices/afx/afs/afs400/afs410/media/AC90-100compliance.pdf | 具有 DME/DME/IRU（Distance Measuring Equipment / Distance Measuring Equipment / Inertial Reference Unit，双测距设备 / 惯性基准组件）输入的 RNAV（Area Navigation，区域导航）系统，且用于 RNAV 航路时符合 AC（Advisory Circular，咨询通告）90-100A《U.S. Terminal and En Route Area Navigation (RNAV) Operations》的设备规定。合规设备表可在以下网站获得：https://www.faa.gov/about/office_org/headquarters_offices/avs/offices/afx/afs/afs400/afs410/media/AC90-100compliance.pdf |
| NOTE: Approved RNAV systems using DME/DME/IRU, without GPS/WAAS position input, may only be used as a substitute means of navigation when specifically authorized by a Notice to Airmen (NOTAM) or other FAA guidance for a specific procedure. The NOTAM or other FAA guidance authorizing the use of DME/DME/IRU systems will also identify any required DME facilities based on an FAA assessment of the DME navigation infrastructure. | 注：经批准但不使用 GPS（Global Positioning System，全球定位系统）/ WAAS（Wide Area Augmentation System，广域增强系统）位置输入、而使用 DME/DME/IRU（Distance Measuring Equipment / Distance Measuring Equipment / Inertial Reference Unit，双测距设备 / 惯性基准组件）的 RNAV（Area Navigation，区域导航）系统，只有在某一特定程序由 NOTAM（Notice to Airmen / Notice to Air Missions，航行通告）或其他 FAA（Federal Aviation Administration，美国联邦航空管理局）指导明确授权时，才可作为替代导航手段使用。授权使用 DME/DME/IRU 系统的 NOTAM（Notice to Airmen / Notice to Air Missions，航行通告）或其他 FAA 指导，还将根据 FAA 对 DME（Distance Measuring Equipment，测距设备）导航基础设施的评估，列明任何必需的 DME 设施。 |

### 3. Uses of Suitable RNAV Systems｜适当 RNAV（Area Navigation，区域导航）系统的用途

| English | 中文 |
|---|---|
| Subject to the operating requirements, operators may use a suitable RNAV system in the following ways. | 在符合运行要求的前提下，运行人可按以下方式使用适当的 RNAV（Area Navigation，区域导航）系统。 |
| Determine aircraft position relative to, or distance from a VOR (see NOTE 6 below), TACAN, NDB, compass locator, DME fix; or a named fix defined by a VOR radial, TACAN course, NDB bearing, or compass locator bearing intersecting a VOR or localizer course. | 确定航空器相对于 VOR（VHF Omni-directional Range，甚高频全向信标）（见下方注 6）、TACAN（Tactical Air Navigation，战术空中导航）、NDB（Non-directional Beacon，无方向信标）、指南针定位台、DME（Distance Measuring Equipment，测距设备）定位点的位置或距离；或确定由 VOR 径向线、TACAN 航线、NDB 方位线，或与 VOR 或 LOC（Localizer，航向道）航线相交的指南针定位台方位线定义的命名定位点。 |
| Navigate to or from a VOR, TACAN, NDB, or compass locator. | 飞向或飞离 VOR（VHF Omni-directional Range，甚高频全向信标）、TACAN（Tactical Air Navigation，战术空中导航）、NDB（Non-directional Beacon，无方向信标）或指南针定位台。 |
| Hold over a VOR, TACAN, NDB, compass locator, or DME fix. | 在 VOR（VHF Omni-directional Range，甚高频全向信标）、TACAN（Tactical Air Navigation，战术空中导航）、NDB（Non-directional Beacon，无方向信标）、指南针定位台或 DME（Distance Measuring Equipment，测距设备）定位点上空等待。 |
| Fly an arc based upon DME. | 飞行基于 DME（Distance Measuring Equipment，测距设备）的圆弧。 |
| NOTE 1: The allowances described in this section apply even when a facility is identified as required on a procedure (for example, “Note ADF required”). | 注 1：即使某设施在程序中标明为必需，例如“注：需要 ADF（Automatic Direction Finder，自动定向仪）”，本节所述允许事项仍然适用。 |
| NOTE 2: These operations do not include lateral navigation on localizer-based courses (including localizer back-course guidance) without reference to raw localizer data. | 注 2：这些运行不包括在不参照原始 LOC（Localizer，航向道）数据的情况下，在基于 LOC（Localizer，航向道）的航线上进行横向导航，包括航向道反航道引导。 |
| NOTE 3: Unless otherwise specified, a suitable RNAV system cannot be used for navigation on procedures that are identified as not authorized (“NA”) without exception by a NOTAM. For example, an operator may not use a RNAV system to navigate on a procedure affected by an expired or unsatisfactory flight inspection, or a procedure that is based upon a recently decommissioned NAVAID. | 注 3：除非另有规定，对于由 NOTAM（Notice to Airmen / Notice to Air Missions，航行通告）无例外标明为不授权（“NA”，Not Authorized）的程序，不得使用适当 RNAV（Area Navigation，区域导航）系统进行导航。例如，对于受过期或不合格飞行校验影响的程序，或基于最近退役 NAVAID（Navigation Aid，导航设施）的程序，运行人不得使用 RNAV（Area Navigation，区域导航）系统导航。 |
| NOTE 4: Pilots may not substitute for the NAVAID (for example, a VOR or NDB) providing lateral guidance for the final approach segment. This restriction does not refer to instrument approach procedures with “or GPS” in the title when using GPS or WAAS. These allowances do not apply to procedures that are identified as not authorized (NA) without exception by a NOTAM, as other conditions may still exist and result in a procedure not being available. | 注 4：飞行员不得用替代方式取代为最后进近航段提供横向引导的 NAVAID（Navigation Aid，导航设施），例如 VOR（VHF Omni-directional Range，甚高频全向信标）或 NDB（Non-directional Beacon，无方向信标）。这一限制不适用于标题中含有“or GPS（Global Positioning System，全球定位系统）”、并使用 GPS 或 WAAS（Wide Area Augmentation System，广域增强系统）的仪表进近程序。这些允许事项不适用于由 NOTAM（Notice to Airmen / Notice to Air Missions，航行通告）无例外标明为不授权（NA，Not Authorized）的程序，因为可能仍存在其他条件导致该程序不可用。 |
| For example, these allowances do not apply to a procedure associated with an expired or unsatisfactory flight inspection, or is based upon a recently decommissioned NAVAID. | 例如，这些允许事项不适用于与过期或不合格飞行校验相关的程序，也不适用于基于最近退役 NAVAID（Navigation Aid，导航设施）的程序。 |
| NOTE 5: Use of a suitable RNAV system as a means to navigate on the final approach segment of an instrument approach procedure based on a VOR, TACAN or NDB signal, is allowable. The underlying NAVAID must be operational and the NAVAID monitored for final segment course alignment. | 注 5：允许使用适当的 RNAV（Area Navigation，区域导航）系统，在基于 VOR（VHF Omni-directional Range，甚高频全向信标）、TACAN（Tactical Air Navigation，战术空中导航）或 NDB（Non-directional Beacon，无方向信标）信号的仪表进近程序最后进近航段上导航。基础 NAVAID（Navigation Aid，导航设施）必须可用，并且必须监视该 NAVAID，以确保最后航段航线对正。 |
| NOTE 6: For the purpose of paragraph c, “VOR” includes VOR, VOR/DME, and VORTAC facilities and “compass locator” includes locator outer marker and locator middle marker. | 注 6：就 c 段而言，“VOR（VHF Omni-directional Range，甚高频全向信标）”包括 VOR、VOR/DME（VHF Omni-directional Range / Distance Measuring Equipment，甚高频全向信标 / 测距设备）和 VORTAC（VOR/TACAN，甚高频全向信标 / 战术空中导航合装台）设施；“指南针定位台”包括定位外指点标和定位中指点标。 |

### 4. Alternate Airport Considerations｜备降机场考虑事项

| English | 中文 |
|---|---|
| For the purposes of flight planning, any required alternate airport must have an available instrument approach procedure that does not require the use of GPS. This restriction includes conducting a conventional approach at the alternate airport using a substitute means of navigation that is based upon the use of GPS. | 就飞行计划而言，任何必需备降机场都必须具备一个可用的、不要求使用 GPS（Global Positioning System，全球定位系统）的仪表进近程序。该限制包括在备降机场使用基于 GPS（Global Positioning System，全球定位系统）的替代导航手段执行传统进近。 |
| For example, these restrictions would apply when planning to use GPS equipment as a substitute means of navigation for an out-of-service VOR that supports an ILS missed approach procedure at an alternate airport. In this case, some other approach not reliant upon the use of GPS must be available. This restriction does not apply to RNAV systems using TSO-C145/-C146 WAAS equipment. For further WAAS guidance, see paragraph 1-1-18. | 例如，当计划在备降机场使用 GPS（Global Positioning System，全球定位系统）设备作为替代导航手段，以替代支持 ILS（Instrument Landing System，仪表着陆系统）复飞程序但已停用的 VOR（VHF Omni-directional Range，甚高频全向信标）时，这些限制将适用。在这种情况下，必须有其他不依赖 GPS（Global Positioning System，全球定位系统）使用的进近可用。该限制不适用于使用 TSO（Technical Standard Order，技术标准令）-C145 / -C146 WAAS（Wide Area Augmentation System，广域增强系统）设备的 RNAV（Area Navigation，区域导航）系统。有关 WAAS（Wide Area Augmentation System，广域增强系统）的进一步指导，见第 1-1-18 段。 |
| For flight planning purposes, TSO-C129() and TSO-C196() equipped users (GPS users) whose navigation systems have fault detection and exclusion (FDE) capability, who perform a preflight RAIM prediction at the airport where the RNAV (GPS) approach will be flown, and have proper knowledge and any required training and/or approval to conduct a GPS-based IAP, may file based on a GPS-based IAP at either the destination or the alternate airport, but not at both locations. | 就飞行计划而言，配备 TSO（Technical Standard Order，技术标准令）-C129() 和 TSO（Technical Standard Order，技术标准令）-C196() 的用户（GPS（Global Positioning System，全球定位系统）用户），如果其导航系统具备 FDE（Fault Detection and Exclusion，故障探测与排除）能力，在将执行 RNAV (GPS)（Area Navigation / Global Positioning System，区域导航 / 全球定位系统）进近的机场进行了飞行前 RAIM（Receiver Autonomous Integrity Monitoring，接收机自主完好性监视）预测，并具备执行基于 GPS（Global Positioning System，全球定位系统）的 IAP（Instrument Approach Procedure，仪表进近程序）所需的适当知识以及任何必要训练和 / 或批准，则可在目的地机场或备降机场之一按基于 GPS（Global Positioning System，全球定位系统）的 IAP（Instrument Approach Procedure，仪表进近程序）提交飞行计划，但不得同时在两个地点都这样提交。 |
| At the alternate airport, pilots may plan for applicable alternate airport weather minimums using: | 在备降机场，飞行员可使用以下方式规划适用的备降机场天气最低标准： |
| 1. Lateral navigation (LNAV) or circling minimum descent altitude (MDA); | 1. LNAV（Lateral Navigation，横向导航）或盘旋 MDA（Minimum Descent Altitude，最低下降高度）； |
| 2. LNAV/vertical navigation (LNAV/VNAV) DA, if equipped with and using approved barometric vertical navigation (baro-VNAV) equipment; | 2. 如果配备并使用经批准的气压 VNAV（Vertical Navigation，垂直导航）设备，则可使用 LNAV/VNAV（Lateral Navigation / Vertical Navigation，横向导航 / 垂直导航）DA（Decision Altitude，决断高度）； |
| 3. RNP 0.3 DA on an RNAV (RNP) IAP, if they are specifically authorized users using approved baro-VNAV equipment and the pilot has verified required navigation performance (RNP) availability through an approved prediction program. | 3. 对于 RNAV (RNP)（Area Navigation / Required Navigation Performance，区域导航 / 所需导航性能）IAP（Instrument Approach Procedure，仪表进近程序）上的 RNP（Required Navigation Performance，所需导航性能）0.3 DA（Decision Altitude，决断高度），仅当其为特别授权用户，使用经批准的气压 VNAV（Vertical Navigation，垂直导航）设备，并且飞行员已通过经批准的预测程序验证 RNP（Required Navigation Performance，所需导航性能）可用性时，方可使用。 |
| If the above conditions cannot be met, any required alternate airport must have an approved instrument approach procedure other than GPS that is anticipated to be operational and available at the estimated time of arrival, and which the aircraft is equipped to fly. | 如果无法满足上述条件，任何必需备降机场必须具备一个非 GPS（Global Positioning System，全球定位系统）的经批准仪表进近程序，该程序预计在预计到达时间可运行且可用，并且航空器配备相应设备能够飞行该程序。 |
| This restriction does not apply to TSO-C145() and TSO-C146() equipped users (WAAS users). For further WAAS guidance, see paragraph 1-1-18. | 该限制不适用于配备 TSO（Technical Standard Order，技术标准令）-C145() 和 TSO（Technical Standard Order，技术标准令）-C146() 的用户（WAAS（Wide Area Augmentation System，广域增强系统）用户）。有关 WAAS（Wide Area Augmentation System，广域增强系统）的进一步指导，见第 1-1-18 段。 |

---

## 1-2-4. Recognizing, Mitigating, and Adapting to GPS Jamming and/or Spoofing｜识别、缓解并适应 GPS（Global Positioning System，全球定位系统）干扰和 / 或欺骗

| English | 中文 |
|---|---|
| The low-strength data transmission signals from GPS satellites are vulnerable to various anomalies that can significantly reduce the reliability of the navigation signal. | GPS（Global Positioning System，全球定位系统）卫星发射的低强度数据传输信号容易受到各种异常影响，这些异常可显著降低导航信号的可靠性。 |
| The GPS signal is vulnerable and has many uses in aviation (e.g., communication, navigation, surveillance, safety systems and automation); therefore, pilots must place additional emphasis on closely monitoring aircraft equipment performance for any anomalies and promptly inform Air Traffic Control (ATC) of any apparent GPS degradation. Pilots should also be prepared to operate without GPS navigation systems. | GPS（Global Positioning System，全球定位系统）信号较为脆弱，且在航空中用途广泛，例如通信、导航、监视、安全系统和自动化。因此，飞行员必须更加重视密切监控航空器设备性能是否存在异常，并及时将任何明显的 GPS（Global Positioning System，全球定位系统）性能下降情况通知 ATC（Air Traffic Control，空中交通管制）。飞行员还应做好在没有 GPS（Global Positioning System，全球定位系统）导航系统情况下运行的准备。 |
| GPS signals are vulnerable to intentional and unintentional interference from a wide variety of sources, including radars, microwave links, ionosphere effects, solar activity, multi-path error, satellite communications, GPS repeaters, and even some systems onboard the aircraft. In general, these types of unintentional interference are localized and intermittent. | GPS（Global Positioning System，全球定位系统）信号容易受到各种来源的有意和无意干扰，包括雷达、微波链路、电离层效应、太阳活动、多路径误差、卫星通信、GPS（Global Positioning System，全球定位系统）转发器，甚至航空器上的某些系统。一般而言，这些无意干扰具有局部性和间歇性。 |
| Of greater and growing concern is the intentional and unauthorized interference of GPS signals by persons using “jammers” or “spoofers” to disrupt air navigation by interfering with the reception of valid satellite signals. | 更令人担忧且日益严重的是，有人使用“干扰器”或“欺骗器”对 GPS（Global Positioning System，全球定位系统）信号进行有意且未经授权的干扰，通过干扰有效卫星信号的接收来破坏空中导航。 |
| NOTE: The U.S. government regularly conducts GPS tests, training activities, and exercises that interfere with GPS signals. These events are geographically limited, coordinated, scheduled, and advertised via GPS and/or WAAS NOTAMS. Operators of GPS aircraft should always check for GPS and/or WAAS NOTAMS for their route of flight. | 注：美国政府定期开展会干扰 GPS（Global Positioning System，全球定位系统）信号的 GPS 测试、训练活动和演习。这些事件在地理范围上受限，且经过协调、安排并通过 GPS（Global Positioning System，全球定位系统）和 / 或 WAAS（Wide Area Augmentation System，广域增强系统）NOTAM（Notice to Airmen / Notice to Air Missions，航行通告）发布。GPS（Global Positioning System，全球定位系统）航空器运行人应始终检查其飞行航路相关的 GPS 和 / 或 WAAS NOTAM（Notice to Airmen / Notice to Air Missions，航行通告）。 |
| Manufacturers, operators, and air traffic controllers should be aware of the general impacts of GPS jamming and/or spoofing, which include, but are not limited to: | 制造商、运行人和空中交通管制员应了解 GPS（Global Positioning System，全球定位系统）干扰和 / 或欺骗的一般影响，包括但不限于： |
| 1. Inability to use GPS for navigation. | 1. 无法使用 GPS（Global Positioning System，全球定位系统）进行导航。 |
| 2. Inability to use hybrid GPS inertial systems for navigation. | 2. 无法使用混合 GPS（Global Positioning System，全球定位系统）惯性系统进行导航。 |
| 3. Loss of, or degraded, performance-based navigation (PBN) capability (e.g., inability to fly required navigation performance (RNP) procedures). | 3. PBN（Performance-Based Navigation，基于性能的导航）能力丧失或下降，例如无法飞行 RNP（Required Navigation Performance，所需导航性能）程序。 |
| 4. Unreliable triggering of Terrain Awareness and Warning Systems (TAWS). | 4. TAWS（Terrain Awareness and Warning System，地形感知与警告系统）触发不可靠。 |
| 5. Inaccurate aircraft position on navigation display (e.g., moving map and electronic flight bag). | 5. 导航显示上的航空器位置不准确，例如移动地图和电子飞行包。 |
| 6. Loss of, or erroneous, Automatic Dependent Surveillance‐Broadcast (ADS-B) outputs. | 6. ADS-B（Automatic Dependent Surveillance-Broadcast，广播式自动相关监视）输出丧失或错误。 |
| 7. Unexpected effects when navigating with conventional NAVAIDS (e.g., if the aircraft is spoofed from the intended flight path, autotuning will not select the nearby NAVAID). | 7. 使用传统 NAVAID（Navigation Aid，导航设施）导航时出现意外影响，例如如果航空器被欺骗而偏离预期航迹，自动调谐将不会选择附近的 NAVAID（Navigation Aid，导航设施）。 |
| 8. Unanticipated position‐dependent flight management system effects (e.g., erroneous insufficient fuel indication). | 8. 出现未预料的位置相关 FMS（Flight Management System，飞行管理系统）影响，例如错误的燃油不足指示。 |
| 9. Failure or degradation of Air Traffic Management (ATM) infrastructure and its associated systems reliant on GPS, resulting in potential airspace infringements and/or route deviations. | 9. 依赖 GPS（Global Positioning System，全球定位系统）的 ATM（Air Traffic Management，空中交通管理）基础设施及其相关系统故障或性能下降，可能导致空域侵入和 / 或航路偏离。 |
| 10. Failure of, or erroneous aircraft clocks (resulting in inability to log on to Controller‐Pilot Data Link Communications CPDLC). | 10. 航空器时钟故障或错误，导致无法登录 CPDLC（Controller-Pilot Data Link Communications，管制员-飞行员数据链通信）。 |
| 11. Erroneous wind and ground speed indications. | 11. 风和地速指示错误。 |
| When flying IFR, pilots should have additional navigation equipment for their intended route to crosscheck their position. Routine checks of position against VOR or DME information, for example, could help detect a compromised GPS signal. Pilots transitioning to VOR navigation in response to GPS anomalies should refer to the Chart Supplement U.S. to identify airports with available conventional approaches associated with the VOR Minimum Operational Network (MON) program. (Reference 1-1-3 f.) | 在 IFR（Instrument Flight Rules，仪表飞行规则）飞行中，飞行员应为预定航路配备额外导航设备，以交叉检查位置。例如，定期将位置与 VOR（VHF Omni-directional Range，甚高频全向信标）或 DME（Distance Measuring Equipment，测距设备）信息对照，可帮助发现受损的 GPS（Global Positioning System，全球定位系统）信号。当飞行员因 GPS（Global Positioning System，全球定位系统）异常而转为 VOR（VHF Omni-directional Range，甚高频全向信标）导航时，应查阅《Chart Supplement U.S.》，以识别具有与 VOR（VHF Omni-directional Range，甚高频全向信标）MON（Minimum Operational Network，最低运行网络）计划相关可用传统进近的机场。参见 1-1-3 f。 |
| Prior to departure, the FAA recommends operators to: | 离场前，FAA（Federal Aviation Administration，美国联邦航空管理局）建议运行人： |
| 1. Be aware of potential risk locations. | 1. 了解潜在风险位置。 |
| 2. Check for any relevant Notices to Airmen (NOTAMs). | 2. 检查任何相关 NOTAM（Notice to Airmen / Notice to Air Missions，航行通告）。 |
| 3. Plan fuel contingencies. | 3. 规划燃油应急方案。 |
| 4. Plan to use conventional NAVAIDs and appropriate arrival/approach procedures at the destination. | 4. 计划在目的地使用传统 NAVAID（Navigation Aid，导航设施）和适当的进场 / 进近程序。 |
| 5. Follow the detailed guidance from the respective Original Equipment Manufacturer (OEM). | 5. 遵循相应 OEM（Original Equipment Manufacturer，原始设备制造商）的详细指导。 |
| During flight, the FAA recommends operators do the following: | 飞行中，FAA（Federal Aviation Administration，美国联邦航空管理局）建议运行人执行以下事项： |
| 1. Be vigilant for any indication that the aircraft's GPS is disrupted by reviewing the manufacturer's guidance for that specific aircraft type and avionics equipage. Verify the aircraft position by means of conventional NAVAIDs, when available. Indications of jamming and/or spoofing may include: | 1. 通过查阅制造商针对特定航空器型号和航空电子设备配置的指导，对航空器 GPS（Global Positioning System，全球定位系统）受扰的任何迹象保持警惕。可用时，通过传统 NAVAID（Navigation Aid，导航设施）核实航空器位置。干扰和 / 或欺骗的迹象可能包括： |
| 1. Changes in actual navigation performance. | 1. 实际导航性能发生变化。 |
| 2. Aircraft clock changes (e.g., incorrect time). | 2. 航空器时钟变化，例如时间不正确。 |
| 3. Incorrect Flight Management System (FMS) position. | 3. FMS（Flight Management System，飞行管理系统）位置不正确。 |
| 4. Large shift in displayed GPS position. | 4. 显示的 GPS（Global Positioning System，全球定位系统）位置出现大幅偏移。 |
| 5. Primary Flight Display (PFD)/Navigation Display (ND) warnings about position error. | 5. PFD（Primary Flight Display，主飞行显示器）/ ND（Navigation Display，导航显示器）出现位置误差警告。 |
| 6. Other aircraft reporting clock issues, position errors, or requesting vectors. | 6. 其他航空器报告时钟问题、位置误差或请求雷达引导。 |
| 2. Assess operational risks and limitations linked to the loss of GPS capability, including any on-board systems requiring inputs from a GPS signal. | 2. 评估与 GPS（Global Positioning System，全球定位系统）能力丧失相关的运行风险和限制，包括任何需要 GPS 信号输入的机载系统。 |
| 3. Ensure NAVAIDs critical to the operation for the intended route/approach are available. | 3. 确保预定航路 / 进近运行所关键的 NAVAID（Navigation Aid，导航设施）可用。 |
| 4. Remain prepared to revert to conventional instrument flight procedures. | 4. 保持随时转回传统仪表飞行程序的准备。 |
| 5. Promptly notify ATC if they experience GPS anomalies. Pilots should not inform ATC of GPS jamming and/or spoofing when flying through known NOTAMed testing areas unless they require ATC assistance. (See paragraph 1-1-13.) | 5. 如遇 GPS（Global Positioning System，全球定位系统）异常，应立即通知 ATC（Air Traffic Control，空中交通管制）。飞行员飞越已通过 NOTAM（Notice to Airmen / Notice to Air Missions，航行通告）公布的已知测试区域时，除非需要 ATC（Air Traffic Control，空中交通管制）协助，否则不应向 ATC 报告 GPS（Global Positioning System，全球定位系统）干扰和 / 或欺骗。见第 1-1-13 段。 |
| Post flight, the FAA recommends operators to: | 飞行后，FAA（Federal Aviation Administration，美国联邦航空管理局）建议运行人： |
| 1. Document any GPS jamming and/or spoofing in the maintenance log to ensure all faults are cleared. | 1. 将任何 GPS（Global Positioning System，全球定位系统）干扰和 / 或欺骗情况记录在维修记录中，以确保所有故障均已排除。 |
| 2. File a detailed report at the reporting site: Report a GPS Anomaly Federal Aviation Administration, www.faa.gov/air_traffic/nas/gps_reports. | 2. 在报告网站提交详细报告：Report a GPS Anomaly Federal Aviation Administration（向美国联邦航空管理局报告 GPS 异常），www.faa.gov/air_traffic/nas/gps_reports。 |

---

## 译注

1. 本文尽量采用航空领域常用中文译法；个别术语保留英文缩写，以便与 FAA（Federal Aviation Administration，美国联邦航空管理局）原文、航图和设备显示一致。
2. 原文中的图示未在本 MD（Markdown）文件中重新绘制，仅保留图号、图题和文字说明。实际图形请查阅 FAA（Federal Aviation Administration，美国联邦航空管理局）官方网页。
3. “NOTAM”在 FAA（Federal Aviation Administration，美国联邦航空管理局）新旧文本中可指 Notice to Airmen 或 Notice to Air Missions，中文通常译为“航行通告”。
