# CIFP 学习与项目应用完整教程（中英双语）
# Comprehensive CIFP Learning & Project Usage Tutorial (Chinese-English Bilingual)

> 面向对象：从零开始的学习者（你完全不需要先懂ARINC 424）  
> Target audience: complete beginners (you do not need prior ARINC 424 knowledge)

---

## 0) 你将学到什么 / What You Will Learn

中文：学完本教程后，你将能够：
1. 说清楚 CIFP 是什么、为什么是航空数据库核心输入。
2. 看懂你项目里的 `FAACIFP18` 和 `IN_CIFP.txt`。
3. 明白 `SIAP / SID / STAR` 在数据层和可视化层怎么连接。
4. 设计一条从 CIFP 到 `waypoints.geojson` 的可复用数据流水线。
5. 知道哪些地方容易出错，以及如何做质量检查。

English:
1. Explain what CIFP is and why it is a core aviation-data input.
2. Read and reason about your local `FAACIFP18` and `IN_CIFP.txt` files.
3. Understand how `SIAP / SID / STAR` map from data to visualization.
4. Design a reusable pipeline from CIFP to `waypoints.geojson`.
5. Know common failure modes and practical QA checks.

---

## 1) 先建立直觉：CIFP 是什么 / Build Intuition First: What CIFP Is

中文：
- CIFP（Coded Instrument Flight Procedures）是 FAA 提供的“编码化仪表飞行程序数据”。
- 它遵循 ARINC 424 导航数据库标准（你的文件名 `FAACIFP18` 对应 FAA 当前提供的 ARINC 424 版本线）。
- 你可以把它理解为：
  - 图纸（航图）是给人看的；
  - CIFP 是给机器（FMS/GPS/导航软件/仿真程序）读的结构化程序数据。
- 它不是直接“给某个具体机载设备即插即用”的终端格式，通常需要二次处理和映射。

English:
- CIFP (Coded Instrument Flight Procedures) is FAA-provided coded procedure data.
- It follows the ARINC 424 navigation database standard (your file name `FAACIFP18` indicates the FAA ARINC 424 line in use).
- Intuition:
  - Charts are human-readable.
  - CIFP is machine-readable structured procedure/navigation data.
- It is usually not a device-specific ready-to-load avionics database without additional processing.

---

## 2) 你的本地数据包到底包含什么 / What Exactly Is in Your Local Package

你当前目录（与你提问路径一致）：
- `data/CIFP/CIFP_260319/FAACIFP18`
- `data/CIFP/CIFP_260319/IN_CIFP.txt`
- `data/CIFP/CIFP_260319/CIFP Readme 2603.pdf`
- `data/CIFP/CIFP_260319/FAA CIFP Disclaimer.pdf`
- `data/CIFP/CIFP_260319/CIFP ATS and Enroute Coverage.pdf`

### 2.1 从文件头读到的关键信息 / Key Metadata from File Header

`FAACIFP18` 头部样例（你的本地文件）：

```text
HDR01FAACIFP18      001P013203972702603  25-FEB-202616:18:27  U.S.A. DOT FAA
HDR04                                 CODED INSTRUMENT FLIGHT PROCEDURES VOLUME 2603  EFFECTIVE 19 MAR 2026
```

中文：这说明你手上的数据周期是 2603，生效日是 2026-03-19。  
English: This indicates cycle 2603, effective date 2026-03-19.

### 2.2 文件规模（你本地统计）/ Local Scale (Your Actual Stats)

- `FAACIFP18`: 397,275 行
- `IN_CIFP.txt`: 14,432 行（含表头）
- `IN_CIFP.txt` 数据行：14,431
- `IN_CIFP.txt` 覆盖机场（唯一 ICAO）：3,074

中文：先记住这个结论：`FAACIFP18` 是“全量明细”，`IN_CIFP.txt` 是“轻量索引/清单”。  
English: Keep this mental model: `FAACIFP18` is full-detail data, `IN_CIFP.txt` is a lightweight index/list.

---

## 3) 先学最容易读懂的：IN_CIFP.txt / Start with the Easiest File: IN_CIFP.txt

### 3.1 结构 / Structure

表头：
```text
Arpt_ICAO       Procedure Type  Procedure Ident
```

示例：
```text
05U     SID     MINES1
05C     STAR    LUCIT3
00R     SIAP    R30
```

中文：
- 第 1 列：机场 ICAO（或本地识别码）
- 第 2 列：程序类别
  - `SIAP` = Standard Instrument Approach Procedure（进近程序）
  - `SID` = Standard Instrument Departure（离场程序）
  - `STAR` = Standard Terminal Arrival Route（进场程序）
- 第 3 列：程序标识（例如 `MINES1`, `LUCIT3`）

English:
- Col 1: Airport ICAO/local identifier
- Col 2: Procedure category
  - `SIAP` = approach
  - `SID` = departure
  - `STAR` = arrival
- Col 3: Procedure identifier (e.g., `MINES1`, `LUCIT3`)

### 3.2 你这份数据中的程序类型分布 / Procedure-Type Distribution in Your File

- SIAP: 10,376
- SID: 2,207
- STAR: 1,848

中文：这对项目很有价值：你能快速知道“进近程序最多”，可优先做进近可视化与验证。  
English: This is highly actionable: approaches dominate, so approach visualization/validation can be prioritized.

---

## 4) 再学主文件：FAACIFP18 / Then Learn the Main File: FAACIFP18

## 4.1 固定宽度记录的概念 / Fixed-Width Record Concept

中文：
- 你的 `FAACIFP18` 每行长度是 133（包含换行）；可视为 132 字节主记录 + 行结束。
- 这意味着它不是 CSV，而是“按列位置切片”的固定宽度文本。
- 解析时应使用 `line[a:b]`（按字符位置切割），不要 `split(',')`。

English:
- In your file, each line length is 133 including newline; effectively a 132-byte fixed record plus line ending.
- This is not CSV; it is position-based fixed-width text.
- Parse with positional slicing (`line[a:b]`), not comma splitting.

### 4.2 你文件里的记录家族（按前缀观察）/ Record Families in Your Data (by prefix)

你本地前缀统计（前 6 字符）显示高频项：
- `SUSAP `: 279,915
- `SUSAEA`: 28,481
- `SUSAUR`: 25,881
- `SUSAER`: 16,777
- `SUSAUC`: 12,835
- `SUSAH `: 6,077

中文：这说明你的周期包里，终端程序相关记录非常多（`SUSAP`），其次是航路/航路点/受控空域等记录家族。  
English: This shows terminal-procedure records (`SUSAP`) dominate, followed by enroute/airspace families.

> 注意 / Note: ARINC 424 的完整字段定义是规范文档级别内容（通常需正式规范文本）；这里采用“项目可用”的工程化读取方式，并用你的真实样例讲解。

---

## 5) 用你的真实样例读一遍 / Read Real Records from Your File

### 5.1 KJFK 相关样例 / KJFK-related sample

```text
SUSAP KJFKK6AJFK     0     145YHN40382374W073464329W013000013 ... JOHN F KENNEDY INTL
SUSAP KJFKK6CAROKE K60    R     N40282056W073540760 ... AROKE
```

中文：
- 可以直接观察到机场（KJFK）、程序标识（例如 K6A 相关）、航路点名（AROKE）和坐标串（N/W 开头）。
- 这些行是你构建程序路径几何（线段点列）的关键原料。

English:
- You can directly see airport (`KJFK`), procedure identity, waypoint names (`AROKE`), and coordinate strings.
- These records are the raw material for constructing procedure geometry (ordered path points).

### 5.2 KLAX 相关样例 / KLAX-related sample

```text
SUSAP KLAXK2ALAX ... LOS ANGELES INTL
SUSAP KLAXK2CADORE ... ADORE
```

中文：同一模式在不同机场一致，说明你可以做通用解析器，不需要为单机场写特例。  
English: Same pattern across airports means a generic parser is feasible; no airport-specific hardcoding needed.

### 5.3 经纬度编码直觉 / Coordinate Encoding Intuition

示例片段：`N40382374W073464329`

中文（工程直觉版）：
- 纬度可按 `40°38'23.74"N` 近似理解；
- 经度可按 `73°46'43.29"W` 近似理解；
- 最终要转十进制度（decimal degrees）用于 GeoJSON/Cesium。

English (engineering intuition):
- Latitude can be interpreted approximately as `40°38'23.74"N`.
- Longitude as `73°46'43.29"W`.
- Convert to decimal degrees for GeoJSON/Cesium.

---

## 6) 这份 CIFP 在你项目里该怎么用 / How CIFP Should Be Used in This Project

你仓库已有明确提示：
- `aeroviz-4d/python/preprocess_airports.py` 注释写明：`waypoints.geojson` 不是它生成，而是来自 ARINC 424 / CIFP 解析。
- 前端通过 `aeroviz-4d/src/hooks/useWaypointLayer.ts` 加载 `/data/waypoints.geojson`。
- 项目总流程文档中也写了：`Nav Canada/FAA CIFP -> waypoints.geojson`。

### 推荐项目数据流水线 / Recommended Project Data Pipeline

中文：
1. 选择周期目录（例如 `data/CIFP/CIFP_260319`）。
2. 用 `IN_CIFP.txt` 做“任务清单”（机场-程序类型-程序名）。
3. 在 `FAACIFP18` 里抽取对应程序的完整点列与属性。
4. 统一坐标转换为 WGS84 十进制度。
5. 生成 `waypoints.geojson`（FeatureCollection）。
6. 放到前端静态目录并加载验证。
7. 用后续调度结果生成 `trajectories.czml`，实现“程序约束 + 4D轨迹”联动可视化。

English:
1. Select cycle folder (e.g., `data/CIFP/CIFP_260319`).
2. Use `IN_CIFP.txt` as the task index (airport-procedure type-procedure id).
3. Extract full path-point sequences from `FAACIFP18` for those procedures.
4. Convert coordinates to WGS84 decimal degrees.
5. Build `waypoints.geojson` (FeatureCollection).
6. Place it into frontend static data and validate rendering.
7. Use scheduling outputs to generate `trajectories.czml` for linked 4D visualization.

---

## 7) 从零到可运行：建议实现步骤 / From Zero to Working: Suggested Implementation Steps

### Step A: 先做“最小可用解析器” / Build a Minimum Viable Parser first

中文：
- 目标不是一次性吃掉全部 ARINC 字段。
- 第一版只做：机场、程序、航路点名、经纬度、程序类型。

English:
- Do not attempt full ARINC semantic coverage in v1.
- First version: airport, procedure id, waypoint name, lat/lon, procedure type.

### Step B: 先支持一个机场 / Start with One Airport

中文：例如只做 `KJFK`，输出该机场 SID/STAR/SIAP 的点。  
English: Start with `KJFK` only and output points for SID/STAR/SIAP.

### Step C: 再扩展到批量机场 / Expand to Multiple Airports

中文：用 `IN_CIFP.txt` 的机场列表批处理。  
English: Batch by airport list from `IN_CIFP.txt`.

### Step D: 加 QA / Add QA

中文：
- 检查坐标范围（lat in [-90,90], lon in [-180,180]）。
- 检查同一程序点序是否连续、无空洞。
- 随机抽 5 个程序人工对照（命名、位置、方向）

English:
- Validate coordinate ranges.
- Validate ordered continuity per procedure.
- Manually spot-check a random sample of procedures.

---

## 8) 给你一个“教学型字段抽取框架” / A Teaching-Oriented Extraction Framework

> 说明 / Note: 下述是“工程框架示意”，不是完整 ARINC 全字段实现。

```python
# Pseudocode / skeleton
for line in faacifp18_lines:
    prefix = line[:6]          # e.g. SUSAP, SUSAEA

    if prefix.startswith("SUSAP"):
        # 1) read airport/procedure identity by fixed slices
        # 2) read waypoint ident and coordinate substring
        # 3) decode N/S/E/W encoded coordinate -> decimal degrees
        # 4) append to {airport, proc_type, proc_ident} ordered list
        pass

# after grouping
# emit GeoJSON FeatureCollection
```

中文：关键是“按固定列位切片 + 分组排序 + 坐标转换 + GeoJSON输出”。  
English: Core pattern is positional slicing + grouping/ordering + coordinate conversion + GeoJSON emission.

---

## 9) 常见坑（你很可能会遇到）/ Common Pitfalls (You Will Likely Hit)

1. 把固定宽度文本当作分隔文本处理。  
   Treating fixed-width records as delimiter-based text.

2. 坐标字符串解析位数错位（经度 3 位度、纬度 2 位度，且有 N/S/E/W 符号）。  
   Mis-parsing coordinate digit widths/signs.

3. 混淆“程序索引文件”和“程序全量明细文件”。  
   Confusing index file (`IN_CIFP.txt`) with full-detail file (`FAACIFP18`).

4. 未考虑 28 天周期更新导致的数据漂移。  
   Ignoring 28-day cycle updates and data drift.

5. 忽略数据使用声明（尤其涉及导航用途时）。  
   Ignoring usage/disclaimer requirements (especially if used for navigation).

---

## 10) 与你项目目标的对齐建议 / Alignment with Your Project Goals

中文：
- 你的论文和系统重点是 4D 轨迹与到达排序可视化，不需要一次性完整复刻航电数据库。
- 对本项目最划算的做法是：
  - 先稳定产出可视化所需 `waypoints.geojson`；
  - 再逐步补充程序语义（如转弯类型、高度/速度约束、RF leg 等）。

English:
- Your thesis/system focus is 4D trajectory + arrival sequencing visualization, not full avionics database replication on day 1.
- Highest ROI path:
  - first, stable `waypoints.geojson` generation;
  - then incrementally add deeper procedure semantics (turn types, altitude/speed constraints, RF legs, etc.).

---

## 11) 权威参考（用于继续深挖）/ Authoritative References

1. FAA CIFP 页面（定义、下载入口、更新节奏、使用说明）  
   https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/cifp/

2. FAA 28-Day NASR Subscription（核对周期生效日）  
   https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/

3. ARINC 424 概述（入门级背景）  
   https://en.wikipedia.org/wiki/ARINC_424

4. 你的本地数据包（最重要）  
   `data/CIFP/CIFP_260319/`

---

## 12) 一页速记 / One-Page Cheat Sheet

中文：
- `IN_CIFP.txt` = 程序清单索引（机场-类型-程序名）
- `FAACIFP18` = 固定宽度全量数据（主解析对象）
- 周期 = 28 天；你当前数据 = 2603，生效 2026-03-19
- 项目落地目标 = `waypoints.geojson`（给 Cesium/React）
- 实施策略 = 先小后大，先可视化后语义增强

English:
- `IN_CIFP.txt` = indexed procedure list (airport-type-id)
- `FAACIFP18` = fixed-width full-detail source (main parser target)
- Update cycle = 28 days; your dataset = cycle 2603, effective 2026-03-19
- Project deliverable = `waypoints.geojson` for Cesium/React
- Strategy = small first, visualization first, semantic depth later

---

如果你愿意，我下一步可以继续给你：
1) 一个“可直接运行”的 Python 解析脚本（先支持 KJFK + KLAX）  
2) 输出 `waypoints.geojson` 的字段规范与示例  
3) 一个可视化对照清单（在前端如何验证解析是否正确）

If you want, next I can provide:
1) a runnable Python parser (starting with KJFK + KLAX),
2) a concrete `waypoints.geojson` schema with examples,
3) a front-end validation checklist to verify parser correctness.
