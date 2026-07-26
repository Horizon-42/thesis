# ADS-B、CIFP 与 EGM96 垂直基准调查

状态：**调查完成，等待修改计划评审，尚未实施**  
日期：2026-07-23  
适用范围：`trajectory_data_process.harvest`、`final_approach`、`flight_scenarios`、
`ts_transformer`、建模结果到 Cesium 的垂直基准边界。

## 1. 摘要结论

这次调查得到五个直接影响实现的结论：

1. OpenSky `geoaltitude` 已经是 GNSS geometric altitude；当前 harvest 保存的就是
   WGS-84 椭球高（HAE）。**获得 HAE 不需要 EGM96。**
2. FAA CIFP 不能笼统描述成“全部是 MSL”。同一份 CIFP 同时包含：
   - 程序限制高度和 Landing Threshold Elevation（正高/MSL）；
   - Path Point/Runway 的 LTP Ellipsoidal Height（WGS-84 HAE）；
   - Path Point continuation 的 LTP Orthometric Height（MSL）。
3. SBAS/LPV Final Approach Segment（FAS）本来就是以 WGS-84 椭球和 LTP
   ellipsoidal height 定义的。因此 ADS-B 最终进近拟合最直接的数据契约是：

   ```text
   ADS-B geoaltitude (HAE)
               对比
   CIFP LTP ellipsoidal height (HAE)
   ```

   这条链不应依赖 geoid grid。
4. EGM96 格网提供的是随经纬度变化的 geoid undulation：

   ```text
   N(lat, lon) = h_HAE - H_orthometric
   H_orthometric = h_HAE - N
   ```

   它不是分别提供 HAE 和 MSL，也不是跑道入口官方高程的替代来源。
5. 对当前 30 km 终端区研究，使用**每跑道、由 CIFP 同点 HAE/MSL 得到的固定偏差**
   足够：KMSY 5 km 最终进近区内 EGM96 的空间变化不超过约 0.10 m，30 km 内不超过
   约 0.61 m。该误差小于 ADS-B 7.62 m 的高度量化步长和约 1.7 m 的拟合不确定度。

因此建议的架构边界是：

- harvest、runway assignment、final-approach fitting：保持 HAE，直接使用 CIFP HAE；
- 进入本地 MSL 建模平面：使用已分配跑道的 CIFP LTP 固定偏差；
- Cesium observed layer：继续保持 HAE；
- EGM96：不再是当前管线的运行依赖，只保留为未来大范围/亚米级模式的可选空间梯度模型。

## 2. 术语与方程

本文使用以下符号：

| 符号 | 含义 |
|---|---|
| `h` / HAE | 相对 WGS-84 reference ellipsoid 的椭球高 |
| `H` / orthometric height | 相对 geoid/官方垂直基准的正高，工程语境通常简称 MSL |
| `N` | geoid undulation，本文采用 `N = h - H` |

换算关系为：

```text
h = H + N
H = h - N
```

“MSL”在软件中常被当成单一概念，但严格来说 EGM96 height、NAVD88 与 FAA 发布的
orthometric height 不是可以无条件互换的同一 realization。本文继续使用项目现有的
“MSL”简称，但所有数据契约必须同时记录其来源。

## 3. 权威来源与字段语义

### 3.1 OpenSky / ADS-B

OpenSky 官方 REST API 把两个高度字段明确分开：

- `baro_altitude`：Barometric altitude in meters；
- `geo_altitude`：Geometric altitude in meters。

来源：[OpenSky Network REST API — State vectors](https://openskynetwork.github.io/opensky-api/rest.html)

本项目历史查询同时请求这两列：

- [`trajectory_data_process/acquisition/opensky_history.py`](../trajectory_data_process/acquisition/opensky_history.py)
  的 `STATE_VECTOR_COLUMNS`；
- [`trajectory_data_process/harvest/tracks.py`](../trajectory_data_process/harvest/tracks.py)
  的 `_to_sample()` 只读取 `geoaltitude`，写入 `Sample.alt_hae_m`；
- 缺少 geometric altitude 的样本被丢弃，不使用 barometric altitude 填补。

所以当前 harvest 的实际契约是：

```text
track.samples[*][3] = OpenSky geoaltitude in metres = HAE
```

ADS-B 并非“只使用一种高度”：运行和 ATC 语境仍大量使用气压高度，而 GNSS/SBAS
几何使用 geometric height。关键不是争论哪一种是“ADS-B高度”，而是保留字段来源并在
比较前统一 datum。

### 3.2 ARINC 424 Runway record

FAA cycle 2603 的
[`CIFP Readme 2603.pdf`](../data/CIFP/CIFP_260319/CIFP%20Readme%202603.pdf)
明确说明该数据集遵循 ARINC 424-18，并列出少量采用 version 19 的例外。仓库中的
[`ARINC424-23.pdf`](../data/CIFP/ARINC424-23.pdf) 用于核对延续至 v23 的字段语义；
不能把 v23 中已经变化的无关列位置整体套到 `FAACIFP18`。

本调查使用的两个 Runway Primary vertical fields 在原始 `FAACIFP18`、当前 `cifparse`
424-18 layout 与 ARINC 424-23 §4.1.10.1（printed page 42）中一致：

| 列 | 字段 | 基准/单位 |
|---|---|---|
| 61–66 | `(LTP) Ellipsoid Height` | WGS-84 HAE，0.1 m |
| 67–71 | `Landing Threshold Elevation` | MSL/正高，1 ft |

这意味着 CIFP Runway record 已经足以为每个发布跑道入口提供 HAE 与 MSL 两个基准，
并不需要从 `runway_thresholds.json` 的 MSL 高程经过 EGM96 反算 HAE。

### 3.3 ARINC 424 Path Point record

`FAACIFP18` 的 Path Point layout 与 ARINC 424-23 §4.1.28（printed pages 80–81）在本文
使用的字段上保持一致：

| 记录 | 列 | 字段 | 基准/单位 |
|---|---|---|---|
| Path Point primary | 38–48 / 49–60 | LTP latitude / longitude | WGS-84 |
| Path Point primary | 61–66 | `(LTP) Ellipsoid Height` | WGS-84 HAE，0.1 m |
| Path Point primary | 67–70 | Glide Path Angle | 0.01° |
| Path Point primary | 103–108 | Path Point TCH | 0.1 ft 或 0.1 m |
| continuation | 35–40 | FPAP Orthometric Height | MSL，0.1 m |
| continuation | 41–46 | LTP Orthometric Height | MSL，0.1 m |

字段定义：

- §5.225 `Ellipsoidal Height`（printed page 262）：相对 WGS-84 ellipsoid 的 surveyed
  point height；
- §5.227 `Orthometric Height`（printed page 263）：相对 Mean Sea Level 的 surveyed
  point height。

当前 parser 已经解析 primary 中的 `ltp_ellipsoidal_height_m`：

- [`trajectory_data_process/harvest/cifp.py`](../trajectory_data_process/harvest/cifp.py)
  的 `_LTP_ELLIPSOIDAL_HEIGHT` 和 `PathPoint.ltp_ellipsoidal_height_m`。

但它尚未解析 continuation 中的 LTP orthometric height；更关键的是，机场加载器没有
使用已经解析出的 ellipsoidal height。

### 3.4 FAS 的垂直几何

ARINC 424-23 §2.2.4（printed pages 9–10）定义：

- Final Approach Flight Path 位于包含三个 precision approach path points 的垂直平面内；
- LTP 由纬度、经度和相对 WGS-84 reference ellipsoid 的高度定义；
- FPCP 的 elevation 是 `LTP ellipsoidal elevation + TCH`；
- GPA 相对 LTP 处 WGS-84 ellipsoid 的切平面定义。

因此，LPV 最终进近的垂直几何并不是先把 GNSS HAE 转成 EGM96 MSL 再建立的。直接用
CIFP LTP HAE 与 ADS-B geometric altitude 拟合，才与 FAS 数据块的定义一致。

## 4. KMSY 原始 CIFP 证据

数据源：
[`data/CIFP/CIFP_260319/FAACIFP18`](../data/CIFP/CIFP_260319/FAACIFP18)，
KMSY records 位于约 266670–266685 行。

### 4.1 Path Point 的同点 HAE/MSL

primary 提供 LTP HAE，紧随其后的 continuation 提供 LTP orthometric height：

| 跑道 | LTP HAE (m) | LTP MSL (m) | `N_CIFP = HAE - MSL` (m) |
|---|---:|---:|---:|
| 02 | -25.7 | +0.4 | -26.1 |
| 11 | -25.2 | +0.9 | -26.1 |
| 20 | -27.0 | -0.9 | -26.1 |
| 29 | -25.9 | +0.2 | -26.1 |

四个独立入口得到相同的 `N_CIFP ≈ -26.1 m`，这是字段位置、符号和单位解析正确的
强交叉验证。

RW29 对应：

- Runway record：约第 266673 行；
- Path Point primary：约第 266684 行；
- Path Point continuation：约第 266685 行。

### 4.2 CIFP 与 `runway_thresholds.json`

当前 [`trajectory_data_process/config/runway_thresholds.json`](../trajectory_data_process/config/runway_thresholds.json)
来自 OurAirports runway geometry，不是 FAA CIFP 的垂直事实来源：

| 跑道 | CIFP Path Point MSL (m) | `runway_thresholds.json` (m) | 备注 |
|---|---:|---:|---|
| 02 | +0.4 | +0.91 | OurAirports runway end |
| 11 | +0.9 | +1.22 | OurAirports runway end |
| 20 | -0.9 | +0.91 | config 使用 opposite-end elevation |
| 29 | +0.2 | +0.63 | config 对 displaced threshold 插值 |

这些值适合机场/runway roster 和几何 fallback，但不应在 CIFP 已有发布值时覆盖 CIFP
的 LTP HAE/MSL。KMSY 20 的符号差异尤其说明“补一个看似合理的跑道高程”会把误差带入
所有后续垂直判断。

## 5. EGM96 是什么，以及它不是什么

### 5.1 格网含义

本次计算使用 PROJ 官方资源：
[`us_nga_egm96_15.tif`](https://cdn.proj.org/us_nga_egm96_15.tif)。

该 GeoTIFF 描述 WGS-84 ellipsoidal height（EPSG:4979）到 EGM96 height
（EPSG:5773）的 vertical grid shift：

- raster size 1440 × 721；
- 全球 15 arc-minute 分辨率；
- 查询点通过格网插值得到 `N(lat, lon)`。

EGM96 提供的是一个全球重力场/geoid 模型。它可以给出空间变化，但不能取代 FAA 在
CIFP 中为某个 surveyed LTP 发布的 HAE 与 orthometric height。

参考：

- [PROJ CDN resource](https://cdn.proj.org/us_nga_egm96_15.tif)
- [GeographicLib online geoid calculator](https://geographiclib.sourceforge.io/cgi-bin/GeoidEval)
- [EPSG:5773 — EGM96 height](https://epsg.io/5773)

### 5.2 KMSY 的绝对值差异

在 KMSY airport center：

```text
N_EGM96 ≈ -27.259 m
N_CIFP  ≈ -26.100 m
difference ≈ -1.159 m
```

这个差异不应被解释成“CIFP错误”或“EGM96错误”。它说明二者的模型、垂直基准
realization、发布分辨率和用途不同，不能在代码里无来源标记地互换。

对最终进近而言，CIFP 的同点、官方发布 LTP HAE/MSL 更接近所定义的 FAS。对大范围
正高转换而言，格网的优势是描述空间梯度，而不是覆盖入口处的 CIFP 发布值。

## 6. 30 km 范围内固定偏差是否足够

### 6.1 测量方法

使用上述 `us_nga_egm96_15.tif` 和 `pyproj` `vgridshift`：

- 以每个机场 config center 为圆心；
- 半径每 500 m 取一圈；
- 每圈方位角每 3° 取样；
- 分别统计 5 km 和 30 km 圆盘内的 sampled minimum/maximum；
- `max deviation` 是相对机场中心 `N` 的最大绝对差。

这些是密集取样结果，不宣称为连续曲面的形式化全局极值；对判断固定偏差误差量级已经
足够。

### 6.2 五机场结果

| airport | center `N` (m) | 5 km range (m) | 5 km max deviation (m) | 30 km range (m) | 30 km max deviation (m) |
|---|---:|---:|---:|---:|---:|
| KRDU | -33.507 | 0.514 | 0.258 | 2.866 | 1.503 |
| KMSY | -27.259 | 0.196 | 0.100 | 1.177 | 0.611 |
| KSJC | -31.985 | 0.168 | 0.086 | 1.338 | 1.061 |
| KSMF | -30.451 | 0.261 | 0.132 | 1.884 | 1.163 |
| KSTL | -31.889 | 0.153 | 0.085 | 0.943 | 0.501 |

### 6.3 与项目误差尺度比较

当前相关尺度：

| 项目 | 量级 |
|---|---:|
| OpenSky `geoaltitude` 量化步长 | 25 ft = 7.62 m |
| established final-approach OLS 典型不确定度 | 约 1.7 m |
| evaluation vertical gate 总宽度 | 9.15 m |
| KMSY 5 km 固定 `N` 的 sampled max error | 约 0.10 m |
| KMSY 30 km 固定 `N` 的 sampled max error | 约 0.61 m |
| 五机场 30 km 最坏 sampled max error（KRDU） | 约 1.50 m |

由此区分三个使用场景：

1. **Runway assignment**：主要由方向和 cross-track arg-min 决定；直接使用 HAE，不需要
   geoid conversion。
2. **Final-approach fitting（窗口约 5 km）**：使用 CIFP LTP HAE；固定偏差即使参与
   MSL 输出，空间误差也只有 0.1–0.26 m，远低于观测误差。
3. **完整 30 km modeling track**：每跑道固定偏差会引入约 0.5–1.5 m 的位置相关误差。
   对当前 ADS-B/优化/TS 研究可接受，但不适合宣称为严格亚米级 geodetic conversion，
   也不应直接用于高精度 terrain clearance。

## 7. 当前代码数据流与根因

### 7.1 已有正确部分

- [`harvest/tracks.py`](../trajectory_data_process/harvest/tracks.py)：保留 OpenSky
  `geoaltitude`，明确标记为 HAE；
- [`harvest/cifp.py`](../trajectory_data_process/harvest/cifp.py)：已经正确解析 Path Point
  LTP position、ellipsoidal height、GPA、course width、TCH；
- [`final_approach`](../final_approach)：要求 track altitude 与 runway-frame elevation
  使用同一个 datum，本身不负责转换；
- [`harvest/observed.py`](../trajectory_data_process/harvest/observed.py)：已经使用一个
  runway-local constant `geoid_undulation_m` 转换整条 observed track；问题只在该常数
  当前来自 EGM96，而不是 CIFP 同点 HAE/MSL。

### 7.2 错误的数据源选择

[`trajectory_data_process/harvest/airports.py`](../trajectory_data_process/harvest/airports.py)
当前流程是：

1. Path Point 存在时使用 CIFP LTP 经纬度；
2. 忽略同一 Path Point 已解析出的 `ltp_ellipsoidal_height_m`；
3. 使用 `runway_thresholds.json` 的 MSL elevation；
4. 调用 `flight_scenarios.datum.geoid_undulation_m()`；
5. 通过 `HAE = config MSL + EGM96 N` 重新构造 HAE。

即：读取了直接事实，却绕过它，用两个不同来源间接重建同一个值。这使 harvest 在已有
CIFP HAE 的情况下仍依赖外部 EGM96 grid，并把 OurAirports/EGM96 与 FAA CIFP 的基准
差异引入 final-approach frame。

### 7.3 网格依赖如何扩散

[`flight_scenarios/datum.py`](../flight_scenarios/datum.py) 把每个 waypoint 从 HAE
逐点转换到 EGM96 height；[`flight_scenarios/fitted_approach.py`](../flight_scenarios/fitted_approach.py)
随后只接受转换后的 MSL flight。于是 `--target-from-fitted-adsb` 本来可以在 HAE 内完成的
拟合，被强制依赖 pyproj + EGM96 grid。

反向边界 [`aeroviz-4d/python/vertical_datum.py`](../aeroviz-4d/python/vertical_datum.py)
又使用同一 grid 把 modeling MSL 转回 Cesium HAE。这个对称设计内部一致，但建立在
“所有 modeling MSL 都是 EGM96 height”的假设上；它没有利用 scenario 已知的 runway
CIFP datum。

### 7.4 文档漂移

[`CLAUDE.md`](../CLAUDE.md) 当前写有“runway thresholds, CIFP altitudes and gates are
MSL”。其中“CIFP procedure altitudes are MSL”成立，但“CIFP altitudes”作为总称不成立，
因为 Runway/Path Point 明确同时包含 ellipsoidal 和 orthometric height。

实施时必须同步修正这一表述，避免后续再次忽略 CIFP HAE。

## 8. 建议的数据契约

### 8.1 Runway/LTP 事实

每个 runway threshold 应同时携带：

```text
latitude / longitude
elevation_hae_m
elevation_msl_m
geoid_undulation_m = elevation_hae_m - elevation_msl_m
vertical_source
position_source
```

来源优先级建议：

1. 有 LPV Path Point：Path Point LTP primary + continuation；
2. 无 Path Point：CIFP Runway primary；
3. CIFP 缺失：应在 harvest 边界明确失败或显式列为不可用，不得悄悄用旧错误数据补齐。

`runway_thresholds.json` 可以继续提供机场中心、runway roster 和非垂直元数据，但不应在
CIFP 已有发布高度时覆盖 CIFP。

### 8.2 Harvest 与 fitting

```text
OpenSky geoaltitude HAE
        +
CIFP runway/LTP HAE
        -> final_approach assignment and fit in HAE
```

assignment 是所有 threshold 上的 relative arg-min；不得加入 approach-quality threshold。
evaluation 才负责 established/gate 的 absolute judgement。

### 8.3 本地 modeling MSL

跑道已经分配后，使用该跑道的：

```text
N_runway = elevation_hae_m - elevation_msl_m
alt_local_msl = alt_hae - N_runway
```

该转换应有新的 provenance tag，明确它是 CIFP LTP anchored local MSL，而不是 EGM96
pointwise conversion。旧的 `_to_msl_egm96` artifacts 不兼容，也不应被自动接受。

### 8.4 Modeling 输出到 Cesium

每个 scenario/evaluation/prediction record 必须携带同一个 `N_runway`。输出 CZML 时：

```text
alt_hae = alt_local_msl + N_runway
```

Observed CZML 本来就是 HAE，不经过这一步。

### 8.5 未来高精度模式

如果将来需要保留 30 km 内 geoid 的空间变化，同时严格锚定 FAA CIFP LTP，可使用：

```text
N_used(p)
  = N_CIFP(LTP)
  + [N_model(p) - N_model(LTP)]
```

其中 `N_model` 可以是经过明确选择的 EGM96/EGM2008/本地 geoid model。这样入口处与
CIFP 完全一致，只借用格网的相对空间梯度。该模式必须显式启用并记录模型名称，不能成为
当前30 km管线的隐式依赖。

## 9. 决策、非目标与风险

### 9.1 本次已确认决策

- Final-approach assignment/fitting 使用 HAE；
- CIFP 是 runway/LTP 垂直事实来源；
- 当前30 km管线采用 per-runway CIFP fixed offset；
- EGM96 从必要运行依赖降为未来可选精度模式；
- 不兼容旧 `_to_msl_egm96` artifacts，不为旧错误 tracks 添加 fallback；
- KMSY 必须从 authoritative harvest 阶段重新生成，而不是 `--evaluate-only` 复用旧 tracks。

### 9.2 非目标

- 本次不修改 barometric altitude/QNH 处理；
- 不把 Cesium observed layer 改成 MSL；
- 不改变 final-approach fit window、arg-min runway assignment 或 evaluation gates；
- 不用 downstream threshold 过滤拟合质量；
- 不把 EGM96 grid 提交到仓库。

### 9.3 风险与验证重点

- Path Point continuation 与 primary 必须按同一记录组正确关联；
- 有多个 LPV approach variant 的同一跑道，其 LTP HAE/MSL/GPA/TCH 必须一致或显式报错；
- 无 Path Point 的 runway 必须从 CIFP Runway record 获得 HAE/MSL，assignment 仍需看到
  所有 threshold；
- scenario、optimizer、TS、evaluation 和 comparison CZML 必须携带同一个 runway-local
  `N`，避免单向修改造成30 m级输出偏移；
- synthetic 数据已经是 MSL，必须保持独立、显式的 source contract；
- 旧 artifacts 应失败并要求重新生成，不能因为“兼容”而被重复或漏做 datum shift。

## 10. 可复核清单

后续 review 或实现前，可独立复核以下事实：

- [ ] OpenSky 官方文档对 `baro_altitude` / `geo_altitude` 的定义；
- [ ] ARINC 424-23 §2.2.4、§4.1.10.1、§4.1.28、§5.225、§5.227；
- [ ] KMSY `FAACIFP18` 266670–266685 行的 Runway/Path Point primary/continuation；
- [ ] `harvest/cifp.py` 已解析但 `airports.py` 未使用的 `ltp_ellipsoidal_height_m`；
- [ ] KMSY `N_CIFP ≈ -26.1 m` 与 `N_EGM96 ≈ -27.259 m` 的来源差异；
- [ ] 五机场5 km/30 km的格网变化量；
- [ ] 当前 `flight_scenarios.datum` 与 `aeroviz-4d/python/vertical_datum.py` 的双向依赖；
- [ ] 旧 tracks 中 earlier overflight 抢占 landing identity 的独立 bug 与本次 datum 修改
  不混在一起验证。

## 11. 引用

1. FAA/Aeronautical Information Services, CIFP Readme volume 2603（数据遵循
   ARINC 424-18，并列出例外）：
   [`data/CIFP/CIFP_260319/CIFP Readme 2603.pdf`](../data/CIFP/CIFP_260319/CIFP%20Readme%202603.pdf)
2. ARINC Specification 424-23, §2.2.4, §4.1.10.1, §4.1.28, §5.225, §5.227
   （用于核对跨revision保持的字段语义）：
   [`data/CIFP/ARINC424-23.pdf`](../data/CIFP/ARINC424-23.pdf)
3. FAA CIFP cycle 2603 raw data：
   [`data/CIFP/CIFP_260319/FAACIFP18`](../data/CIFP/CIFP_260319/FAACIFP18)
4. OpenSky Network REST API, State vectors：
   <https://openskynetwork.github.io/opensky-api/rest.html>
5. PROJ EGM96 15-minute vertical grid：
   <https://cdn.proj.org/us_nga_egm96_15.tif>
6. EPSG:5773, EGM96 height：<https://epsg.io/5773>
7. GeographicLib GeoidEval：
   <https://geographiclib.sourceforge.io/cgi-bin/GeoidEval>
8. 项目 altitude data contract：
   [`trajectory_data_process/README.md`](../trajectory_data_process/README.md)
9. 当前 HAE→MSL 实现：
   [`flight_scenarios/datum.py`](../flight_scenarios/datum.py)
10. 当前 MSL→HAE 实现：
   [`aeroviz-4d/python/vertical_datum.py`](../aeroviz-4d/python/vertical_datum.py)
