# Five U.S. Airport Profiles for AeroViz-4D

本文档介绍 AeroViz-4D 当前美国机场组合的选择逻辑：4 个已在项目中使用的机场
`KRDU`, `KSJC`, `KSMF`, `KSTL`，以及建议新增的第 5 个美国机场 `KMSY`。

This document introduces the AeroViz-4D U.S. airport set: the four airports
already used by the project, `KRDU`, `KSJC`, `KSMF`, and `KSTL`, plus the
recommended fifth U.S. airport, `KMSY`.

## Selection Logic

选择目标不是找美国最大的机场，而是找一组**规模相近、空间分布不同、航运结构有差异、
地形和地表渲染问题互补**的机场。这样可以让同一套 AeroViz-4D 数据管线和 Cesium
渲染逻辑在不同场景下被检验，同时避免 ATL, LAX, JFK 这类超大型机场带来的数据量和
渲染复杂度过早主导系统设计。

The goal is not to choose the largest U.S. airports. The goal is to build a set
of airports with similar scale but different geography, air-service structure,
and terrain/rendering behavior. This keeps the AeroViz-4D data pipeline and
Cesium runtime comparable across cases, while avoiding mega-hub scale dominating
the system design too early.

规模比较使用 FAA CY2024 commercial service enplanements。按 FAA 页面说明，截至
2026-05-31，CY2024 是最新的最终可比数据；CY2025 初步数据预计 2026 年 6 月公布，
最终数据预计 2026 年 8 月后公布。

Scale comparison uses FAA CY2024 commercial service enplanements. As of
2026-05-31, CY2024 is the latest final comparable FAA dataset; preliminary
CY2025 data is expected in June 2026, with final CY2025 data expected later in
August 2026.

| Airport | Role | FAA CY2024 enplanements | FAA hub size | Primary diversity value |
| --- | --- | ---: | --- | --- |
| KSTL / STL | Existing U.S. airport | 7,807,362 | Medium | Large Midwest Class B airfield with complex multi-runway geometry |
| KRDU / RDU | Existing U.S. airport | 7,584,394 | Medium | Research Triangle growth market with passenger, cargo, and terrain-clearance relevance |
| KSMF / SMF | Existing U.S. airport | 6,679,426 | Medium | Inland Northern California valley airport with passenger, cargo, and international service mix |
| KMSY / MSY | Recommended fifth airport | 6,537,092 | Medium | Gulf Coast delta airport with Class B traffic and low-elevation wetland/water boundaries |
| KSJC / SJC | Existing U.S. airport | 5,822,019 | Medium | Bay Area urban airport under SFO Class B with parallel-runway and GA/corporate complexity |

## KRDU - Raleigh-Durham International Airport

**规模匹配 / Scale fit**

KRDU 的 CY2024 FAA 登机量为 7,584,394，属于 medium hub，和 KSTL, KSMF, KMSY,
KSJC 落在同一研究量级。它足够繁忙，可以代表成熟的美国中型枢纽机场；但又不至于像
大型枢纽那样让轨迹、程序、地形和影像数据规模失控。

KRDU recorded 7,584,394 CY2024 FAA enplanements and is a medium hub. It is busy
enough to represent a mature U.S. mid-sized hub, but it is still bounded enough
for repeatable trajectory, procedure, terrain, and imagery experiments.

**空间分布多样性 / Spatial diversity**

KRDU 位于北卡罗来纳 Research Triangle 区域，补充了美国东南部内陆 Piedmont
地貌。它和加州机场、密西西比河流域机场、墨西哥湾沿岸机场都不同，适合作为温和起伏
地形、森林和郊区开发混合场景的代表。

KRDU adds the inland Southeast and North Carolina Piedmont to the set. It differs
from the California, Mississippi River corridor, and Gulf Coast examples, making
it useful for moderate relief, wooded terrain, and suburban airport development.

**航运多样性 / Air-service diversity**

RDU 官方资料显示机场有约 360 次每日起降、57 个 nonstop destinations、17 家主要
航空公司，并且 2024 年货运超过 101,000 tons。这让它不只是客运样本，也能支撑货运、
航班增长和终端区运行压力的讨论。

RDU's official materials list about 360 daily arrivals and departures, 57
nonstop destinations, 17 major airlines, and more than 101,000 tons of cargo in
2024. That makes it useful beyond passenger traffic: it also supports cargo,
growth, and terminal-area operations analysis.

**地形和渲染价值 / Terrain and rendering value**

KRDU 对 DSM/DEM 对比特别有价值：地形整体不剧烈，但树木、建筑、跑道边界和
no-data 外缘会在垂直 exaggeration 下变得明显。它适合测试本项目的 airport-local
terrain bounds、DSM 厚度、hillshade、以及缩放时 terrain LOD 边界稳定性。

KRDU is valuable for DSM/DEM comparison. The terrain is not dramatic, but trees,
buildings, runway edges, and no-data margins become visible under vertical
exaggeration. It is a good stress case for airport-local terrain bounds, DSM
surface thickness, hillshade, and terrain LOD stability during zooming.

**在 AeroViz-4D 中的角色 / Role in AeroViz-4D**

KRDU 是当前最适合作为 terrain pipeline 回归测试的美国基准机场。它可以验证“局部
高精地形 + World Terrain 背景 + 程序/轨迹叠加”的核心工作流。

KRDU should remain the U.S. baseline for terrain-pipeline regression. It verifies
the core workflow of local high-resolution terrain, World Terrain context, and
procedure/trajectory overlays.

## KSJC - Norman Y. Mineta San Jose International Airport

**规模匹配 / Scale fit**

KSJC 的 CY2024 FAA 登机量为 5,822,019，仍是 medium hub，但处在本组机场的下沿。
这很有用：它能测试同一套管线在稍小但仍有复杂空域和高密度城市边界的机场上是否稳健。

KSJC recorded 5,822,019 CY2024 FAA enplanements. It is still a medium hub, but it
sits near the lower end of this set. That is useful because it tests the same
pipeline on a slightly smaller airport that still has complex airspace and dense
urban boundaries.

**空间分布多样性 / Spatial diversity**

KSJC 位于硅谷城市走廊，靠近旧金山湾、盐池、城市建筑和山麓地形。它提供了一个典型
的西海岸高密度城市机场样本，和 KRDU 的内陆林地、KSMF 的河谷平原、KSTL 的中西部
枢纽、KMSY 的沿海湿地形成对照。

KSJC sits in the Silicon Valley urban corridor near San Francisco Bay, salt
ponds, dense buildings, and foothills. It provides the West Coast dense-urban
case, contrasting with KRDU's inland tree cover, KSMF's valley plain, KSTL's
Midwest hub geometry, and KMSY's coastal wetlands.

**航运多样性 / Air-service diversity**

FAA From the Flight Deck 将 SJC 描述为 multi-use airport：以 air carrier 为主，
但也有明显的 general aviation 和 corporate aviation。机场由接近平行的 12L/30R 与
12R/30L 构成，处于 Class C 空域，并位于 San Francisco Class B 下方。

FAA From the Flight Deck describes SJC as a multi-use airport: predominantly air
carrier, but with a sizable general and corporate aviation presence. The airport
has closely spaced parallel runways 12L/30R and 12R/30L, is Class C, and lies
under San Francisco Class B.

**地形和渲染价值 / Terrain and rendering value**

KSJC 的价值不在山地高差，而在复杂边界：城市、湾水面、盐池、道路、建筑物和低矮地形
交织。它适合检查局部地形 footprint、影像贴合、平坦区域 hillshade 可读性，以及
DSM 与 DEM 在城市环境中的视觉差异。

KSJC is valuable less for mountain relief and more for boundary complexity:
urban fabric, bay water, salt ponds, roads, buildings, and low terrain are
intermixed. It is useful for testing terrain footprints, imagery alignment,
hillshade readability over flat areas, and DSM-vs-DEM differences in a city.

**在 AeroViz-4D 中的角色 / Role in AeroViz-4D**

KSJC 是“城市低地 + 复杂邻近空域”的美国样本。它尤其适合验证机场边界、影像、DSM/DEM
和程序线在密集地物中的可读性。

KSJC is the U.S. dense-urban lowland and neighboring-airspace case. It is
especially useful for validating airport boundaries, imagery, DSM/DEM rendering,
and procedure-line readability in dense surface detail.

## KSMF - Sacramento International Airport

**规模匹配 / Scale fit**

KSMF 的 CY2024 FAA 登机量为 6,679,426，和 KMSY 非常接近，也接近 KSJC 和 KRDU。
它提供了一个规模适中但增长明显的西海岸内陆机场样本。

KSMF recorded 6,679,426 CY2024 FAA enplanements, very close to KMSY and still
near KSJC and KRDU. It provides a mid-sized but growing inland West Coast case.

**空间分布多样性 / Spatial diversity**

KSMF 位于 Sacramento Valley，地形低平、空间开阔，和 KSJC 的城市湾区环境不同。
它让美国机场组合不只覆盖沿海加州，也覆盖加州内陆河谷、农业/郊区过渡带和相对平坦的
大范围地表。

KSMF sits in the Sacramento Valley, with low, open terrain that differs from
KSJC's urban Bay Area setting. It expands the California coverage from coastal
urban terrain to inland valley terrain, agricultural/suburban transition, and
large low-relief surfaces.

**航运多样性 / Air-service diversity**

SMF 官方页面列出 51 个 nonstop destinations；机场历史页称其服务 12 家主要承运人，
航空公司页面同时列出 passenger airlines 和 cargo carriers。Pilot information 页面
还说明 SMF 有 24-hour staffed control tower and ATC services、场高 27 ft MSL。

SMF's official site lists 51 nonstop destinations. Its history page says the
airport is served by 12 major carriers, and its airline page lists both passenger
airlines and cargo carriers. The pilot-information page also notes a 24-hour
staffed control tower and ATC services, with an airfield elevation of 27 ft MSL.

**地形和渲染价值 / Terrain and rendering value**

KSMF 是平坦地形渲染的好样本。真正的挑战不是大起伏，而是低 relief 下 hillshade 是否
还能表达微小坡度、DSM/DEM 之间是否过度夸张、以及机场外围大面积平坦区域是否保持
视觉稳定。

KSMF is a strong flat-terrain rendering case. The challenge is not dramatic
relief; it is whether hillshade can still communicate subtle slopes, whether
DSM/DEM differences become over-amplified, and whether large flat airport
surroundings remain visually stable.

**在 AeroViz-4D 中的角色 / Role in AeroViz-4D**

KSMF 是“西海岸内陆平原 + 增长型中型机场”的样本。它适合验证低起伏地形下的渲染效率、
地形材质对比度和机场级数据包大小控制。

KSMF is the inland West Coast plain and growing mid-sized airport case. It is
well suited for validating rendering efficiency, terrain-material contrast, and
airport-level data package size under low-relief conditions.

## KSTL - St. Louis Lambert International Airport

**规模匹配 / Scale fit**

KSTL 的 CY2024 FAA 登机量为 7,807,362，是本组里规模最大的一个，但仍然是 medium
hub。它给这组机场提供了上沿样本：比 KSMF, KMSY, KSJC 更忙，但仍和它们处在同一
可比阶层。

KSTL recorded 7,807,362 CY2024 FAA enplanements, the largest in this group but
still a medium hub. It provides the upper end of the set: busier than KSMF,
KMSY, and KSJC, while remaining comparable.

**空间分布多样性 / Spatial diversity**

KSTL 位于密苏里州 St. Louis 都市区，补充中西部和 Mississippi/Missouri River
走廊语境。它与加州机场、东南部 KRDU、墨西哥湾 KMSY 都不同，能帮助避免美国样本过度
集中在东西海岸。

KSTL adds the Midwest and the Mississippi/Missouri River corridor. It differs
from the California airports, the Southeast KRDU case, and the Gulf Coast KMSY
case, reducing coastal bias in the U.S. subset.

**航运多样性 / Air-service diversity**

FAA From the Flight Deck 将 STL 描述为 Missouri 最繁忙机场，交通包含 scheduled
air carriers、air taxis，以及少量 general aviation 和 military aircraft。它也是
Class B 空域机场，拥有三条平行跑道加一条交叉跑道，地面几何和流量混合都较复杂。

FAA From the Flight Deck describes STL as Missouri's busiest airport, with
scheduled air carriers, air taxis, and a smaller mix of general aviation and
military aircraft. It is also a Class B airport with three parallel runways plus
one intersecting runway, giving it complex geometry and traffic mix.

**地形和渲染价值 / Terrain and rendering value**

KSTL 的价值主要在机场几何和中西部城市地貌，而不是高山地形。多跑道、多滑行道、
城市/工业边界和宽阔地表可以检验 runway layer、procedure layer、obstacle layer 与
terrain layer 同屏时的可读性和性能。

KSTL's value is mainly airfield geometry and Midwest urban terrain rather than
mountain relief. Multiple runways, taxiways, urban/industrial edges, and broad
surface areas test readability and performance when runway, procedure, obstacle,
and terrain layers are shown together.

**在 AeroViz-4D 中的角色 / Role in AeroViz-4D**

KSTL 是“复杂跑道几何 + 中西部 Class B”的美国样本。它适合压力测试多跑道机场的图层
组织和渲染密度。

KSTL is the complex-runway Midwest Class B case. It is useful for stress-testing
layer organization and rendering density at a multi-runway airport.

## KMSY - Louis Armstrong New Orleans International Airport

**规模匹配 / Scale fit**

KMSY 的 CY2024 FAA 登机量为 6,537,092，和 KSMF 的 6,679,426 非常接近，也和
KSJC, KRDU, KSTL 保持同一 medium hub 等级。因此，它可以作为第 5 个美国机场加入，
而不会破坏现有组合的规模一致性。

KMSY recorded 6,537,092 CY2024 FAA enplanements, very close to KSMF's 6,679,426
and still in the same medium-hub class as KSJC, KRDU, and KSTL. It can be added
as the fifth U.S. airport without breaking the scale balance of the set.

**空间分布多样性 / Spatial diversity**

KMSY 最大的增量价值是空间分布：它把美国样本扩展到 Gulf Coast 和 Mississippi Delta
区域。这个区域有低海拔、湿地、水体、堤坝、排水渠和城市边缘，比 KRDU、KSJC、KSMF、
KSTL 都更适合测试近海低地机场。

KMSY's largest contribution is geographic. It adds the Gulf Coast and
Mississippi Delta region, with low elevation, wetlands, water bodies, levees,
drainage canals, and urban edges. That makes it a better coastal-lowland test
than KRDU, KSJC, KSMF, or KSTL.

**航运多样性 / Air-service diversity**

FAA From the Flight Deck 将 MSY 描述为以 air carrier 为主的机场，位于 New Orleans
市中心以西约 11 miles，有两条跑道、两个 FBO、新客运航站楼，并处于 Class B 空域。
MSY 官方航空统计还把 domestic scheduled、international scheduled 和 charter 分开
列示，适合作为旅游目的地和国际/季节性服务组合的样本。

FAA From the Flight Deck describes MSY as primarily air carrier, about 11 miles
west of downtown New Orleans, with two runways, two FBOs, a new passenger
terminal, and Class B airspace. MSY's official airline statistics also separate
domestic scheduled, international scheduled, and charter traffic, making it a
useful tourism and international/seasonal service case.

**地形和渲染价值 / Terrain and rendering value**

KMSY 对 terrain pipeline 很有价值，因为它会暴露低海拔和水陆边界问题：DEM/DSM
在湿地、水面、机场填土、道路和排水设施之间的 no-data、插值和边界裁剪更容易被看见。
它也能补充 KRDU 那类 DSM 树冠厚度问题，转而测试“平但边界复杂”的沿海地形。

KMSY is valuable for the terrain pipeline because it exposes low-elevation and
land/water boundary issues. DEM/DSM no-data handling, interpolation, and
footprint clipping around wetlands, water, airport fill, roads, and drainage
features become more visible. It complements KRDU's DSM canopy-thickness issues
with a flat but boundary-complex coastal case.

**在 AeroViz-4D 中的角色 / Role in AeroViz-4D**

KMSY 推荐作为新增第 5 个美国机场。它的主要任务不是增加一个“更大机场”，而是增加一个
新的地理和地表类型：Gulf Coast lowland airport。

KMSY is the recommended fifth U.S. airport. Its purpose is not to add a larger
airport, but to add a new geography and surface type: a Gulf Coast lowland
airport.

## Combined Value

这 5 个美国机场形成一个互补组合：

These five U.S. airports form a complementary set:

- KRDU: 东南部 Piedmont、增长型中型机场、terrain pipeline 回归基准。
- KRDU: Southeast Piedmont, growing mid-sized airport, terrain-pipeline baseline.
- KSJC: 湾区城市低地、复杂邻近空域、城市 DSM/DEM 可读性样本。
- KSJC: Bay Area urban lowland, neighboring-airspace complexity, urban DSM/DEM readability case.
- KSMF: 加州内陆河谷、低 relief、大面积平坦地表和增长型机场样本。
- KSMF: California inland valley, low relief, broad flat surfaces, and growth-market case.
- KSTL: 中西部 Class B、多跑道复杂几何、图层密度压力测试样本。
- KSTL: Midwest Class B, complex multi-runway geometry, and layer-density stress case.
- KMSY: Gulf Coast lowland、湿地/水陆边界、国际/旅游目的地航运样本。
- KMSY: Gulf Coast lowland, wetland/land-water boundaries, and international/tourism-service case.

从渲染效率角度看，这个组合也比较合理：全部机场都是 medium hub 量级，机场局部地形
和程序数据包可以保持在可管理范围内；但它们的地理和地表差异足够大，可以发现只在一种
机场上看不出来的 Cesium LOD、DSM/DEM footprint、hillshade、影像贴合和图层遮挡问题。

From a rendering-efficiency perspective, the set is also reasonable. All five
airports are medium-hub scale, so local terrain and procedure packages can stay
manageable. At the same time, their geographic and surface differences are large
enough to expose Cesium LOD, DSM/DEM footprint, hillshade, imagery alignment,
and layer-occlusion issues that would not appear in a single-airport test.

## Sources

- FAA, Passenger Boarding (Enplanement) and All-Cargo Data for U.S. Airports:
  <https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger>
- FAA, CY2024 commercial service enplanements PDF:
  <https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/arp-cy2024-commercial-service-enplanements.pdf>
- FAA From the Flight Deck, SJC:
  <https://www.faa.gov/flight_deck/sjc>
- FAA From the Flight Deck, STL:
  <https://www.faa.gov/flight_deck/stl>
- FAA From the Flight Deck, MSY:
  <https://www.faa.gov/flight_deck/msy>
- RDU official facts:
  <https://www.rdu.com/facts/>
- RDU official cargo statistics:
  <https://www.rdu.com/fbo-and-cargo/cargo-statistics/>
- Sacramento International Airport official pilot information:
  <https://flysmf.gov/about/pilot-information/>
- Sacramento International Airport official home page:
  <https://flysmf.gov/>
- Sacramento International Airport official airlines page:
  <https://flysmf.gov/flight-and-travel/airlines>
- Sacramento International Airport official history page:
  <https://flysmf.gov/about/history/>
- MSY official airport data and statistics:
  <https://flymsy.com/business/newsroom/airport-data-and-statistics/>
