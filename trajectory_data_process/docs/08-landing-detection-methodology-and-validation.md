# 08 · 落地数据集方法论:bbox 几何提取的可信度与验证指南

> 面向论文的方法论说明 + 可执行的验证清单。
> 回答一个核心问题:**「直接用 bbox 下载再重建轨迹、按几何判定落地」这套做法,在论文层面说服力够吗?是不是该改用 full-track 的到达机场元数据(`estarrivalairport`)?**

---

## 0. TL;DR(先给结论)

1. 当前 **bbox + 几何重建 + 几何落地判定** 的做法在论文层面**站得住**,但它的说服力**不来自「数据源选得对」,而来自「有没有验证检测器」**。
2. **full-track 的元数据不是它的上位替代**——恰恰相反,对「按跑道分」的研究,元数据不够用:
   - `estarrivalairport` 是 OpenSky **估计**出来的 noisy label;
   - 它只到**机场级**,给不了**跑道级**(23R vs 05L),而本项目整条 pipeline 是按 runway threshold 分的。
3. 所以这**不是二选一**:几何跑道判定绕不过去。正确姿势是**混合**——几何做主判定,元数据降级成**独立交叉验证 / provenance 富集层**。
4. 让它有说服力的关键动作:**几何↔元数据一致性验证 + 阈值敏感性分析 + 权威源(航图/OurAirports)对照 + 覆盖统计与局限声明**。其中方向那一支的工具**已经就绪**(`heading_rejected` + `summarize_heading_rejected.py`)。

---

## 1. 背景:我们在提取什么、为什么

本项目从观测 ADS-B 中提取**按跑道入口(runway threshold)归属的真实进近轨迹**,用于:

- 观测 vs 优化 vs 仿真的三方可视化对比;
- 给优化器提供真实的 initial / target 状态;
- 验证模型。

因此数据集的真实需求是:**归属正确(到具体跑道)、终端区(最后 ~30 km / 最终进近)几何保真、终端段完整**。全程 gate-to-gate 的巡航段与本研究无关。

当前实现(`download_landings.py` → `landings.py`):

- 以机场为中心、`--radius-km`(默认 30 km)的 **bbox** 查询 OpenSky Trino 历史库(`state_vectors_data4`,**不含**到达机场元数据);
- 按 `icao24` + 900 s 时间间隔切分,**重建轨迹**;
- **几何判定落地**(`processing/czml_export.py`):最近点落在 threshold 的 `RUNWAY_THRESHOLD_RADIUS_M`(1 km)内 + 下降 ≥ `descent_margin_m`(300 m)+ 航向对齐 ≤ `DEFAULT_HEADING_TOLERANCE_DEG`(20°)+ 最近点 ≤ `landing_max_agl_m`(1500 m AGL);
- 方向不符但其余像落地的轨迹**不丢**,存 `*_heading_rejected.json` 供审查。

---

## 2. 两种路线

| | A · bbox + 几何(现状) | B · full-track 元数据 |
|---|---|---|
| 数据源 | `state_vectors_data4` bbox 查询 | flights 表 / `airport=` join(带 `est{arr,dep}airport`) |
| 归属依据 | 相对 threshold 的位置/航向/下降(几何) | OpenSky 估计的到达机场字段 |
| 归属粒度 | **跑道级** | **机场级**(无跑道) |
| 对元数据的依赖 | 无(独立) | 强(且是估计值) |
| 下载量 | 只终端区 | 整条 gate-to-gate |
| 可审计性 | 高(判据透明可复现) | 低(启发式不透明) |

---

## 3. 核心判断:元数据是「机场级」,需求是「跑道级」

`estarrivalairport` 有两个硬伤,正好卡在研究目标上:

1. **它是估计值。** OpenSky 用「航迹末端最近机场」启发式反推,GA、备降、复飞、缺失都不少;文献普遍把它当 noisy label,而非 ground truth。
2. **它只到机场,不到跑道。** 本 pipeline 按 **runway threshold** 分(每条跑道 N 个落地、per-runway 对比)。元数据给你 `CYYC`,给不了 `23R` vs `05L`。

**结论:即使切到元数据,你仍必须在其上再做一遍几何跑道判定。** 元数据顶多把「是不是到本场」先筛一道,替代不了核心逻辑。**几何判定不可绕过——这不是二选一。**

反过来,现状可以正面包装成方法论**优点**:

> 我们不依赖 OpenSky 估计的、且仅到机场级的 `estarrivalairport`,而是用相对跑道入口的**直接几何证据**(1 km 邻近 + 航向对齐 + 下降)做**跑道级**归属,依据可审计、可复现。

---

## 4. 现状路线的效度威胁(诚实清单)

审稿人不会攻击「为什么用 bbox」,会攻击「你手调的检测器怎么保证对」。当前软肋:

- **阈值手调**(1 km / 20° / 300 m / 1500 m AGL):是否系统评估过
  - **假阳**:高空掠过、复飞(go-around)、掠过邻近/平行跑道被误判为落地?
  - **假阴**:近地 ADS-B 稀疏、缺 `geoaltitude` 导致真落地被漏?
- **轨迹重建**(icao24 + 900 s gap 切分):holding、missed-approach + 二次进近如何处理?会不会误切 / 误并?
- **近地覆盖偏差**:ADS-B 到地面很稀,最近点常还在几百到 ~2000 ft,**几乎观测不到真正接地**——必须明说。
- **抽样偏差**:往回扫到凑够 N 个,是按**可得性**抽样,非随机,可能偏向覆盖好的机型/时段。

> ⚠️ 关键:**换成元数据一个都解决不了**——元数据本身也是启发式、也没跑道、也一样受近地覆盖限制。**「换数据源」不是解药,「加验证」才是。**

---

## 5. 让它有说服力:验证计划

按投入产出比排序。

### 5.1 几何 ↔ 元数据 一致性交叉验证 ★最值钱

对一个样本,**单独**去 flights 表 / arrival API 拉 `estarrivalairport`,报:

> 几何判为 `CYYC` 的落地里,在元数据存在的样本上一致率 = **X%**;不一致的 (100−X)% 主要是复飞 / 备降 / 元数据缺失。

**两个独立方法一致**是很强的效度论证,直接回答「你怎么知道检测器对」。
（实现要点:bbox 查询用的 `state_vectors_data4` **不含**到达机场字段,交叉验证需按 `icao24`+时间**另查 flights 表**。）

### 5.2 阈值敏感性分析

对 1 km / 20° / 300 m 各扫几档,证明结论对合理取值**稳健**。

- **方向这一支的工具已就绪**:`*_heading_rejected.json` + `summarize_heading_rejected.py` 已输出「误差直方图 + 假杀信号」;把它写成**「假杀率随 `--heading-tolerance-deg` 的变化曲线」**即可(被拒轨迹已落盘,调阈值不用重新下载,重跑脚本即看新分布)。
- 关注头号信号 `geometry_ok_track_bad`(几何已对准、只 ADS-B track 不符 → 很可能是被噪声误杀的真落地)。

### 5.3 权威源 / 航图对照

- 跑道 threshold 几何(入口坐标 / 真航向)来自 OurAirports(`build_runway_config.py`),可与**公布航图**对照;
- 项目已有对公布 RNAV 值的 golden 测试(见 `aeroviz-4d/python` 的 CIFP 章节),说明所用跑道几何与现实一致——**引用它**佐证「几何落点确实落在跑道上」。

### 5.4 覆盖统计 + 局限声明

- 每条进近的**点数**、**最近点高度**分布;
- 坦白「**未观测到接地**」这一 ADS-B 固有限制,并说明判据为何用「对齐下降」而非「接地高度」。

---

## 6. 推荐架构:几何主判定 + 元数据交叉验证层

```
                      ┌───────────────────────────┐
  bbox state vectors  │  几何落地判定(主判定)     │  → accepted / heading_rejected
   (state_vectors_    │  跑道级、可审计            │
        data4)        └──────────────┬────────────┘
                                     │  按 icao24 + 时间回查
                      ┌──────────────▼────────────┐
   flights 表 /       │  元数据交叉验证(佐证层)   │  → 一致 = 双重证据
   arrival API        │  arr_airport / callsign /  │  → 不一致 = 单独存档待审
   (est arr airport)  │  起飞机场(provenance)     │     (仿 heading_rejected)
                      └───────────────────────────┘
```

- **主判定**:保留几何(跑道级,不可替代)。
- **加一层元数据**:
  - (a) 一致 → 双重独立证据;
  - (b) 不一致 → 像 `heading_rejected` 一样单独存下来审;
  - (c) 顺带补 provenance(从哪来、什么航班)——对 **CTA** 语境尤其有用。

这样论文里可写成:「几何直接归属(可审计、跑道级)+ 独立元数据交叉验证(一致率 X%)+ 阈值敏感性 + 航图对照」。

---

## 7. 论文写作模板(可直接改写)

> **数据来源与归属方法。** 我们从 OpenSky 历史库以机场为中心的终端区包围盒(半径 30 km)获取原始 ADS-B 状态向量,并按机身地址(`icao24`)与时间间隔重建轨迹。落地归属采用**直接几何判据**——相对跑道入口的横向邻近(≤ 1 km)、航向对齐(≤ 20°,同时用轨迹几何航向与 ADS-B 航迹双重校验)与持续下降(≥ 300 m)——而非依赖 OpenSky 估计的、仅到机场级的 `estarrivalairport` 字段,从而获得**可审计的跑道级**归属。
>
> **效度验证。** (i) 在存在到达机场元数据的样本上,几何归属与 `estarrivalairport` 的一致率为 **X%**,不一致主要源于复飞/备降/元数据缺失;(ii) 归属对判据阈值不敏感(航向容差在 15°–30°、邻近半径在 0.5–2 km 区间内结论稳定);(iii) 所用跑道入口几何与公布航图一致;(iv) 我们如实报告 ADS-B 近地覆盖限制(最近观测点中位高度 ~___ ft AGL,未观测到接地),故判据以对齐下降而非接地高度为准。

---

## 8. 落地清单(TODO)

- [ ] **几何↔元数据一致性脚本**:复用 `download_landings` 已判出的落地,按 `icao24`+时间查 OpenSky flights 表,输出一致率 + 不一致清单(风格对齐 `summarize_heading_rejected.py`)。
- [ ] **阈值敏感性**:对 `--heading-tolerance-deg`(及 1 km / 300 m)扫档,基于 `*_heading_rejected.json` 画「假杀率 vs 阈值」。
- [ ] **覆盖统计**:每条进近的点数 / 最近点 AGL 直方图 + 中位数。
- [ ] **provenance 富集**(可选):把 flights 表的 `callsign`/起飞机场并入每条落地记录。
- [ ] **局限声明**:在论文与 README 写清近地覆盖 + 抽样偏差。

---

## 附:相关代码 / 产物索引

| 关注点 | 位置 |
|---|---|
| bbox 下载 + 反扫 | `download_landings.py`, `landings.py::download_airport_landings` |
| 几何落地判定 | `processing/czml_export.py::trajectory_to_czml_flight` / `_is_landing_geometry` / `_annotate_heading` |
| 判据常量 | `landings.py`(`RUNWAY_THRESHOLD_RADIUS_M` 等),`czml_export.py`(`DEFAULT_HEADING_TOLERANCE_DEG`) |
| 方向假杀审查 | `*_heading_rejected.json`(产物),`summarize_heading_rejected.py`(汇总) |
| 跑道几何来源 | `build_runway_config.py` → `config/runway_thresholds.json`(OurAirports) |
| 航图 golden 对照 | `aeroviz-4d/python`(CIFP / RNAV 公布值测试) |
