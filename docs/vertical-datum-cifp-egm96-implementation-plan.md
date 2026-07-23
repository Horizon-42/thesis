# CIFP 双垂直基准与最终进近管线实施计划

状态：**等待评审，垂直基准重构尚未实施**  
日期：2026-07-23  
前置调查：[`vertical-datum-cifp-egm96-investigation.md`](vertical-datum-cifp-egm96-investigation.md)

> 实施门槛：本文先固定问题边界、数据契约、修改顺序和验收条件。收到评审确认前，
> 不开始本文所列的业务代码、schema、测试或数据重生成修改。

## 1. 目标

本次修改要同时完成两件相互关联、但必须独立验证的工作：

1. 修正最终进近航段与 landing identity/crop 的关联，避免较早飞越跑道的样本抢占真正的
   最后进近，使 `arrivals` 截断掉可拟合航段；
2. 以 FAA CIFP 的跑道入口 HAE/MSL 双高程建立唯一垂直基准契约，使 runway assignment
   和 final-approach fitting 全程使用 HAE，跑道分配后再用每跑道固定偏差进入本地 MSL
   建模平面，并取消当前生产路径对 EGM96 格网的依赖。

最终结果应允许“几何上可以拟合、但质量不满足 evaluation gate”的航迹正常进入场景：
`final_approach` 负责给出数学事实，跑道分配负责相对选择，只有 `evaluation` 负责绝对质量
判断。

## 2. 必要背景

### 2.1 原始失败不是“UAL1850 拟合质量不合格”

最初失败为：

```text
ValueError: flight 'UAL1850' has no usable fitted final approach for runway '29'
```

已定位的因果链是：

1. UAL1850 在真正的最后进近之前曾更接近 RW29 threshold；
2. `final_approach` 可以从航迹的**最后一个连续 inbound pass**中找到并拟合真正的最后进近；
3. 旧 landing anchor 逻辑却在**整条航迹的所有样本**上按 threshold 距离做 `argmin`；
4. 较早飞越的离散样本比最后进近末端样本更靠近 threshold，因此取得了
   `landing_sample_index`；
5. `arrivals` 按这个错误 index 截断，真正的最后进近被裁掉；
6. `flight_scenarios --target-from-fitted-adsb` 收到错误裁剪的航迹后没有足够的可拟合末端，
   最终抛出上述错误。

这里存在两个不同的 `argmin`，不能混为一谈：

| 名称 | 比较对象 | 用途 | 结论 |
|---|---|---|---|
| runway-assignment arg-min | 同一航迹对所有候选跑道的 final-pass cross-track score | 分配唯一跑道 | 正确，必须保留 |
| landing-sample arg-min | 一条航迹的所有样本到已分配 threshold 的距离 | 生成 landing identity/crop anchor | 旧实现错误，必须限制在已拟合的最后 inbound pass 之后 |

当前工作树已经有 landing identity 的定向修复：

- [`final_approach/fit.py`](../final_approach/fit.py) 的 `SegmentFit` 保存原始
  `first_sample_index` / `last_sample_index`；
- [`trajectory_data_process/harvest/classify.py`](../trajectory_data_process/harvest/classify.py)
  从已分配 fit 的最后样本开始寻找 landing endpoint；
- [`trajectory_data_process/harvest/tests/test_classify.py`](../trajectory_data_process/harvest/tests/test_classify.py)
  覆盖“较早飞越 + 真正最后进近”的回归场景。

本文计划会保留、补全并端到端验证这项修复。它不等同于尚未实施的 CIFP/垂直基准重构。

### 2.2 四层职责必须分开

| 层 | 输入/基准 | 负责什么 | 不负责什么 |
|---|---|---|---|
| landing classification | ADS-B HAE + 全部 runway HAE frames | 判断是否为本场降落、是否有结构上可拟合的候选 | 不判断是否满足 evaluation 的 established/gates |
| final-approach fit | 同一 datum 下的 track + runway frame | 选择最后连续 inbound pass，执行 OLS，返回 crossing、slope、uncertainty 和源 sample indices | 不输出合格/不合格 verdict，不读取 regulation thresholds |
| runway assignment | 所有候选跑道的可用 fit | 用相对 cross-track score 做全跑道 arg-min，并单独报告 ambiguous/unassignable | 不以 approach quality gate 过滤后再分配 |
| evaluation | 与上面相同的 fit 几何 | 应用绝对 established/FAA gates，并把 `not_established` 计入统计 | 不反向决定哪些航迹可以被 assignment 或 fitted-ADSB scenario 看见 |

`LandingScreen` 只允许保留“这是否像一次本场降落”的宽松分类边界；它不得演变成
evaluation gate，也不得要求 fitted-ADSB target 先通过 evaluation 才能生成。

### 2.3 已确认的高度语义

- OpenSky `geoaltitude` 是 geometric altitude，harvest 保存为 WGS-84 椭球高 HAE；
- CIFP Path Point primary 包含 LTP HAE；
- CIFP Path Point continuation 包含同一 LTP 的 orthometric height（本文简称 MSL）；
- CIFP Runway primary 同时包含 LTP ellipsoidal height 和 landing threshold elevation；
- `N = HAE - MSL`，所以 `MSL = HAE - N`；
- 获得 ADS-B HAE 不需要 EGM96；EGM96 格网只是在任意经纬度给出模型化的空间变化。

KMSY CIFP 提供的同点事实为：

| Runway | HAE (m) | MSL (m) | `N = HAE - MSL` (m) |
|---|---:|---:|---:|
| 02 | -25.7 | +0.4 | -26.1 |
| 11 | -25.2 | +0.9 | -26.1 |
| 20 | -27.0 | -0.9 | -26.1 |
| 29 | -25.9 | +0.2 | -26.1 |

KMSY 最终进近 5 km 范围内 EGM96 的 sampled variation 约为 0.10 m，30 km 内约为
0.61 m；五个研究机场在 30 km 内的最坏 sampled deviation 约为 1.50 m。对当前
7.62 m 量化的 ADS-B 高度和约 1.7 m 的拟合不确定度，每跑道固定 CIFP offset 足够；
它不用于宣称亚米级大范围 geodetic conversion 或 terrain clearance。

### 2.4 当前错误的数据流

```text
OpenSky geoaltitude HAE
        |
        +-------------------------------> harvest tracks (HAE)

CIFP Path Point HAE --已解析但被 airports.py 忽略--+
                                                    |
runway_thresholds.json MSL + EGM96 N ---------------+
        -> 间接重建 runway HAE
        -> assignment / fit

harvest arrival HAE
        -> flight_scenarios.datum 逐点 EGM96 HAE→MSL
        -> fitted_approach 只接受 _to_msl_egm96 tag
        -> scenario / optimization / TS / evaluation (MSL)
        -> vertical_datum.py 逐点 EGM96 MSL→HAE
        -> comparison CZML
```

问题不只是“缺少格网”：当前实现已经读取 CIFP 的直接 HAE 事实，却忽略它，再把
OurAirports MSL 与 EGM96 组合成另一个 HAE；下游又把 fitted-ADSB 强制绑定到 EGM96
转换后的 tag。这混合了不同来源/realization，也给当前任务增加了不必要的运行依赖。

### 2.5 目标数据流

```text
CIFP Path Point primary + continuation
              或 CIFP Runway primary
        -> runway threshold {HAE, MSL, N, provenance}

OpenSky geoaltitude HAE + runway HAE
        -> landing classification
        -> final inbound fit
        -> all-runway relative arg-min assignment
        -> fitted ADS-B crossing (仍为 HAE)

跑道分配后：N_runway = runway HAE - runway MSL
        -> waypoint_local_msl = waypoint_hae - N_runway
        -> scenario / optimizer / TS / evaluation
        -> output_hae = output_local_msl + N_runway
        -> Cesium comparison CZML
```

Observed CZML 继续直接使用 ADS-B HAE，不做转换。

### 2.6 `runway_thresholds.json` 的保留职责

[`trajectory_data_process/config/runway_thresholds.json`](../trajectory_data_process/config/runway_thresholds.json)
继续承担：

- 机场和 active-runway roster；
- runway ident、heading、displaced-threshold 等非垂直元数据；
- 必要的几何交叉检查。

它不再作为跑道入口 HAE/MSL 的垂直事实来源，也不能在 CIFP 缺少必要垂直字段时静默
补值。无 Path Point 的跑道仍应参加 assignment，但其 HAE/MSL 来自 CIFP Runway primary；
若同时缺少 LPV TCH/GPA，则继续明确排除在 model-ready LPV arrivals 之外。

## 3. 范围

### 3.1 In scope

- 解析 CIFP Path Point continuation 的 LTP orthometric height；
- 解析 CIFP Runway primary 的 threshold position、HAE 和 MSL；
- 将 `Runway`、arrival `runway_target` 和下游 records 改为显式双基准契约；
- assignment 与 fitted-ADSB fit 使用 HAE；
- 分配后用同一 per-runway CIFP offset 进入/离开 local MSL；
- 清除生产路径的 EGM96/pyproj/grid 依赖；
- 保留并验证 final-inbound landing identity 修复；
- 拒绝旧错误 schema、旧 altitude tag 和缺少必要 provenance 的 artifacts；
- 全量重新 harvest KMSY 并验证 UAL1850，之后按评审决定推广到其他机场；
- 更新与该契约直接相关的 README、`CLAUDE.md` 和数据格式文档。

### 3.2 Out of scope

- 不改变 OpenSky barometric altitude/QNH 处理；
- 不把 observed CZML 改成 MSL；
- 不改变 OLS 形式、fit window、最小样本数或 uncertainty 算法；
- 不给 runway arg-min 增加 evaluation quality thresholds；
- 不修改 established/FAA gate 的数值；
- 不提交 EGM96 grid；
- 不为旧错误 tracks、`_to_msl_egm96` records 或旧 manifest schema 增加兼容分支；
- 不把 fixed-offset 结果宣传为高精度 terrain/obstacle clearance 高度。

## 4. 拟定数据契约

字段名在评审确认后一次性落地；不保留旧字段 alias。

### 4.1 CIFP parser 输出

`PathPoint` 至少增加：

```text
ltp_ellipsoidal_height_m
ltp_orthometric_height_m
```

新增 CIFP Runway record 对象，至少包含：

```text
airport
runway
latitude
longitude
ltp_ellipsoidal_height_m
landing_threshold_elevation_msl_m
```

Parser 必须按 CIFP cycle 所遵循的 ARINC 424-18 record layout 解码；仓库中的
ARINC 424-23 文档只用于核对本次涉及字段的延续语义，不能把完整 v23 layout 直接套到
`FAACIFP18`。

### 4.2 Runtime `Runway`

建议的 canonical 字段为：

```text
lat
lon
elevation_hae_m
elevation_msl_m
hae_minus_msl_m          # N = elevation_hae_m - elevation_msl_m
position_source
vertical_source
course_deg
threshold_crossing_height_m
published_glidepath_deg
```

使用 `hae_minus_msl_m` 而不是继续把该值命名为 EGM96 `geoid_undulation_m`，目的是让
字段本身明确表达数学方向，不暗示其来源是某个 geoid grid。

必须保持以下 invariant：

```text
abs(elevation_hae_m - elevation_msl_m - hae_minus_msl_m) <= numeric_tolerance
```

### 4.3 Arrival `runway_target`

每条 model-ready arrival 必须携带：

```json
{
  "lat": 29.0,
  "lon": -90.0,
  "elevation_hae_m": -25.9,
  "elevation_msl_m": 0.2,
  "hae_minus_msl_m": -26.1,
  "course_deg": 290.0,
  "threshold_crossing_height_m": 15.0,
  "published_glidepath_deg": 3.0,
  "position_source": "faa_cifp_path_point",
  "vertical_source": "faa_cifp_path_point"
}
```

示例数值只展示 schema；实际值直接来自相应 CIFP record。manifest/record schema 需要
升级，使缺少上述双基准/provenance 的旧 artifacts 在 loader 边界失败，而不是进入 fallback。

### 4.4 Altitude provenance

建议固定三类 source contract：

| 数据 | `altitude_source` | datum |
|---|---|---|
| 原始 harvested ADS-B | `opensky_history_geoaltitude_m` | HAE |
| 经已分配跑道 CIFP offset 转换的 modeling track | `opensky_history_geoaltitude_m_to_local_msl_cifp_threshold` | runway-local MSL |
| 现有 synthetic 数据 | `synthetic` | 已有 synthetic MSL contract |

旧 `opensky_history_geoaltitude_m_to_msl_egm96` 必须报错并提示重新 harvest/regenerate；
不得视为“已经是 MSL”继续接受。

### 4.5 CIFP 来源优先级

1. Path Point primary + 与其成组的 continuation：提供 LTP position、HAE、MSL、TCH、GPA；
2. Path Point 不可用时，CIFP Runway primary：提供 threshold position、HAE、MSL；
3. CIFP 无法提供完整 HAE/MSL 时：该机场/跑道在 harvest 边界明确失败或明确列为不可用，
   不从历史 tracks 或 OurAirports elevation 猜测垂直值。

同一跑道有多个 LPV variant 时，LTP position、HAE、MSL、TCH/GPA 若不一致，parser/loader
必须给出可审计错误，不能继续沿用“first record wins”掩盖冲突。

## 5. Action items

- [ ] **1. 先固定 CIFP 双基准 parser 测试。** 新增
  `trajectory_data_process/harvest/tests/test_cifp.py`，用定长 fixture 和 KMSY 原始记录
  验证 Path Point primary/continuation 关联、Runway primary 字段、负 HAE、MSL 单位/符号、
  多 variant 一致性及 malformed/missing continuation 的失败行为。

- [ ] **2. 实现 CIFP Runway 与 Path Point continuation 解析。** 修改
  [`trajectory_data_process/harvest/cifp.py`](../trajectory_data_process/harvest/cifp.py)，
  保留现有 fixed-column decode confidence checks，增加双高程的单位、范围和记录组校验；
  不用 ARINC v23 的无关列布局覆盖 FAA cycle 的 v18 layout。

- [ ] **3. 重构机场/跑道加载边界。** 修改
  [`trajectory_data_process/harvest/airports.py`](../trajectory_data_process/harvest/airports.py)，
  按 CIFP 来源优先级构建 `Runway` 的 HAE、MSL、`hae_minus_msl_m` 和 provenance；移除
  `config MSL + EGM96 -> HAE` 的间接重建。`runway_thresholds.json` 只保留 roster 与
  非垂直元数据职责。

- [ ] **4. 完成 landing identity/crop 修复并删除旧数据 fallback。** 保留
  `SegmentFit` 的原始 sample indices 和 `classify.py` 的 final-pass endpoint 搜索；修改
  [`trajectory_data_process/harvest/arrivals.py`](../trajectory_data_process/harvest/arrivals.py)，
  让 `_anchor_index()` 缺少合法 `landing_sample_index` 时直接报“必须重新 harvest”，删除
  整条 track 上重新做距离 `argmin` 的兼容逻辑。

- [ ] **5. 升级 manifest/record 与 `runway_target` schema。** 在 harvest 输出中写入
  `elevation_hae_m`、`elevation_msl_m`、`hae_minus_msl_m`、`position_source`、
  `vertical_source`，并升级相关 schema version；loader 必须拒绝旧 schema、旧 EGM96 tag、
  缺字段和违反 `HAE - MSL = N` invariant 的输入。

- [ ] **6. 将 fitted-ADSB 拟合移到 HAE，再显式进入 local MSL。** 修改
  [`flight_scenarios/fitted_approach.py`](../flight_scenarios/fitted_approach.py) 和
  [`flight_scenarios/build.py`](../flight_scenarios/build.py)：先用 arrival HAE waypoints 与
  `runway_target.elevation_hae_m` 拟合 crossing；再用该 target 的固定
  `hae_minus_msl_m` 转换 crossing、initial state 和 modeling waypoints。一个可拟合但不满足
  evaluation gate 的航迹仍必须能产生 fitted target。

- [ ] **7. 统一所有 modeling consumers 的固定 offset。** 修改
  [`trajectory_data_process/harvest/observed.py`](../trajectory_data_process/harvest/observed.py)、
  `flight_scenarios` scenario records、`4dTrajectory/ts_transformer/dataset.py`、evaluation
  metadata 及 optimizer/prediction record metadata，使同一 flight 从入口到输出始终携带并
  使用同一个 runway-local `hae_minus_msl_m`；synthetic 路径保持显式独立。

- [ ] **8. 统一输出到 Cesium 的逆变换并隔离 EGM96。** 修改
  [`aeroviz-4d/python/build_scenario_comparison_czml.py`](../aeroviz-4d/python/build_scenario_comparison_czml.py)
  等 record-derived CZML 路径，使用 record 的 `HAE = local MSL + N_runway`；observed
  reference 继续旁路转换。生产代码不再导入 pyproj/EGM96。现有 EGM96 能力按评审决定
  保留在 diagnostic/research-only 模块，不得被生产 loader 隐式调用。

- [ ] **9. 执行分层测试与契约审计。** 先跑 parser/airport/datum/fit/arrival 单元测试，再跑
  `flight_scenarios`、evaluation、TS、CZML 和 scenario-pipeline 集成测试；用 `rg` 确认生产
  路径没有 `_to_msl_egm96`、`pyproj` 或 whole-track anchor fallback，且 assignment 没有
  引入 evaluation gate。

- [ ] **10. 全量重采集并生成验证数据。** 先用原实验相同时间窗/数量完整 re-harvest
  KMSY，不使用 `--evaluate-only` 复用旧 tracks；重建 arrivals、observed evaluation、
  fitted-ADSB scenarios、optimizer/TS 所需派生结果和 comparison CZML，逐项验证 UAL1850。
  KMSY canary 通过后，再按评审决定重建其余四机场。

## 6. 测试矩阵

| 层 | 核心用例 | 预期结果 |
|---|---|---|
| CIFP Path Point parser | primary + continuation，含 KMSY 负 HAE | 正确得到同点 HAE/MSL 与 `N=-26.1 m` |
| CIFP Runway parser | 无 LPV Path Point 的 runway | 仍得到 assignment 所需 position、HAE、MSL |
| variant consistency | 同 runway 多个 LPV records 一致/冲突 | 一致时合并；冲突时明确失败 |
| airport source selection | Path Point 完整、仅 Runway 完整、CIFP 不完整 | 按优先级选择；不完整时无 config/EGM96 猜测 |
| datum math | 正/负 `N`、空 track、round trip | `MSL=HAE-N`、`HAE=MSL+N`，误差仅浮点级 |
| final approach | 几何可拟合但 evaluation 不合格 | fit/target 仍生成；evaluation 单独判不合格 |
| landing identity | 较早飞越比最后进近更靠近 threshold | anchor 来自已拟合 final pass，crop 保留最后进近 |
| old artifact rejection | 旧 schema、旧 EGM96 tag、无 `landing_sample_index` | 失败并要求完整 re-harvest/regenerate |
| cross-consumer contract | scenario、TS、evaluation、CZML 使用同一 `N_runway` | 无重复/遗漏转换，无约 26–33 m 跳变 |
| KMSY integration | UAL1850 / RW29 | fitted-ADSB scenario 成功生成且使用正确 final pass |

实施后的测试命令必须先使用仓库环境解析脚本激活 thesis environment：

```bash
source scripts/activate_aeroviz_env.sh
python -m pytest \
  final_approach/tests \
  trajectory_data_process/harvest/tests \
  flight_scenarios/tests \
  evaluation/tests
python -m pytest 4dTrajectory/ts_transformer/tests
python -m pytest aeroviz-4d/python/tests
```

实际实施时先运行与改动直接相关的测试文件，再运行上述较宽测试；若暴露无关失败，按
repository change-scope 规则只报告，不顺手修改。

## 7. 数据重生成顺序

```text
1. CIFP parser/airport tests pass
2. schema + fixed-offset consumer tests pass
3. KMSY full harvest（重新读取 authoritative OpenSky history + CIFP）
4. tracks/manifest.json + track records
5. arrivals/manifest.json + arrival records
6. observed evaluation/CZML
7. fitted-ADSB scenarios
8. optimization / TS derivative outputs（仅本次验证需要的范围）
9. comparison CZML
10. UAL1850 与 batch-level invariants 审计
11. 再决定是否全量重建 KRDU、KSJC、KSMF、KSTL
```

不得从旧 KMSY tracks 运行 `--evaluate-only`，因为旧 track 的 landing identity、crop 和
vertical provenance 本身就是本次要淘汰的错误数据。

## 8. 验收标准

- [ ] KMSY 四条跑道从 CIFP 解析出的 HAE/MSL/N 与调查表一致；
- [ ] `load_airport()` 不再调用 EGM96，不再用 OurAirports MSL 重建 CIFP HAE；
- [ ] assignment 与 fitted-ADSB 的 `TrackPoint.alt_m`/`RunwayFrame.elevation_m` 都是 HAE；
- [ ] runway assignment 仍是全候选跑道的 relative arg-min，没有 evaluation-quality 预筛；
- [ ] evaluation 仍独立报告 `established`/gate verdict，包括质量差但可拟合的航迹；
- [ ] UAL1850 的 landing identity 与 crop 来自最后 inbound pass，fitted-ADSB scenario 成功；
- [ ] `arrivals._anchor_index()` 不再兼容缺少新 index 的旧 tracks；
- [ ] 每个 model-ready `runway_target` 同时携带 HAE、MSL、`N` 和来源；
- [ ] scenario、TS、evaluation、optimizer/prediction 输出与 comparison CZML 使用同一个
  per-runway offset，HAE↔local-MSL round trip 无 26–33 m 级跳变；
- [ ] 生产管线在没有 EGM96 grid、没有 PROJ network 的环境中仍能完成当前 30 km 工作流；
- [ ] 旧 `_to_msl_egm96` artifacts、旧 schema 和缺 provenance 输入均明确失败；
- [ ] 相关单元/集成测试通过，`git diff --check` 通过，文档同步更新。

## 9. 风险与防护

- **Continuation 关联错误：** fixed-column 数据可产生“看似合理”的数字；用 KMSY 四跑道
  一致 `N=-26.1 m`、record grouping 和 range checks 三重校验。
- **不同物理点的 HAE/MSL 被拼接：** Path Point 与 Runway record 只能在同点一致性通过时
  组合；否则整组 fallback 或失败，不单独借一个高程字段。
- **符号错误：** 所有转换只使用命名明确的 `hae_minus_msl_m`，以 KMSY 负 `N` 和 round-trip
  测试固定公式方向。
- **只改一个 consumer：** `N_runway` 必须进入 scenario source metadata，并由 TS、evaluation、
  optimizer/prediction/CZML 读取同一字段，不在下游重新查询机场或 geoid。
- **误把 coverage 当质量：** `fit is None` 只描述样本/航段不可拟合；evaluation verdict 不得
  反向影响 assignment、arrival roster 或 fitted target 生成。
- **旧数据污染：** schema/tag/index 缺失直接失败，KMSY 从 harvest 起完整重建。
- **未来精度需求：** EGM96 可保留为 diagnostic 或未来 gradient model，但不能重新成为
  当前生产路径的隐式依赖。未来若启用空间梯度，应使用
  `N_CIFP(LTP) + [N_model(p) - N_model(LTP)]`，保证 LTP 处仍锚定 CIFP。

## 10. Review questions

1. **Path Point continuation 缺失时的规则：**是否确认仅当 CIFP Runway record 的位置与 HAE
   在发布精度内证明为同一点时，才允许用其 MSL 补齐 Path Point；否则整组使用 Runway
   record 或直接失败？**建议：确认。**
2. **EGM96 代码的归宿：**是完全删除 production API，还是保留 diagnostic/research-only
   工具？**建议：保留诊断能力，但任何 production module 不得导入或自动调用。**
3. **数据 rollout：**实现后先完整 re-harvest KMSY 作为 canary，还是直接重建五机场？
   **建议：先 KMSY，UAL1850 和 batch invariants 通过后再重建其余四机场。**

评审确认这三项以及第 4 节字段命名后，再进入实施阶段。
