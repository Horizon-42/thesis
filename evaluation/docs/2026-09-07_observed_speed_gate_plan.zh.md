# 观测速度门：实测载荷因子 + METAR 顶风修正（2026-09-07）

分支 `dev-observed-load-factor-metar`（worktree `.claude/worktrees/lnav-vnav`，基线
`e836575` on `dev-leg-ctrl`）。承接 `2026-09-07_review_fix_plan.zh.md`：v7 把速度门下边锚在穿越
载荷因子上，但观测记录没有控制量，被声明为 1 g，于是新门在 ADS-B 数据上是空操作。ADS-B 评估
是方法的 ground truth，所以这一阶段让观测记录（1）实测 n，（2）用 METAR 顶风把地速代理修成
空速估计。每项：代码 → 测试 → opus 评审 → 提交；最后一起 schema bump。

## 1. 状态表

| # | 项 | 状态 | 提交 | 备注 |
|---|---|---|---|---|
| O1 | 观测记录的窗口反演 n：穿越前最后 20 s 实测样本上线性拟合 ψ̇、γ̇，n = hypot(cos γ + Vγ̇/g, V cos γ ψ̇/g)；来源 `adsb_kinematics`，窗口事实写在行上 | 已评审、提交 `8bf9f5c` | `8bf9f5c` | 窗口样本不足 4 个 ⇒ 声明 1 g；五机场实测见速度门文档 §3.5 |
| O2 | METAR 顶风：IEM ASOS 归档抓取脚本 + 评估读时按落地时间就近关联，空速估计 = 地速 + 顶风分量，±5 kt 声明不确定度，`speed_marginal` 计数 | 已评审、提交 `8bf9f5c`；v9 去掉了 ±5 kt/`speed_marginal` 层（§6） | `8bf9f5c` | 五机场 2026-04-30 至 07-23 的观测已抓取到 `data/metar/` |
| O3 | 五机场重评估的 n 分布与风修正前后的速度判定对比，写进 `BASELINE_SPEED_GATE_RESULTS.md` | 完成 | | 只写 scratchpad，不覆盖盘上报告 |
| O3 结果 | 风修正让"过慢"簇缩小 3 到 5 倍，剩下 737 家族的"过快"簇（中位 V_est/Vs1g(MLW) 1.37–1.41），A320 家族居中，A21N 仍慢 | 已写入 `BASELINE_SPEED_GATE_RESULTS.md` §9 | | 91–97 % 行有 30 min 内的 METAR |
| O4 | 用公开发布的机型 V_ref 取代失速模型锚（A 方案）：FAA Aircraft Characteristics Database（2024-10 版）每个 ICAO 机型在 MALW 处的 FSB 进近速度（含襟翼构型上下值），按 √(m/MALW) 缩放；计算记录用穿越质量，观测记录（质量未知）用机型公布质量区间 [最小质量, MALW]；来源文件存 `data/reference_speeds/`，出处索引 `docs/reference_speeds/README.md`，机器表 `aircraft/reference_speeds.json` | 已评审（opus）、提交 `53ae8e8`；结果见 §6.5 | `53ae8e8` | 判定保持二值；删除 ±5 kt 不确定度层；B 方案（观测空速标定）只在 FAA 表无该机型时启用，目前机队 40 型全部在表内，未触发 |
| — | schema v7 → v8 → v9（四处镜像），合并到 `dev-leg-ctrl` | v9 已 bump（`53ae8e8`）；分支已吸收 `dev-leg-ctrl`（`ebda491`），合入主树等 campaign 结束 | | 观测与计算判定都会变，按项目规则 bump |

## 2. 依据

- 观测记录的 V/ψ/γ 由 `flight_scenarios.start_state.state_samples_from_track` 在居中窗口上最小
  二乘拟合，不是逐点差分；在 20 s 窗口上再拟合斜率，高度量化 7.62 m 对 76 m 下降量的斜率误差
  约 10 %，折到 n 只有 0.002；ψ̇ 项在 700 m 基线上约 0.005。之前"γ̇ 只有噪声"的说法针对瞬时值，
  不成立于窗口拟合。
- 地球曲率/自转的输运项（`ts_transformer/flyability._transport_rates`）量级 V/R ≈ 1e-5 rad/s，
  折到 n 约 7e-5，忽略并注明。
- 右截尾穿越（KRDU 966/996 条航迹在入口前中位 325 m 结束）的穿越点是直线外推，n 按定义为 1；
  实测窗口给出的是最后观测处的 n，行上写 `ends_m_before_threshold`。
- 风：OpenSky 状态向量无空速、无真航向，做不了风三角。IEM ASOS 归档（`mesonet.agron.iastate.edu
  /cgi-bin/request/asos.py`，`report_type=3,4`，UTC）给出 drct（真北）、sknt、gust。跑道航向
  `AssessmentContext.runway_course_deg` 是真航向。顶风分量 = W·cos(drct − course)。
- 质量（观测窗口锚在机型着陆质量上，A21N 残差）不在本阶段。

## 3. 决策

- 观测 n 的来源标签：`adsb_kinematics`（窗口拟合）或 `assumed_1g`（样本不足）。观测 criterion id
  恢复为 `vref_1p23_vs_at_n_to_vref_1g_plus_20kt_ground_speed_proxy`，因为窗口现在真的锚在 n 上。
- 下边仍用 max(n, 1)，与计算记录同一规则。
- 风关联在评估读取时做（`evaluation/wind.py`），不改 harvest 产物：观测记录不重写，
  arrivals roster 不动。最近一次观测在 ±30 min 内才使用；VRB 或缺测 ⇒ 该行退回地速代理，
  行上与批量计数都写明，不是静默回退。
- 风修正后的判定量 `crossing_airspeed_estimate_ms`，criterion id 加 `_metar_airspeed_estimate`；
  不确定度 ±5 kt 作为声明值，`speed_marginal` 统计距边界 5 kt 内的行。
- METAR 文件放 `data/metar/<ICAO>/asos_<start>_<end>.csv` + 同名 `.provenance.json`（URL、抓取时间、
  行数）；`data/` 已 git-ignore，与 CIFP 同一处理方式。CLI `--metar-root`，默认 `data/metar`，
  目录缺失时观测批次按代理判定并在报告里写明。

## 4. 改动清单

- O1：`aircraft/kinematics.py`（新，`load_factor_from_rates`）；`evaluation/stats.py`
  `linear_slope`；`evaluation/arrival.py` `_observed_load_factor`；`evaluation/speed_gate.py`
  常量与 id；`evaluation/metrics.py` 聚合与 methodology；`evaluation/tests/factories.py`
  多样本观测航迹工厂；`test_speed_gate.py`、`test_arrival.py`；`THRESHOLD_SPEED_GATE.md` §3.5、
  `evaluation/CLAUDE.md`。
- O2：`trajectory_data_process/metar/fetch_iem_asos.py`（新）；`evaluation/wind.py`（新）；
  `evaluation/metrics.py`、`cli.py`、`__main__.py`、`visualize.py`、`run_all_evaluations.py`
  的 `--metar-root`；前端镜像字段；测试与文档。
- 落地时间范围（tracks manifest 的 assigned 行）：见 §5。

## 5. 数据范围

tracks manifest 的 assigned 落地时间与抓取结果（`data/metar/<ICAO>/asos_<start>_<end>.csv`，
前后各留一天）：

| 机场 | assigned | 首次落地 | 末次落地 | ASOS 观测行 |
|---|---|---|---|---|
| KRDU | 16,056 | 2026-05-01 00:02 | 2026-07-21 21:58 | 2,259 |
| KSJC | 11,157 | 2026-05-01 00:01 | 2026-07-22 14:59 | 2,205 |
| KSTL | 8,769 | 2026-05-10 12:06 | 2026-07-22 16:54 | 2,144 |
| KSMF | 4,490 | 2026-05-17 00:05 | 2026-07-22 16:58 | 1,663 |
| KMSY | 4,150 | 2026-05-16 18:05 | 2026-07-21 19:55 | 1,920 |

O1 实测（v8 代码，风修正关闭）：见 `THRESHOLD_SPEED_GATE.md` §3.5 的表。

## 6. O4 设计：公开 V_ref 窗口（2026-09-07 决定）

用户决定（原话要点）：不要画蛇添足；A 方案为主、B 方案兜底；速度限必须是百分百确定的数，不做概率化
的三值判定；用到的 V_ref 必须下载或至少引用源文件，单独放一个文档文件夹，来源可查。

### 6.1 为什么换锚

§8 的失速模型（OpenAP 机翼面积 × 按吨位分桶的 Cl_max）不是任何机型的公布数据：737 桶的 Cl_max 2.7
给出 B738 的 1.23·Vs1g(MLW) = 134 kt，而公布的 V_ref 是 140–147 kt，于是风修正后的观测 737 家族
系统性"过快"。这些飞机全部正常落地，30 % 的失败率是窗口的错。公布的机型进近速度就是这个窗口该
锚的量，且每个数字都能在一跳内核对。

### 6.2 来源（全部已下载，`docs/reference_speeds/README.md` 逐条列 URL、抓取日期、sha256、页码）

- **主来源：FAA Office of Airports, Aircraft Characteristics Database, "Aircraft Characteristics
  (October 2024)"**（xlsx，125,570 B）。按 ICAO 机型给出 `Approach_Speed_knot`（定义：MALW 处进近
  速度，FSB 或厂商文件中各着陆襟翼构型里的最高值）、`Approach_Speed_minimum/maximum_knot`（双构型
  机型的上下值）、`MALW_lb`。机队 40 型全部在表内，含公务机与 C172。
- **厂商佐证：** Airbus AC（A319/A320/A321，3-5-0 "Final Approach Speed"：MLW 处的门槛指示空速，
  A320-200 136 kt @ 66,000 kg 等）；Boeing ACAP（737NG/737 MAX/757/767/777/787：2.1 节 MLW 与
  OEW，不给进近速度）；Embraer E175 APM（Table 2.1：MLW 34,000 kg、BOW 21,500 kg）；Bombardier
  CRJ900 APM（00-02-01：MLW 33,340 kg、最小飞行质量 20,412 kg；00-03-03 图 3/4：V_ref 随质量曲线）。
- **第三佐证：** Eurocontrol Aircraft Performance Database 各机型页的 `Landing Vat (IAS)`。
- 最小质量（观测窗口下边）：厂商公布的最小飞行质量 > 厂商 OEW/BOW > 厂商公开规格页 > OpenAP 2.4
  `aircraft/<type>.yml` 的 OEW（注明）；都没有则该机型的观测速度判定为 indeterminate 并写明原因。

### 6.3 窗口（确定性，每个量在行上）

    V_ref,lo(m) = V_min · √(m / MALW)        # FAA minimum（最大襟翼）或主值
    V_ref,hi(m) = V_max · √(m / MALW)        # FAA maximum（较小襟翼）或主值
    计算记录（穿越质量 m 已知）：  [V_ref,lo(m) · √max(n, 1),        V_ref,hi(m) + 20 kt]
    观测记录（质量未知）：        [V_ref,lo(m_min) · √max(n, 1),    V_ref,hi(MALW) + 20 kt]

- 下边随载荷因子 √n 抬升（O1 的实测 n 保留），上边是 1 g 的能量准则（不变）。
- 观测窗口宽（B738 约 116–167 kt CAS）是质量未知这一事实的诚实表述：任何正确落地的航班都在其中，
  越界即真异常或数据错误，逐条可解释；模型记录质量已知，窗口 20 kt 加襟翼差。
- 判定二值 pass/fail（indeterminate 只用于"无法判定"：无机型、无穿越速度、无最小质量）。删除
  `speed_uncertainty_ms`、`speed_marginal`、`speed_uncertainty_unknown` 与 ±5 kt 声明；保留
  `speed_margin_ms`（到最近边界的带符号距离，一个数）。风修正保留：它是确定性修正，不是概率层。
- 行上写 `reference_typecode`、`reference_sources`（三项事实的来源 id）、`vref_low_ms`、`vref_high_ms`、`mass_basis`
  （`crossing_mass` / `type_mass_range`）、`speed_lower_ms`、`speed_upper_ms`。

### 6.4 改动清单

- `aircraft/reference_speeds.json`（机器表，每个字段带 source id）+ `aircraft/reference_speeds.py`
  （加载与 √(m/MALW) 缩放，stdlib）；`docs/reference_speeds/`（README、摘录、fetch 脚本）。
- `evaluation/speed_gate.py` 重写：锚换成公开 V_ref，criterion id 改名，`SpeedGateBounds` 字段改名。
- 记录机型键：观测 `source.aircraft_type`（harvest/observed.py 写的 identity typecode）、计算
  `source.dynamics_typecode`（flight_scenarios.build 写的模型所飞机型）。`source.landing_aero`
  评估不再读：`records.py` 去掉边界校验，`harvest/observed.py` 不再写，
  `4dTrajectory/optimization/backfill_landing_aero.py` 及其测试删除（盘上 70,267 条优化记录本来就
  带 `dynamics_typecode`，重生成报告即可判定）。`flight_scenarios.build` 的块保留（它记录的是模型
  自身速度下限用的 Cl_max，属动力学出处）。
- `metrics.py`/`__main__.py`/`visualize.py`/前端 `evaluationReport.ts` + `EvaluationReportWindow.tsx`
  /ts `lateral_eligibility.py`：schema v9，去掉不确定度字段，改 bounds 字段。
- 测试：`test_speed_gate.py` 重写（工厂记录改带 `aircraft_type`/`dynamics_typecode`，窗口从
  `reference_speeds` 推导）；`test_wind.py`、`test_observed_load_factor.py`、`test_records.py` 调整；
  新 `aircraft/tests/test_reference_speeds.py`（表完整性：每型有 source、MALW>min_mass、速度单调）。
- 文档：`THRESHOLD_SPEED_GATE.md` §2/§3.1/§4/§9 重写，`BASELINE_SPEED_GATE_RESULTS.md` §10（五机场
  重评估：预期通过率接近 100 %，残差逐条解释），`evaluation/CLAUDE.md`、`README.md`、根 `CLAUDE.md`
  open item、`trajectory_data_process/CLAUDE.md`、`flight_scenarios/CLAUDE.md`、CHANGELOG、followups #19。

### 6.5 结果（2026-09-07，`observed_v9_2026-09-07/`）

| 机场 | 判定行 | 通过 | v9 通过率 | v8 通过率 |
|---|---|---|---|---|
| KRDU | 10,030 | 9,567 | 95.4 % | 71.1 % |
| KSJC | 7,308 | 7,212 | 98.7 % | 80.8 % |
| KSTL | 7,151 | 6,946 | 97.1 % | 71.1 % |
| KSMF | 3,814 | 3,708 | 97.2 % | 75.9 % |
| KMSY | 3,484 | 3,355 | 96.3 % | 70.3 % |
| 机队 | 31,787 | 30,788 | **96.9 %** | 73.8 % |

残余 999 条失败：992 快、7 慢；超出上边中位 2.2 kt（p90 7.2）；747 条的原始地速在窗内，是塔台顶风
修正把它推出去的，集中在大风日（KSMF 05-18 顶风 21.5 kt：124 条中 23 条失败）；机型簇 E75L（302）、
B737（221）、A319、E170/E190 都是 FAA 表只给单一（最大襟翼）速度的机型，减小襟翼构型的 V_ref 更高——
Eurocontrol Vat 高 6–7 kt——按 A 方案规则保留 FAA 值，列为唯一待补来源项（followups #21）。737 家族
的"过快"簇与 A21N 的"过慢"残差都消失。详见 `BASELINE_SPEED_GATE_RESULTS.md` §10。
