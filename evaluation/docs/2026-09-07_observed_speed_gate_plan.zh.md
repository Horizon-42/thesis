# 观测速度门：实测载荷因子 + METAR 顶风修正（2026-09-07）

分支 `dev-observed-load-factor-metar`（worktree `.claude/worktrees/lnav-vnav`，基线
`e836575` on `dev-leg-ctrl`）。承接 `2026-09-07_review_fix_plan.zh.md`：v7 把速度门下边锚在穿越
载荷因子上，但观测记录没有控制量，被声明为 1 g，于是新门在 ADS-B 数据上是空操作。ADS-B 评估
是方法的 ground truth，所以这一阶段让观测记录（1）实测 n，（2）用 METAR 顶风把地速代理修成
空速估计。每项：代码 → 测试 → opus 评审 → 提交；最后一起 schema bump。

## 1. 状态表

| # | 项 | 状态 | 提交 | 备注 |
|---|---|---|---|---|
| O1 | 观测记录的窗口反演 n：穿越前最后 20 s 实测样本上线性拟合 ψ̇、γ̇，n = hypot(cos γ + Vγ̇/g, V cos γ ψ̇/g)；来源 `adsb_kinematics`，窗口事实写在行上 | 代码+测试+文档完成，待评审 | | 窗口样本不足 4 个 ⇒ 声明 1 g；五机场实测见速度门文档 §3.5 |
| O2 | METAR 顶风：IEM ASOS 归档抓取脚本 + 评估读时按落地时间就近关联，空速估计 = 地速 + 顶风分量，±5 kt 声明不确定度，`speed_marginal` 计数 | 代码+测试+文档完成，待评审 | | 五机场 2026-04-30 至 07-23 的观测已抓取到 `data/metar/` |
| O3 | 五机场重评估的 n 分布与风修正前后的速度判定对比，写进 `BASELINE_SPEED_GATE_RESULTS.md` | 完成 | | 只写 scratchpad，不覆盖盘上报告 |
| O3 结果 | 风修正让"过慢"簇缩小 3 到 5 倍，剩下 737 家族的"过快"簇（中位 V_est/Vs1g(MLW) 1.37–1.41），A320 家族居中，A21N 仍慢 | 已写入 `BASELINE_SPEED_GATE_RESULTS.md` §9 | | 91–97 % 行有 30 min 内的 METAR |
| O4 | 在修正后的空速上重新推导各机型的窗口锚（§8 的 A320 Cl_max 3.0 是在含风地速上标定的；737 桶 2.7 现在是反向离群），并处理 A21N 质量 | 待做，需要机型 V_ref 表或运行着陆质量数据 | | 在此之前不要把观测速度失败率当飞行行为引用 |
| — | schema v7 → v8（四处镜像），合并到 `dev-leg-ctrl` | v8 已就位，合并待做 | | 观测判定会变，按项目规则 bump |

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
