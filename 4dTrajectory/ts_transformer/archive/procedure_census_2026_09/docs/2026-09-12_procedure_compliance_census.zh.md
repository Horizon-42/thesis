# 观测航迹对编码 RNAV(GPS) 进近程序的完整符合率普查（2026-09-12）

回答"完全符合进近程序的航迹有哪些"。脚本 `docs/measure_procedure_compliance.py`（只读，全部 42,650 条 v5 到达，
五个机场，存储航迹全段，即 30 km 裁剪范围到落地），通过全部判据的航班清单在
`2026-09-12_procedure_compliance_tier_C_flights.json`（按 `机场 跑道` 分组，每条为 `[flight_key, 过渡段]`）。

## 判据（嵌套三层，镜像优化器对程序的读法：`approach_constraints` / `procedure_segments`）

- **A 五边段**：从 FAF 上游起一直留在 k=0.5 的 LPV 角走廊内直到入口前 300 m（`d_join ≥ d_FAF`），
  且 FAF 以内每个采样都在下滑道窗口 [−60, +120] m 内（相对 TDZE + TCH + d·tan GPA）。
- **B = A + IF**：到 IF 的最近距离 ≤ 926 m（k·RNP = 0.5 × 1 NM，优化器的 FAF 前定位盘），经过 IF 时的高度满足
  其编码约束（容差 30 m），IF 到 FAF 之间每个采样不低于 FAF 的编码下限（优化器的 leg 下限）。
- **C = B + 一条编码过渡段**：至少对一条过渡段，落在该航班存储范围内（定位点到入口的径向距离 ≤ 第一个存储采样的）的
  每个过渡定位点都按顺序在 926 m 内经过、经过时高度满足编码约束、到达每个定位点前的 leg 下限成立、最后一个过渡定位点到 IF
  之间 IF 的下限成立。没有任何过渡定位点在存储范围内的航班记为 **C 不可判定**（其过渡段在 30 km 存储范围之外），不算失败。

**C 是"存储范围内的完整程序"，不是"从 IAF 起的完整程序"**：所有离轴 IAF（OTTOS 38 km、BUTTS 34–36 km、ESELL 41–42 km、
BORED 41 km 等）都在 30 km 裁剪范围外，用现有存储无法判定。要判定 IAF 需要把 harvest 的裁剪半径放大到 45 km 以上重新下载。

高度：存储的 `opensky_history_geoaltitude_m` 是 HAE，按跑道目标的 `hae_minus_msl_m` 转成 MSL 后与编码高度（MSL）比较；这是
几何高度，而下限是按气压高度飞的，30 m 容差不覆盖温度偏差（5 月偏暖时几何高度高于指示高度，所以"低于下限"的读数不是温度造成的）。
高度离群点按 `harvest.altitude_filter` 在读取时修复。

## 结果

| 机场 跑道 | n | A 五边 | B +IF | C / 可判定 | C 占可判定 | IF（沿航道距离，下限） | 存储范围内的过渡定位点 |
|---|---:|---:|---:|---:|---:|---|---|
| KRDU 05L | 3,076 | 94.1% | 55.3% | 408 / 3,046 | 13.4% | SCHOO 15.9 km, ≥ 914 m | ACHWDR: CHWDR 27 km/BOULE 21 km; AOTTOS: CHWDR 27 km/BOULE 21 km |
| KRDU 05R | 1,491 | 83.4% | 47.9% | 193 / 1,481 | 13.0% | PECIT 15.8 km, ≥ 914 m | ABEICH: BEICH 26 km/RATTY 21 km; AOTTOS: BEICH 26 km/RATTY 21 km |
| KRDU 23L | 3,230 | 81.7% | 50.2% | 525 / 3,226 | 16.3% | DOCAT 16.0 km, ≥ 914 m | ABUTTS: DINTS 20 km; ADUWON: DUWON 26 km/DINTS 20 km; AESELL: DUWON 26 km/DINTS 20 km |
| KRDU 23R | 6,638 | 94.6% | 65.6% | 1,570 / 6,626 | 23.7% | PRSTN 15.7 km, ≥ 914 m | ABUTTS: WARMS 29 km/DABKE 21 km; AESELL: MARBY 26 km/DABKE 21 km; AMARBY: MARBY 26 km/DABKE 21 km |
| KSJC 12R | 565 | 35.8% | 28.1% | 0 / 0 | — | ARTAQ 13.6 km, ≥ 549 m | 无（过渡定位点全部在 30 km 存储范围外） |
| KSJC 30L | 9,560 | 78.9% | 76.7% | 6,781 / 9,513 | 71.3% | ZULUP 22.2 km, ≥ 853 m | ABORED: SWIGS 30 km/KLIDE 28 km; AKLIDE: KLIDE 28 km |
| KSJC 30R | 943 | 67.2% | 65.5% | 571 / 936 | 61.0% | KIRVE 15.3 km, ≥ 792 m | ABORED: SWIGS 30 km/KLIDE 28 km; AKLIDE: KLIDE 28 km |
| KSTL 11 | 645 | 94.0% | 11.9% | 0 / 396 | 0.0% | HILPI 21.6 km, ≥ 1067 m | ASISDE: SISDE 30 km |
| KSTL 12L | 3,943 | 92.5% | 8.0% | 0 / 0 | — | EUBIE 24.8 km, ≥ 1219 m | 无（过渡定位点全部在 30 km 存储范围外） |
| KSTL 12R | 262 | 82.1% | 22.5% | 0 / 0 | — | NAIRN 19.6 km, ≥ 1097 m | 无（过渡定位点全部在 30 km 存储范围外） |
| KSTL 29 | 96 | 95.8% | 63.5% | 0 / 0 | — | JIGIM 16.4 km, ≥ 1067 m | 无（过渡定位点全部在 30 km 存储范围外） |
| KSTL 30L | 121 | 81.8% | 43.0% | 0 / 0 | — | ZETGO 17.6 km, ≥ 1067 m | 无（过渡定位点全部在 30 km 存储范围外） |
| KSTL 30R | 3,676 | 93.8% | 48.3% | 0 / 1,179 | 0.0% | EXALE 16.1 km, ≥ 1067 m | ATYRSH: TYRSH 30 km |
| KSMF 17L | 1,142 | 73.6% | 58.5% | 55 / 1,137 | 4.8% | DIRBE 14.3 km, ≥ 579 m | ATENCO: TENCO 26 km |
| KSMF 17R | 2,484 | 59.1% | 48.0% | 40 / 2,469 | 1.6% | FAPIN 13.8 km, ≥ 549 m | ATENCO: TENCO 28 km |
| KSMF 35L | 593 | 71.8% | 36.1% | 0 / 0 | — | ELMAC 21.2 km, ≥ 914 m | 无（过渡定位点全部在 30 km 存储范围外） |
| KMSY 02 | 284 | 58.5% | 5.6% | 0 / 0 | — | ROYUL 23.6 km, ≥ 610 m | 无（过渡定位点全部在 30 km 存储范围外） |
| KMSY 11 | 2,614 | 80.4% | 40.1% | 0 / 0 | — | FIGUR 24.5 km, ≥ 610 m | 无（过渡定位点全部在 30 km 存储范围外） |
| KMSY 29 | 1,247 | 66.6% | 29.1% | 0 / 0 | — | HELMT 20.6 km, ≥ 914 m | 无（过渡定位点全部在 30 km 存储范围外） |
| **合计（表内跑道）** | **42,610** | **83.0%** | **52.4%** | **10,143 / 30,009** | **33.8%** | | |

跑道 n < 50 的（KSJC 12L、KSTL 06、KMSY 20）略去，脚本输出里有。

读法：
- 完整程序（C）集中在 **KSJC 30L/30R**（71 % / 61 % 的可判定航班：KLIDE 28 km → ZULUP/KIRVE → 五边，即数据里"进 25 km 环时已在五边上"
  的那批）和 **KRDU**（13–24 %：沿轴线的 BOULE/RATTY/DINTS/DABKE 21 km 过渡）。KSMF 只有 2–5 %（TENCO）。
- **KSTL、KMSY 一条都没有**，但不是不符合，是不可判定：它们的过渡定位点全部在 30 km 之外（SISDE/TYRSH 正好在 30 km 裁剪边上，
  可判定的 396 / 1,179 架无一在 926 m 内经过，属裁剪边缘效应）。
- B（从 IF 起的程序，也就是优化器实际施加的横向行）全机队 52 %，跑道间 8 %（KSTL 12L：IF EUBIE 在 24.8 km、下限 4000 ft，
  引导航班在其内侧切入、高度更低）到 77 %（KSJC 30L）。
- A（五边）83 %，与 2026-09-04 的测量一致。

## 失败原因（A 航班里 B 失败的原因；B 航班里 C 失败的原因，取最接近通过的那条过渡段）

| 机场 跑道 | 错过 IF 盘 | IF 高度不符 | IF→FAF 下限 | 过渡：错过定位盘 | 过渡：高度不符 | 过渡：leg 下限 |
|---|---:|---:|---:|---:|---:|---:|
| KRDU 05L | 645 | 996 | 340 | 914 | 369 | 1 |
| KRDU 05R | 329 | 427 | 134 | 364 | 155 | 0 |
| KRDU 23L | 601 | 754 | 330 | 584 | 511 | 1 |
| KRDU 23R | 1,139 | 1,591 | 630 | 1,619 | 1,163 | 4 |
| KSJC 12R | 38 | 0 | 6 | 0 | 0 | 0 |
| KSJC 30L | 197 | 3 | 4 | 539 | 5 | 0 |
| KSJC 30R | 8 | 7 | 5 | 46 | 0 | 0 |
| KSTL 11 | 523 | 286 | 8 | 10 | 0 | 0 |
| KSTL 12L | 3,169 | 2,695 | 21 | 0 | 0 | 0 |
| KSTL 12R | 117 | 127 | 2 | 0 | 0 | 0 |
| KSTL 29 | 15 | 25 | 1 | 0 | 0 | 0 |
| KSTL 30L | 41 | 28 | 0 | 0 | 0 | 0 |
| KSTL 30R | 1,195 | 1,424 | 10 | 188 | 0 | 0 |
| KSMF 17L | 151 | 1 | 26 | 305 | 306 | 0 |
| KSMF 17R | 265 | 1 | 15 | 873 | 275 | 0 |
| KSMF 35L | 198 | 166 | 9 | 0 | 0 | 0 |
| KMSY 02 | 150 | 0 | 0 | 0 | 0 | 0 |
| KMSY 11 | 1,046 | 0 | 28 | 0 | 0 | 0 |
| KMSY 29 | 451 | 339 | 14 | 0 | 0 | 0 |

两类原因各占一半：**横向**（错过 926 m 盘：被引导在定位点内侧切入）和**垂直**（经过定位点时低于编码下限：引导高度按最低引导高度
指定，低于程序台阶；KRDU 的 A 航班在 IF 处相对 3000 ft 下限的高度余量 p10 为 −250 m）。leg 下限几乎从不单独失败。

## 与 2026-09-04 测量的关系

`measure_procedure_adherence.py` 报告的"0.0 % 经过离轴 IAF"是切片产物：它只读 25 km 到达切片，而所有离轴 IAF 在 29–42 km 外，
几何上不可能命中；其对照表把优化器的 `_prefaf_fix_rows`（轴上的 IF 盘）错对应到"经过离轴 IAF"。本普查按优化器实际的行逐项测：
优化器在 FAF 上游只施加 IF 盘、汇入窗和 leg 高度下限，入口 IAF 横向自由。设计文档 §1–§2 与 `OPEN_ITEMS.md` 里的那几行待改写。

## 复现

```bash
for A in KRDU KSJC KSTL KSMF KMSY; do
  conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/measure_procedure_compliance.py $A --stride 1 --out /tmp/census_$A.json
done
```
每机场 2–8 s。输出 JSON 含每航班的分层判定、`c_reasons`、`b_checks`、IF 经过高度与最近距离、汇入处航迹角。
