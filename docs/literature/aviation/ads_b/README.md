# ADS-B 介绍资料

日期：2026-10-16
问题：本项目（数据来自 OpenSky 的 ADS-B 历史库）引用 ADS-B 时，最适合引哪份介绍文档？

## 文件

| 文件 | 来源 | 页数 | 用途 |
|---|---|---|---|
| `papers/Sun2021_1090_Megahertz_Riddle_2nd_ed.pdf` | J. Sun，《The 1090 Megahertz Riddle》第 2 版，TU Delft OPEN，2021，CC BY-NC-SA 4.0，DOI 10.34641/mg.11 | 160 | ADS-B 与 Mode S 的原理和消息格式（第 7.2 节：GNSS 高度与气压高度之差） |
| `papers/Schafer2014_Bringing_Up_OpenSky_IPSN.pdf` | Schäfer 等，IPSN 2014 | 12 | 本项目数据源 OpenSky 网络的描述 |

## 官方文件（`official/`）

| 文件 | 来源 | 页数 | 用途 |
|---|---|---|---|
| `official/FAA_AC_90-114B_ADS-B_Operations.pdf` | FAA 咨询通告 AC 90-114B，2019-12-30 | 59 | **介绍 ADS-B 本身的首选**：第 2 章 "Overview and System Description"（系统架构、1090ES 与 UAT 两个频率、ADS-B Out/In、ADS-R、TIS-B、FIS-B）；第 3 章讲美国空域要求（14 CFR 91.225/91.227） |
| `official/FAA_AC_20-165B_ADS-B_Out_Airworthiness.pdf` | FAA AC 20-165B，2015-12-07 | 113 | ADS-B Out 设备的适航批准；只在需要引用设备性能要求时用 |
| `official/ICAO_APAC_2024_SP02_Annexes_and_Documents.pdf` | ICAO 亚太办 2024 ADS-B 研讨会，SP02 | 69 | ICAO 的 ADS-B 文件体系综述（Annex 10 Vol IV、Doc 9871、Doc 9924 等）；是简报，不是标准本身 |

## 没有下载到的

- ICAO Doc 9924《航空监视手册》是付费文件，ICAO 商店有售，仓库里没有。
- ICAO NACC 的 ADS-B Out 概述简报（`P01-OverviewADSBOut-ENG.pdf`）和 `ICAO provisions related to ADS-B`：2026-10-16 主机 `www2023.icao.int` 无法解析，备用地址 404。
- ICAO Annex 10 Vol IV、Doc 9871、RTCA DO-260B、EUROCAE ED-102A 都是付费标准。

## 重新下载

`./download.sh`（PDF 不入 git）。
