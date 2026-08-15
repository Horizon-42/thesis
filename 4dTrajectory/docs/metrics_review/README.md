# 轨迹预测指标审核：本地参考资料索引

审核日期：2026-08-15。

本目录保存本次指标审核实际阅读全文或关键章节后的本地副本。版本信息以发布机构的
正式页面和 PDF 的 document-control/edition-history 页为准，不以搜索结果摘要为准。

| 文件 | 发布机构与当前版本核对 | 本次精读范围 | SHA-256 |
|---|---|---|---|
| [EUROCONTROL-GUID-199-FDPS-Ed1.0-2024.pdf](EUROCONTROL-GUID-199-FDPS-Ed1.0-2024.pdf) | EUROCONTROL，GUID-199，Edition 1.0，validity date 2024-07-03；[当前官方发布页](https://www.eurocontrol.int/publication/eurocontrol-guidance-material-flight-data-processing-system-fdps)列为 2024-07-25 发布 | §5.13，PDF pp. 50–52；Annex A §§3.2–3.4，PDF pp. 108–114 | `4eea2e47ca1edd6cc5d91532d206a378ddfc69654de23028fb0603deaf26e28a` |
| [EUROCONTROL-SPEC-0143-Trajectory-Prediction-Ed2.0.pdf](EUROCONTROL-SPEC-0143-Trajectory-Prediction-Ed2.0.pdf) | EUROCONTROL，SPEC-0143，Edition 2.0；[当前官方发布页](https://www.eurocontrol.int/publication/eurocontrol-specification-trajectory-prediction)仍列 2017-03-03、Version 2.0 | §4 与对应 measurement annex；用于核对 2024 guidance 的来源与延续关系 | `134ac864f8b7ec4efbbc317d625f0e767dd5f024702dfa9ebd96ead81f2caf44` |
| [NASA-20110003565-Validation-Methodology-for-Aircraft-Trajectory-Predictors.pdf](NASA-20110003565-Validation-Methodology-for-Aircraft-Trajectory-Predictors.pdf) | NASA NTRS document 20110003565，2010 AIAA conference paper；[NTRS 记录](https://ntrs.nasa.gov/citations/20110003565) | §II.A–C，PDF pp. 2–4；§VI.B，PDF pp. 17–19 | `fc2a2c323ac1465e1c897f83896ffcf72f1dbe90b607510ab0ef846618b72512` |
| [Kendall-Gal-Cipolla-2018-Multitask-Uncertainty-Loss.pdf](Kendall-Gal-Cipolla-2018-Multitask-Uncertainty-Loss.pdf) | CVPR 2018 原论文；[CVF 论文页](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html) | §3，尤其 Eq. (1)–(7)，printed pp. 7484–7485 | `059f6c07acfe8b187ee1f74cdaf6a28fefbf78c3f8508fdf4eeda94d5ebdf92f` |
| [Ivanovic-Pavone-2019-Trajectron.pdf](Ivanovic-Pavone-2019-Trajectron.pdf) | ICCV 2019 原论文；[CVF 论文页](https://openaccess.thecvf.com/content_ICCV_2019/html/Ivanovic_The_Trajectron_Probabilistic_Multi-Agent_Trajectory_Modeling_With_Dynamic_Spatiotemporal_Graphs_ICCV_2019_paper.html) | §4 的 ADE/FDE、best-of-N 与 distributional NLL 讨论，printed pp. 2379–2381 | `75e1d1982157934f193a49fec467ae0f84b1f89b3f246b812df4e8ed60f4f1f9` |

## 版本关系

2024 FDPS Guidance 比 2017 Trajectory Prediction Specification 更新，并说明其内容是对
旧 specification 的重构，同时加入 FF-ICE R1、EPP 与 Network 4DT CONOPS 的成熟要求。
因此，本次设计以 2024 §5.13/Annex A 为 EUROCONTROL 侧的主要依据；2017 文件只保留为
可追溯的前身。两份文件都面向 ATM/FDPS 系统，不应被误写成机器学习模型的通用排行榜，
也不应把其中中高空、按预测分钟给出的数值直接移植为机场最终进近的通过门限。

NASA、CVPR、ICCV 文献是固定版本的研究论文，不存在“用后来的版次替换原论文”的问题。
其作用分别是支撑航空轨迹预测验证方法、多任务 loss 权重风险，以及确定性/概率性预测
需要不同正式指标。
