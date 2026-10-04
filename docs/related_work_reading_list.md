# Related Work 论文清单

依据：`docs/Final Proposal.md`（物理约束的生成式 4D 进近航迹预测：控制参数生成 → 可微运动学 → 可微惩罚约束 → 多机协调；对比三类基线；三个评价维度）。
范围：只选仓库 `docs/literature/` 与 `docs/models/` 中**已有**的论文，共 33 篇。路径相对于 `docs/literature/`（PDF 不入 git，缺失时跑对应文件夹的 `download.sh`）。
★ = 必引；其余为择引。

**引用来源说明**：作者、标题、会议/期刊、DOI/arXiv 均摘自各文件夹 `README.md` 里已核对的条目。标注「†」的条目，仓库 README 没有完整引用，其作者/标题来自文件名或我的记忆，写进论文前请打开 PDF 核对。

## 组织结构

```
2.1 终端空域轨迹预测的演进      物理/优化 → 确定性序列模型 → 概率/生成模型 → 上下文条件（同时给出三类基线）
2.2 物理信息与可微运动学        对应核心组件 1：生成控制参数，再由运动学积分出轨迹
2.3 约束的可微表达              对应核心组件 2：物理边界与运行规则作为可微惩罚
2.4 多机交互与间隔              对应核心组件 3：多机协调与间隔评价
2.5 序列生成与闭环训练          生成式建模的方法背景（可选，一小段）
2.6 数据集
```

每节末尾的"空白"一句可直接改写成本文的定位句。

---

## 2.1 终端空域轨迹预测的演进

**(a) 物理 / 优化基线**（基线类别 1）

| # | 完整引用 | 路径 | 用途 |
|---|---|---|---|
| ★ | J. Sun, J. M. Hoekstra, J. Ellerbroek, "OpenAP: An Open-Source Aircraft Performance Model for Air Transportation Studies and Simulations," *Aerospace* 7(8):104, 2020. DOI 10.3390/aerospace7080104 | `control_normalization/papers/OpenAP_Sun2020_…pdf` | 开源性能模型；本项目运动学与推力/阻力的来源 |
| ★ | A. Nuic, D. Poles, V. Mouillet, "BADA: An Advanced Aircraft Performance Model for Present and Future ATM Systems," *Int. J. Adaptive Control and Signal Processing* 24(10):850–866, 2010. DOI 10.1002/acs.1176 † 标题 | `control_normalization/papers/BADA-overview_Nuic2010_…pdf` | 经典物理法（Total Energy Model）；可与 EUROCONTROL, *BADA 3 User Manual rev. 3.7*, 2009（同目录 `BADA3-UM_EUROCONTROL2009_…pdf`）并引 |

**(b) 确定性序列模型**（基线类别 2：LSTM / TCN / Transformer）

| # | 完整引用 | 路径 | 用途 |
|---|---|---|---|
| ★ | W. Zeng, Z. Quan, Z. Zhao, C. Xie, X. Lu, "A Deep Learning Approach for Aircraft Trajectory Prediction in Terminal Airspace," *IEEE Access* 8:151250–151266, 2020. DOI 10.1109/ACCESS.2020.3016289 | `prediction_horizons/papers/Zeng2020_…pdf` | 终端区（广州白云）ADS-B 深度学习预测的典型设置 |
| ★ | S. Yoon, K. Lee, "Multi-Agent Inverted Transformer for Flight Trajectory Prediction" (MAIFormer), *IEEE T-ITS*, 2026. DOI 10.1109/TITS.2026.3689290; arXiv:2509.21004 | `prediction_horizons/papers/MAIFormer_YoonLee2026_…pdf` | 近期 iTransformer 类终端区（仁川）多机预测，对应本仓库的 Transformer 基线 |
| | L. Ma, S. Tian, "A Hybrid CNN-LSTM Model for Aircraft 4D Trajectory Prediction," *IEEE Access* 8:134668–134680, 2020. DOI 10.1109/ACCESS.2020.3010963 | `prediction_horizons/papers/MaTian2020_…pdf` | **注意是航路（青岛→北京），不是终端区**；作 4D 预测的先例 |
| | Y. Pang, N. Xu, Y. Liu, "Aircraft Trajectory Prediction using LSTM Neural Network with Embedded Convolutional Layer," *Annual Conf. of the PHM Society* 11(1), 2019. DOI 10.36001/phmconf.2019.v11i1.849 | `multimodal_intent/papers/PangXuLiu2019_…pdf` | 择引；单假设 LSTM，输入飞行计划 + 天气 |
| | L. Huang, S. Zhang, Y. Zhang, Y. Zhang, Y. Yin, "Aircraft Landing Time Prediction with Deep Learning on Trajectory Images," *13th SESAR Innovation Days (SIDS 2023)*; arXiv:2401.01083 | `prediction_horizons/papers/Huang2023_…pdf` | 到达时间维度的先例（新加坡樟宜，以入口时间为目标），对应评价维度 1 |

**(c) 概率 / 生成模型**（基线类别 3）

| # | 完整引用 | 路径 | 用途 |
|---|---|---|---|
| ★ | Y. Liu, M. Hansen, "Predicting Aircraft Trajectories: A Deep Generative Convolutional Recurrent Neural Networks Approach" (DeepTP), arXiv:1812.11670, 2018（PDF 未标注刊物） | `multimodal_intent/papers/DeepTP_…pdf` | 终端区深度生成式预测的开创工作（GMM + beam search） |
| ★ | J. Xiang, J. Chen, "Data-driven Probabilistic Trajectory Learning with High Temporal Resolution in Terminal Airspace," arXiv:2409.17359, 2024 | `prediction_horizons/papers/XiangChen2024_…pdf` | 1 s 分辨率的终端区概率模型，数据设定最接近本文 |
| ★ | Prutsch, Schinagl, Possegger, "ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace," *ICRA 2026*; arXiv:2603.16550 | `procedure_hard_constraints/papers/prutsch2026_ascent_terminal_tp.pdf` | K 假设 + WTA 的终端区预测；仍直接预测坐标 |
| | Petit, Torun, Brusset, Kam, Bayen, "FlowATC: Aircraft Trajectory Prediction via Flow Matching," 2026 | `controller_instruction_learning/papers/FlowATC_…pdf` | 择引：flow matching 航迹生成（全部作者名与刊物以 README #15 为准） |

**(d) 上下文条件（跑道 / 天气 / 交通）**，Proposal 明确提到

| # | 完整引用 | 路径 | 用途 |
|---|---|---|---|
| ★ | J. Patrikar, B. Moon, J. Oh, S. Scherer, "Predicting Like A Pilot: Dataset and Method to Predict Socially-Aware Aircraft Trajectories in Non-Towered Terminal Airspace" (TrajAir / TrajAirNet), *ICRA 2022*. DOI 10.1109/ICRA46639.2022.9811972; arXiv:2109.15158 | `hierarchical_prediction/papers/TrajAirNet_…pdf` | 条件化于交通的终端区预测 |
| | Hodgkin, Pepper, Thomas, "Conditioning Aircraft Trajectory Prediction on Meteorological Data with a Physics-Informed ML Approach," *AIAA SciTech 2026*. DOI 10.2514/6.2026-1792; arXiv:2601.03152 | `control_normalization/papers/MetConditioning_Hodgkin2026_…pdf` | 条件化于气象 |

> **空白**：以上方法几乎都直接预测未来坐标，物理可行性与程序合规靠隐式学习；`prediction_horizons/README.md` 还指出没有工作把到达航迹飞到跑道入口并在那里评分。

## 2.2 物理信息与可微运动学

| # | 完整引用 | 路径 | 用途 |
|---|---|---|---|
| ★ | M. Raissi, P. Perdikaris, G. E. Karniadakis, "Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations," arXiv:1711.10561, 2017（后续合并发表于 *J. Comput. Phys.* 378, 2019）† | `../models/PINN/Physics Informed Deep Learning (Part I)….pdf` | PINN 的源头；说明本文是"运动学积分 + 惩罚"，不是 PDE 残差 |
| ★ | H. Cui, T. Nguyen, F.-C. Chou, T.-H. Lin, J. Schneider, D. Bradley, N. Djuric, "Deep Kinematic Models for Kinematically Feasible Vehicle Trajectory Predictions," *ICRA 2020*. DOI 10.1109/ICRA40945.2020.9197560 | `control_normalization/papers/DKM_Cui2020_…pdf` | 预测加速度与转向，经运动学模型积分得轨迹的直接先例（车辆域） |
| ★ | T. Salzmann, B. Ivanovic, P. Chakravarty, M. Pavone, "Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data," *ECCV 2020*; arXiv:2001.03093 † 作者 | `control_normalization/papers/Trajectron++_Salzmann2020_…pdf` | 同样预测控制并经动力学积分，且含多智能体 |
| ★ | Hodgkin et al., "Probabilistic Simulation of Aircraft Descent via a Physics-Informed Machine Learning Approach," arXiv:2504.02529, 2025 † 作者与标题 | `control_normalization/papers/DescentPIML_Hodgkin2025_…pdf` | 航空域内最接近的"物理 + ML 概率生成"（学习阻力与 CAS 函数，参数化 BADA 方程） |
| | Pepper, Thomas, "Learning Generative Models for Climbing Aircraft from Radar Data," *J. Aerospace Information Systems* 21(6):474–481, 2024. DOI 10.2514/1.I011359; arXiv:2309.14941 | `control_normalization/papers/ClimbGenerative_Pepper2024_…pdf` | 择引：对 BADA 推力的学习修正；讨论参数的不可辨识性 |
| | Alligier, Gianazza, Durand, "Learning the Aircraft Mass and Thrust to Improve the Ground-Based Trajectory Prediction of Climbing Flights," *Transportation Research Part C* 36:45–60, 2013. DOI 10.1016/j.trc.2013.08.006 | `control_normalization/papers/MassThrustLearning_Alligier2013_…pdf` | 择引：从历史学习推力设定与等效质量 |

> **空白**：已有物理信息工作集中在爬升/下降/巡航，没有一个针对带程序约束、飞到跑道入口的进近。

## 2.3 约束的可微表达

| # | 完整引用 | 路径 | 用途 |
|---|---|---|---|
| ★ | P. Márquez-Neila, M. Salzmann, P. Fua, "Imposing Hard Constraints on Deep Networks: Promises and Limitations," arXiv:1706.02025, 2017 | `procedure_hard_constraints/papers/marquezneila_hard_constraints_1706.02025.pdf` | 论证软惩罚不保证满足约束 |
| ★ | F. Fioretto, T. W. K. Mak, P. Van Hentenryck, "Lagrangian Duality for Constrained Deep Learning," *ECML-PKDD 2020* † 作者 | `procedure_hard_constraints/papers/fioretto2020_lagrangian_duality_constrained_dl.pdf` | 约束学习的对偶/惩罚框架 |
| ★ | L. Chamon, A. Ribeiro, "Probably Approximately Correct Constrained Learning," *NeurIPS 2020* | `procedure_hard_constraints/papers/chamon2020_pac_constrained_learning.pdf` | 约束学习的理论依据 |
| | W. Xiao et al., "BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control," *IEEE T-RO*, 2023; arXiv:2111.11277 † 标题 | `procedure_hard_constraints/papers/xiao2023_barriernet_2111.11277.pdf` | 择引：可微障碍函数层（硬约束一侧） |
| | H. Kervadec et al., "Constrained Deep Networks: Lagrangian Optimization via Log-Barrier Extensions," *ICPR 2022* | `procedure_hard_constraints/papers/kervadec2022_log_barrier_extensions.pdf` | 择引：惩罚形式的选择 |

> **空白**：约束学习与安全层的工作在机器人和驾驶域；航空轨迹预测里尚无把 LPV 走廊、下滑道窗口等程序规则作为可微惩罚的做法（见 `procedure_hard_constraints/README.md` 的五条结论）。

## 2.4 多机交互与间隔

| # | 完整引用 | 路径 | 用途 |
|---|---|---|---|
| ★ | FAA, *Order JO 7110.65BB Air Traffic Control*, Change 3, Chap. 5 Sec. 5 "Radar Separation"（2026-07-09 生效） | `arrival_separation/papers/7110.65BB_chap5_section_5_Radar_Separation.html` | 间隔最小值的原文来源，评价维度 3 的定义依据，须引段号 |
| ★ | H. Erzberger, Y. Itoh, *Design Principles and Algorithms for Air Traffic Arrival Scheduling*, NASA TP-2014 † 编号 | `runway_assignment/nasa_eurocontrol/NASA_TP-2014_Erzberger_Itoh_…pdf` | 到达调度/协调的经典 |
| | Lee, Balakrishnan, "Tradeoffs in Scheduling Terminal-Area Operations," *Proc. IEEE*, 2008 | `runway_assignment/papers/Lee_Balakrishnan_2008_…pdf` | 择引：优化类到达调度基线 |
| ★ | K. Kim, S. Yoon, K. Lee, "Probabilistic Multi-Agent Aircraft Landing Time Prediction," *AIAA SciTech 2026*; arXiv:2512.08281 | `multi_agent_interaction/papers/MALTP_Kim2025_…pdf` | 多机 + 到达时间的概率预测 |
| ★ | S. Jung, A. Hardy, M. J. Kochenderfer, "Inferring Traffic Models in Terminal Airspace from Flight Tracks and Procedures," arXiv:2303.09981（v3, 2025；PDF 未标注刊物） | `multi_agent_interaction/papers/PairwiseTerminal_Jung2023_…pdf` | 从航迹与程序推断终端区成对交互 |
| ★ | D. J. Groot, J. Ellerbroek, J. M. Hoekstra, "Using Relative State Transformer Models for Multi-Agent Reinforcement Learning in Air Traffic Control," *SESAR Innovation Days 2022*, pp. 1–9 | `multi_agent_interaction/papers/RelStateTransformer_Groot2022_…pdf` | 相对状态注意力的航空多机模型 |
| | M. Brittain, X. Yang, P. Wei, "A Deep Multi-Agent Reinforcement Learning Approach to Autonomous Separation Assurance," arXiv:2003.08353, 2020 | `multi_agent_interaction/papers/D2MAV-A_Brittain2020_…pdf` | 择引：多智能体 RL 间隔保持 |
| ★ | S. Suo, S. Regalado, S. Casas, R. Urtasun, "TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors," *CVPR 2021*, pp. 10400–10409; arXiv:2101.06557 | `multi_agent_interaction/papers/TrafficSim_Suo2021_…pdf` | 联合生成并显式惩罚碰撞的先例 |

> **空白**：多机研究要么只预测着陆时间，要么用 RL 做间隔而不做航迹预测；"联合生成 + 间隔约束 + 到跑道入口评分"未见。

## 2.5 序列生成与闭环训练（可选）

| # | 完整引用 | 路径 | 用途 |
|---|---|---|---|
| ★ | A. Seff, B. Cera, D. Chen, M. Ng, A. Zhou, N. Nayakanti, K. S. Refaat, R. Al-Rfou, B. Sapp, "MotionLM: Multi-Agent Motion Forecasting as Language Modeling," *ICCV 2023*; arXiv:2309.16534 | `manoeuvre_tokens/papers/MotionLM_Seff2023_…pdf` | 多智能体运动当语言建模的生成式范式 |
| ★ | Z. Zhang, P. Karkus, M. Igl, W. Ding, Y. Chen, B. Ivanovic, M. Pavone, "Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models" (CAT-K), *CVPR 2025*; arXiv:2412.05334 | `trajectory_as_language/papers/CATK_Zhang2025_…pdf` | 闭环微调：下一词预训练之外让模型"说到做到" |

篇幅紧时 2.5 并入 2.1(c) 一句带过。

## 2.6 数据集

| # | 完整引用 | 路径 | 用途 |
|---|---|---|---|
| ★ | J. Patrikar et al., "TartanAviation: Image, Speech, and ADS-B Trajectory Datasets for Terminal Airspace Operations," 2024 † 作者与刊物 | `controller_instruction_learning/papers/TartanAviation_Patrikar2024_…pdf` | 开放的终端空域 ADS-B 数据集（CC BY 4.0），与本文数据设定可比 |

---

## 篇幅与缺口

- 只引 ★ 约 25 篇，已覆盖三类基线、三个核心组件和三个评价维度。
- 三个评价维度的依据：到达时间 = Huang 2023 + MALTP；运行合规 = 7110.65 + 约束学习三篇；多机安全 = 7110.65 Ch.5 + TrafficSim。
- 仓库里没有、但 Proposal 提到的内容：OpenSky Network 的原始引用、USGS DEM、IATA 安全报告，需另补。
- 旧版 `Proposal.md` 点名的基线 GMM / VAE / TCN，仓库里没有对应的原始论文，作基线引用时需另补。
