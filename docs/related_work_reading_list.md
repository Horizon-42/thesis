# Related Work 论文清单

依据：`docs/Final Proposal.md`（物理约束的生成式 4D 进近航迹预测：控制参数生成 → 可微运动学 → 可微惩罚约束 → 多机协调；对比三类基线；三个评价维度）。
范围：只选仓库 `docs/literature/` 与 `docs/models/` 中**已有**的论文，共 33 篇。路径相对于 `docs/literature/`（PDF 不入 git，缺失时跑对应文件夹的 `download.sh`）。
★ = 必引（论证链上不可缺）；其余为择引（篇幅允许时补充）。

## 组织结构（对应 Related Work 的小节）

```
2.1 终端空域轨迹预测的演进        → 对应 Proposal 第 3 段，并给出三类基线
      物理/优化 → 确定性序列模型 → 概率/生成模型 → 上下文条件
2.2 物理信息与可微运动学          → 对应核心组件 1：生成控制参数，再由运动学积分出轨迹
2.3 约束的可微表达                → 对应核心组件 2：物理边界与运行规则作为可微惩罚
2.4 多机交互与间隔                → 对应核心组件 3：多机协调与间隔评价
2.5 序列生成与闭环训练            → 生成式建模的方法背景（可选，一小段）
2.6 数据与评价                    → ADS-B 数据集、评价维度
```

每节结尾的"空白"一句即本文的定位，可直接改写成小节收尾句。

---

## 2.1 终端空域轨迹预测的演进

**(a) 物理 / 优化基线**（基线类别 1）

| 论文 | 路径 | 用途 |
|---|---|---|
| ★ OpenAP — Sun 2020 | `control_normalization/papers/OpenAP_Sun2020_...pdf` | 开源性能模型；本项目运动学与推力/阻力参数的来源 |
| ★ BADA 3 用户手册 + Nuic 2010 概述 | `control_normalization/papers/BADA3-UM_EUROCONTROL2009_...pdf`、`BADA-overview_Nuic2010_...pdf` | 经典"物理法"轨迹预测的代表；与 OpenAP 并列引用即可 |

**(b) 确定性序列模型**（基线类别 2：LSTM / TCN / Transformer）

| 论文 | 路径 | 用途 |
|---|---|---|
| ★ Zeng 2020 终端区深度学习预测 | `prediction_horizons/papers/Zeng2020_...pdf` | 终端区 LSTM 预测的典型设置 |
| Ma & Tian 2020 CNN-LSTM 4D 预测 | `prediction_horizons/papers/MaTian2020_...pdf` | 同上，4D（含时间）预测 |
| Pang 2019 LSTM + 卷积 | `multimodal_intent/papers/PangXuLiu2019_...pdf` | 择引 |
| ★ MAIFormer — Yoon & Lee 2026 | `prediction_horizons/papers/MAIFormer_YoonLee2026_...pdf` | 近期 iTransformer 类终端区多机预测，对应本仓库的 Transformer 基线 |
| Huang 2023 着陆时间预测 | `prediction_horizons/papers/Huang2023_...pdf` | "到达时间"维度的先例（对应评价维度 1） |

**(c) 概率 / 生成模型**（基线类别 3）

| 论文 | 路径 | 用途 |
|---|---|---|
| ★ DeepTP — Liu & Hansen 2018 | `multimodal_intent/papers/DeepTP_...pdf` | 深度生成式航迹预测的开创工作 |
| ★ Xiang & Chen 2024 概率轨迹学习 | `prediction_horizons/papers/XiangChen2024_...pdf` | 高时间分辨率终端区概率模型，最接近的数据设定 |
| ★ ASCENT — Prutsch 2026 | `procedure_hard_constraints/papers/prutsch2026_ascent_terminal_tp.pdf` | 终端区 K 假设 WTA 预测，仍直接预测坐标 |
| FlowATC — Petit 2026 flow matching | `controller_instruction_learning/papers/FlowATC_...pdf` | 择引：flow matching 的航迹生成 |

**上下文条件（跑道/天气/交通）**：Proposal 明确提到。

| 论文 | 路径 | 用途 |
|---|---|---|
| ★ TrajAirNet — Patrikar 2021 | `hierarchical_prediction/papers/TrajAirNet_...pdf` | 条件化于交通的终端区预测 |
| Hodgkin 2026 气象条件化 | `control_normalization/papers/MetConditioning_Hodgkin2026_...pdf` | 条件化于气象的物理信息 ML |

> **空白**：以上方法几乎都直接预测未来坐标，物理可行性与程序合规靠隐式学习；`prediction_horizons/README.md` 还指出没有工作把到达航迹飞到跑道入口并在那里评分。

## 2.2 物理信息与可微运动学

| 论文 | 路径 | 用途 |
|---|---|---|
| ★ PINN — Raissi 2019 | `models/PINN/Physics Informed Deep Learning (Part I)...pdf` | 物理信息学习的源头；说明本文与 PDE 型 PINN 的区别（本文是运动学积分 + 惩罚，而非 PDE 残差） |
| ★ Deep Kinematic Models — Cui 2020 | `control_normalization/papers/DKM_Cui2020_...pdf` | 预测控制量、经运动学模型积分得轨迹的直接先例（车辆域） |
| ★ Trajectron++ — Salzmann 2020 | `control_normalization/papers/Trajectron++_Salzmann2020_...pdf` | 同样预测控制并经动力学积分，且含多智能体 |
| ★ Hodgkin 2025 下降段物理信息 ML | `control_normalization/papers/DescentPIML_Hodgkin2025_...pdf` | 航空域内最接近的"物理 + ML 概率生成" |
| Pepper 2024 爬升段生成模型 | `control_normalization/papers/ClimbGenerative_Pepper2024_...pdf` | 择引：由雷达数据学习的生成式航空模型 |
| Alligier 2013 质量与推力学习 | `control_normalization/papers/MassThrustLearning_Alligier2013_...pdf` | 择引：从数据学习性能参数 |

> **空白**：已有物理信息工作多在爬升/下降/巡航，没有一个针对"带程序约束的进近到跑道入口"。

## 2.3 约束的可微表达

| 论文 | 路径 | 用途 |
|---|---|---|
| ★ Márquez-Neila 2017 硬约束 | `procedure_hard_constraints/papers/marquezneila_hard_constraints_1706.02025.pdf` | 指出"软惩罚不保证满足约束"，论证惩罚法的局限 |
| ★ Fioretto 2020 拉格朗日对偶 | `procedure_hard_constraints/papers/fioretto2020_lagrangian_duality_constrained_dl.pdf` | 约束学习的对偶/惩罚框架 |
| ★ Chamon 2020 约束学习 | `procedure_hard_constraints/papers/chamon2020_pac_constrained_learning.pdf` | 约束学习的理论依据 |
| BarrierNet — Xiao 2023 | `procedure_hard_constraints/papers/xiao2023_barriernet_2111.11277.pdf` | 择引：可微障碍函数层（硬约束一侧） |
| Kervadec 2022 log-barrier | `procedure_hard_constraints/papers/kervadec2022_log_barrier_extensions.pdf` | 择引：惩罚形式的选择 |

> **空白**：约束学习/安全层的工作在机器人与驾驶域；航空轨迹预测尚无把 LPV 走廊、下滑道窗口等程序规则作为可微惩罚的做法（见 `procedure_hard_constraints/README.md` 五条结论）。

## 2.4 多机交互与间隔

| 论文 | 路径 | 用途 |
|---|---|---|
| ★ 7110.65BB Ch.5 Sec.5 雷达间隔 | `arrival_separation/papers/7110.65BB_chap5_section_5_Radar_Separation.html` | 间隔最小值的原文来源（评价维度 3 的定义依据，须引段号） |
| ★ Erzberger & Itoh 2014 到达调度原则 | `runway_assignment/nasa_eurocontrol/NASA_TP-2014_Erzberger_Itoh_...pdf` | 到达调度/协调的经典 |
| Lee & Balakrishnan 2008 终端区调度折中 | `runway_assignment/papers/Lee_Balakrishnan_2008_...pdf` | 择引：优化类到达调度基线 |
| ★ MALTP — Kim 2025 多机着陆时间 | `multi_agent_interaction/papers/MALTP_Kim2025_...pdf` | 多机 + 到达时间的概率预测 |
| ★ Jung 2023 终端空域成对交通模型 | `multi_agent_interaction/papers/PairwiseTerminal_Jung2023_...pdf` | 从航迹与程序推断终端区成对交互 |
| ★ RelStateTransformer — Groot 2022 | `multi_agent_interaction/papers/RelStateTransformer_Groot2022_...pdf` | 相对状态注意力的航空多机模型 |
| D2MAV-A — Brittain 2020 | `multi_agent_interaction/papers/D2MAV-A_Brittain2020_...pdf` | 择引：多智能体 RL 间隔保持（"多智能体协调"一支） |
| ★ TrafficSim — Suo 2021 | `multi_agent_interaction/papers/TrafficSim_Suo2021_...pdf` | 多智能体联合生成并显式惩罚碰撞的先例 |

> **空白**：多机研究要么只预测着陆时间，要么用 RL 做间隔而不做航迹预测；"联合生成 + 间隔约束 + 到跑道入口评分"未见。

## 2.5 序列生成与闭环训练（可选，一小段）

| 论文 | 路径 | 用途 |
|---|---|---|
| ★ MotionLM — Seff 2023 | `manoeuvre_tokens/papers/MotionLM_...pdf` | 把多智能体运动当语言建模的生成式范式 |
| ★ CAT-K — Zhang 2025 | `trajectory_as_language/papers/CATK_...pdf` | 闭环微调：下一词预训练之外让模型"说到做到" |

若篇幅紧，2.5 可并入 2.1(c) 一句带过。

## 2.6 数据与评价

| 论文 | 路径 | 用途 |
|---|---|---|
| ★ TartanAviation — Patrikar 2024 | `controller_instruction_learning/papers/TartanAviation_...pdf` | 终端空域 ADS-B 数据集，与本文数据设定可比 |

---

## 控制篇幅的建议

- 只引 ★：约 25 篇，已覆盖三类基线、三个核心组件与三个评价维度。
- 评价维度 1/2/3 分别对应 Huang 2023 + MALTP（到达时间）、7110.65 + 约束学习（合规）、7110.65 Ch.5 + TrafficSim（间隔）。
- 仓库中没有、但 Proposal 提到的内容：OpenSky 数据网络的原始引用、USGS DEM、IATA 安全报告。这三项需另行补引，未列入上表。
- Proposal.md（旧版）中点名的基线 GMM / VAE / TCN，仓库里无对应原始论文；若要作为基线引用，需另补。
